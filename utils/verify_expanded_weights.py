#!/usr/bin/env python3
"""
Verify that an expanded model's weights match the original model's weights.

Supports:
1. Layer Expansion: Verifies original layers and duplicated layers.
2. MoE Expert Expansion: Verifies routers and expert weights.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from tqdm import tqdm


def load_index(model_dir: Path):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            return json.load(f)
    return None


def get_layer_index(param_name: str) -> int | None:
    m = re.search(r"model\.layers\.(\d+)\.", param_name)
    if m:
        return int(m.group(1))
    return None


def get_expert_info(param_name: str) -> tuple[int, int, str] | None:
    m = re.search(r"model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.(.*)", param_name)
    if m:
        return int(m.group(1)), int(m.group(2)), m.group(3)
    return None


def is_router_param(param_name: str) -> bool:
    suffixes = (
        "mlp.router.classifier.weight",
        "mlp.gate.weight",
        "mlp.router.e_score_correction_bias",
        "mlp.gate.e_score_correction_bias",
    )
    return param_name.endswith(suffixes)


class ModelWeightLoader:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.index = load_index(model_dir)
        self.weight_map = self.index["weight_map"] if self.index else None
        
        if not self.weight_map:
            # Single file model
            files = list(model_dir.glob("*.safetensors"))
            if not files:
                raise FileNotFoundError(f"No safetensors found in {model_dir}")
            self.weight_map = {k: files[0].name for k in safe_open(files[0], framework="pt").keys()}
            
        self.shards = {}
        # Group parameters by shard for faster access
        self.params_by_shard = defaultdict(list)
        for name, shard in self.weight_map.items():
            self.params_by_shard[shard].append(name)

    def get_tensor(self, name: str):
        if name not in self.weight_map:
            return None
        shard_name = self.weight_map[name]
        if shard_name not in self.shards:
            self.shards[shard_name] = safe_open(self.model_dir / shard_name, framework="pt")
        return self.shards[shard_name].get_tensor(name)


def set_layer_index(param_name: str, new_index: int) -> str:
    """Change the layer index in a parameter name. e.g. model.layers.0.xxx → model.layers.5.xxx"""
    return re.sub(
        r"model\.layers\.(\d+)\.",
        f"model.layers.{new_index}.",
        param_name,
    )


def get_source_name_for_layers(exp_name, original_layers, target_layers, mapping):
    """Determine the source parameter name in the original model for a given expanded parameter name."""
    l_idx = get_layer_index(exp_name)
    if l_idx is None:
        return exp_name # Non-layer param
    
    if l_idx < original_layers:
        return exp_name # Original layer param
    
    # New layer param
    new_layer_offset = l_idx - original_layers
    if new_layer_offset < len(mapping):
        src_idx = mapping[new_layer_offset]
        return set_layer_index(exp_name, src_idx)
    
    return None


def get_source_name_for_experts(exp_name, original_experts, target_experts):
    """Determine the source parameter name in the original model for an expanded expert model."""
    info = get_expert_info(exp_name)
    if info:
        l_idx, e_idx, rest = info
        if e_idx < original_experts:
            return exp_name # Original expert (though we check duplicates too)
        
        # Could be a duplicate of a routed expert or a shifted zero-shot expert
        # We need to know if it's in the [original_experts, target_experts) range
        if e_idx < target_experts:
            # Duplicate of a routed expert
            src_e_idx = e_idx % original_experts
            return f"model.layers.{l_idx}.mlp.experts.{src_e_idx}.{rest}"
        else:
            # Shifted zero-shot expert
            src_e_idx = e_idx - (target_experts - original_experts)
            return f"model.layers.{l_idx}.mlp.experts.{src_e_idx}.{rest}"
            
    if is_router_param(exp_name):
        return exp_name # Routers are checked specially in verify_experts due to shape change
        
    return exp_name


def verify_layers(orig_loader, exp_loader, original_layers, target_layers, copy_source):
    print(f"\nVerifying Layer Expansion: {original_layers} -> {target_layers}")
    
    # 1. Parse copy source
    num_new = target_layers - original_layers
    if copy_source is None or copy_source.lower() == "seq":
        mapping = [i % original_layers for i in range(num_new)]
    elif "," in copy_source:
        mapping = [int(x) for x in copy_source.split(",")]
    else:
        mapping = [int(copy_source)] * num_new
    
    mismatches = []
    
    # 2. Iterate shard-by-shard through the EXPANDED model
    # This ensures we read the expanded model linearly (the bottleneck)
    exp_shards = sorted(exp_loader.params_by_shard.keys())
    
    for shard_name in tqdm(exp_shards, desc="Verifying Expanded Shards"):
        with safe_open(exp_loader.model_dir / shard_name, framework="pt") as sf_exp:
            for exp_name in exp_loader.params_by_shard[shard_name]:
                src_name = get_source_name_for_layers(exp_name, original_layers, target_layers, mapping)
                
                if src_name is None:
                    mismatches.append(f"Could not determine source for: {exp_name}")
                    continue
                
                t_exp = sf_exp.get_tensor(exp_name)
                t_orig = orig_loader.get_tensor(src_name)
                
                if t_orig is None:
                    mismatches.append(f"Source parameter {src_name} not found for: {exp_name}")
                elif not torch.equal(t_exp, t_orig):
                    mismatches.append(f"Value mismatch: {exp_name} (should match {src_name})")

    return mismatches


def verify_experts(orig_loader, exp_loader):
    print("\nVerifying MoE Expert Expansion")
    
    with open(orig_loader.model_dir / "config.json") as f:
        orig_config = json.load(f)
    with open(exp_loader.model_dir / "config.json") as f:
        exp_config = json.load(f)
    
    orig_experts = orig_config.get("n_routed_experts") or orig_config.get("n_experts")
    exp_experts = exp_config.get("n_routed_experts") or exp_config.get("n_experts")
    zero_experts = orig_config.get("zero_expert_num", 0)
    
    print(f"Experts: {orig_experts} -> {exp_experts}, Zero experts: {zero_experts}")
    
    mismatches = []
    expansion_factor = exp_experts // orig_experts

    # Iterate shard-by-shard through the EXPANDED model
    exp_shards = sorted(exp_loader.params_by_shard.keys())
    
    for shard_name in tqdm(exp_shards, desc="Verifying Expanded Shards"):
        with safe_open(exp_loader.model_dir / shard_name, framework="pt") as sf_exp:
            for exp_name in exp_loader.params_by_shard[shard_name]:
                # 1. Special case: Routers (shape changed, cannot use simple torch.equal)
                if is_router_param(exp_name):
                    t_exp = sf_exp.get_tensor(exp_name)
                    t_orig = orig_loader.get_tensor(exp_name)
                    
                    if t_orig is None:
                        mismatches.append(f"Source router {exp_name} not found")
                        continue
                        
                    # Check expanded real experts part
                    real_orig = t_orig[:orig_experts]
                    real_exp = t_exp[:exp_experts]
                    
                    for f in range(expansion_factor):
                        part = real_exp[f*orig_experts : (f+1)*orig_experts]
                        if not torch.equal(real_orig, part):
                            mismatches.append(f"Router value mismatch in real part (factor {f}): {exp_name}")
                    
                    # Check zero experts part
                    if zero_experts > 0:
                        zero_orig = t_orig[orig_experts:]
                        zero_exp = t_exp[exp_experts:]
                        if not torch.equal(zero_orig, zero_exp):
                            mismatches.append(f"Router value mismatch in zero part: {exp_name}")
                    continue

                # 2. Expert or general params
                src_name = get_source_name_for_experts(exp_name, orig_experts, exp_experts)
                
                t_exp = sf_exp.get_tensor(exp_name)
                t_orig = orig_loader.get_tensor(src_name)
                
                if t_orig is None:
                    mismatches.append(f"Source parameter {src_name} not found for: {exp_name}")
                elif not torch.equal(t_exp, t_orig):
                    mismatches.append(f"Value mismatch: {exp_name} (should match {src_name})")

    return mismatches


def main():
    parser = argparse.ArgumentParser(description="Verify expanded model weights")
    parser.add_argument("--orig_dir", type=str, required=True, help="Original model directory")
    parser.add_argument("--exp_dir", type=str, required=True, help="Expanded model directory")
    parser.add_argument("--type", type=str, choices=["layers", "experts"], required=True, help="Expansion type")
    
    # For layers
    parser.add_argument("--orig_layers", type=int, default=28)
    parser.add_argument("--target_layers", type=int, default=56)
    parser.add_argument("--copy_source", type=str, default="seq")
    
    args = parser.parse_args()
    
    orig_loader = ModelWeightLoader(Path(args.orig_dir))
    exp_loader = ModelWeightLoader(Path(args.exp_dir))
    
    if args.type == "layers":
        mismatches = verify_layers(orig_loader, exp_loader, args.orig_layers, args.target_layers, args.copy_source)
    else:
        mismatches = verify_experts(orig_loader, exp_loader)
        
    if mismatches:
        print(f"\n❌ Verification FAILED with {len(mismatches)} mismatches!")
        for m in mismatches[:20]:
            print(f"  - {m}")
        if len(mismatches) > 20:
            print(f"  ... and {len(mismatches) - 20} more")
    else:
        print("\n✅ Verification SUCCESSFUL! All weights match perfectly.")


if __name__ == "__main__":
    main()
