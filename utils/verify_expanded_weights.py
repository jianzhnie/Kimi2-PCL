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


def verify_non_layer_params(orig_loader, exp_loader):
    print("\nVerifying non-layer parameters (Embedding, Final Norm, etc.)")
    mismatches = []
    non_layer_params = [k for k in orig_loader.weight_map.keys() if get_layer_index(k) is None]
    
    for name in tqdm(non_layer_params, desc="Non-layer Params"):
        t_orig = orig_loader.get_tensor(name)
        t_exp = exp_loader.get_tensor(name)
        
        if t_exp is None:
            mismatches.append(f"Missing in expanded: {name}")
        elif not torch.equal(t_orig, t_exp):
            mismatches.append(f"Value mismatch: {name}")
    return mismatches


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
    
    # 2. Check non-layer params first
    mismatches.extend(verify_non_layer_params(orig_loader, exp_loader))
    
    # 3. Check original layers (0 to original_layers-1)
    print(f"Checking original layers [0, {original_layers})...")
    orig_params = [k for k in orig_loader.weight_map.keys() if get_layer_index(k) is not None]
    
    for name in tqdm(orig_params, desc="Original Layers"):
        l_idx = get_layer_index(name)
        if l_idx >= original_layers: continue
        
        t_orig = orig_loader.get_tensor(name)
        t_exp = exp_loader.get_tensor(name)
        
        if t_exp is None:
            mismatches.append(f"Missing in expanded: {name}")
        elif not torch.equal(t_orig, t_exp):
            mismatches.append(f"Value mismatch: {name}")

    # 4. Check new layers (original_layers to target_layers-1)
    print(f"Checking expanded layers [{original_layers}, {target_layers})...")
    for i, src_idx in enumerate(tqdm(mapping, desc="Expanded Layers")):
        new_idx = original_layers + i
        
        src_prefix = f"model.layers.{src_idx}."
        src_params = [k for k in orig_loader.weight_map.keys() if k.startswith(src_prefix)]
        
        for src_name in src_params:
            exp_name = set_layer_index(src_name, new_idx)
            
            t_src = orig_loader.get_tensor(src_name)
            t_exp = exp_loader.get_tensor(exp_name)
            
            if t_exp is None:
                mismatches.append(f"Missing in expanded: {exp_name}")
            elif not torch.equal(t_src, t_exp):
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

    # 1. Check all parameters that are NOT experts or routers (Attention, Norms, Embedding, etc.)
    print("Checking non-expert/non-router parameters...")
    all_orig_params = orig_loader.weight_map.keys()
    for name in tqdm(all_orig_params, desc="General Params"):
        if get_expert_info(name) is not None or is_router_param(name):
            continue
            
        t_orig = orig_loader.get_tensor(name)
        t_exp = exp_loader.get_tensor(name)
        
        if t_exp is None:
            mismatches.append(f"Missing in expanded: {name}")
        elif not torch.equal(t_orig, t_exp):
            mismatches.append(f"Value mismatch: {name}")
    
    # 2. Check experts
    expert_params = [k for k in orig_loader.weight_map.keys() if get_expert_info(k) is not None]
    
    for name in tqdm(expert_params, desc="Experts"):
        l_idx, e_idx, rest = get_expert_info(name)
        t_orig = orig_loader.get_tensor(name)
        
        if e_idx < orig_experts:
            # Routed expert: Check original position and all duplicated copies
            for f in range(expansion_factor):
                target_idx = e_idx + f * orig_experts
                target_name = f"model.layers.{l_idx}.mlp.experts.{target_idx}.{rest}"
                t_exp = exp_loader.get_tensor(target_name)
                if t_exp is None:
                    mismatches.append(f"Missing in expanded: {target_name}")
                elif not torch.equal(t_orig, t_exp):
                    mismatches.append(f"Value mismatch: {target_name} (should match original expert {e_idx})")
        else:
            # Zero-shot expert: Check shifted position
            new_e_idx = e_idx + (exp_experts - orig_experts)
            new_name = f"model.layers.{l_idx}.mlp.experts.{new_e_idx}.{rest}"
            t_exp = exp_loader.get_tensor(new_name)
            if t_exp is None or not torch.equal(t_orig, t_exp):
                mismatches.append(f"Value mismatch: {new_name} (should match original zero-shot expert {e_idx})")

    # 3. Check routers
    router_params = [k for k in orig_loader.weight_map.keys() if is_router_param(k)]
    for name in tqdm(router_params, desc="Routers"):
        t_orig = orig_loader.get_tensor(name)
        t_exp = exp_loader.get_tensor(name)
        
        if t_exp is None:
            mismatches.append(f"Missing router in expanded: {name}")
            continue
            
        # Router has shape [num_routed + zero, hidden]
        # Check expanded real experts part
        real_orig = t_orig[:orig_experts]
        real_exp = t_exp[:exp_experts]
        
        for f in range(expansion_factor):
            part = real_exp[f*orig_experts : (f+1)*orig_experts]
            if not torch.equal(real_orig, part):
                mismatches.append(f"Router value mismatch in real part (factor {f}): {name}")
        
        # Check zero experts part (should be at the end)
        if zero_experts > 0:
            zero_orig = t_orig[orig_experts:]
            zero_exp = t_exp[exp_experts:]
            if not torch.equal(zero_orig, zero_exp):
                mismatches.append(f"Router value mismatch in zero part: {name}")

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
