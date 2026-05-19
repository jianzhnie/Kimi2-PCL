#!/usr/bin/env python3
"""
Expand the number of MoE experts in a sharded safetensors model.

For a model with N experts, creates a 2N-expert version by:
1. Duplicating each expert weight (expert E and expert E+N will be identical initially)
2. Expanding the router classifier weights by concatenating them with themselves.
3. Expanding the score correction bias similarly.

Processes shards one at a time to keep memory usage manageable.
"""

import argparse
import json
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from tqdm import tqdm

from utils.shared import (
    EXPERT_COUNT_KEYS,
    auto_detect_shard_size,
    find_expert_count,
    get_expert_info,
    get_nbytes_from_meta,
    is_router_param,
    load_config,
    load_index,
    read_safetensors_header,
    tensor_nbytes,
)

TOPK_KEYS = ["moe_topk", "num_experts_per_tok", "top_k"]


def build_expert_target_map(
    original_experts: int,
    target_experts: int,
) -> dict[int, list[int]]:
    """Build source expert -> target expert indices for all newly added experts."""
    targets: dict[int, list[int]] = defaultdict(list)
    for new_idx in range(original_experts, target_experts):
        src_idx = new_idx % original_experts
        targets[src_idx].append(new_idx)
    return dict(targets)


def validate_expert_layout(index: dict, original_experts: int, zero_expert_num: int) -> dict[int, list[int]]:
    """Validate that each MoE layer has contiguous expert indices.

    Expected: [0, original_experts) or [0, original_experts + zero_expert_num).
    """
    experts_by_layer: dict[int, set[int]] = defaultdict(set)
    for param_name in index["weight_map"]:
        info = get_expert_info(param_name)
        if info is None:
            continue
        layer_idx, expert_idx, _ = info
        experts_by_layer[layer_idx].add(expert_idx)

    if not experts_by_layer:
        print(
            "ERROR: No expert parameters matching 'model.layers.<idx>.mlp.experts.<idx>.' "
            "were found in the index.",
            file=sys.stderr,
        )
        sys.exit(1)

    expected_routed = list(range(original_experts))
    expected_total = list(range(original_experts + zero_expert_num))

    validated: dict[int, list[int]] = {}
    for layer_idx, expert_indices in sorted(experts_by_layer.items()):
        actual = sorted(expert_indices)
        if actual != expected_routed and actual != expected_total:
            print(
                f"ERROR: Layer {layer_idx} has expert indices {actual[:8]}"
                f"{'...' if len(actual) > 8 else ''}, but expected contiguous "
                f"indices 0-{original_experts - 1} or 0-{original_experts + zero_expert_num - 1}.",
                file=sys.stderr,
            )
            sys.exit(1)
        validated[layer_idx] = actual
    return validated


def validate_router_shape(param_name: str, shape: list[int], total_routed: int) -> None:
    """Ensure router tensors have the expected first dimension (real experts + zero experts)."""
    if not shape:
        print(f"ERROR: Router tensor {param_name} has an empty shape.", file=sys.stderr)
        sys.exit(1)
    if shape[0] != total_routed:
        print(
            f"ERROR: Router tensor {param_name} has shape {shape}, expected first "
            f"dimension {total_routed} (n_routed_experts + zero_expert_num).",
            file=sys.stderr,
        )
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Expand MoE experts by duplicating weights and expanding routers"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the original model directory",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to output the expanded model",
    )
    parser.add_argument(
        "--target_experts",
        type=int,
        default=None,
        help="Target number of experts. Defaults to double the original.",
    )
    parser.add_argument(
        "--target_topk",
        type=int,
        default=None,
        help="Target moe_topk (number of activated experts per token). "
             "Defaults to unchanged.",
    )
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    if not model_dir.exists():
        print(f"ERROR: Model directory not found: {model_dir}", file=sys.stderr)
        sys.exit(1)

    # ── Load config & index ──────────────────────────────────────────────
    config = load_config(model_dir)
    index = load_index(model_dir)

    if not index:
        print("ERROR: Model is not sharded. This script handles sharded models.", file=sys.stderr)
        sys.exit(1)

    expert_count_key, original_experts, zero_expert_num = find_expert_count(config)
    if original_experts == 0:
        print(
            "ERROR: Could not find any expert count key in config.json. "
            f"Tried: {', '.join(EXPERT_COUNT_KEYS)}",
            file=sys.stderr,
        )
        sys.exit(1)

    total_routed = original_experts + zero_expert_num

    target_experts = args.target_experts if args.target_experts is not None else original_experts * 2

    if target_experts <= original_experts:
        print(
            f"ERROR: target_experts ({target_experts}) must be greater than "
            f"original_experts ({original_experts}).",
            file=sys.stderr,
        )
        sys.exit(1)

    if target_experts % original_experts != 0:
        print(f"ERROR: Target experts ({target_experts}) must be a multiple of original ({original_experts})")
        sys.exit(1)

    expansion_factor = target_experts // original_experts
    target_zero_expert_num = zero_expert_num * expansion_factor

    print(f"Expanding experts: {original_experts} -> {target_experts} (factor: {expansion_factor})")
    if zero_expert_num > 0:
        print(f"Zero experts: {zero_expert_num} -> {target_zero_expert_num} "
              f"(router dim: {total_routed} -> {target_experts + target_zero_expert_num})")

    shard_files = sorted(set(index["weight_map"].values()))
    experts_by_layer = validate_expert_layout(index, original_experts, zero_expert_num)
    source_to_targets = build_expert_target_map(original_experts, target_experts)
    print(f"Detected {len(experts_by_layer)} MoE layer(s) with {original_experts} experts each")

    # ── Update & write config ───────────────────────────────────────────
    updated_keys = []
    for key in EXPERT_COUNT_KEYS:
        if key in config:
            config[key] = target_experts
            updated_keys.append(key)

    if zero_expert_num > 0:
        config["zero_expert_num"] = target_zero_expert_num
        updated_keys.append("zero_expert_num")

    if args.target_topk is not None:
        topk_updated = False
        for key in TOPK_KEYS:
            if key in config:
                old_val = config[key]
                config[key] = args.target_topk
                print(f"Updating topk: {key} {old_val} → {args.target_topk}")
                updated_keys.append(key)
                topk_updated = True
                break
        if not topk_updated:
            config["moe_topk"] = args.target_topk
            print(f"Adding topk: moe_topk = {args.target_topk}")
            updated_keys.append("moe_topk")

    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"WARNING: Output directory already exists and is not empty: {output_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)
    print(f"Updated config.json written (keys updated: {', '.join(updated_keys)}).")

    # ── Pass 1: Scan headers to determine output layout ─────────────────
    print("\nPass 1/2: Scanning headers to determine output layout...")
    target_shard_size = auto_detect_shard_size(model_dir, shard_files)
    total_output_bytes = 0
    print(f"Target shard size: {target_shard_size / 1e9:.2f} GB")

    num_output_shards = 1
    current_bytes = 0
    total_original = 0
    total_duplicated = 0

    for shard_file in tqdm(shard_files, desc="Scanning"):
        shard_path = model_dir / shard_file
        if not shard_path.exists():
            tqdm.write(f"  WARNING: {shard_file} not found — skipping")
            continue

        header = read_safetensors_header(shard_path)
        for key, (dtype, shape) in header.items():
            nbytes = get_nbytes_from_meta(dtype, shape)

            if is_router_param(key):
                validate_router_shape(key, shape, total_routed)
                # Router expands: real experts (×expansion_factor) + zero experts (×expansion_factor)
                bytes_per_expert = nbytes // total_routed
                new_nbytes = bytes_per_expert * (target_experts + target_zero_expert_num)
                if new_nbytes + current_bytes > target_shard_size and current_bytes > 0:
                    num_output_shards += 1
                    current_bytes = 0
                current_bytes += new_nbytes
                total_output_bytes += new_nbytes
                total_original += 1
            elif info := get_expert_info(key):
                _, expert_idx, _ = info
                if nbytes + current_bytes > target_shard_size and current_bytes > 0:
                    num_output_shards += 1
                    current_bytes = 0
                current_bytes += nbytes
                total_output_bytes += nbytes
                total_original += 1

                if expert_idx < original_experts:
                    for _ in source_to_targets.get(expert_idx, []):
                        if nbytes + current_bytes > target_shard_size and current_bytes > 0:
                            num_output_shards += 1
                            current_bytes = 0
                        current_bytes += nbytes
                        total_output_bytes += nbytes
                        total_duplicated += 1
                elif zero_expert_num > 0:
                    for _ in range(expansion_factor - 1):
                        if nbytes + current_bytes > target_shard_size and current_bytes > 0:
                            num_output_shards += 1
                            current_bytes = 0
                        current_bytes += nbytes
                        total_output_bytes += nbytes
                        total_duplicated += 1
            else:
                if nbytes + current_bytes > target_shard_size and current_bytes > 0:
                    num_output_shards += 1
                    current_bytes = 0
                current_bytes += nbytes
                total_output_bytes += nbytes
                total_original += 1

    print(
        f"Output plan: {total_original:,} original + {total_duplicated:,} duplicated "
        f"= {total_original + total_duplicated:,} tensors"
    )
    print(
        f"Planned output size: {total_output_bytes / 1e9:.2f} GB across "
        f"{num_output_shards} shard(s) "
        f"(~{total_output_bytes / num_output_shards / 1e9:.2f} GB each)"
    )

    # ── Pass 2: Process and write ───────────────────────────────────────
    new_weight_map: dict[str, str] = {}
    output_shard_idx = 1
    current_tensors: dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush_shard():
        nonlocal output_shard_idx, current_tensors, current_bytes
        if not current_tensors:
            return
        shard_name = f"model-{output_shard_idx:05d}-of-{num_output_shards:05d}.safetensors"
        output_path = output_dir / shard_name
        save_file(current_tensors, str(output_path))
        for t_name in current_tensors:
            new_weight_map[t_name] = shard_name
        output_shard_idx += 1
        current_tensors.clear()
        current_bytes = 0

    print("\nPass 2/2: Processing and writing tensors...")
    for shard_file in tqdm(shard_files, desc="Input shards"):
        with safe_open(str(model_dir / shard_file), framework="pt", device="cpu") as sf:
            for key in sf.keys():
                tensor = sf.get_tensor(key)
                nbytes = tensor_nbytes(tensor)

                if is_router_param(key):
                    validate_router_shape(key, list(tensor.shape), total_routed)
                    if zero_expert_num > 0:
                        real_part = tensor[:original_experts]
                        zero_part = tensor[original_experts:]
                        expanded_real = torch.cat([real_part] * expansion_factor, dim=0)
                        expanded_zero = torch.cat([zero_part] * expansion_factor, dim=0)
                        new_tensor = torch.cat([expanded_real, expanded_zero], dim=0)
                    else:
                        new_tensor = torch.cat([tensor] * expansion_factor, dim=0)
                    new_nbytes = tensor_nbytes(new_tensor)
                    if current_bytes + new_nbytes > target_shard_size and current_tensors:
                        flush_shard()
                    current_tensors[key] = new_tensor
                    current_bytes += new_nbytes

                elif info := get_expert_info(key):
                    layer_idx, expert_idx, rest = info
                    if expert_idx < original_experts:
                        if current_bytes + nbytes > target_shard_size and current_tensors:
                            flush_shard()
                        current_tensors[key] = tensor
                        current_bytes += nbytes

                        for new_expert_idx in source_to_targets.get(expert_idx, []):
                            new_key = f"model.layers.{layer_idx}.mlp.experts.{new_expert_idx}.{rest}"
                            if current_bytes + nbytes > target_shard_size and current_tensors:
                                flush_shard()
                            current_tensors[new_key] = tensor.clone()
                            current_bytes += nbytes
                    else:
                        # Zero-expert: shift to new indices after expanded routed experts
                        base_new_idx = expert_idx - original_experts + target_experts
                        new_key = f"model.layers.{layer_idx}.mlp.experts.{base_new_idx}.{rest}"
                        if current_bytes + nbytes > target_shard_size and current_tensors:
                            flush_shard()
                        current_tensors[new_key] = tensor
                        current_bytes += nbytes

                        # Additional copies for expanded zero-expert slots
                        zero_offset = expert_idx - original_experts
                        for f in range(1, expansion_factor):
                            copy_idx = target_experts + zero_offset + f * zero_expert_num
                            copy_key = f"model.layers.{layer_idx}.mlp.experts.{copy_idx}.{rest}"
                            if current_bytes + nbytes > target_shard_size and current_tensors:
                                flush_shard()
                            current_tensors[copy_key] = tensor.clone()
                            current_bytes += nbytes

                else:
                    if current_bytes + nbytes > target_shard_size and current_tensors:
                        flush_shard()
                    current_tensors[key] = tensor
                    current_bytes += nbytes

    flush_shard()

    actual_shards = output_shard_idx - 1
    if actual_shards != num_output_shards:
        print(
            f"WARNING: Predicted {num_output_shards} shards but wrote {actual_shards}. "
            "Adjusting shard names and index..."
        )
        for i in range(1, actual_shards + 1):
            old_name = output_dir / f"model-{i:05d}-of-{num_output_shards:05d}.safetensors"
            new_name = output_dir / f"model-{i:05d}-of-{actual_shards:05d}.safetensors"
            if old_name.exists() and old_name != new_name:
                old_name.rename(new_name)
        num_output_shards = actual_shards

    # ── Write new index ──────────────────────────────────────────────────
    metadata = {**index.get("metadata", {})}
    metadata["total_size"] = total_output_bytes
    new_index = {
        "metadata": metadata,
        "weight_map": {
            k: re.sub(
                r"-of-\d+\.safetensors",
                f"-of-{num_output_shards:05d}.safetensors",
                v,
            )
            for k, v in new_weight_map.items()
        },
    }
    with open(output_dir / "model.safetensors.index.json", "w") as f:
        json.dump(new_index, f, indent=2)

    # ── Copy auxiliary files ────────────────────────────────────────────
    skip_suffixes = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".h5")
    skip_names = {"model.safetensors.index.json", "config.json"}
    for fpath in model_dir.iterdir():
        if fpath.is_file() and fpath.suffix not in skip_suffixes and fpath.name not in skip_names:
            shutil.copy2(fpath, output_dir / fpath.name)
            print(f"  Copied: {fpath.name}")

    print("\nVerification:")
    print(f"  Expert count key used: {expert_count_key or 'n_routed_experts'}")
    print(f"  Config experts: {target_experts} (real) + {target_zero_expert_num} (zero) = {target_experts + target_zero_expert_num} total routed")
    if args.target_topk is not None:
        print(f"  Topk: {args.target_topk}")
    print(f"  Output shards: {num_output_shards}")
    print(f"\nDone! Output saved to: {output_dir}")

if __name__ == "__main__":
    main()
