#!/usr/bin/env python3
# Copyright 2025 — utility script for this repo.
"""
Verify that local `models/modeling_deepseek.DeepseekV3ForCausalLM` matches a HuggingFace
sharded safetensors checkpoint: key names, per-shard keys vs index, and tensor shapes.

Does not load full weight tensors; uses safetensors slices for shapes only.

Usage:
  python check_model_weights.py /path/to/model_folder

Requires: torch, transformers, safetensors, accelerate
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _require(pkg: str, import_error: ImportError) -> None:
    raise SystemExit(
        f"Missing dependency {pkg!r}: {import_error}\n"
        f"Install with: pip install {pkg}"
    ) from import_error


try:
    import torch
except ImportError as e:
    _require("torch", e)

try:
    from safetensors import safe_open
except ImportError as e:
    _require("safetensors", e)

try:
    from accelerate import init_empty_weights
except ImportError as e:
    _require("accelerate", e)

from models.configuration_deepseek_1t import DeepseekV3Config
from models.modeling_deepseek import DeepseekV3ForCausalLM


def _shard_paths(ckpt: Path) -> Tuple[List[Path], Path | None]:
    """Return list of .safetensors shard files and optional index path."""
    index = ckpt / "model.safetensors.index.json"
    single = ckpt / "model.safetensors"
    if index.is_file():
        with open(index, encoding="utf-8") as f:
            idx = json.load(f)
        weight_map = idx.get("weight_map", {})
        files = sorted({weight_map[k] for k in weight_map})
        return [ckpt / name for name in files], index
    if single.is_file():
        return [single], None
    raise FileNotFoundError(
        f"No model.safetensors.index.json or model.safetensors under {ckpt}"
    )


def _read_specs_from_shard(path: Path) -> Dict[str, Tuple[int, ...]]:
    """Tensor name -> shape, without loading full tensors."""
    out: Dict[str, Tuple[int, ...]] = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            out[key] = tuple(sl.get_shape())
    return out


def _build_empty_model(config: DeepseekV3Config) -> "torch.nn.Module":
    with init_empty_weights():
        model = DeepseekV3ForCausalLM(config)
    return model


def _expected_state_specs(
    model: "torch.nn.Module",
) -> Tuple[Dict[str, Tuple[int, ...]], Set[str]]:
    """state_dict key -> shape; and set of all keys."""
    sd = model.state_dict()
    specs = {k: tuple(v.shape) for k, v in sd.items()}
    return specs, set(sd.keys())


def _compare_shapes(
    expected: Dict[str, Tuple[int, ...]],
    ckpt: Dict[str, Tuple[int, ...]],
    label: str,
) -> List[str]:
    problems: List[str] = []
    for k in sorted(expected.keys() & ckpt.keys()):
        if expected[k] != ckpt[k]:
            problems.append(
                f"  {label} shape mismatch {k!r}: model {expected[k]} vs checkpoint {ckpt[k]}"
            )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "checkpoint",
        type=Path,
        help="Directory containing config.json and model *.safetensors shards",
    )
    ap.add_argument(
        "--skip-shape-check",
        action="store_true",
        help="Only compare key sets, not shapes (faster if you only need key coverage)",
    )
    args = ap.parse_args()
    ckpt: Path = args.checkpoint.expanduser().resolve()

    if not ckpt.is_dir():
        raise SystemExit(f"Not a directory: {ckpt}")

    cfg_path = ckpt / "config.json"
    if not cfg_path.is_file():
        raise SystemExit(f"Missing config.json in {ckpt}")

    print(f"Checkpoint: {ckpt}")
    print(f"Loading config from {cfg_path}")
    config = DeepseekV3Config.from_pretrained(str(ckpt))

    print("Building empty DeepseekV3ForCausalLM (no weight memory) …")
    model = _build_empty_model(config)
    expected_specs, expected_keys = _expected_state_specs(model)
    print(f"  Model state_dict keys: {len(expected_keys)}")

    shards, index_path = _shard_paths(ckpt)
    print(f"  Safetensors shards: {len(shards)}")
    if index_path:
        print(f"  Index: {index_path.name}")

    # Load weight_map from index if present
    weight_map: Dict[str, str] = {}
    if index_path and index_path.is_file():
        with open(index_path, encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

    # Per-file: keys + shapes; global union; overlap detection
    key_to_shard: Dict[str, str] = {}
    union_specs: Dict[str, Tuple[int, ...]] = {}
    for shard_path in shards:
        if not shard_path.is_file():
            raise SystemExit(f"Missing shard file: {shard_path}")
        rel = shard_path.name
        print(f"\n--- Shard file: {rel} ---")
        specs = _read_specs_from_shard(shard_path)
        print(f"  Tensors in file: {len(specs)}")
        for k in specs:
            if k in key_to_shard:
                raise SystemExit(
                    f"Duplicate key across shards: {k!r} in {key_to_shard[k]!r} and {rel!r}"
                )
            key_to_shard[k] = rel
            union_specs[k] = specs[k]

        # Cross-check index weight_map for this file
        if weight_map:
            mapped_here = sorted(k for k, fn in weight_map.items() if fn == rel)
            keys_here = set(specs.keys())
            if set(mapped_here) != keys_here:
                only_map = sorted(keys_here - set(mapped_here))
                only_idx = sorted(set(mapped_here) - keys_here)
                if only_map:
                    print(f"  WARN: keys in file but not in weight_map for this file (first 10): {only_map[:10]}")
                if only_idx:
                    print(f"  WARN: keys in weight_map for this file but missing in file (first 10): {only_idx[:10]}")

    ckpt_keys = set(union_specs.keys())
    missing_in_ckpt = expected_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - expected_keys

    print("\n" + "=" * 60)
    print("SUMMARY: keys vs model.state_dict()")
    print("=" * 60)
    print(f"  Expected (model): {len(expected_keys)}")
    print(f"  In checkpoint:    {len(ckpt_keys)}")
    print(f"  Missing in checkpoint: {len(missing_in_ckpt)}")
    if missing_in_ckpt:
        for k in sorted(missing_in_ckpt)[:200]:
            print(f"    - {k}")
        if len(missing_in_ckpt) > 200:
            print(f"    ... and {len(missing_in_ckpt) - 200} more")
    print(f"  Extra in checkpoint (not in model): {len(extra_in_ckpt)}")
    if extra_in_ckpt:
        for k in sorted(extra_in_ckpt)[:200]:
            print(f"    + {k}")
        if len(extra_in_ckpt) > 200:
            print(f"    ... and {len(extra_in_ckpt) - 200} more")

    if not args.skip_shape_check:
        shape_issues = _compare_shapes(expected_specs, union_specs, "tensor")
        print("\n" + "=" * 60)
        print("SUMMARY: shape mismatches (intersection)")
        print("=" * 60)
        if not shape_issues:
            print("  None — all overlapping keys match shapes.")
        else:
            print(f"  Found {len(shape_issues)}:")
            for line in shape_issues[:500]:
                print(line)
            if len(shape_issues) > 500:
                print(f"  ... and {len(shape_issues) - 500} more")

    ok = not missing_in_ckpt and not extra_in_ckpt
    if not args.skip_shape_check:
        shape_ok = not _compare_shapes(expected_specs, union_specs, "tensor")
        ok = ok and shape_ok

    print("\n" + "=" * 60)
    if ok:
        print("RESULT: OK — model definition and checkpoint keys/shapes match.")
    else:
        print("RESULT: MISMATCH — see details above.")
    print("=" * 60)

    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
