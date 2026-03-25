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
    raise SystemExit(f'Missing dependency {pkg!r}: {import_error}\n'
                     f'Install with: pip install {pkg}') from import_error


try:
    import torch
except ImportError as e:
    _require('torch', e)

try:
    from safetensors import safe_open
except ImportError as e:
    _require('safetensors', e)

try:
    from accelerate import init_empty_weights
except ImportError as e:
    _require('accelerate', e)


def _shard_paths(ckpt: Path) -> Tuple[List[Path], Path | None]:
    """Return list of .safetensors shard files and optional index path."""
    index = ckpt / 'model.safetensors.index.json'
    single = ckpt / 'model.safetensors'
    if index.is_file():
        with open(index, encoding='utf-8') as f:
            idx = json.load(f)
        weight_map = idx.get('weight_map', {})
        files = sorted({weight_map[k] for k in weight_map})
        return [ckpt / name for name in files], index
    if single.is_file():
        return [single], None
    raise FileNotFoundError(
        f'No model.safetensors.index.json or model.safetensors under {ckpt}')


def _read_specs_from_shard(path: Path) -> Dict[str, Tuple[int, ...]]:
    """Tensor name -> shape, without loading full tensors."""
    out: Dict[str, Tuple[int, ...]] = {}
    with safe_open(path, framework='pt', device='cpu') as f:
        for key in f.keys():
            sl = f.get_slice(key)
            out[key] = tuple(sl.get_shape())
    return out


def _build_empty_model(config) -> 'torch.nn.Module':
    with init_empty_weights():
        from models.modeling_deepseek import DeepseekV3ForCausalLM
        model = DeepseekV3ForCausalLM(config)
    return model


def _expected_state_specs(
    model: 'torch.nn.Module', ) -> Tuple[Dict[str, Tuple[int, ...]], Set[str]]:
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
                f'  {label} shape mismatch {k!r}: model {expected[k]} vs checkpoint {ckpt[k]}'
            )
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        'checkpoint',
        type=Path,
        help='Directory containing config.json and model *.safetensors shards',
    )
    ap.add_argument(
        '--skip-shape-check',
        action='store_true',
        help=
        'Only compare key sets, not shapes (faster if you only need key coverage)',
    )
    ap.add_argument(
        '--strict-index',
        action='store_true',
        help=
        'Treat any mismatch between shard contents and index weight_map as an error',
    )
    ap.add_argument(
        '--report-limit',
        type=int,
        default=200,
        help='Maximum number of items to print per mismatch category',
    )
    args = ap.parse_args()
    ckpt: Path = args.checkpoint.expanduser().resolve()

    if not ckpt.is_dir():
        raise SystemExit(f'Not a directory: {ckpt}')

    cfg_path = ckpt / 'config.json'
    if not cfg_path.is_file():
        raise SystemExit(f'Missing config.json in {ckpt}')

    print(f'Checkpoint: {ckpt}')
    print(f'Loading config from {cfg_path}')
    from models.configuration_deepseek_1t import DeepseekV3Config
    config = DeepseekV3Config.from_pretrained(str(ckpt))

    print('Building empty DeepseekV3ForCausalLM (no weight memory) …')
    model = _build_empty_model(config)
    expected_specs, expected_keys = _expected_state_specs(model)
    print(f'  Model state_dict keys: {len(expected_keys)}')

    shards, index_path = _shard_paths(ckpt)
    print(f'  Safetensors shards: {len(shards)}')
    if index_path:
        print(f'  Index: {index_path.name}')

    # Load weight_map from index if present
    weight_map: Dict[str, str] = {}
    if index_path and index_path.is_file():
        with open(index_path, encoding='utf-8') as f:
            weight_map = json.load(f).get('weight_map', {})

    # Per-file: keys + shapes; global union; overlap detection
    key_to_shard: Dict[str, str] = {}
    union_specs: Dict[str, Tuple[int, ...]] = {}
    per_file_mismatches: List[str] = []
    per_file_shape_mismatches: List[str] = []
    observed_dims = {
        'q_proj_out': set(),
        'k_proj_out': set(),
        'v_proj_out': set(),
        'o_proj_in': set(),
        'q_layernorm_dim': set(),
        'k_layernorm_dim': set(),
    }
    for shard_path in shards:
        if not shard_path.is_file():
            raise SystemExit(f'Missing shard file: {shard_path}')
        rel = shard_path.name
        print(f'\n--- Shard file: {rel} ---')
        specs = _read_specs_from_shard(shard_path)
        print(f'  Tensors in file: {len(specs)}')
        for k in specs:
            if k in key_to_shard:
                raise SystemExit(
                    f'Duplicate key across shards: {k!r} in {key_to_shard[k]!r} and {rel!r}'
                )
            key_to_shard[k] = rel
            union_specs[k] = specs[k]
            if k.endswith('.q_proj.weight'):
                observed_dims['q_proj_out'].add(specs[k][0])
            elif k.endswith('.k_proj.weight'):
                observed_dims['k_proj_out'].add(specs[k][0])
            elif k.endswith('.v_proj.weight'):
                observed_dims['v_proj_out'].add(specs[k][0])
            elif k.endswith('.o_proj.weight'):
                observed_dims['o_proj_in'].add(specs[k][1])
            elif k.endswith('.q_layernorm.weight'):
                observed_dims['q_layernorm_dim'].add(specs[k][0])
            elif k.endswith('.k_layernorm.weight'):
                observed_dims['k_layernorm_dim'].add(specs[k][0])

        # Cross-check index weight_map for this file
        if weight_map:
            mapped_here = sorted(k for k, fn in weight_map.items()
                                 if fn == rel)
            keys_here = set(specs.keys())
            mapped_set = set(mapped_here)
            if mapped_set != keys_here:
                only_map = sorted(keys_here - mapped_set)
                only_idx = sorted(mapped_set - keys_here)
                if only_map:
                    msg = f'Index mismatch: keys present in {rel} but not mapped in index (first {min(10, len(only_map))}): {only_map[:10]}'
                    print(f'  {msg}')
                    per_file_mismatches.append(msg)
                if only_idx:
                    msg = f'Index mismatch: keys mapped to {rel} in index but missing in file (first {min(10, len(only_idx))}): {only_idx[:10]}'
                    print(f'  {msg}')
                    per_file_mismatches.append(msg)

            # Per-file shapes vs model for overlapping keys
            expected_specs_subset = {
                k: v
                for k, v in expected_specs.items() if k in keys_here
            }
            for k in sorted(expected_specs_subset.keys() & keys_here):
                if expected_specs_subset[k] != specs[k]:
                    per_file_shape_mismatches.append(
                        f'  {rel} shape mismatch {k!r}: model {expected_specs_subset[k]} vs checkpoint {specs[k]}'
                    )

    ckpt_keys = set(union_specs.keys())
    missing_in_ckpt = expected_keys - ckpt_keys
    extra_in_ckpt = ckpt_keys - expected_keys

    print('\n' + '=' * 60)
    print('SUMMARY: keys vs model.state_dict()')
    print('=' * 60)
    print(f'  Expected (model): {len(expected_keys)}')
    print(f'  In checkpoint:    {len(ckpt_keys)}')
    print(f'  Missing in checkpoint: {len(missing_in_ckpt)}')
    if missing_in_ckpt:
        for k in sorted(missing_in_ckpt)[:args.report_limit]:
            print(f'    - {k}')
        if len(missing_in_ckpt) > args.report_limit:
            print(
                f'    ... and {len(missing_in_ckpt) - args.report_limit} more')
    print(f'  Extra in checkpoint (not in model): {len(extra_in_ckpt)}')
    if extra_in_ckpt:
        for k in sorted(extra_in_ckpt)[:args.report_limit]:
            print(f'    + {k}')
        if len(extra_in_ckpt) > args.report_limit:
            print(f'    ... and {len(extra_in_ckpt) - args.report_limit} more')

    if not args.skip_shape_check:
        shape_issues = _compare_shapes(expected_specs, union_specs, 'tensor')
        print('\n' + '=' * 60)
        print('SUMMARY: shape mismatches (intersection)')
        print('=' * 60)
        if not shape_issues:
            print('  None — all overlapping keys match shapes.')
        else:
            print(
                f'  Found {len(shape_issues)} (showing up to {min(500, args.report_limit)}):'
            )
            for line in shape_issues[:min(500, args.report_limit)]:
                print(line)
            if len(shape_issues) > args.report_limit:
                print(
                    f'  ... and {len(shape_issues) - args.report_limit} more')

        print('\n' + '=' * 60)
        print('SUMMARY: per-file shape mismatches')
        print('=' * 60)
        if not per_file_shape_mismatches:
            print('  None — all per-file overlapping keys match shapes.')
        else:
            print(
                f'  Found {len(per_file_shape_mismatches)} (showing up to {min(500, args.report_limit)}):'
            )
            for line in per_file_shape_mismatches[:min(500, args.report_limit
                                                       )]:
                print(line)
            if len(per_file_shape_mismatches) > args.report_limit:
                print(
                    f'  ... and {len(per_file_shape_mismatches) - args.report_limit} more'
                )
        print('\n' + '=' * 60)
        print('HEAD DIM DIAGNOSTICS')
        print('=' * 60)
        exp_q_out = config.num_attention_heads * (config.qk_nope_head_dim +
                                                  config.qk_rope_head_dim)
        exp_k_out = getattr(
            config, 'num_key_value_heads', config.num_attention_heads) * (
                config.qk_nope_head_dim + config.qk_rope_head_dim)
        exp_v_out = getattr(config, 'num_key_value_heads',
                            config.num_attention_heads) * config.v_head_dim
        exp_o_in = config.num_attention_heads * config.v_head_dim
        print(
            f"  Expected q_proj out: {exp_q_out}  | Observed: {sorted(observed_dims['q_proj_out'])}"
        )
        print(
            f"  Expected k_proj out: {exp_k_out}  | Observed: {sorted(observed_dims['k_proj_out'])}"
        )
        print(
            f"  Expected v_proj out: {exp_v_out}  | Observed: {sorted(observed_dims['v_proj_out'])}"
        )
        print(
            f"  Expected o_proj in : {exp_o_in}   | Observed: {sorted(observed_dims['o_proj_in'])}"
        )
        print(
            f"  Expected q_ln dim  : {config.qk_nope_head_dim + config.qk_rope_head_dim} | Observed: {sorted(observed_dims['q_layernorm_dim'])}"
        )
        print(
            f"  Expected k_ln dim  : {config.qk_nope_head_dim + config.qk_rope_head_dim} | Observed: {sorted(observed_dims['k_layernorm_dim'])}"
        )

    ok = not missing_in_ckpt and not extra_in_ckpt
    if not args.skip_shape_check:
        shape_ok = not _compare_shapes(expected_specs, union_specs, 'tensor')
        ok = ok and shape_ok
        ok = ok and not per_file_shape_mismatches
    if args.strict_index and weight_map:
        ok = ok and not per_file_mismatches

    print('\n' + '=' * 60)
    if ok:
        print(
            'RESULT: OK — model definition and checkpoint keys/shapes match.')
    else:
        print('RESULT: MISMATCH — see details above.')
    print('=' * 60)

    raise SystemExit(0 if ok else 1)


if __name__ == '__main__':
    main()
