#!/usr/bin/env python3
# Copyright 2025 — utility script for this repo.
"""
Kimi2-PCL 模型权重验证与配置检查工具

功能：
1. 验证 HF checkpoint 与本地模型定义的键名和形状是否匹配
2. 估算模型参数量（支持 1T/100B 等规模）
3. 验证配置文件之间的一致性
4. 验证预训练脚本与配置的一致性

用法：
  # 验证 checkpoint 与模型定义
  python check_model_weights.py /path/to/model_folder

  # 估算参数量
  python check_model_weights.py --estimate-params

  # 验证配置一致性
  python check_model_weights.py --verify-config

  # 执行所有检查
  python check_model_weights.py --verify-all

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

# Also add parent directory (repo root) to path for imports
REPO_PARENT = REPO_ROOT.parent
if str(REPO_PARENT) not in sys.path:
    sys.path.insert(0, str(REPO_PARENT))


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

# accelerate is only needed for checkpoint validation, not for param estimation
init_empty_weights = None


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
    global init_empty_weights
    if init_empty_weights is None:
        try:
            from accelerate import init_empty_weights as _init
            init_empty_weights = _init
        except ImportError as e:
            _require('accelerate', e)
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        type=Path,
        help='Directory containing config.json and model *.safetensors shards',
    )
    parser.add_argument(
        '--estimate-params',
        action='store_true',
        help='Estimate model parameter count without loading checkpoint',
    )
    parser.add_argument(
        '--verify-config',
        action='store_true',
        help='Verify consistency between configuration files',
    )
    parser.add_argument(
        '--verify-all',
        action='store_true',
        help='Run all verification checks (config + params)',
    )
    parser.add_argument(
        '--skip-shape-check',
        action='store_true',
        help=
        'Only compare key sets, not shapes (faster if you only need key coverage)',
    )
    parser.add_argument(
        '--strict-index',
        action='store_true',
        help=
        'Treat any mismatch between shard contents and index weight_map as an error',
    )
    parser.add_argument(
        '--report-limit',
        type=int,
        default=200,
        help='Maximum number of items to print per mismatch category',
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent

    # If no specific action requested, require checkpoint
    if not (args.estimate_params or args.verify_config or args.verify_all):
        if args.checkpoint is None:
            parser.error(
                'checkpoint is required unless using --estimate-params, --verify-config, or --verify-all'
            )
        return _main_check_checkpoint(args)

    exit_code = 0

    # Config verification
    if args.verify_config or args.verify_all:
        ok = verify_config_consistency(repo_root)
        if not ok:
            exit_code = 1

        verify_pretrain_script_consistency(repo_root)

    # Parameter estimation
    if args.estimate_params or args.verify_all:
        print_parameter_estimate()

    # Checkpoint check (if provided)
    if args.checkpoint is not None:
        ckpt_code = _main_check_checkpoint(args)
        if ckpt_code != 0:
            exit_code = ckpt_code

    return exit_code


def estimate_model_params(
    vocab_size: int = 163840,
    hidden_size: int = 7168,
    num_layers: int = 32,
    num_attention_heads: int = 64,
    num_key_value_heads: int = 32,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    intermediate_size: int = 18432,
    moe_intermediate_size: int = 12288,
    num_experts: int = 128,
    first_k_dense_replace: int = 2,
    n_shared_experts: int = 1,
) -> dict:
    """估算模型各部分的参数量"""

    params = {}

    # Embedding 层
    params['embedding'] = vocab_size * hidden_size

    # 每层参数
    per_layer_params = {}

    # Attention 参数
    q_head_dim = qk_nope_head_dim + qk_rope_head_dim
    q_proj = hidden_size * num_attention_heads * q_head_dim
    k_proj = hidden_size * num_key_value_heads * q_head_dim
    v_proj = hidden_size * num_key_value_heads * v_head_dim
    o_proj = num_attention_heads * v_head_dim * hidden_size
    qk_layernorm = num_attention_heads * q_head_dim + num_key_value_heads * q_head_dim

    per_layer_params[
        'attention'] = q_proj + k_proj + v_proj + o_proj + qk_layernorm

    # LayerNorm 参数
    per_layer_params['layernorm'] = 2 * hidden_size  # input_ln + pre_mlp_ln

    # MLP 参数 (MoE 或 Dense)
    moe_layers = num_layers - first_k_dense_replace
    dense_layers = first_k_dense_replace

    # Dense MLP (每层)
    dense_mlp = hidden_size * intermediate_size * 3  # gate, up, down
    per_layer_params['dense_mlp'] = dense_mlp

    # MoE 层 (每层)
    # Router
    router = hidden_size * num_experts
    # Shared experts
    shared_experts = hidden_size * moe_intermediate_size * 2 + moe_intermediate_size * hidden_size
    # Routed experts
    routed_experts = num_experts * (hidden_size * moe_intermediate_size * 2 +
                                    moe_intermediate_size * hidden_size)
    per_layer_params['moe_mlp'] = router + shared_experts + routed_experts

    # 所有层总参数
    dense_total = dense_layers * (per_layer_params['attention'] +
                                  per_layer_params['layernorm'] +
                                  per_layer_params['dense_mlp'])
    moe_total = moe_layers * (per_layer_params['attention'] +
                              per_layer_params['layernorm'] +
                              per_layer_params['moe_mlp'])
    params['layers'] = dense_total + moe_total

    # 输出层
    params['lm_head'] = hidden_size * vocab_size
    params['final_layernorm'] = hidden_size

    # 总参数量
    params['total'] = sum([
        params['embedding'], params['layers'], params['lm_head'],
        params['final_layernorm']
    ])

    params['per_layer'] = per_layer_params
    params['moe_layers'] = moe_layers
    params['dense_layers'] = dense_layers

    return params


def verify_config_consistency(repo_root: Path) -> bool:
    """验证配置文件之间的一致性"""
    from models.configuration_deepseek_1t import DeepseekV3Config

    # 加载 Python 配置类默认值
    cfg = DeepseekV3Config()

    # 加载 JSON 配置
    json_path = repo_root / 'models' / 'config_1t.json'
    if not json_path.is_file():
        print(f"Warning: {json_path} not found, skipping JSON config check")
        return True

    with open(json_path) as f:
        json_cfg = json.load(f)

    print('=' * 60)
    print('配置文件一致性验证')
    print('=' * 60)

    mismatches = []

    key_mappings = [
        ('vocab_size', 'vocab_size'),
        ('hidden_size', 'hidden_size'),
        ('intermediate_size', 'intermediate_size'),
        ('moe_intermediate_size', 'moe_intermediate_size'),
        ('num_hidden_layers', 'num_hidden_layers'),
        ('num_attention_heads', 'num_attention_heads'),
        ('num_key_value_heads', 'num_key_value_heads'),
        ('n_routed_experts', 'n_routed_experts'),
        ('n_shared_experts', 'n_shared_experts'),
        ('first_k_dense_replace', 'first_k_dense_replace'),
        ('max_position_embeddings', 'max_position_embeddings'),
        ('rope_theta', 'rope_theta'),
    ]
    # Optional mappings - only check if both exist
    optional_mappings = [
        ('qk_nope_head_dim', 'qk_nope_head_dim'),
        ('qk_rope_head_dim', 'qk_rope_head_dim'),
        ('v_head_dim', 'v_head_dim'),
    ]

    for py_attr, json_key in key_mappings:
        py_val = getattr(cfg, py_attr)
        json_val = json_cfg.get(json_key)

        if py_val != json_val:
            mismatches.append((py_attr, py_val, json_val))

    # Check optional mappings only if they exist in both
    for py_attr, json_key in optional_mappings:
        if hasattr(cfg, py_attr) and json_key in json_cfg:
            py_val = getattr(cfg, py_attr)
            json_val = json_cfg.get(json_key)
            if py_val != json_val:
                mismatches.append((py_attr, py_val, json_val))

    if mismatches:
        print('❌ 发现配置不匹配:')
        for attr, py_val, json_val in mismatches:
            print(f"  {attr}: Python={py_val}, JSON={json_val}")
        return False
    else:
        print('✓ 所有配置参数一致')
        return True


def verify_pretrain_script_consistency(repo_root: Path) -> bool:
    """验证预训练脚本参数与配置的一致性"""
    script_path = repo_root / 'scripts' / 'pretrain_kimi2_1t_4k.sh'

    if not script_path.is_file():
        print(f"Warning: {script_path} not found, skipping script check")
        return True

    print('')
    print('=' * 60)
    print('预训练脚本与配置一致性验证')
    print('=' * 60)

    # 读取脚本内容
    with open(script_path) as f:
        script = f.read()

    # 检查关键参数是否存在
    key_params = [
        'NUM_LAYERS=32',
        'hidden-size 7168',
        'ffn-hidden-size 18432',
        'vocab-size 163840',
        'num-experts 128',
        'moe-ffn-hidden-size 12288',
        'num-attention-heads 64',
        'first-k-dense-replace 2',
        'rotary-base 50000',
    ]

    all_found = True
    for param in key_params:
        if param in script:
            print(f"  ✓ Found: {param}")
        else:
            print(f"  ⚠ Not found: {param}")
            all_found = False

    if all_found:
        print('\n✓ 预训练脚本参数完整')
    else:
        print('\n⚠ 部分参数可能需要手动检查')

    return True


def print_parameter_estimate():
    """打印参数量估算"""
    print('')
    print('=' * 60)
    print('模型参数量估算 (Kimi2-1T)')
    print('=' * 60)

    params = estimate_model_params()
    per_layer = params['per_layer']

    print(f"\n层配置:")
    print(f"  Dense 层数: {params['dense_layers']}")
    print(f"  MoE 层数:   {params['moe_layers']}")

    print(f"\n每层参数量:")
    print(
        f"  Attention:   {per_layer['attention']:>15,} ({per_layer['attention']/1e9:.2f}B)"
    )
    print(f"  LayerNorm:   {per_layer['layernorm']:>15,}")
    print(
        f"  Dense MLP:   {per_layer['dense_mlp']:>15,} ({per_layer['dense_mlp']/1e9:.2f}B)"
    )
    print(
        f"  MoE MLP:     {per_layer['moe_mlp']:>15,} ({per_layer['moe_mlp']/1e9:.2f}B)"
    )

    print(f"\n总参数量:")
    print(
        f"  Embedding 层:       {params['embedding']:>15,} ({params['embedding']/1e9:.2f}B)"
    )
    print(
        f"  Transformer 层:     {params['layers']:>15,} ({params['layers']/1e9:.2f}B)"
    )
    print(
        f"  LM Head:            {params['lm_head']:>15,} ({params['lm_head']/1e9:.2f}B)"
    )
    print(f"  Final LayerNorm:    {params['final_layernorm']:>15,}")
    print('-' * 60)
    print(
        f"  总计:               {params['total']:>15,} ({params['total']/1e9:.2f}B)"
    )

    # 计算 BF16 模型大小
    bf16_bytes = params['total'] * 2  # BF16 = 2 bytes
    print(f"\nBF16 模型大小:        {bf16_bytes/1e9:.2f} GB")
    print(f"FP32 模型大小:        {params['total']*4/1e9:.2f} GB")


def _main_check_checkpoint(args):
    """原有的 checkpoint 验证逻辑"""
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

    weight_map: Dict[str, str] = {}
    if index_path and index_path.is_file():
        with open(index_path, encoding='utf-8') as f:
            weight_map = json.load(f).get('weight_map', {})

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
        # Safely get head dimension attributes with defaults
        qk_nope_head_dim = getattr(config, 'qk_nope_head_dim', 128)
        qk_rope_head_dim = getattr(config, 'qk_rope_head_dim', 64)
        v_head_dim = getattr(config, 'v_head_dim', 128)
        exp_q_out = config.num_attention_heads * (qk_nope_head_dim +
                                                  qk_rope_head_dim)
        exp_k_out = getattr(
            config, 'num_key_value_heads',
            config.num_attention_heads) * (qk_nope_head_dim + qk_rope_head_dim)
        exp_v_out = getattr(config, 'num_key_value_heads',
                            config.num_attention_heads) * v_head_dim
        exp_o_in = config.num_attention_heads * v_head_dim
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
            f"  Expected q_ln dim  : {qk_nope_head_dim + qk_rope_head_dim} | Observed: {sorted(observed_dims['q_layernorm_dim'])}"
        )
        print(
            f"  Expected k_ln dim  : {qk_nope_head_dim + qk_rope_head_dim} | Observed: {sorted(observed_dims['k_layernorm_dim'])}"
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

    return 0 if ok else 1


if __name__ == '__main__':
    main()
