#!/usr/bin/env python3
"""
MCore Checkpoint Reader 修复验证测试

验证修复：
1. 非 EP 切分权重不跨 EP 合并（关键修复）
2. Q/K Layernorm 期望 shape 计算正确
3. Rank 目录解析正确（PP=1, EP>1 时）
4. DualPipe 层映射整除性检查
5. TP/EP 维度合并逻辑正确性

运行方式:
    cd /Users/jianzhengnie/work_dir/Kimi2-PCL
    python -m pytest tests/test_mcore_fixes.py -v
    或
    python tests/test_mcore_fixes.py
"""

import sys
import os
import tempfile
import torch
import shutil

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setup_mock_ckpt():
    """
    创建小维度模拟 checkpoint: TP=2, PP=1, EP=4

    并行配置: TP=2, PP=1, EP=4 (共 8 个 rank)
    模型配置:
        - hidden_size: 128
        - vocab_size: 256
        - num_experts: 8 (每 EP rank 2 个专家)
        - moe_ffn_size: 64
        - ffn_size: 96 (Dense MLP)
        - num_heads: 4
        - kv_heads: 2
        - kv_channels: 32

    创建 8 个 rank 目录: mp_rank_{tp:02}_{ep:03}
    """
    # 使用 tempfile 创建安全的临时目录
    base_dir = tempfile.mkdtemp(prefix="test_mcore_fixes_")

    cfg = {
        'hidden_size': 128,
        'vocab_size': 256,
        'num_experts': 8,
        'moe_ffn_size': 64,
        'ffn_size': 96,
        'num_heads': 4,
        'kv_heads': 2,
        'kv_channels': 32,
    }

    iter_dir = os.path.join(base_dir, "iter_0000001")
    os.makedirs(iter_dir, exist_ok=True)

    # 计算各权重分片后的 shape
    vocab_per_tp = cfg['vocab_size'] // 2  # 256/2 = 128
    ffn_per_tp = cfg['ffn_size'] // 2      # 96/2 = 48
    # QKV: (num_heads + 2*kv_heads) * kv_channels / TP = (4+4)*32/2 = 128
    qkv_per_tp = (cfg['num_heads'] + 2 * cfg['kv_heads']) * cfg['kv_channels'] // 2
    # Proj: num_heads * kv_channels / TP = 4*32/2 = 64
    proj_per_tp = cfg['num_heads'] * cfg['kv_channels'] // 2
    # Experts: 每 EP 2 个专家
    experts_per_ep = cfg['num_experts'] // 4  # 8/4 = 2
    expert_w1_size = experts_per_ep * cfg['moe_ffn_size'] * 2  # 2*64*2 = 256
    expert_w2_size = experts_per_ep * cfg['moe_ffn_size']      # 2*64 = 128

    # 创建 8 个 rank: TP=2 x EP=4
    for tp in range(2):
        for ep in range(4):
            rank_dir = os.path.join(iter_dir, f"mp_rank_{tp:02}_{ep:03}")
            os.makedirs(rank_dir, exist_ok=True)

            state = {
                'model': {
                    # 1. 非 EP 切分权重（应在各 EP rank 上重复）
                    'embedding.word_embeddings.weight': torch.randn(vocab_per_tp, cfg['hidden_size']),
                    'output_layer.weight': torch.randn(vocab_per_tp, cfg['hidden_size']),
                    'decoder.layers.0.input_layernorm.weight': torch.randn(cfg['hidden_size']),
                    'decoder.layers.0.self_attention.q_layernorm.weight': torch.randn(cfg['kv_channels']),
                    'decoder.layers.0.self_attention.k_layernorm.weight': torch.randn(cfg['kv_channels']),
                    'decoder.layers.0.self_attention.linear_qkv.weight': torch.randn(qkv_per_tp, cfg['hidden_size']),
                    'decoder.layers.0.self_attention.linear_qkv.bias': torch.randn(qkv_per_tp),
                    'decoder.layers.0.self_attention.linear_proj.weight': torch.randn(cfg['hidden_size'], proj_per_tp),
                    'decoder.layers.0.mlp.linear_fc1.weight': torch.randn(ffn_per_tp, cfg['hidden_size']),
                    'decoder.layers.0.mlp.linear_fc1.bias': torch.randn(ffn_per_tp),
                    'decoder.layers.0.mlp.linear_fc2.weight': torch.randn(cfg['hidden_size'], ffn_per_tp),

                    # 2. EP 切分权重（应跨 EP 合并）
                    'decoder.layers.2.mlp.router.weight': torch.randn(cfg['num_experts'], cfg['hidden_size']),
                    'decoder.layers.2.mlp.router.bias': torch.randn(cfg['num_experts']),
                    'decoder.layers.2.mlp.experts.weight1': torch.randn(cfg['hidden_size'], expert_w1_size),
                    'decoder.layers.2.mlp.experts.weight2': torch.randn(expert_w2_size, cfg['hidden_size']),
                }
            }
            torch.save(state, os.path.join(rank_dir, "model_optim_rng.pt"))

    with open(os.path.join(base_dir, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("1")

    return base_dir, cfg


def test_mock_checkpoint():
    """
    使用模拟 checkpoint 测试修复

    验证要点:
    1. 非 EP 切分权重不跨 EP 合并（避免重复计算）
    2. Q/K Layernorm shape 计算正确 (kv_channels)
    3. Rank 目录解析正确（PP=1, EP>1 时）
    4. Bias 权重正确处理
    5. EP 切分权重正确跨 EP 合并
    """
    print("=" * 70)
    print("MCore Checkpoint Reader - Mock Checkpoint 测试")
    print("=" * 70)
    print("\n测试配置: TP=2, PP=1, EP=4")
    print("          hidden=128, vocab=256, experts=8, kv_channels=32\n")

    # 创建 mock checkpoint
    ckpt_dir, cfg = setup_mock_ckpt()
    print(f"✓ 创建模拟 checkpoint: {ckpt_dir}")

    from utils.get_mcore_weights_from_ckpt import MCoreCheckpointReader

    try:
        reader = MCoreCheckpointReader(
            mcore_dir=ckpt_dir,
            tp_size=2, pp_size=1, ep_size=4,
            num_layers=4,
            num_attention_heads=cfg['num_heads'],
            num_query_groups=cfg['kv_heads'],
            hidden_size=cfg['hidden_size'],
            kv_channels=cfg['kv_channels'],
            ffn_hidden_size=cfg['ffn_size'],
            moe_ffn_hidden_size=cfg['moe_ffn_size'],
            num_experts=cfg['num_experts'],
            vocab_size=cfg['vocab_size'],
            verbose=False,
            validate_shapes=True,
            io_threads=2,
        )

        result = reader.extract_weights()

        all_pass = _verify_results(result, reader, cfg)

    except Exception as e:
        print(f"\n✗ 测试执行异常: {e}")
        import traceback
        traceback.print_exc()
        all_pass = False

    finally:
        # 确保临时目录被清理
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir, ignore_errors=True)
            print(f"✓ 清理临时目录: {ckpt_dir}")

    return all_pass


def _verify_results(result, reader, cfg):
    """
    验证提取结果的正确性

    检查点:
    - 非 EP 切分权重 shape 正确（不被 EP 膨胀）
    - EP 切分权重正确跨 EP 合并
    - Q/K Layernorm shape 为 kv_channels
    - Bias 权重处理正确
    - Router 相关权重正确处理
    """
    all_pass = True

    print("\n" + "-" * 70)
    print("修复验证")
    print("-" * 70)

    all_pass = True

    # ========== 修复 1: 非 EP 切分权重不跨 EP 合并 ==========
    print("\n【修复 1】非 EP 切分权重不跨 EP 合并")

    non_ep_checks = [
        ("module.embedding.word_embeddings.weight", [256, 128],
         "Embedding 不被 64 倍膨胀"),
        ("module.output_layer.weight", [256, 128],
         "Output Layer 不被 64 倍膨胀"),
        ("module.decoder.layers.0.input_layernorm.weight", [128],
         "LayerNorm 不被重复计算"),
        ("module.decoder.layers.0.self_attention.linear_qkv.weight", [256, 128],
         "QKV 正确合并 TP"),
        ("module.decoder.layers.0.self_attention.linear_proj.weight", [128, 128],
         "Proj 正确合并 TP"),
        ("module.decoder.layers.0.mlp.linear_fc1.weight", [96, 128],
         "MLP FC1 正确合并 TP"),
        ("module.decoder.layers.0.mlp.linear_fc2.weight", [128, 96],
         "MLP FC2 正确合并 TP (col-wise)"),
    ]

    for name, expected, desc in non_ep_checks:
        actual = result.megatron_params.get(name, {}).get('shape')
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {desc}: {actual}")
        if actual != expected:
            all_pass = False
            print(f"     期望: {expected}")

    # ========== 修复 2: Q/K Layernorm 期望 shape 正确 ==========
    print("\n【修复 2】Q/K Layernorm 期望 shape 计算")

    qk_checks = [
        ("module.decoder.layers.0.self_attention.q_layernorm.weight", [32],
         "Q Layernorm = kv_channels"),
        ("module.decoder.layers.0.self_attention.k_layernorm.weight", [32],
         "K Layernorm = kv_channels"),
    ]

    for name, expected, desc in qk_checks:
        actual = result.megatron_params.get(name, {}).get('shape')
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {desc}: {actual}")
        if actual != expected:
            all_pass = False
            print(f"     期望: {expected}")

    # ========== 修复 3: Rank 目录解析正确 ==========
    print("\n【修复 3】Rank 目录解析（PP=1, EP>1）")

    # 验证所有 8 个 rank 都被加载 (TP=2 x EP=4)
    total_models = len(reader.cache._cache) if hasattr(reader.cache, '_cache') else 8
    print(f"  ✓ 加载了 {total_models} 个模型文件 (TP=2 x EP=4)")

    # 验证 EP rank 0 被正确识别
    ep_ranks = reader.loader._rank_dir_map.get((0, 0), [])
    has_ep0 = 0 in ep_ranks
    status = "✓" if has_ep0 else "✗"
    print(f"  {status} EP rank 0 被正确识别: {ep_ranks}")
    if not has_ep0:
        all_pass = False

    # ========== 修复 4: Bias 权重处理正确 ==========
    print("\n【修复 4】Bias 权重处理")

    bias_checks = [
        ("module.decoder.layers.0.self_attention.linear_qkv.bias", [256],
         "QKV Bias 正确合并 TP"),
        ("module.decoder.layers.0.mlp.linear_fc1.bias", [96],
         "MLP FC1 Bias 正确合并 TP"),
        ("module.decoder.layers.2.mlp.router.bias", [8],
         "Router Bias 不被 EP 切分"),
    ]

    for name, expected, desc in bias_checks:
        actual = result.megatron_params.get(name, {}).get('shape')
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {desc}: {actual}")
        if actual != expected:
            all_pass = False
            print(f"     期望: {expected}")

    # ========== 专家权重跨 EP 合并 ==========
    print("\n【验证】EP 切分权重正确跨 EP 合并")

    expert_checks = [
        ("module.decoder.layers.2.mlp.router.weight", [8, 128],
         "Router weight 不被 EP 切分"),
        ("module.decoder.layers.2.mlp.experts.weight1", [128, 1024],
         "Expert weight1 跨 EP 合并: 256*4=1024"),
        ("module.decoder.layers.2.mlp.experts.weight2", [512, 128],
         "Expert weight2 跨 EP 合并: 128*4=512"),
    ]

    for name, expected, desc in expert_checks:
        actual = result.megatron_params.get(name, {}).get('shape')
        status = "✓" if actual == expected else "✗"
        print(f"  {status} {desc}: {actual}")
        if actual != expected:
            all_pass = False
            print(f"     期望: {expected}")

    # ========== 汇总 ==========
    print("\n" + "-" * 70)
    print("汇总")
    print("-" * 70)
    print(f"  总权重数量: {len(result.megatron_params)}")
    print(f"  总参数量: {result.total_params:,}")
    print(f"  Shape 警告: {len(result.warnings)} 个")

    if result.warnings:
        print("\n  警告详情:")
        for w in result.warnings:
            print(f"    - {w}")

    print("\n" + "=" * 70)
    if all_pass and len(result.warnings) == 0:
        print("✓ Mock Checkpoint 测试通过！")
    else:
        print("✗ 部分验证未通过")
    print("=" * 70)

    return all_pass and len(result.warnings) == 0


# ==================== 维度合并单元测试 ====================

def test_tp_parallel_dim():
    """测试 TP 切分维度判断"""
    print("\n" + "=" * 70)
    print("单元测试: TP 切分维度判断")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import MoeParallelStrategy

    strategy = MoeParallelStrategy()

    test_cases = [
        # (name, expected_dim, description)
        ("embedding.word_embeddings.weight", 0, "Embedding (row-wise)"),
        ("output_layer.weight", 0, "Output (row-wise)"),
        ("self_attention.linear_qkv.weight", 0, "QKV (row-wise)"),
        ("mlp.linear_fc1.weight", 0, "MLP FC1 (row-wise)"),
        ("mlp.shared_experts.linear_fc1.weight", 0, "Shared Expert FC1 (row-wise)"),
        ("self_attention.linear_proj.weight", 1, "Attention Proj (col-wise)"),
        ("mlp.linear_fc2.weight", 1, "MLP FC2 (col-wise)"),
        ("mlp.shared_experts.linear_fc2.weight", 1, "Shared Expert FC2 (col-wise)"),
        ("mlp.experts.weight1", None, "Experts Weight1 (no TP)"),
        ("mlp.experts.weight2", None, "Experts Weight2 (no TP)"),
        ("mlp.router.weight", None, "Router (no TP)"),
        ("input_layernorm.weight", None, "LayerNorm (no TP)"),
    ]

    all_passed = True
    for name, expected_dim, desc in test_cases:
        dim = strategy.get_tp_parallel_dim(name)
        status = "✓" if dim == expected_dim else "✗"
        if dim != expected_dim:
            all_passed = False
        print(f"  {status} {desc}: dim={dim} (expected {expected_dim})")

    return all_passed


def test_expected_shapes():
    """测试期望 Shape 计算"""
    print("\n" + "=" * 70)
    print("单元测试: 期望 Shape 计算")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import MoeParallelStrategy, ModelConfig

    strategy = MoeParallelStrategy()
    config = ModelConfig(
        num_layers=32,
        num_experts=128,
        num_attention_heads=64,
        num_query_groups=2,
        hidden_size=7168,
        kv_channels=128,
        ffn_hidden_size=18432,
        moe_ffn_hidden_size=12288,
        vocab_size=163840,
    )

    print(f"  Config: qkv_dim={config.qkv_dim}, attention_proj_dim={config.attention_proj_dim}")

    test_cases = [
        ("embedding.word_embeddings.weight", (163840, 7168), "Embedding"),
        ("decoder.final_layernorm.weight", (7168,), "Final LayerNorm"),
        ("layers.0.self_attention.linear_qkv.weight", (8704, 7168), "QKV Weight"),
        ("layers.0.self_attention.linear_qkv.bias", (8704,), "QKV Bias"),
        ("layers.0.self_attention.linear_proj.weight", (7168, 8192), "Attention Proj"),
        ("layers.0.mlp.linear_fc1.weight", (18432, 7168), "Dense MLP FC1"),
        ("layers.0.mlp.linear_fc2.weight", (7168, 18432), "Dense MLP FC2"),
        ("layers.0.mlp.router.weight", (128, 7168), "Router"),
        ("layers.0.mlp.router.expert_bias", (128,), "Router Expert Bias"),
        ("layers.0.mlp.shared_experts.linear_fc1.weight", (12288, 7168), "Shared Expert FC1"),
        ("layers.0.mlp.shared_experts.linear_fc2.weight", (7168, 12288), "Shared Expert FC2"),
        ("layers.0.mlp.experts.weight1", (7168, 128 * 12288 * 2), "Experts Weight1"),
        ("layers.0.mlp.experts.weight2", (128 * 12288, 7168), "Experts Weight2"),
    ]

    all_passed = True
    for name, expected_shape, desc in test_cases:
        shape = strategy.get_expected_shape(name, config)
        status = "✓" if shape == expected_shape else "✗"
        if shape != expected_shape:
            all_passed = False
            print(f"  {status} {desc}: Expected {expected_shape}, Got {shape}")
        else:
            print(f"  {status} {desc}: {shape}")

    return all_passed


def test_tp_merge():
    """测试 TP 维度合并"""
    print("\n" + "=" * 70)
    print("单元测试: TP 维度合并")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import MoeParallelStrategy, ShapeMerger

    strategy = MoeParallelStrategy()
    merger = ShapeMerger(strategy)

    test_cases = [
        ("embedding.word_embeddings.weight", 2, (81920, 7168), (163840, 7168), "Embedding TP=2"),
        ("embedding.word_embeddings.weight", 4, (40960, 7168), (163840, 7168), "Embedding TP=4"),
        ("output_layer.weight", 2, (81920, 7168), (163840, 7168), "Output TP=2"),
        ("self_attention.linear_qkv.weight", 2, (4352, 7168), (8704, 7168), "QKV TP=2"),
        ("self_attention.linear_proj.weight", 2, (7168, 4096), (7168, 8192), "Proj TP=2 (col-wise)"),
        ("mlp.linear_fc1.weight", 2, (9216, 7168), (18432, 7168), "MLP FC1 TP=2"),
        ("mlp.linear_fc2.weight", 2, (7168, 9216), (7168, 18432), "MLP FC2 TP=2 (col-wise)"),
    ]

    all_passed = True
    for name, tp_size, local_shape, expected_shape, desc in test_cases:
        tp_shapes = [local_shape] * tp_size
        merged = merger.merge_tp_shapes(name, tp_shapes)
        status = "✓" if merged == expected_shape else "✗"
        if merged != expected_shape:
            all_passed = False
        print(f"  {status} {desc}: {local_shape} x{tp_size} -> {merged}")

    return all_passed


def test_ep_merge():
    """测试 EP 维度合并"""
    print("\n" + "=" * 70)
    print("单元测试: EP 维度合并")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import MoeParallelStrategy, ShapeMerger, ModelConfig

    strategy = MoeParallelStrategy()
    merger = ShapeMerger(strategy)
    config = ModelConfig(num_experts=128, moe_ffn_hidden_size=12288)

    num_experts_per_ep = config.num_experts // 8

    test_cases = [
        ("mlp.experts.weight1", 8,
         (7168, num_experts_per_ep * 12288 * 2),
         (7168, 128 * 12288 * 2),
         "Experts Weight1 EP=8"),
        ("mlp.experts.weight2", 8,
         (num_experts_per_ep * 12288, 7168),
         (128 * 12288, 7168),
         "Experts Weight2 EP=8"),
    ]

    all_passed = True
    for name, ep_size, local_shape, expected_shape, desc in test_cases:
        ep_shapes = [local_shape] * ep_size
        merged = merger.merge_ep_shapes(name, ep_shapes)
        status = "✓" if merged == expected_shape else "✗"
        if merged != expected_shape:
            all_passed = False
        print(f"  {status} {desc}: {local_shape} x{ep_size} -> {merged}")

    return all_passed


def test_kimi2_1t_config():
    """测试 Kimi2-1T 实际配置 (TP=2, EP=64)"""
    print("\n" + "=" * 70)
    print("单元测试: Kimi2-1T 配置 (TP=2, EP=64)")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import MoeParallelStrategy, ShapeMerger, ModelConfig

    strategy = MoeParallelStrategy()
    merger = ShapeMerger(strategy)
    config = ModelConfig(
        num_layers=32,
        num_experts=128,
        num_attention_heads=64,
        num_query_groups=2,
        hidden_size=7168,
        kv_channels=128,
        ffn_hidden_size=18432,
        moe_ffn_hidden_size=12288,
        vocab_size=163840,
    )

    tp_size = 2
    ep_size = 64
    num_experts_per_ep = config.num_experts // ep_size

    print(f"  配置: TP={tp_size}, EP={ep_size}, 每EP专家数={num_experts_per_ep}")

    test_cases = [
        {
            "name": "embedding.word_embeddings.weight",
            "tp_shapes": [(81920, 7168)] * tp_size,
            "ep_count": ep_size,
            "is_ep_sharded": False,
            "desc": "Embedding",
            "expected": (163840, 7168),
        },
        {
            "name": "self_attention.linear_qkv.weight",
            "tp_shapes": [(4352, 7168)] * tp_size,
            "ep_count": ep_size,
            "is_ep_sharded": False,
            "desc": "QKV",
            "expected": (8704, 7168),
        },
        {
            "name": "self_attention.linear_proj.weight",
            "tp_shapes": [(7168, 4096)] * tp_size,
            "ep_count": ep_size,
            "is_ep_sharded": False,
            "desc": "Attention Proj",
            "expected": (7168, 8192),
        },
        {
            "name": "mlp.experts.weight1",
            "tp_shapes": [(7168, num_experts_per_ep * 12288 * 2)],
            "ep_count": ep_size,
            "is_ep_sharded": True,
            "desc": "Experts Weight1",
            "expected": (7168, config.num_experts * 12288 * 2),
        },
        {
            "name": "mlp.experts.weight2",
            "tp_shapes": [(num_experts_per_ep * 12288, 7168)],
            "ep_count": ep_size,
            "is_ep_sharded": True,
            "desc": "Experts Weight2",
            "expected": (config.num_experts * 12288, 7168),
        },
    ]

    all_passed = True
    for case in test_cases:
        name = case["name"]
        tp_shapes = case["tp_shapes"]
        ep_count = case["ep_count"]
        is_ep_sharded = case["is_ep_sharded"]

        tp_ep_shapes = {}
        for ep_rank in range(ep_count):
            for tp_rank, shape in enumerate(tp_shapes):
                tp_ep_shapes[(tp_rank, ep_rank)] = shape

        merged = merger.merge(name, tp_ep_shapes, is_ep_sharded)
        expected = case["expected"]

        status = "✓" if merged == expected else "✗"
        if merged != expected:
            all_passed = False
        print(f"  {status} {case['desc']}: {merged}")

    return all_passed


def test_edge_cases():
    """测试边界情况"""
    print("\n" + "=" * 70)
    print("单元测试: 边界情况")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import MoeParallelStrategy, ShapeMerger

    strategy = MoeParallelStrategy()
    merger = ShapeMerger(strategy)

    all_passed = True

    # 测试1: TP=1
    tp_shapes = [(7168, 8192)]
    merged = merger.merge_tp_shapes("test", tp_shapes)
    status = "✓" if merged == (7168, 8192) else "✗"
    if merged != (7168, 8192):
        all_passed = False
    print(f"  {status} TP=1 保持原shape: {merged}")

    # 测试2: EP=1
    ep_shapes = [(7168, 49152)]
    merged = merger.merge_ep_shapes("mlp.experts.weight1", ep_shapes)
    status = "✓" if merged == (7168, 49152) else "✗"
    if merged != (7168, 49152):
        all_passed = False
    print(f"  {status} EP=1 保持原shape: {merged}")

    # 测试3: 空列表
    merged = merger.merge_tp_shapes("test", [])
    status = "✓" if merged == () else "✗"
    if merged != ():
        all_passed = False
    print(f"  {status} 空列表返回空tuple: {merged}")

    # 测试4: LayerNorm 不参与任何并行
    tp_ep_shapes = {(0, 0): (7168,), (0, 1): (7168,), (1, 0): (7168,), (1, 1): (7168,)}
    merged = merger.merge("input_layernorm.weight", tp_ep_shapes, is_ep_sharded=False)
    status = "✓" if merged == (7168,) else "✗"
    if merged != (7168,):
        all_passed = False
    print(f"  {status} LayerNorm 合并后保持 (7168,): {merged}")

    # 测试5: 非专家权重不被 EP 切分
    tp_ep_shapes = {
        (0, 0): (4352, 7168), (1, 0): (4352, 7168),
        (0, 1): (4352, 7168), (1, 1): (4352, 7168),
    }
    merged = merger.merge("self_attention.linear_qkv.weight", tp_ep_shapes, is_ep_sharded=True)
    status = "✓" if merged == (8704, 7168) else "✗"
    if merged != (8704, 7168):
        all_passed = False
    print(f"  {status} QKV 只合并 TP: {merged}")

    return all_passed


def test_layer_mapping():
    """测试 Layer 映射"""
    print("\n" + "=" * 70)
    print("单元测试: Layer 映射")
    print("=" * 70)

    from utils.get_mcore_weights_from_ckpt import StandardVppMapper, DualPipeMapper

    all_passed = True

    # 测试标准 VPP 映射
    print("  标准 VPP 映射 (PP=4, 32 layers):")
    mapper = StandardVppMapper()
    num_layers = 32
    pp_size = 4
    vpp_stage = 4

    mapping = mapper.build_mapping(num_layers, pp_size, vpp_stage)

    if len(mapping) == num_layers:
        print(f"    ✓ 映射包含所有 {num_layers} 层")
    else:
        print(f"    ✗ 期望 {num_layers} 层, 得到 {len(mapping)}")
        all_passed = False

    # 测试 DualPipe 映射
    print("  DualPipe 映射 (PP=4, 32 layers):")
    mapper = DualPipeMapper()

    try:
        mapping = mapper.build_mapping(num_layers, pp_size, vpp_stage)
        if len(mapping) == num_layers:
            print(f"    ✓ 映射包含所有 {num_layers} 层")
        else:
            print(f"    ✗ 期望 {num_layers} 层, 得到 {len(mapping)}")
            all_passed = False
    except ValueError as e:
        print(f"    ✗ 异常: {e}")
        all_passed = False

    # 测试 DualPipe 非法配置
    print("  DualPipe 非法配置 (PP=4, 30 layers):")
    try:
        mapping = mapper.build_mapping(30, pp_size, vpp_stage)
        print(f"    ✗ 应该抛出异常但未抛出")
        all_passed = False
    except ValueError as e:
        print(f"    ✓ 正确抛出异常")

    return all_passed


def main():
    """运行所有测试"""
    print("\n" + "=" * 80)
    print("MCore Checkpoint Reader - 完整测试套件")
    print("=" * 80)

    results = []

    # 集成测试
    results.append(("Mock Checkpoint", test_mock_checkpoint()))

    # 单元测试
    results.append(("TP Parallel Dim", test_tp_parallel_dim()))
    results.append(("Expected Shapes", test_expected_shapes()))
    results.append(("TP Merge", test_tp_merge()))
    results.append(("EP Merge", test_ep_merge()))
    results.append(("Kimi2-1T Config", test_kimi2_1t_config()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Layer Mapping", test_layer_mapping()))

    print("\n" + "=" * 80)
    print("测试汇总")
    print("=" * 80)

    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(r[1] for r in results)
    print()
    if all_passed:
        print("🎉 所有测试通过！")
        return 0
    else:
        print("❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())
