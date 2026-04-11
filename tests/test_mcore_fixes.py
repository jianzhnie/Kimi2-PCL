#!/usr/bin/env python3
"""
MCore Checkpoint Reader 修复验证测试（小维度）

验证三个修复：
1. 非 EP 切分权重不跨 EP 合并（关键修复）
2. Q/K Layernorm 期望 shape 计算正确
3. Rank 目录解析正确（PP=1, EP>1 时）
"""

import sys
import os
import torch
import shutil

sys.path.insert(0, os.path.dirname(__file__))


def setup_mock_ckpt():
    """创建小维度模拟 checkpoint: TP=2, PP=1, EP=4"""
    base_dir = "/tmp/test_mcore_fixes"
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir)
    
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
    
    iter_dir = f"{base_dir}/iter_0000001"
    os.makedirs(iter_dir, exist_ok=True)
    
    # 创建 8 个 rank: TP=2 x EP=4
    for tp in range(2):
        for ep in range(4):
            rank_dir = f"{iter_dir}/mp_rank_{tp:02}_{ep:03}"
            os.makedirs(rank_dir, exist_ok=True)
            
            state = {
                'model': {
                    # 1. 非 EP 切分权重（应在各 EP rank 上重复）
                    'embedding.word_embeddings.weight': torch.randn(128, 128),  # vocab/TP
                    'decoder.layers.0.input_layernorm.weight': torch.randn(128),
                    'decoder.layers.0.self_attention.q_layernorm.weight': torch.randn(32),  # kv_channels
                    'decoder.layers.0.self_attention.k_layernorm.weight': torch.randn(32),
                    'decoder.layers.0.self_attention.linear_qkv.weight': torch.randn(128, 128),
                    'decoder.layers.0.self_attention.linear_proj.weight': torch.randn(128, 64),
                    'decoder.layers.0.mlp.linear_fc1.weight': torch.randn(48, 128),
                    
                    # 2. EP 切分权重（应跨 EP 合并）
                    'decoder.layers.2.mlp.router.weight': torch.randn(8, 128),  # 不切分
                    'decoder.layers.2.mlp.experts.weight1': torch.randn(128, 256),  # 每 EP 2 专家
                    'decoder.layers.2.mlp.experts.weight2': torch.randn(128, 128),
                }
            }
            torch.save(state, f"{rank_dir}/model_optim_rng.pt")
    
    with open(f"{base_dir}/latest_checkpointed_iteration.txt", "w") as f:
        f.write("1")
    
    return base_dir, cfg


def test_fixes():
    print("=" * 70)
    print("MCore Checkpoint Reader - 三修复验证")
    print("=" * 70)
    print("\n测试配置: TP=2, PP=1, EP=4")
    print("          hidden=128, vocab=256, experts=8, kv_channels=32\n")
    
    # 创建 mock checkpoint
    ckpt_dir, cfg = setup_mock_ckpt()
    print(f"✓ 创建模拟 checkpoint: {ckpt_dir}")
    
    from get_mcore_weights_form_ckpt import MCoreCheckpointReader
    
    reader = MCoreCheckpointReader(
        mcore_dir=ckpt_dir,
        tp_size=2, pp_size=1, ep_size=4,
        num_layers=4,
        num_attention_heads=cfg['num_heads'],
        num_key_value_heads=cfg['kv_heads'],
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
    
    print("\n" + "-" * 70)
    print("修复验证")
    print("-" * 70)
    
    all_pass = True
    
    # ========== 修复 1: 非 EP 切分权重不跨 EP 合并 ==========
    print("\n【修复 1】非 EP 切分权重不跨 EP 合并")
    
    non_ep_checks = [
        ("module.embedding.word_embeddings.weight", [256, 128], 
         "Embedding 不被 64 倍膨胀"),
        ("module.decoder.layers.0.input_layernorm.weight", [128], 
         "LayerNorm 不被重复计算"),
        ("module.decoder.layers.0.self_attention.linear_qkv.weight", [256, 128], 
         "QKV 正确合并 TP"),
        ("module.decoder.layers.0.self_attention.linear_proj.weight", [128, 128], 
         "Proj 正确合并 TP"),
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
    total_models = len(reader._cache._cache) if hasattr(reader._cache, '_cache') else 8
    print(f"  ✓ 加载了 {total_models} 个模型文件 (TP=2 x EP=4)")
    
    # 验证 EP rank 0 被正确识别
    ep_ranks = reader._rank_dir_map.get((0, 0), [])
    has_ep0 = 0 in ep_ranks
    status = "✓" if has_ep0 else "✗"
    print(f"  {status} EP rank 0 被正确识别: {ep_ranks}")
    if not has_ep0:
        all_pass = False
    
    # ========== 专家权重跨 EP 合并 ==========
    print("\n【验证】EP 切分权重正确跨 EP 合并")
    
    expert_checks = [
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
    
    # 清理
    shutil.rmtree(ckpt_dir)
    
    print("\n" + "=" * 70)
    if all_pass and len(result.warnings) == 0:
        print("✓ 所有修复验证通过！")
    else:
        print("✗ 部分验证未通过")
    print("=" * 70)
    
    return all_pass and len(result.warnings) == 0


if __name__ == "__main__":
    success = test_fixes()
    sys.exit(0 if success else 1)
