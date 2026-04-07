#!/usr/bin/env python3
"""
测试 get_mcore_weights.py 的功能
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_mcore_weights import (
    MCoreWeightExtractor,
    _resolve_iter_dir,
    _mp_prefix,
    _get_torch_dtype_name,
    _get_tensor_size,
)


class TestGetMCoreWeights(unittest.TestCase):
    """测试 get_mcore_weights.py 的功能"""

    def setUp(self):
        """设置测试环境，创建临时目录"""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, 'checkpoint')
        os.makedirs(self.checkpoint_dir)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def create_mock_checkpoint(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        num_layers: int = 2,
    ):
        """创建模拟的 MCore checkpoint 结构"""
        # 创建 iter_0000001 目录
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir, exist_ok=True)

        # 创建 latest_checkpointed_iteration.txt
        with open(os.path.join(self.checkpoint_dir, 'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                for ep_rank in range(ep_size):
                    # 构建 mp_rank 目录名
                    prefix = _mp_prefix(tp_rank, pp_rank, ep_rank, tp_size, pp_size, ep_size)
                    rank_dir = os.path.join(iter_dir, prefix)
                    os.makedirs(rank_dir, exist_ok=True)

                    # 创建模拟的 state dict
                    state_dict = self._create_mock_state_dict(
                        tp_rank, pp_rank, ep_rank, tp_size, pp_size, ep_size, num_layers
                    )
                    
                    # 保存 checkpoint
                    ckpt_path = os.path.join(rank_dir, 'model_optim_rng.pt')
                    torch.save({'model': state_dict}, ckpt_path)

    def _create_mock_state_dict(
        self,
        tp_rank: int,
        pp_rank: int,
        ep_rank: int,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        num_layers: int,
    ) -> dict:
        """创建模拟的 state dict"""
        state = {}
        
        # 只在 PP rank 0 添加 embedding
        if pp_rank == 0:
            vocab_size = 100
            hidden_size = 64
            state['embedding.word_embeddings.weight'] = torch.randn(
                vocab_size // tp_size, hidden_size, dtype=torch.float16
            )
        
        # 计算该 PP rank 的层数
        layers_per_pp = num_layers // pp_size
        start_layer = pp_rank * layers_per_pp
        
        for local_idx in range(layers_per_pp):
            layer_idx = start_layer + local_idx
            
            # Attention
            state[f'decoder.layers.{local_idx}.self_attention.linear_qkv.weight'] = torch.randn(
                192, 64, dtype=torch.float16
            )
            state[f'decoder.layers.{local_idx}.self_attention.linear_proj.weight'] = torch.randn(
                64, 64, dtype=torch.float16
            )
            state[f'decoder.layers.{local_idx}.self_attention.q_layernorm.weight'] = torch.randn(
                64, dtype=torch.float32
            )
            state[f'decoder.layers.{local_idx}.self_attention.k_layernorm.weight'] = torch.randn(
                64, dtype=torch.float32
            )
            
            # LayerNorm
            state[f'decoder.layers.{local_idx}.input_layernorm.weight'] = torch.randn(
                64, dtype=torch.float32
            )
            state[f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'] = torch.randn(
                64, dtype=torch.float32
            )
            
            # MLP (假设 Dense MLP)
            state[f'decoder.layers.{local_idx}.mlp.linear_fc1.weight'] = torch.randn(
                256, 64, dtype=torch.float16
            )
            state[f'decoder.layers.{local_idx}.mlp.linear_fc2.weight'] = torch.randn(
                64, 128, dtype=torch.float16
            )
        
        # 只在最后一个 PP rank 添加 output
        if pp_rank == pp_size - 1:
            state['decoder.final_layernorm.weight'] = torch.randn(64, dtype=torch.float32)
            state['output_layer.weight'] = torch.randn(
                100 // tp_size, 64, dtype=torch.float16
            )
        
        return state

    def test_resolve_iter_dir(self):
        """测试迭代目录解析"""
        # 测试 latest 文件指向
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir)
        with open(os.path.join(self.checkpoint_dir, 'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')
        
        resolved = _resolve_iter_dir(self.checkpoint_dir)
        self.assertEqual(resolved, iter_dir)

    def test_mp_prefix(self):
        """测试 mp_rank 前缀生成"""
        # TP only
        self.assertEqual(_mp_prefix(0, 0, 0, 1, 1, 1), 'mp_rank_00')
        # TP + PP
        self.assertEqual(_mp_prefix(1, 2, 0, 2, 4, 1), 'mp_rank_01_002')
        # TP + EP
        self.assertEqual(_mp_prefix(1, 0, 3, 2, 1, 8), 'mp_rank_01_003')
        # TP + PP + EP
        self.assertEqual(_mp_prefix(1, 2, 3, 2, 4, 8), 'mp_rank_01_002_003')

    def test_get_torch_dtype_name(self):
        """测试数据类型名称获取"""
        self.assertEqual(_get_torch_dtype_name(torch.float16), 'float16')
        self.assertEqual(_get_torch_dtype_name(torch.bfloat16), 'bfloat16')
        self.assertEqual(_get_torch_dtype_name(torch.float32), 'float32')
        self.assertEqual(_get_torch_dtype_name(torch.int64), 'int64')

    def test_get_tensor_size(self):
        """测试张量大小计算"""
        # float16: 2 bytes per element
        tensor = torch.randn(10, 20, dtype=torch.float16)
        self.assertEqual(_get_tensor_size(tensor), 10 * 20 * 2)
        
        # float32: 4 bytes per element
        tensor = torch.randn(10, 20, dtype=torch.float32)
        self.assertEqual(_get_tensor_size(tensor), 10 * 20 * 4)

    def test_extract_weights_simple(self):
        """测试简单的权重提取"""
        self.create_mock_checkpoint(tp_size=1, pp_size=1, ep_size=1, num_layers=2)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
        )
        
        result = extractor.extract_weights()
        
        # 验证结果结构
        self.assertIn('total_size', result.metadata)
        self.assertGreater(result.total_size, 0)
        self.assertGreater(len(result.weight_map), 0)

    def test_extract_weights_with_parallelism(self):
        """测试带并行配置的权重提取"""
        self.create_mock_checkpoint(tp_size=2, pp_size=2, ep_size=1, num_layers=4)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=2,
            pp_size=2,
            ep_size=1,
            verbose=False,
        )
        
        result = extractor.extract_weights()
        
        # 验证结果
        self.assertGreater(result.total_size, 0)
        self.assertGreater(len(result.weight_map), 0)

    def test_json_output(self):
        """测试 JSON 输出"""
        self.create_mock_checkpoint(tp_size=1, pp_size=1, ep_size=1, num_layers=2)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
        )
        
        output = extractor.get_json_output()
        
        # 验证 JSON 结构
        self.assertIn('metadata', output)
        self.assertIn('weight_map', output)
        
        # 验证 metadata 字段
        metadata = output['metadata']
        self.assertIn('total_size', metadata)
        
        # 验证 weight_map 字段
        weight_map = output['weight_map']
        self.assertGreater(len(weight_map), 0)
        
        # 验证每个权重的字段
        for name, info in weight_map.items():
            self.assertIn('shape', info)
            self.assertIn('dtype', info)
            self.assertIsInstance(info['shape'], list)
            self.assertIsInstance(info['dtype'], str)

    def test_save_json(self):
        """测试保存 JSON 到文件"""
        self.create_mock_checkpoint(tp_size=1, pp_size=1, ep_size=1, num_layers=2)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
        )
        
        output_path = os.path.join(self.test_dir, 'weights_info.json')
        extractor.save_json(output_path)
        
        # 验证文件存在且格式正确
        self.assertTrue(os.path.exists(output_path))
        
        with open(output_path, 'r') as f:
            data = json.load(f)
        
        self.assertIn('metadata', data)
        self.assertIn('weight_map', data)

    def test_hf_mapping(self):
        """测试 HF 格式权重名称映射"""
        self.create_mock_checkpoint(tp_size=1, pp_size=1, ep_size=1, num_layers=1)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
        )
        
        output = extractor.get_json_output()
        weight_map = output['weight_map']
        
        # 验证关键权重被正确映射
        self.assertIn('model.embed_tokens.weight', weight_map)
        self.assertIn('model.norm.weight', weight_map)
        self.assertIn('lm_head.weight', weight_map)
        self.assertIn('model.layers.0.input_layernorm.weight', weight_map)
        self.assertIn('model.layers.0.post_attention_layernorm.weight', weight_map)
        self.assertIn('model.layers.0.self_attn.o_proj.weight', weight_map)
        self.assertIn('model.layers.0.mlp.down_proj.weight', weight_map)

    def create_mock_vpp_checkpoint(
        self,
        tp_size: int = 1,
        pp_size: int = 2,
        vpp_size: int = 2,
        num_layers: int = 4,
    ):
        """创建模拟的 VPP MCore checkpoint 结构
        
        模拟 VPP 场景，每个 PP rank 有多个 VPP stage，
        每个 VPP stage 中的权重使用局部 layer 索引（从 0 开始）
        所有 VPP stages 存储在同一个 checkpoint 文件中
        """
        # 创建 iter_0000001 目录
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir, exist_ok=True)

        # 创建 latest_checkpointed_iteration.txt
        with open(os.path.join(self.checkpoint_dir, 'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        # 计算每个 PP/VPP 的层数
        layers_per_pp = num_layers // pp_size
        layers_per_vpp = layers_per_pp // vpp_size

        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                prefix = f'mp_rank_{tp_rank:02}_{pp_rank:03}'
                rank_dir = os.path.join(iter_dir, prefix)
                os.makedirs(rank_dir, exist_ok=True)

                # VPP 场景：所有 VPP stages 的权重存储在同一个文件中
                full_state = {}
                
                for vpp_rank in range(vpp_size):
                    # 创建模拟的 state dict，使用局部 layer 索引
                    state_dict = {}
                    
                    for local_idx in range(layers_per_vpp):
                        # local_idx 是局部索引（0, 1, ...）
                        # 实际保存的权重名称使用 local_idx
                        state_dict[f'decoder.layers.{local_idx}.self_attention.linear_proj.weight'] = torch.randn(
                            64, 64, dtype=torch.float16
                        )
                        state_dict[f'decoder.layers.{local_idx}.input_layernorm.weight'] = torch.randn(
                            64, dtype=torch.float32
                        )
                        state_dict[f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'] = torch.randn(
                            64, dtype=torch.float32
                        )
                    
                    # VPP 场景：权重存储在 modelX 键下
                    full_state[f'model{vpp_rank}'] = state_dict
                
                ckpt_path = os.path.join(rank_dir, 'model_optim_rng.pt')
                torch.save(full_state, ckpt_path)

    def test_vpp_layer_mapping(self):
        """测试 VPP 场景下的 layer 映射 - 关键修复验证
        
        此测试验证：
        1. 不同 VPP stage 中的相同 local_idx 能正确映射到不同的全局 layer id
        2. 不会只提取 layer 0 和 1，而是提取所有层
        """
        # 创建 8 层的 VPP checkpoint，PP=2，VPP=2（总共 4 个 stages）
        self.create_mock_vpp_checkpoint(tp_size=1, pp_size=2, vpp_size=2, num_layers=8)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=2,
            ep_size=1,
            vpp_stage=2,  # 每个 VPP stage 有 2 层
            num_layers=8,
            verbose=False,
        )
        
        output = extractor.get_json_output()
        weight_map = output['weight_map']
        metadata = output['metadata']
        
        # 验证所有 8 层的权重都被提取
        layer_ids = set()
        for name in weight_map.keys():
            match = __import__('re').match(r'model\.layers\.(\d+)\.', name)
            if match:
                layer_ids.add(int(match.group(1)))
        
        # 关键验证：应该有 8 个不同的 layer id（0-7），而不是只有 2 个
        self.assertEqual(len(layer_ids), 8, 
                        f"应该提取所有 8 层的权重，但实际只提取了 {len(layer_ids)} 层: {sorted(layer_ids)}")
        self.assertEqual(sorted(layer_ids), list(range(8)))

    def test_tp_weight_merging(self):
        """测试 TP (Tensor Parallel) 权重合并功能
        
        验证：
        1. 所有 TP rank 的权重被正确合并
        2. 合并后的形状是原始形状（而非切分后的形状）
        """
        # 创建 TP=2, PP=1 的 checkpoint
        self.create_mock_checkpoint(tp_size=2, pp_size=1, ep_size=1, num_layers=2)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=2,
            pp_size=1,
            ep_size=1,
            verbose=False,
        )
        
        output = extractor.get_json_output()
        weight_map = output['weight_map']
        metadata = output['metadata']
        
        # 验证 embedding 的形状已合并
        # TP=2 时，每个 rank 的 vocab_size 是 50，合并后应该是 100
        embed_shape = weight_map['model.embed_tokens.weight']['shape']
        self.assertEqual(embed_shape[0], 100, 
                        f"Embedding 第0维应该合并为 100，实际是 {embed_shape[0]}")
        
        # 验证 lm_head 的形状已合并
        lm_head_shape = weight_map['lm_head.weight']['shape']
        self.assertEqual(lm_head_shape[0], 100,
                        f"LM head 第0维应该合并为 100，实际是 {lm_head_shape[0]}")
        
        # 验证 metadata 包含合并信息
        self.assertIn('note', metadata)
        self.assertIn('TP=2', metadata['note'])

    def test_fused_weight_expansion(self):
        """测试融合权重的展开
        
        验证:
        1. linear_qkv 展开为 q_proj, k_proj, v_proj
        2. linear_fc1 展开为 gate_proj, up_proj
        """
        self.create_mock_checkpoint(tp_size=1, pp_size=1, ep_size=1, num_layers=2)
        
        extractor = MCoreWeightExtractor(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
        )
        
        output = extractor.get_json_output()
        weight_map = output['weight_map']
        
        # 验证 QKV 展开
        self.assertIn('model.layers.0.self_attn.q_proj.weight', weight_map)
        self.assertIn('model.layers.0.self_attn.k_proj.weight', weight_map)
        self.assertIn('model.layers.0.self_attn.v_proj.weight', weight_map)
        
        # 验证 MLP gate_up 展开
        self.assertIn('model.layers.0.mlp.gate_proj.weight', weight_map)
        self.assertIn('model.layers.0.mlp.up_proj.weight', weight_map)
        
        # 验证 down_proj 直接映射（不是融合权重）
        self.assertIn('model.layers.0.mlp.down_proj.weight', weight_map)


if __name__ == '__main__':
    unittest.main()
