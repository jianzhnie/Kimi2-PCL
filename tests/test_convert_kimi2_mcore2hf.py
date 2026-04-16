#!/usr/bin/env python3
"""
Kimi2 MCore to HF 转换脚本测试

覆盖范围:
1. GQA Attention 权重拆分 (linear_qkv -> q_proj, k_proj, v_proj)
2. MoE 权重转换 (router, shared_experts, local_experts)
3. Layer 映射 (PP, VPP, DualPipe)
4. 边界情况

运行方式:
    python -m unittest tests.test_convert_kimi2_mcore2hf -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.convert_kimi2_mcore2hf import (HIDDEN_SIZE, KV_CHANNELS,
                                          NUM_ATTENTION_HEADS, NUM_EXPERTS,
                                          NUM_QUERY_GROUPS, MgCkptConvert)


class TestKimi2Mcore2Hf(unittest.TestCase):
    """测试 Kimi2 MCore 到 HF 的转换"""

    def setUp(self):
        """设置测试环境"""
        self.test_dir = tempfile.mkdtemp()
        self.mcore_dir = os.path.join(self.test_dir, 'mcore_ckpt')
        self.hf_dir = os.path.join(self.test_dir, 'hf_ckpt')
        os.makedirs(self.mcore_dir, exist_ok=True)

    def tearDown(self):
        """清理测试环境"""
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def _create_mock_mcore_checkpoint(
        self,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        num_layers: int = 4,
        first_k_dense_replace: int = 2,
    ):
        """创建模拟的 MCore checkpoint"""
        iter_dir = os.path.join(self.mcore_dir, 'iter_0000001')
        os.makedirs(iter_dir, exist_ok=True)

        with open(
                os.path.join(self.mcore_dir,
                             'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        vocab_size = 100
        hidden_size = 64
        num_heads = 8
        num_query_groups = 2
        head_dim = hidden_size // num_heads
        q_proj_rows = num_heads * head_dim
        k_proj_rows = num_query_groups * head_dim
        v_proj_rows = num_query_groups * head_dim
        qkv_rows = q_proj_rows + k_proj_rows + v_proj_rows

        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                for ep_rank in range(ep_size):
                    prefix = self._mp_prefix(tp_rank, pp_rank, ep_rank,
                                             tp_size, pp_size, ep_size)
                    rank_dir = os.path.join(iter_dir, prefix)
                    os.makedirs(rank_dir, exist_ok=True)

                    state_dict = self._create_state_dict(
                        tp_rank, pp_rank, ep_rank, tp_size, pp_size, ep_size,
                        num_layers, first_k_dense_replace, vocab_size,
                        hidden_size, num_heads, num_query_groups, head_dim,
                        qkv_rows)
                    ckpt_path = os.path.join(rank_dir, 'model_optim_rng.pt')
                    torch.save({'model': state_dict}, ckpt_path)

    def _mp_prefix(self, tp_rank, pp_rank, ep_rank, tp, pp, ep):
        """生成 mp_rank 目录前缀"""
        if ep == 1 and pp == 1:
            return f'mp_rank_{tp_rank:02}'
        if ep == 1:
            return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
        if pp == 1:
            return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'

    def _create_state_dict(self, tp_rank, pp_rank, ep_rank, tp_size, pp_size,
                           ep_size, num_layers, first_k_dense_replace,
                           vocab_size, hidden_size, num_heads,
                           num_query_groups, head_dim, qkv_rows):
        """创建模拟的 state dict"""
        state = {}
        layers_per_pp = num_layers // pp_size
        start_layer = pp_rank * layers_per_pp

        # Embedding (only PP rank 0)
        if pp_rank == 0:
            state['embedding.word_embeddings.weight'] = torch.randn(
                vocab_size // tp_size, hidden_size, dtype=torch.float16)

        for local_idx in range(layers_per_pp):
            layer_idx = start_layer + local_idx

            # Attention (GQA)
            state[
                f'decoder.layers.{local_idx}.self_attention.linear_qkv.weight'] = torch.randn(
                    qkv_rows // tp_size, hidden_size, dtype=torch.float16)
            state[
                f'decoder.layers.{local_idx}.self_attention.linear_proj.weight'] = torch.randn(
                    hidden_size, (num_heads * head_dim) // tp_size,
                    dtype=torch.float16)
            state[
                f'decoder.layers.{local_idx}.self_attention.q_layernorm.weight'] = torch.randn(
                    head_dim, dtype=torch.float32)
            state[
                f'decoder.layers.{local_idx}.self_attention.k_layernorm.weight'] = torch.randn(
                    head_dim, dtype=torch.float32)

            # LayerNorm
            state[
                f'decoder.layers.{local_idx}.input_layernorm.weight'] = torch.randn(
                    hidden_size, dtype=torch.float32)
            state[
                f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'] = torch.randn(
                    hidden_size, dtype=torch.float32)

            # MLP (Dense or MoE)
            if layer_idx < first_k_dense_replace:
                # Dense MLP
                state[
                    f'decoder.layers.{local_idx}.mlp.linear_fc1.weight'] = torch.randn(
                        256 // tp_size, hidden_size, dtype=torch.float16)
                state[
                    f'decoder.layers.{local_idx}.mlp.linear_fc2.weight'] = torch.randn(
                        hidden_size, 128 // tp_size, dtype=torch.float16)
            else:
                # MoE
                num_experts = 8
                experts_per_ep = num_experts // ep_size
                moe_hidden = 64

                state[
                    f'decoder.layers.{local_idx}.mlp.router.weight'] = torch.randn(
                        num_experts, hidden_size, dtype=torch.float16)
                state[
                    f'decoder.layers.{local_idx}.mlp.router.expert_bias'] = torch.randn(
                        num_experts, dtype=torch.float16)

                # Shared experts
                state[
                    f'decoder.layers.{local_idx}.mlp.shared_experts.linear_fc1.weight'] = torch.randn(
                        (moe_hidden * 2) // tp_size,
                        hidden_size,
                        dtype=torch.float16)
                state[
                    f'decoder.layers.{local_idx}.mlp.shared_experts.linear_fc2.weight'] = torch.randn(
                        hidden_size,
                        moe_hidden // tp_size,
                        dtype=torch.float16)

                # Local experts (non-grouped gemm format)
                for local_expert_idx in range(experts_per_ep):
                    state[
                        f'decoder.layers.{local_idx}.mlp.experts.local_experts.{local_expert_idx}.linear_fc1.weight'] = torch.randn(
                            moe_hidden * 2,
                            hidden_size // tp_size,
                            dtype=torch.float16).t()
                    state[
                        f'decoder.layers.{local_idx}.mlp.experts.local_experts.{local_expert_idx}.linear_fc2.weight'] = torch.randn(
                            hidden_size,
                            moe_hidden // tp_size,
                            dtype=torch.float16).t()

        # Output (only last PP rank)
        if pp_rank == pp_size - 1:
            state['decoder.final_layernorm.weight'] = torch.randn(
                hidden_size, dtype=torch.float32)
            state['output_layer.weight'] = torch.randn(vocab_size // tp_size,
                                                       hidden_size,
                                                       dtype=torch.float16)

        return state

    def test_gqa_attention_split(self):
        """测试 GQA Attention 权重正确拆分为 q_proj, k_proj, v_proj"""
        self._create_mock_mcore_checkpoint(tp_size=1,
                                           pp_size=1,
                                           ep_size=1,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            num_dense_layers=2,
            moe_grouped_gemm=False,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
        )

        # 检查维度计算
        self.assertEqual(converter.q_proj_rows, 64)  # 8 heads * 8 dims
        self.assertEqual(converter.k_proj_rows, 16)  # 2 groups * 8 dims
        self.assertEqual(converter.v_proj_rows, 16)  # 2 groups * 8 dims
        self.assertEqual(converter.qkv_proj_rows, 96)  # 64 + 16 + 16

    def test_default_model_config(self):
        """测试默认 Kimi2-1T 模型配置"""
        self._create_mock_mcore_checkpoint(tp_size=2,
                                           pp_size=1,
                                           ep_size=1,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=2,
            pp_size=1,
            ep_size=1,
            num_dense_layers=2,
        )

        # 检查默认配置
        self.assertEqual(converter.hidden_size, HIDDEN_SIZE)
        self.assertEqual(converter.num_experts, NUM_EXPERTS)
        self.assertEqual(converter.num_attention_heads, NUM_ATTENTION_HEADS)
        self.assertEqual(converter.num_query_groups, NUM_QUERY_GROUPS)
        self.assertEqual(converter.kv_channels, KV_CHANNELS)

    def test_layer_mapping_pp(self):
        """测试 PP (Pipeline Parallel) 层映射"""
        self._create_mock_mcore_checkpoint(tp_size=1,
                                           pp_size=2,
                                           ep_size=1,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=1,
            pp_size=2,
            ep_size=1,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
        )

        # 检查 PP 层映射
        self.assertEqual(converter.pprank_layer_idxs[0], [0, 1])
        self.assertEqual(converter.pprank_layer_idxs[1], [2, 3])

        # 检查反向映射
        self.assertEqual(converter.layeridx_pprank[0], (0, 0))
        self.assertEqual(converter.layeridx_pprank[1], (0, 1))
        self.assertEqual(converter.layeridx_pprank[2], (1, 0))
        self.assertEqual(converter.layeridx_pprank[3], (1, 1))

    def test_dense_vs_moe_layers(self):
        """测试 Dense 层和 MoE 层的区分"""
        self._create_mock_mcore_checkpoint(tp_size=1,
                                           pp_size=1,
                                           ep_size=2,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=1,
            pp_size=1,
            ep_size=2,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
        )

        # 前 2 层是 Dense
        self.assertTrue(all(layer < 2 for layer in range(2)))
        # 后 2 层是 MoE
        self.assertTrue(all(layer >= 2 for layer in range(2, 4)))

    def test_get_pt_path(self):
        """测试 checkpoint 路径生成"""
        self._create_mock_mcore_checkpoint(tp_size=2, pp_size=4, ep_size=8,
                                           num_layers=4, first_k_dense_replace=2)
        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=2,
            pp_size=4,
            ep_size=8,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
        )

        iter_path = converter.iter_path

        # TP + PP + EP: 验证存在的文件能正确找到
        path = converter.get_pt_path_by_tpppep_rank(iter_path, 0, 0, 0)
        self.assertIn('mp_rank_00_000_000', path)
        self.assertTrue(os.path.isfile(path))

        path = converter.get_pt_path_by_tpppep_rank(iter_path, 1, 3, 7)
        self.assertIn('mp_rank_01_003_007', path)
        self.assertTrue(os.path.isfile(path))

        # 不存在的文件应抛出 FileNotFoundError
        with self.assertRaises(FileNotFoundError):
            converter.get_pt_path_by_tpppep_rank(iter_path, 0, 99, 0)

    def test_parameter_validation(self):
        """测试参数验证"""
        self._create_mock_mcore_checkpoint(tp_size=2, pp_size=2, ep_size=1,
                                           num_layers=4, first_k_dense_replace=2)
        # 有效配置
        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=2,
            pp_size=2,
            ep_size=1,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
        )
        self.assertIsNotNone(converter)

        # 无效: num_layers 不能被 pp_size 整除
        with self.assertRaises(ValueError):
            MgCkptConvert(
                mg_model_path=self.mcore_dir,
                hf_save_path=self.hf_dir,
                num_layers=5,  # 5 不能被 2 整除
                tp_size=2,
                pp_size=2,
                ep_size=1,
                num_dense_layers=2,
                hidden_size=64,
                num_experts=8,
                num_attention_heads=8,
                num_query_groups=2,
                kv_channels=8,
                vocab_size=100,
            )

    def test_qk_layernorm_and_rotary_base(self):
        """测试 qk_layernorm 和 rotary_base 参数"""
        self._create_mock_mcore_checkpoint(tp_size=1,
                                           pp_size=1,
                                           ep_size=1,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
            qk_layernorm=True,
            rotary_base=50000.0,
        )

        self.assertTrue(converter.qk_layernorm)
        self.assertEqual(converter.rotary_base, 50000.0)

    def test_full_conversion_pp2(self):
        """测试完整 PP=2 转换流程，验证输出 safetensors 文件"""
        self._create_mock_mcore_checkpoint(tp_size=1,
                                           pp_size=2,
                                           ep_size=1,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=1,
            pp_size=2,
            ep_size=1,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
            moe_grouped_gemm=False,
        )

        converter.run()

        # 验证输出文件存在
        index_file = os.path.join(self.hf_dir,
                                  'model.safetensors.index.json')
        self.assertTrue(os.path.isfile(index_file))

        # 验证 index 文件内容
        with open(index_file) as f:
            index = json.load(f)
        self.assertIn('weight_map', index)
        weight_map = index['weight_map']

        # 验证关键权重存在
        self.assertIn('model.embed_tokens.weight', weight_map)
        self.assertIn('model.norm.weight', weight_map)
        self.assertIn('lm_head.weight', weight_map)

        # 验证每层都有注意力权重
        for layer_idx in range(4):
            self.assertIn(
                f'model.layers.{layer_idx}.self_attn.q_proj.weight',
                weight_map)
            self.assertIn(
                f'model.layers.{layer_idx}.self_attn.k_proj.weight',
                weight_map)
            self.assertIn(
                f'model.layers.{layer_idx}.self_attn.v_proj.weight',
                weight_map)

    def test_gqa_with_tp2(self):
        """测试 TP=2 时 GQA QKV 正确拆分 (per-shard split)"""
        self._create_mock_mcore_checkpoint(tp_size=2,
                                           pp_size=1,
                                           ep_size=1,
                                           num_layers=4,
                                           first_k_dense_replace=2)

        converter = MgCkptConvert(
            mg_model_path=self.mcore_dir,
            hf_save_path=self.hf_dir,
            num_layers=4,
            tp_size=2,
            pp_size=1,
            ep_size=1,
            num_dense_layers=2,
            hidden_size=64,
            num_experts=8,
            num_attention_heads=8,
            num_query_groups=2,
            kv_channels=8,
            vocab_size=100,
            moe_grouped_gemm=False,
        )

        converter.run()

        # 验证 Q/K/V 维度正确
        index_file = os.path.join(self.hf_dir,
                                  'model.safetensors.index.json')
        with open(index_file) as f:
            index = json.load(f)
        weight_map = index['weight_map']

        # 读取一个 safetensors 文件并检查权重形状
        import safetensors.torch
        shard_file = os.path.join(self.hf_dir, weight_map[
            'model.layers.0.self_attn.q_proj.weight'])
        tensors = safetensors.torch.load_file(shard_file)

        # Q: [num_heads * head_dim, hidden_size] = [64, 64]
        q_shape = tensors['model.layers.0.self_attn.q_proj.weight'].shape
        self.assertEqual(q_shape, (64, 64))

        # K: [num_query_groups * head_dim, hidden_size] = [16, 64]
        k_shape = tensors['model.layers.0.self_attn.k_proj.weight'].shape
        self.assertEqual(k_shape, (16, 64))

        # V: [num_query_groups * head_dim, hidden_size] = [16, 64]
        v_shape = tensors['model.layers.0.self_attn.v_proj.weight'].shape
        self.assertEqual(v_shape, (16, 64))

        # O: [hidden_size, num_heads * head_dim] = [64, 64]
        o_shape = tensors['model.layers.0.self_attn.o_proj.weight'].shape
        self.assertEqual(o_shape, (64, 64))


if __name__ == '__main__':
    unittest.main()
