#!/usr/bin/env python3
"""
MCore Checkpoint Reader 完整测试套件

覆盖范围:
1. 基础工具函数 (_resolve_iter_dir, _mp_prefix, dtype/size 计算)
2. 简单/并行权重提取 (TP, PP, EP)
3. JSON 输出与保存
4. MCore 权重名称格式
5. VPP / DualPipe Layer 映射
6. TP/EP 维度合并逻辑
7. 期望 Shape 计算 (含 SwiGLU *2)
8. 边界情况

运行方式:
    python -m unittest tests.test_get_mcore_weights_from_ckpt -v
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.get_mcore_weights_from_ckpt import (
    DualPipeMapper,  # noqa: E402
    MCoreCheckpointReader,
    ModelConfig,
    MoeParallelStrategy,
    ShapeMerger,
    StandardVppMapper,
    _get_tensor_size,
    _get_torch_dtype_name,
    _mp_prefix,
    _resolve_iter_dir)


class TestGetMCoreWeightsFromCkpt(unittest.TestCase):
    """测试 get_mcore_weights_from_ckpt.py 的功能"""

    def setUp(self):
        """设置测试环境，创建临时目录"""
        self.test_dir = tempfile.mkdtemp()
        self.checkpoint_dir = os.path.join(self.test_dir, 'checkpoint')
        os.makedirs(self.checkpoint_dir, exist_ok=True)

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
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir, exist_ok=True)

        with open(
                os.path.join(self.checkpoint_dir,
                             'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                for ep_rank in range(ep_size):
                    prefix = _mp_prefix(tp_rank, pp_rank, ep_rank, tp_size,
                                        pp_size, ep_size)
                    rank_dir = os.path.join(iter_dir, prefix)
                    os.makedirs(rank_dir, exist_ok=True)

                    state_dict = self._create_mock_state_dict(
                        tp_rank, pp_rank, ep_rank, tp_size, pp_size, ep_size,
                        num_layers)
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
        vocab_size = 100
        hidden_size = 64

        # 只在 PP rank 0 添加 embedding
        if pp_rank == 0:
            state['embedding.word_embeddings.weight'] = torch.randn(
                vocab_size // tp_size, hidden_size, dtype=torch.float16)

        layers_per_pp = num_layers // pp_size
        start_layer = pp_rank * layers_per_pp

        for local_idx in range(layers_per_pp):
            layer_idx = start_layer + local_idx

            # Attention
            state[
                f'decoder.layers.{local_idx}.self_attention.linear_qkv.weight'] = torch.randn(
                    192, 64, dtype=torch.float16)
            state[
                f'decoder.layers.{local_idx}.self_attention.linear_proj.weight'] = torch.randn(
                    64, 64, dtype=torch.float16)
            state[
                f'decoder.layers.{local_idx}.self_attention.q_layernorm.weight'] = torch.randn(
                    64, dtype=torch.float32)
            state[
                f'decoder.layers.{local_idx}.self_attention.k_layernorm.weight'] = torch.randn(
                    64, dtype=torch.float32)

            # LayerNorm
            state[
                f'decoder.layers.{local_idx}.input_layernorm.weight'] = torch.randn(
                    64, dtype=torch.float32)
            state[
                f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'] = torch.randn(
                    64, dtype=torch.float32)

            # MLP (Dense MLP，使用 SwiGLU 融合格式)
            state[
                f'decoder.layers.{local_idx}.mlp.linear_fc1.weight'] = torch.randn(
                    256, 64, dtype=torch.float16)
            state[
                f'decoder.layers.{local_idx}.mlp.linear_fc2.weight'] = torch.randn(
                    64, 128, dtype=torch.float16)

        # 只在最后一个 PP rank 添加 output
        if pp_rank == pp_size - 1:
            state['decoder.final_layernorm.weight'] = torch.randn(
                64, dtype=torch.float32)
            state['output_layer.weight'] = torch.randn(vocab_size // tp_size,
                                                       64,
                                                       dtype=torch.float16)

        return state

    def create_mock_vpp_checkpoint(
        self,
        tp_size: int = 1,
        pp_size: int = 2,
        vpp_size: int = 2,
        num_layers: int = 4,
    ):
        """创建模拟的 VPP MCore checkpoint 结构"""
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir, exist_ok=True)

        with open(
                os.path.join(self.checkpoint_dir,
                             'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        layers_per_pp = num_layers // pp_size
        layers_per_vpp = layers_per_pp // vpp_size

        for tp_rank in range(tp_size):
            for pp_rank in range(pp_size):
                prefix = f'mp_rank_{tp_rank:02}_{pp_rank:03}'  # noqa: E231
                rank_dir = os.path.join(iter_dir, prefix)
                os.makedirs(rank_dir, exist_ok=True)

                full_state = {}
                for vpp_rank in range(vpp_size):
                    state_dict = {}
                    for local_idx in range(layers_per_vpp):
                        state_dict[
                            f'decoder.layers.{local_idx}.self_attention.linear_proj.weight'] = torch.randn(
                                64, 64, dtype=torch.float16)
                        state_dict[
                            f'decoder.layers.{local_idx}.input_layernorm.weight'] = torch.randn(
                                64, dtype=torch.float32)
                        state_dict[
                            f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'] = torch.randn(
                                64, dtype=torch.float32)
                    full_state[f'model{vpp_rank}'] = state_dict

                ckpt_path = os.path.join(rank_dir, 'model_optim_rng.pt')
                torch.save(full_state, ckpt_path)

    def create_mock_moe_checkpoint(
        self,
        tp_size: int = 2,
        pp_size: int = 1,
        ep_size: int = 4,
        num_layers: int = 4,
    ) -> dict:
        """
        创建带 MoE 的模拟 checkpoint，用于 EP 集成测试。
        返回 cfg dict 供 reader 初始化使用。
        """
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir, exist_ok=True)

        with open(
                os.path.join(self.checkpoint_dir,
                             'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        cfg = {
            'hidden_size': 128,
            'vocab_size': 256,
            'num_experts': 8,
            'moe_ffn_hidden_size': 64,
            'ffn_hidden_size': 96,
            'num_attention_heads': 4,
            'num_query_groups': 2,
            'kv_channels': 32,
        }

        vocab_per_tp = cfg['vocab_size'] // tp_size
        ffn_per_tp_fc1 = cfg['ffn_hidden_size'] * 2 // tp_size  # SwiGLU
        ffn_per_tp_fc2 = cfg['ffn_hidden_size'] // tp_size
        qkv_per_tp = (cfg['num_attention_heads'] + 2 *
                      cfg['num_query_groups']) * cfg['kv_channels'] // tp_size
        proj_per_tp = cfg['num_attention_heads'] * cfg['kv_channels'] // tp_size
        experts_per_ep = cfg['num_experts'] // ep_size
        expert_w1_size = experts_per_ep * cfg['moe_ffn_hidden_size'] * 2
        expert_w2_size = experts_per_ep * cfg['moe_ffn_hidden_size']

        for tp in range(tp_size):
            for ep in range(ep_size):
                rank_dir = os.path.join(
                    iter_dir, f'mp_rank_{tp:02}_{ep:03}')  # noqa: E231
                os.makedirs(rank_dir, exist_ok=True)

                state = {
                    'model': {
                        # 非 EP 切分权重（各 EP rank 上重复）
                        'embedding.word_embeddings.weight':
                        torch.randn(vocab_per_tp, cfg['hidden_size']),
                        'output_layer.weight':
                        torch.randn(vocab_per_tp, cfg['hidden_size']),
                        'decoder.layers.0.input_layernorm.weight':
                        torch.randn(cfg['hidden_size']),
                        'decoder.layers.0.self_attention.q_layernorm.weight':
                        torch.randn(cfg['kv_channels']),
                        'decoder.layers.0.self_attention.k_layernorm.weight':
                        torch.randn(cfg['kv_channels']),
                        'decoder.layers.0.self_attention.linear_qkv.weight':
                        torch.randn(qkv_per_tp, cfg['hidden_size']),
                        'decoder.layers.0.self_attention.linear_qkv.bias':
                        torch.randn(qkv_per_tp),
                        'decoder.layers.0.self_attention.linear_proj.weight':
                        torch.randn(cfg['hidden_size'], proj_per_tp),
                        'decoder.layers.0.mlp.linear_fc1.weight':
                        torch.randn(ffn_per_tp_fc1, cfg['hidden_size']),
                        'decoder.layers.0.mlp.linear_fc1.bias':
                        torch.randn(ffn_per_tp_fc1),
                        'decoder.layers.0.mlp.linear_fc2.weight':
                        torch.randn(cfg['hidden_size'], ffn_per_tp_fc2),

                        # EP 相关权重
                        'decoder.layers.2.mlp.router.weight':
                        torch.randn(cfg['num_experts'], cfg['hidden_size']),
                        'decoder.layers.2.mlp.router.bias':
                        torch.randn(cfg['num_experts']),
                        'decoder.layers.2.mlp.experts.weight1':
                        torch.randn(cfg['hidden_size'], expert_w1_size),
                        'decoder.layers.2.mlp.experts.weight2':
                        torch.randn(expert_w2_size, cfg['hidden_size']),
                    }
                }
                torch.save(state, os.path.join(rank_dir, 'model_optim_rng.pt'))

        return cfg

    # ------------------------------------------------------------------
    # 基础工具函数测试
    # ------------------------------------------------------------------

    def test_resolve_iter_dir(self):
        """测试迭代目录解析"""
        iter_dir = os.path.join(self.checkpoint_dir, 'iter_0000001')
        os.makedirs(iter_dir)
        with open(
                os.path.join(self.checkpoint_dir,
                             'latest_checkpointed_iteration.txt'), 'w') as f:
            f.write('1')

        resolved = _resolve_iter_dir(self.checkpoint_dir)
        self.assertEqual(resolved, iter_dir)

    def test_mp_prefix(self):
        """测试 mp_rank 前缀生成"""
        self.assertEqual(_mp_prefix(0, 0, 0, 1, 1, 1), 'mp_rank_00')
        self.assertEqual(_mp_prefix(1, 2, 0, 2, 4, 1), 'mp_rank_01_002')
        self.assertEqual(_mp_prefix(1, 0, 3, 2, 1, 8), 'mp_rank_01_003')
        self.assertEqual(_mp_prefix(1, 2, 3, 2, 4, 8), 'mp_rank_01_002_003')

    def test_get_torch_dtype_name(self):
        """测试数据类型名称获取"""
        self.assertEqual(_get_torch_dtype_name(torch.float16), 'float16')
        self.assertEqual(_get_torch_dtype_name(torch.bfloat16), 'bfloat16')
        self.assertEqual(_get_torch_dtype_name(torch.float32), 'float32')
        self.assertEqual(_get_torch_dtype_name(torch.int64), 'int64')

    def test_get_tensor_size(self):
        """测试张量大小计算"""
        self.assertEqual(_get_tensor_size((10, 20), 'float16'), 10 * 20 * 2)
        self.assertEqual(_get_tensor_size((10, 20), 'float32'), 10 * 20 * 4)
        self.assertEqual(_get_tensor_size((10, 20), 'bfloat16'), 10 * 20 * 2)

    # ------------------------------------------------------------------
    # 集成测试
    # ------------------------------------------------------------------

    def test_extract_weights_simple(self):
        """测试简单的权重提取"""
        self.create_mock_checkpoint(tp_size=1,
                                    pp_size=1,
                                    ep_size=1,
                                    num_layers=2)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        result = reader.extract_weights()
        self.assertGreater(result.total_params, 0)
        self.assertGreater(result.total_size, 0)
        self.assertGreater(len(result.megatron_params), 0)

    def test_extract_weights_with_parallelism(self):
        """测试带并行配置的权重提取"""
        self.create_mock_checkpoint(tp_size=2,
                                    pp_size=2,
                                    ep_size=1,
                                    num_layers=4)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=2,
            pp_size=2,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        result = reader.extract_weights()
        self.assertGreater(result.total_params, 0)
        self.assertGreater(len(result.megatron_params), 0)

    def test_json_output(self):
        """测试 JSON 输出"""
        self.create_mock_checkpoint(tp_size=1,
                                    pp_size=1,
                                    ep_size=1,
                                    num_layers=2)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        output = reader.get_json_output()
        self.assertIn('megatron_params', output)
        self.assertIn('metadata', output)

        metadata = output['metadata']
        self.assertIn('total_params', metadata)
        self.assertIn('total_size_bytes', metadata)
        self.assertIn('model_config', metadata)
        self.assertIn('parallel_config', metadata)

        megatron_params = output['megatron_params']
        self.assertGreater(len(megatron_params), 0)
        for name, info in megatron_params.items():
            self.assertIn('shape', info)
            self.assertIn('dtype', info)
            self.assertIn('requires_grad', info)
            self.assertIsInstance(info['shape'], list)
            self.assertIsInstance(info['dtype'], str)
            self.assertIsInstance(info['requires_grad'], bool)

    def test_save_json(self):
        """测试保存 JSON 到文件"""
        self.create_mock_checkpoint(tp_size=1,
                                    pp_size=1,
                                    ep_size=1,
                                    num_layers=2)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        output_path = os.path.join(self.test_dir, 'weights_info.json')
        reader.save_json(output_path)
        self.assertTrue(os.path.exists(output_path))

        with open(output_path, 'r') as f:
            data = json.load(f)
        self.assertIn('megatron_params', data)
        self.assertIn('metadata', data)

    def test_mcore_name_format(self):
        """测试 MCore 权重名称格式"""
        self.create_mock_checkpoint(tp_size=1,
                                    pp_size=1,
                                    ep_size=1,
                                    num_layers=2)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        output = reader.get_json_output()
        megatron_params = output['megatron_params']

        for name in megatron_params.keys():
            self.assertTrue(name.startswith('module.'),
                            f"MCore 权重名称应以 'module.' 开头: {name}")

        self.assertIn('module.embedding.word_embeddings.weight',
                      megatron_params)
        self.assertIn('module.decoder.final_layernorm.weight', megatron_params)
        self.assertIn('module.output_layer.weight', megatron_params)
        self.assertIn('module.decoder.layers.0.input_layernorm.weight',
                      megatron_params)
        self.assertIn(
            'module.decoder.layers.0.self_attention.linear_qkv.weight',
            megatron_params)

    def test_tp_weight_merging(self):
        """测试 TP (Tensor Parallel) 权重合并功能"""
        self.create_mock_checkpoint(tp_size=2,
                                    pp_size=1,
                                    ep_size=1,
                                    num_layers=2)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=2,
            pp_size=1,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        output = reader.get_json_output()
        megatron_params = output['megatron_params']

        embed_shape = megatron_params[
            'module.embedding.word_embeddings.weight']['shape']
        self.assertEqual(embed_shape[0], 100)

        output_shape = megatron_params['module.output_layer.weight']['shape']
        self.assertEqual(output_shape[0], 100)

    def test_layer_index_conversion(self):
        """测试 layer index 转换（PP 场景）"""
        self.create_mock_checkpoint(tp_size=1,
                                    pp_size=2,
                                    ep_size=1,
                                    num_layers=4)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=2,
            ep_size=1,
            num_layers=4,
            verbose=False,
            validate_shapes=False,
        )
        output = reader.get_json_output()
        megatron_params = output['megatron_params']

        layer_ids = set()
        import re
        for name in megatron_params.keys():
            match = re.match(r'module\.decoder\.layers\.(\d+)\.', name)
            if match:
                layer_ids.add(int(match.group(1)))

        self.assertEqual(sorted(layer_ids), [0, 1, 2, 3])

    def test_fused_weights_not_expanded(self):
        """测试融合权重不会被展开（与 get_mcore_weights.py 的区别）"""
        self.create_mock_checkpoint(tp_size=1,
                                    pp_size=1,
                                    ep_size=1,
                                    num_layers=2)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=1,
            ep_size=1,
            verbose=False,
            validate_shapes=False,
        )
        output = reader.get_json_output()
        megatron_params = output['megatron_params']

        self.assertIn(
            'module.decoder.layers.0.self_attention.linear_qkv.weight',
            megatron_params)
        self.assertNotIn('module.decoder.layers.0.self_attn.q_proj.weight',
                         megatron_params)
        self.assertIn('module.decoder.layers.0.mlp.linear_fc1.weight',
                      megatron_params)
        self.assertNotIn('module.decoder.layers.0.mlp.gate_proj.weight',
                         megatron_params)

    def test_vpp_layer_mapping(self):
        """测试 VPP 场景下的 layer 映射"""
        self.create_mock_vpp_checkpoint(tp_size=1,
                                        pp_size=2,
                                        vpp_size=2,
                                        num_layers=8)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=1,
            pp_size=2,
            ep_size=1,
            num_layers=8,
            vpp_stage=2,
            verbose=False,
            validate_shapes=False,
        )
        output = reader.get_json_output()
        megatron_params = output['megatron_params']

        layer_ids = set()
        import re
        for name in megatron_params.keys():
            match = re.match(r'module\.decoder\.layers\.(\d+)\.', name)
            if match:
                layer_ids.add(int(match.group(1)))

        self.assertEqual(
            len(layer_ids), 8,
            f"应该提取所有 8 层的权重，但实际只提取了 {len(layer_ids)} 层: {sorted(layer_ids)}")
        self.assertEqual(sorted(layer_ids), list(range(8)))

    def test_ep_and_non_ep_weight_merging(self):
        """测试 EP 场景下非 EP 切分权重不跨 EP 合并，EP 切分权重正确跨 EP 合并"""
        cfg = self.create_mock_moe_checkpoint(tp_size=2,
                                              pp_size=1,
                                              ep_size=4,
                                              num_layers=4)
        reader = MCoreCheckpointReader(
            mcore_dir=self.checkpoint_dir,
            tp_size=2,
            pp_size=1,
            ep_size=4,
            num_layers=4,
            num_attention_heads=cfg['num_attention_heads'],
            num_query_groups=cfg['num_query_groups'],
            hidden_size=cfg['hidden_size'],
            kv_channels=cfg['kv_channels'],
            ffn_hidden_size=cfg['ffn_hidden_size'],
            moe_ffn_hidden_size=cfg['moe_ffn_hidden_size'],
            num_experts=cfg['num_experts'],
            vocab_size=cfg['vocab_size'],
            verbose=False,
            validate_shapes=True,
            io_threads=2,
        )
        result = reader.extract_weights()
        self.assertEqual(len(result.warnings), 0,
                         f"不应有 shape warnings: {result.warnings}")

        params = result.megatron_params

        # 非 EP 切分权重
        self.assertEqual(
            params['module.embedding.word_embeddings.weight']['shape'],
            [256, 128])
        self.assertEqual(params['module.output_layer.weight']['shape'],
                         [256, 128])
        self.assertEqual(
            params['module.decoder.layers.0.input_layernorm.weight']['shape'],
            [128])
        self.assertEqual(
            params['module.decoder.layers.0.self_attention.linear_qkv.weight']
            ['shape'], [256, 128])
        self.assertEqual(
            params['module.decoder.layers.0.self_attention.linear_proj.weight']
            ['shape'], [128, 128])
        self.assertEqual(
            params['module.decoder.layers.0.mlp.linear_fc1.weight']['shape'],
            [192, 128])
        self.assertEqual(
            params['module.decoder.layers.0.mlp.linear_fc2.weight']['shape'],
            [128, 96])

        # Q/K Layernorm
        self.assertEqual(
            params['module.decoder.layers.0.self_attention.q_layernorm.weight']
            ['shape'], [32])
        self.assertEqual(
            params['module.decoder.layers.0.self_attention.k_layernorm.weight']
            ['shape'], [32])

        # Bias
        self.assertEqual(
            params['module.decoder.layers.0.self_attention.linear_qkv.bias']
            ['shape'], [256])
        self.assertEqual(
            params['module.decoder.layers.0.mlp.linear_fc1.bias']['shape'],
            [192])
        self.assertEqual(
            params['module.decoder.layers.2.mlp.router.bias']['shape'], [8])

        # EP 切分权重
        self.assertEqual(
            params['module.decoder.layers.2.mlp.router.weight']['shape'],
            [8, 128])
        self.assertEqual(
            params['module.decoder.layers.2.mlp.experts.weight1']['shape'],
            [128, 1024])
        self.assertEqual(
            params['module.decoder.layers.2.mlp.experts.weight2']['shape'],
            [512, 128])

    # ------------------------------------------------------------------
    # 单元测试: TP/EP 维度与合并
    # ------------------------------------------------------------------

    def test_tp_parallel_dimension(self):
        """测试 TP 切分维度判断"""
        strategy = MoeParallelStrategy()
        cases = [
            ('embedding.word_embeddings.weight', 0),
            ('output_layer.weight', 0),
            ('self_attention.linear_qkv.weight', 0),
            ('mlp.linear_fc1.weight', 0),
            ('mlp.shared_experts.linear_fc1.weight', 0),
            ('self_attention.linear_proj.weight', 1),
            ('mlp.linear_fc2.weight', 1),
            ('mlp.shared_experts.linear_fc2.weight', 1),
            ('mlp.experts.weight1', None),
            ('mlp.experts.weight2', None),
            ('mlp.router.weight', None),
            ('input_layernorm.weight', None),
        ]
        for name, expected_dim in cases:
            with self.subTest(name=name):
                self.assertEqual(strategy.get_tp_parallel_dim(name),
                                 expected_dim)

    def test_expected_shape_calculation(self):
        """测试期望 Shape 计算（含 SwiGLU *2 修正）"""
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
        cases = [
            ('embedding.word_embeddings.weight', (163840, 7168)),
            ('decoder.final_layernorm.weight', (7168, )),
            ('layers.0.self_attention.linear_qkv.weight', (8704, 7168)),
            ('layers.0.self_attention.linear_qkv.bias', (8704, )),
            ('layers.0.self_attention.linear_proj.weight', (7168, 8192)),
            # SwiGLU: linear_fc1 为 gate+up 拼接，dim0 = ffn_hidden_size * 2
            ('layers.0.mlp.linear_fc1.weight', (36864, 7168)),
            ('layers.0.mlp.linear_fc2.weight', (7168, 18432)),
            ('layers.0.mlp.router.weight', (128, 7168)),
            ('layers.0.mlp.router.expert_bias', (128, )),
            # Shared experts 同样受 SwiGLU 影响
            ('layers.0.mlp.shared_experts.linear_fc1.weight', (24576, 7168)),
            ('layers.0.mlp.shared_experts.linear_fc2.weight', (7168, 12288)),
            ('layers.0.mlp.experts.weight1', (7168, 128 * 12288 * 2)),
            ('layers.0.mlp.experts.weight2', (128 * 12288, 7168)),
        ]
        for name, expected_shape in cases:
            with self.subTest(name=name):
                self.assertEqual(strategy.get_expected_shape(name, config),
                                 expected_shape)

    def test_tp_shape_merging(self):
        """测试 TP 维度合并"""
        strategy = MoeParallelStrategy()
        merger = ShapeMerger(strategy)
        cases = [
            ('embedding.word_embeddings.weight', 2, (81920, 7168), (163840,
                                                                    7168)),
            ('output_layer.weight', 2, (81920, 7168), (163840, 7168)),
            ('self_attention.linear_qkv.weight', 2, (4352, 7168), (8704,
                                                                   7168)),
            ('self_attention.linear_proj.weight', 2, (7168, 4096), (7168,
                                                                    8192)),
            ('mlp.linear_fc1.weight', 2, (18432, 7168), (36864, 7168)),
            ('mlp.linear_fc2.weight', 2, (7168, 9216), (7168, 18432)),
        ]
        for name, tp_size, local_shape, expected_shape in cases:
            with self.subTest(name=name, tp_size=tp_size):
                tp_shapes = [local_shape] * tp_size
                merged = merger.merge_tp_shapes(name, tp_shapes)
                self.assertEqual(merged, expected_shape)

    def test_ep_shape_merging(self):
        """测试 EP 维度合并"""
        strategy = MoeParallelStrategy()
        merger = ShapeMerger(strategy)
        config = ModelConfig(num_experts=128, moe_ffn_hidden_size=12288)
        ep_size = 8
        num_experts_per_ep = config.num_experts // ep_size
        cases = [
            ('mlp.experts.weight1', ep_size,
             (7168, num_experts_per_ep * 12288 * 2), (7168, 128 * 12288 * 2)),
            ('mlp.experts.weight2', ep_size, (num_experts_per_ep * 12288,
                                              7168), (128 * 12288, 7168)),
        ]
        for name, ep_size_val, local_shape, expected_shape in cases:
            with self.subTest(name=name, ep_size=ep_size_val):
                ep_shapes = [local_shape] * ep_size_val
                merged = merger.merge_ep_shapes(name, ep_shapes)
                self.assertEqual(merged, expected_shape)

    def test_kimi2_1t_full_merge(self):
        """测试 Kimi2-1T 配置 (TP=2, EP=64) 下的完整合并"""
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

        cases = [
            {
                'name': 'embedding.word_embeddings.weight',
                'tp_shapes': [(81920, 7168)] * tp_size,
                'ep_count': ep_size,
                'is_ep_sharded': False,
                'expected': (163840, 7168),
            },
            {
                'name': 'self_attention.linear_qkv.weight',
                'tp_shapes': [(4352, 7168)] * tp_size,
                'ep_count': ep_size,
                'is_ep_sharded': False,
                'expected': (8704, 7168),
            },
            {
                'name': 'self_attention.linear_proj.weight',
                'tp_shapes': [(7168, 4096)] * tp_size,
                'ep_count': ep_size,
                'is_ep_sharded': False,
                'expected': (7168, 8192),
            },
            {
                'name': 'mlp.experts.weight1',
                'tp_shapes': [(7168, num_experts_per_ep * 12288 * 2)],
                'ep_count': ep_size,
                'is_ep_sharded': True,
                'expected': (7168, config.num_experts * 12288 * 2),
            },
            {
                'name': 'mlp.experts.weight2',
                'tp_shapes': [(num_experts_per_ep * 12288, 7168)],
                'ep_count': ep_size,
                'is_ep_sharded': True,
                'expected': (config.num_experts * 12288, 7168),
            },
        ]
        for case in cases:
            with self.subTest(name=case['name']):
                tp_ep_shapes = {}
                for ep_rank in range(case['ep_count']):
                    for tp_rank, shape in enumerate(case['tp_shapes']):
                        tp_ep_shapes[(tp_rank, ep_rank)] = shape
                merged = merger.merge(case['name'], tp_ep_shapes,
                                      case['is_ep_sharded'])
                self.assertEqual(merged, case['expected'])

    def test_merger_edge_cases(self):
        """测试 ShapeMerger 边界情况"""
        strategy = MoeParallelStrategy()
        merger = ShapeMerger(strategy)

        # TP=1 保持原 shape
        self.assertEqual(merger.merge_tp_shapes('test', [(7168, 8192)]),
                         (7168, 8192))

        # EP=1 保持原 shape
        self.assertEqual(
            merger.merge_ep_shapes('mlp.experts.weight1', [(7168, 49152)]),
            (7168, 49152))

        # 空列表返回空 tuple
        self.assertEqual(merger.merge_tp_shapes('test', []), ())

        # LayerNorm 不参与任何并行，多 rank 合并后应保持不变
        tp_ep_shapes = {
            (0, 0): (7168, ),
            (0, 1): (7168, ),
            (1, 0): (7168, ),
            (1, 1): (7168, )
        }
        self.assertEqual(
            merger.merge('input_layernorm.weight',
                         tp_ep_shapes,
                         is_ep_sharded=False), (7168, ))

        # 非专家权重即使 is_ep_sharded=True 也不应跨 EP 合并（只合并 TP）
        tp_ep_shapes = {
            (0, 0): (4352, 7168),
            (1, 0): (4352, 7168),
            (0, 1): (4352, 7168),
            (1, 1): (4352, 7168),
        }
        self.assertEqual(
            merger.merge('self_attention.linear_qkv.weight',
                         tp_ep_shapes,
                         is_ep_sharded=True), (8704, 7168))

    # ------------------------------------------------------------------
    # 单元测试: Layer 映射
    # ------------------------------------------------------------------

    def test_layer_mappers(self):
        """测试 StandardVPP 和 DualPipe Layer 映射"""
        num_layers = 32
        pp_size = 4
        vpp_stage = 4

        # Standard VPP
        mapper = StandardVppMapper()
        mapping = mapper.build_mapping(num_layers, pp_size, vpp_stage)
        self.assertEqual(len(mapping), num_layers)

        # DualPipe
        mapper = DualPipeMapper()
        mapping = mapper.build_mapping(num_layers, pp_size, vpp_stage)
        self.assertEqual(len(mapping), num_layers)

        # DualPipe 非法配置应抛出异常
        with self.assertRaises(ValueError):
            mapper.build_mapping(30, pp_size, vpp_stage)


if __name__ == '__main__':
    unittest.main()
