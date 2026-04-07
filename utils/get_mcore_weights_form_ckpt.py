#!/usr/bin/env python3
"""
MCore 格式权重读取与保存工具

从 MCore checkpoint 中提取权重信息（名称、形状、数据类型），并保存为 JSON 格式。
输出 keys 与 hf_weights_info.json 对齐。

用法：
  python get_mcore_weights_form_ckpt.py /path/to/mcore/checkpoint \
    --tp 2 --pp 8 --ep 64 \
    --num-layers 32 \
    --output weights_info.json

输出格式（与 hf_weights_info.json 对齐）：
{
  "metadata": {
    "total_size": 2059316667904
  },
  "weight_map": {
    "model.embed_tokens.weight": {
      "shape": [163840, 7168],
      "dtype": "bfloat16"
    },
    ...
  }
}
"""

import argparse
import inspect
import json
import logging
import math
import os
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import torch

logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int,
               ep: int) -> str:
    """生成 mp_rank 目录前缀。"""
    if ep == 1 and pp == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


def _resolve_iter_dir(load_dir: str) -> str:
    """解析迭代目录，支持 latest 指针和默认迭代目录。"""
    latest = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    if os.path.isfile(latest):
        with open(latest) as f:
            it = f.read().strip()
        try:
            it_num = int(it)
        except ValueError as e:
            raise ValueError(f"latest_checkpointed_iteration.txt 内容无效: '{it}'") from e
        latest_dir = os.path.join(load_dir, f'iter_{it_num:07d}')
        if os.path.isdir(latest_dir):
            return latest_dir

    # 尝试默认迭代目录
    for it_name in ['iter_0000001', 'iter_0000000']:
        it_path = os.path.join(load_dir, it_name)
        if os.path.isdir(it_path):
            return it_path

    # 如果输入路径本身就是迭代目录
    if os.path.basename(load_dir).startswith('iter_'):
        return load_dir

    raise FileNotFoundError(
        f'无法定位迭代目录: {load_dir}\n'
        f'请确保目录包含 latest_checkpointed_iteration.txt 或 iter_XXXXXXX 子目录'
    )


def _torch_load_compat(path: str, disable_mmap: bool = False) -> dict:
    """兼容不同 PyTorch 版本的 torch.load 调用。"""
    base = {'map_location': 'cpu'}
    sig = None
    try:
        sig = inspect.signature(torch.load)
    except Exception:
        pass

    support_weights_only = bool(sig and 'weights_only' in sig.parameters)
    support_mmap = bool(sig and 'mmap' in sig.parameters and not disable_mmap)

    candidates: list[dict] = []

    # 首选：weights_only=True, mmap=True
    if support_weights_only:
        kw = dict(base)
        kw['weights_only'] = True
        if support_mmap:
            kw['mmap'] = True
        candidates.append(kw)

    # 备选：weights_only=False, mmap=True
    kw = dict(base)
    if support_weights_only:
        kw['weights_only'] = False
    if support_mmap:
        kw['mmap'] = True
    candidates.append(kw)

    # 最后备选：最基本的参数
    kw = dict(base)
    if support_weights_only:
        kw['weights_only'] = False
    candidates.append(kw)

    last_error = None
    for kw in candidates:
        try:
            return torch.load(path, **kw)
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"无法加载 checkpoint: {path}") from last_error


def _get_torch_dtype_name(dtype: torch.dtype) -> str:
    """获取 PyTorch 数据类型的字符串名称。"""
    dtype_map = {
        torch.float16: 'float16',
        torch.bfloat16: 'bfloat16',
        torch.float32: 'float32',
        torch.float64: 'float64',
        torch.int8: 'int8',
        torch.int16: 'int16',
        torch.int32: 'int32',
        torch.int64: 'int64',
        torch.uint8: 'uint8',
        torch.bool: 'bool',
    }
    return dtype_map.get(dtype, str(dtype).replace('torch.', ''))


class MCoreCheckpointReader:
    """MCore Checkpoint 读取器 - 读取权重信息并保存为 JSON（与 hf_weights_info.json 对齐）"""

    def __init__(
        self,
        mcore_dir: str,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        num_layers: Optional[int] = None,
        num_experts: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        hidden_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        moe_ffn_hidden_size: Optional[int] = None,
        first_k_dense_replace: int = 2,
        disable_mmap: bool = False,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.mcore_dir = mcore_dir
        self.iter_dir = _resolve_iter_dir(mcore_dir)

        # 并行配置
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size

        # 模型配置（默认值与 hf_weights_info.json 一致）
        self.num_layers = num_layers or 32
        self.num_experts = num_experts or 128
        self.num_attention_heads = num_attention_heads or 64
        self.num_key_value_heads = num_key_value_heads or 2  # GQA: 2 KV heads
        self.hidden_size = hidden_size or 7168
        self.ffn_hidden_size = ffn_hidden_size or 18432
        self.moe_ffn_hidden_size = moe_ffn_hidden_size or 12288
        self.first_k_dense_replace = first_k_dense_replace

        # 计算 head_dim
        self.head_dim = self.hidden_size // self.num_attention_heads

        self.disable_mmap = disable_mmap

        # 内部状态
        self._rank_dir_map: Dict[Tuple[int, int], List[int]] = {}

        if self.verbose:
            logger.info('Resolved iter_dir: %s', self.iter_dir)

    def _resolve_rank_ckpt_path(self, tp_rank: int, pp_rank: int,
                                ep_rank: Optional[int]) -> str:
        """解析 rank checkpoint 路径。"""
        candidates: list[str] = []
        if ep_rank is not None:
            candidates.append(
                _mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size,
                           self.pp_size, self.ep_size))
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}')
            candidates.append(f'mp_rank_{tp_rank:02}_{ep_rank:03}_{pp_rank:03}')
            candidates.append(f'mp_rank_{tp_rank:02}_{ep_rank:03}')
        else:
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}_000')
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}_001')
            candidates.append(f'mp_rank_{tp_rank:02}_{pp_rank:03}')
            candidates.append(f'mp_rank_{tp_rank:02}')

        for p in candidates:
            path = os.path.join(self.iter_dir, p, 'model_optim_rng.pt')
            if os.path.isfile(path):
                return path

        raise FileNotFoundError(
            f'无法定位 rank 文件: tp={tp_rank}, pp={pp_rank}, ep={ep_rank}, '
            f'iter_dir={self.iter_dir}'
        )

    def _build_rank_dir_map(self) -> None:
        """构建 rank 目录映射。"""
        mp_dirs: list[str] = []
        try:
            for d in os.listdir(self.iter_dir):
                if d.startswith('mp_rank_'):
                    mp_dirs.append(d)
        except FileNotFoundError:
            mp_dirs = []

        rank_map: Dict[Tuple[int, int], set] = defaultdict(set)
        for d in mp_dirs:
            parts = d.split('_')
            if len(parts) < 3:
                continue
            try:
                tp = int(parts[2])
            except ValueError:
                continue

            idxs = []
            for p in parts[3:]:
                try:
                    idxs.append(int(p))
                except ValueError:
                    idxs.append(None)

            pp = None
            ep = None
            for v in idxs:
                if v is None:
                    continue
                if pp is None and v < self.pp_size:
                    pp = v
                    continue
                if ep is None and v < self.ep_size:
                    ep = v

            if pp is None and self.pp_size == 1:
                pp = 0
            if ep is None and self.ep_size == 1:
                ep = 0
            if pp is None or ep is None:
                continue

            rank_map[(tp, pp)].add(ep)

        self._rank_dir_map = {k: sorted(list(v)) for k, v in rank_map.items()}

        if self.verbose:
            logger.info('Rank dir map size=%d', len(self._rank_dir_map))

    def _load_rank_state(self, tp_rank: int, pp_rank: int,
                         ep_rank: Optional[int]) -> Dict[str, torch.Tensor]:
        """加载单个 rank 的 state。"""
        ckpt_path = self._resolve_rank_ckpt_path(tp_rank, pp_rank, ep_rank)
        state = _torch_load_compat(ckpt_path, disable_mmap=self.disable_mmap)
        return state.get('model', state)

    def _is_moe_layer(self, layer_id: int) -> bool:
        """判断指定层是否是 MoE 层。"""
        return layer_id >= self.first_k_dense_replace

    def _get_tp_parallel_dim(self, mcore_name: str) -> Optional[int]:
        """确定权重在哪个维度上进行张量并行切分。"""
        # Embedding: 在第0维切分 (vocab_size)
        if 'embedding.word_embeddings.weight' in mcore_name:
            return 0

        # Output layer: 在第0维切分 (vocab_size)
        if 'output_layer.weight' in mcore_name:
            return 0

        # QKV projection: 在第0维切分
        if 'self_attention.linear_qkv.weight' in mcore_name:
            return 0
        if 'self_attention.linear_qkv.bias' in mcore_name:
            return 0

        # Output projection: 在第1维切分 (input dim)
        if 'self_attention.linear_proj.weight' in mcore_name:
            return 1

        # MLP fc1 (gate_up 融合): 在第0维切分
        if 'mlp.linear_fc1.weight' in mcore_name:
            return 0
        if 'mlp.linear_fc1.bias' in mcore_name:
            return 0

        # MLP fc2 (down): 在第1维切分
        if 'mlp.linear_fc2.weight' in mcore_name:
            return 1

        # Shared experts
        if 'shared_experts.linear_fc1.weight' in mcore_name:
            return 0
        if 'shared_experts.linear_fc2.weight' in mcore_name:
            return 1

        # MoE experts (local_experts 格式)
        if 'experts.local_experts' in mcore_name and 'linear_fc1.weight' in mcore_name:
            return 0
        if 'experts.local_experts' in mcore_name and 'linear_fc2.weight' in mcore_name:
            return 1

        # MoE experts (grouped_gemm 格式)
        if 'experts.weight1' in mcore_name:
            return 1  # [hidden, num_local*intermed*2] -> 在第1维切分
        if 'experts.weight2' in mcore_name:
            return 0  # [num_local*intermed, hidden] -> 在第0维切分

        # Router: 不切分
        if 'router.weight' in mcore_name:
            return None
        if 'router.expert_bias' in mcore_name:
            return None

        # LayerNorm: 不切分
        if 'layernorm' in mcore_name.lower():
            return None

        return None

    def _merge_tp_shapes(self, name: str,
                         tp_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """合并所有 TP rank 的形状，恢复原始形状。"""
        if not tp_shapes:
            return ()

        if len(tp_shapes) == 1:
            return tp_shapes[0]

        parallel_dim = self._get_tp_parallel_dim(name)
        if parallel_dim is None:
            # 不切分，返回第一个 shape
            return tp_shapes[0]

        # 合并指定维度的形状
        base_shape = list(tp_shapes[0])
        total_dim = sum(s[parallel_dim] for s in tp_shapes)
        base_shape[parallel_dim] = total_dim
        return tuple(base_shape)

    def _compute_rotary_emb_shape(self) -> List[int]:
        """计算 rotary_emb.inv_freq 的形状。
        
        在 HF 模型中，rotary_emb.inv_freq 通常是 head_dim // 2
        """
        return [self.head_dim // 2]

    def _add_rotary_emb_weights(self, weight_map: Dict[str, Dict[str, Any]], 
                                 layer_id: int, dtype: str, total_size: int) -> int:
        """添加 rotary_emb.inv_freq 权重。"""
        shape = self._compute_rotary_emb_shape()
        name = f'model.layers.{layer_id}.self_attn.rotary_emb.inv_freq'
        
        # 计算大小（float32 是 4 字节）
        numel = 1
        for dim in shape:
            numel *= dim
        size_bytes = numel * 4  # float32
        
        weight_map[name] = {
            'shape': shape,
            'dtype': dtype,
        }
        
        return total_size + size_bytes

    def extract_weights(self) -> Dict[str, Any]:
        """提取所有权重信息（与 hf_weights_info.json 对齐）。"""
        self._build_rank_dir_map()

        weight_map: Dict[str, Dict[str, Any]] = {}
        total_size = 0

        if self.verbose:
            logger.info("开始提取权重信息...")

        # 遍历所有 PP ranks
        for pp_rank in range(self.pp_size):
            # 确定要加载的 EP ranks
            ep_ranks_to_load: set = set()
            for tp_rank in range(self.tp_size):
                eps = self._rank_dir_map.get((tp_rank, pp_rank), [0])
                ep_ranks_to_load.update(eps)

            # 加载所有需要的 TP 和 EP ranks
            all_states: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

            for tp_rank in range(self.tp_size):
                for ep_rank in ep_ranks_to_load:
                    try:
                        state = self._load_rank_state(tp_rank, pp_rank, ep_rank)
                        all_states[(tp_rank, ep_rank)] = state
                    except FileNotFoundError:
                        continue

            if not all_states:
                logger.warning("PP rank %d: 没有加载到任何状态", pp_rank)
                continue

            # 收集所有权重名称
            all_weight_names: set = set()
            for state in all_states.values():
                all_weight_names.update(
                    k for k in state.keys() if isinstance(state[k], torch.Tensor))

            # 处理每个权重
            for name in all_weight_names:
                # 收集所有 TP rank 的 shape 和 dtype
                tp_shapes = []
                dtype = None

                for (tp_rank, ep_rank), state in all_states.items():
                    if name in state:
                        tensor = state[name]
                        if isinstance(tensor, torch.Tensor):
                            tp_shapes.append(tuple(tensor.shape))
                            if dtype is None:
                                dtype = _get_torch_dtype_name(tensor.dtype)

                if not tp_shapes:
                    continue

                # 合并 shape
                merged_shape = self._merge_tp_shapes(name, tp_shapes)

                # 计算大小
                numel = 1
                for dim in merged_shape:
                    numel *= dim

                # 估计 element size
                if dtype in ['float16', 'bfloat16']:
                    element_size = 2
                elif dtype == 'float32':
                    element_size = 4
                else:
                    element_size = 2

                size_bytes = numel * element_size

                # 确定 layer id
                layer_match = re.match(r'decoder\.layers\.(\d+)\.', name)
                if layer_match:
                    local_idx = int(layer_match.group(1))
                    # 计算全局 layer id
                    if self.pp_size > 1:
                        layers_per_pp = self.num_layers // self.pp_size
                        layer_id = pp_rank * layers_per_pp + local_idx
                    else:
                        layer_id = local_idx
                else:
                    layer_id = 0

                # 转换 MCore 名称为 HF 名称
                # Embedding
                if name == 'embedding.word_embeddings.weight':
                    hf_name = 'model.embed_tokens.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # Final LayerNorm
                if name == 'decoder.final_layernorm.weight':
                    hf_name = 'model.norm.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # Output layer
                if name == 'output_layer.weight':
                    hf_name = 'lm_head.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # Decoder layers
                layer_match = re.match(r'decoder\.layers\.(\d+)\.(.*)', name)
                if not layer_match:
                    continue

                local_idx = int(layer_match.group(1))
                rest = layer_match.group(2)

                # 计算全局 layer id
                if self.pp_size > 1:
                    layers_per_pp = self.num_layers // self.pp_size
                    layer_id = pp_rank * layers_per_pp + local_idx
                else:
                    layer_id = local_idx

                prefix = f'model.layers.{layer_id}'
                is_moe = self._is_moe_layer(layer_id)

                # Self Attention
                if rest == 'self_attention.linear_proj.weight':
                    hf_name = f'{prefix}.self_attn.o_proj.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                        # 同时添加 rotary_emb.inv_freq
                        total_size = self._add_rotary_emb_weights(
                            weight_map, layer_id, dtype, total_size) - size_bytes
                    continue

                if rest == 'self_attention.linear_qkv.weight':
                    # 展开 QKV
                    qkv_out_dim, hidden_size = merged_shape
                    
                    # 计算各部分的输出维度
                    q_out = self.num_attention_heads * self.head_dim
                    k_out = self.num_key_value_heads * self.head_dim
                    v_out = self.num_key_value_heads * self.head_dim
                    
                    # 验证形状是否匹配
                    expected_total = q_out + k_out + v_out
                    if qkv_out_dim != expected_total:
                        # 如果形状不匹配，按比例分配
                        ratio_q = self.num_attention_heads / (
                            self.num_attention_heads + 2 * self.num_key_value_heads)
                        ratio_kv = self.num_key_value_heads / (
                            self.num_attention_heads + 2 * self.num_key_value_heads)
                        q_out = int(qkv_out_dim * ratio_q)
                        k_out = int(qkv_out_dim * ratio_kv)
                        v_out = qkv_out_dim - q_out - k_out
                    
                    q_shape = (q_out, hidden_size)
                    k_shape = (k_out, hidden_size)
                    v_shape = (v_out, hidden_size)
                    
                    for proj_name, proj_shape in [('q_proj', q_shape), ('k_proj', k_shape), ('v_proj', v_shape)]:
                        hf_proj_name = f'{prefix}.self_attn.{proj_name}.weight'
                        if hf_proj_name not in weight_map:
                            proj_numel = 1
                            for dim in proj_shape:
                                proj_numel *= dim
                            proj_size = proj_numel * element_size
                            weight_map[hf_proj_name] = {
                                'shape': list(proj_shape),
                                'dtype': dtype,
                            }
                            total_size += proj_size
                    continue

                if rest == 'self_attention.q_layernorm.weight':
                    hf_name = f'{prefix}.self_attn.q_layernorm.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                if rest == 'self_attention.k_layernorm.weight':
                    hf_name = f'{prefix}.self_attn.k_layernorm.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # LayerNorm
                if rest == 'input_layernorm.weight':
                    hf_name = f'{prefix}.input_layernorm.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                if rest in ('pre_mlp_layernorm.weight', 'post_attention_layernorm.weight'):
                    hf_name = f'{prefix}.post_attention_layernorm.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # MLP - Dense (非 MoE 层或 Shared Experts)
                if rest == 'mlp.linear_fc1.weight':
                    # 检查是否是 shared_experts 的权重（MoE 层中的 Dense MLP）
                    if is_moe:
                        # 这是 shared_experts
                        fused_dim, hidden_size = merged_shape
                        single_dim = fused_dim // 2
                        
                        for proj_name, proj_shape in [('gate_proj', (single_dim, hidden_size)), 
                                                       ('up_proj', (single_dim, hidden_size))]:
                            hf_proj_name = f'{prefix}.mlp.shared_experts.{proj_name}.weight'
                            if hf_proj_name not in weight_map:
                                proj_numel = 1
                                for dim in proj_shape:
                                    proj_numel *= dim
                                proj_size = proj_numel * element_size
                                weight_map[hf_proj_name] = {
                                    'shape': list(proj_shape),
                                    'dtype': dtype,
                                }
                                total_size += proj_size
                    else:
                        # 这是 Dense MLP
                        fused_dim, hidden_size = merged_shape
                        single_dim = fused_dim // 2
                        
                        for proj_name, proj_shape in [('gate_proj', (single_dim, hidden_size)), 
                                                       ('up_proj', (single_dim, hidden_size))]:
                            hf_proj_name = f'{prefix}.mlp.{proj_name}.weight'
                            if hf_proj_name not in weight_map:
                                proj_numel = 1
                                for dim in proj_shape:
                                    proj_numel *= dim
                                proj_size = proj_numel * element_size
                                weight_map[hf_proj_name] = {
                                    'shape': list(proj_shape),
                                    'dtype': dtype,
                                }
                                total_size += proj_size
                    continue

                if rest == 'mlp.linear_fc2.weight':
                    if is_moe:
                        # shared_experts down_proj
                        hf_name = f'{prefix}.mlp.shared_experts.down_proj.weight'
                    else:
                        # Dense MLP down_proj
                        hf_name = f'{prefix}.mlp.down_proj.weight'
                    
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # Shared Experts (显式指定的 shared_experts)
                if rest == 'mlp.shared_experts.linear_fc1.weight':
                    fused_dim, hidden_size = merged_shape
                    single_dim = fused_dim // 2
                    
                    for proj_name, proj_shape in [('gate_proj', (single_dim, hidden_size)), 
                                                   ('up_proj', (single_dim, hidden_size))]:
                        hf_proj_name = f'{prefix}.mlp.shared_experts.{proj_name}.weight'
                        if hf_proj_name not in weight_map:
                            proj_numel = 1
                            for dim in proj_shape:
                                proj_numel *= dim
                            proj_size = proj_numel * element_size
                            weight_map[hf_proj_name] = {
                                'shape': list(proj_shape),
                                'dtype': dtype,
                            }
                            total_size += proj_size
                    continue

                if rest == 'mlp.shared_experts.linear_fc2.weight':
                    hf_name = f'{prefix}.mlp.shared_experts.down_proj.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # Router (仅 MoE 层)
                if rest == 'mlp.router.weight':
                    hf_name = f'{prefix}.mlp.gate.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                if rest == 'mlp.router.expert_bias':
                    hf_name = f'{prefix}.mlp.gate.e_score_correction_bias'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # MoE Experts - local_experts 格式
                expert_fc1_match = re.match(
                    r'mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight', rest)
                if expert_fc1_match:
                    local_expert_id = int(expert_fc1_match.group(1))
                    num_local_experts = self.num_experts // self.ep_size
                    
                    # 计算全局专家 ID
                    for (tp_rank, ep_rank), state in all_states.items():
                        if name in state:
                            global_expert_id = ep_rank * num_local_experts + local_expert_id
                            break
                    else:
                        global_expert_id = local_expert_id

                    fused_dim, hidden_size = merged_shape
                    single_dim = fused_dim // 2
                    
                    for proj_name, proj_shape in [('gate_proj', (single_dim, hidden_size)), 
                                                   ('up_proj', (single_dim, hidden_size))]:
                        hf_proj_name = f'{prefix}.mlp.experts.{global_expert_id}.{proj_name}.weight'
                        if hf_proj_name not in weight_map:
                            proj_numel = 1
                            for dim in proj_shape:
                                proj_numel *= dim
                            proj_size = proj_numel * element_size
                            weight_map[hf_proj_name] = {
                                'shape': list(proj_shape),
                                'dtype': dtype,
                            }
                            total_size += proj_size
                    continue

                expert_fc2_match = re.match(
                    r'mlp\.experts\.local_experts\.(\d+)\.linear_fc2\.weight', rest)
                if expert_fc2_match:
                    local_expert_id = int(expert_fc2_match.group(1))
                    num_local_experts = self.num_experts // self.ep_size
                    
                    for (tp_rank, ep_rank), state in all_states.items():
                        if name in state:
                            global_expert_id = ep_rank * num_local_experts + local_expert_id
                            break
                    else:
                        global_expert_id = local_expert_id

                    hf_name = f'{prefix}.mlp.experts.{global_expert_id}.down_proj.weight'
                    if hf_name not in weight_map:
                        weight_map[hf_name] = {
                            'shape': list(merged_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
                    continue

                # MoE Experts - grouped_gemm 格式
                if rest == 'mlp.experts.weight1':
                    num_local_experts = self.num_experts // self.ep_size
                    
                    for (tp_rank, ep_rank), state in all_states.items():
                        if name in state:
                            expert_offset = ep_rank * num_local_experts
                            
                            # weight1: [hidden, num_local * intermed*2]
                            hidden_size = merged_shape[0]
                            intermed_2x_total = merged_shape[1]
                            intermed_2x = intermed_2x_total // num_local_experts
                            intermed = intermed_2x // 2
                            
                            for local_id in range(num_local_experts):
                                global_expert_id = expert_offset + local_id
                                gate_shape = (intermed, hidden_size)
                                up_shape = (intermed, hidden_size)
                                
                                for proj_name, proj_shape in [('gate_proj', gate_shape), ('up_proj', up_shape)]:
                                    hf_proj_name = f'{prefix}.mlp.experts.{global_expert_id}.{proj_name}.weight'
                                    if hf_proj_name not in weight_map:
                                        proj_numel = 1
                                        for dim in proj_shape:
                                            proj_numel *= dim
                                        proj_size = proj_numel * element_size
                                        weight_map[hf_proj_name] = {
                                            'shape': list(proj_shape),
                                            'dtype': dtype,
                                        }
                                        total_size += proj_size
                            break
                    continue

                if rest == 'mlp.experts.weight2':
                    num_local_experts = self.num_experts // self.ep_size
                    
                    for (tp_rank, ep_rank), state in all_states.items():
                        if name in state:
                            expert_offset = ep_rank * num_local_experts
                            
                            # weight2: [num_local * intermed, hidden]
                            intermed_total = merged_shape[0]
                            hidden_size = merged_shape[1]
                            intermed = intermed_total // num_local_experts
                            
                            for local_id in range(num_local_experts):
                                global_expert_id = expert_offset + local_id
                                down_shape = (hidden_size, intermed)
                                
                                hf_proj_name = f'{prefix}.mlp.experts.{global_expert_id}.down_proj.weight'
                                if hf_proj_name not in weight_map:
                                    proj_numel = 1
                                    for dim in down_shape:
                                        proj_numel *= dim
                                    proj_size = proj_numel * element_size
                                    weight_map[hf_proj_name] = {
                                        'shape': list(down_shape),
                                        'dtype': dtype,
                                    }
                                    total_size += proj_size
                            break
                    continue

        if self.verbose:
            logger.info("提取完成，共 %d 个权重，总大小: %d bytes (%.2f GB)",
                       len(weight_map), total_size, total_size / 1e9)

        return {
            'metadata': {
                'total_size': total_size,
            },
            'weight_map': weight_map,
        }

    def save_json(self, output_path: str, indent: int = 2) -> None:
        """保存权重信息到 JSON 文件。"""
        output = self.extract_weights()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=indent, ensure_ascii=False)

        if self.verbose:
            logger.info("JSON 输出已保存到: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='读取 MCore 格式权重并保存为 JSON（与 hf_weights_info.json 对齐）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 (TP=1, PP=1, EP=1)
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt --output weights_info.json

  # 指定并行配置 (Kimi2-1T 默认配置: TP=2, PP=8, EP=64)
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt \\
    --tp 2 --pp 8 --ep 64 \\
    --num-layers 32 \\
    --num-attention-heads 64 --num-key-value-heads 32 \\
    --hidden-size 7168 --num-experts 128 \\
    --output weights_info.json

  # 静默模式
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt --output weights_info.json --quiet
        """
    )

    parser.add_argument('mcore_dir',
                        type=str,
                        help='MCore checkpoint 目录')
    parser.add_argument('--tp',
                        type=int,
                        default=1,
                        help='Tensor 并行大小 (默认: 1)')
    parser.add_argument('--pp',
                        type=int,
                        default=1,
                        help='Pipeline 并行大小 (默认: 1)')
    parser.add_argument('--ep',
                        type=int,
                        default=1,
                        help='Expert 并行大小 (默认: 1)')
    parser.add_argument('--num-layers',
                        type=int,
                        default=32,
                        help='模型的总层数 (默认: 32)')
    parser.add_argument('--num-attention-heads',
                        type=int,
                        default=64,
                        help='注意力头数 (默认: 64)')
    parser.add_argument('--num-key-value-heads',
                        type=int,
                        default=2,
                        help='KV 头数 (GQA, 默认: 2)')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=7168,
                        help='隐藏层维度 (默认: 7168)')
    parser.add_argument('--ffn-hidden-size',
                        type=int,
                        default=18432,
                        help='Dense FFN 隐藏层维度 (默认: 18432)')
    parser.add_argument('--moe-ffn-hidden-size',
                        type=int,
                        default=12288,
                        help='MoE FFN 隐藏层维度 (默认: 12288)')
    parser.add_argument('--num-experts',
                        type=int,
                        default=128,
                        help='MoE 专家总数 (默认: 128)')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=2,
                        help='前 K 层使用 Dense MLP (默认: 2)')
    parser.add_argument('--disable-mmap',
                        action='store_true',
                        help='禁用 torch.load 的 mmap')
    parser.add_argument('--output',
                        '-o',
                        type=str,
                        required=True,
                        help='输出 JSON 文件路径')
    parser.add_argument('--quiet', action='store_true', help='静默模式')

    args = parser.parse_args()

    reader = MCoreCheckpointReader(
        mcore_dir=args.mcore_dir,
        tp_size=args.tp,
        pp_size=args.pp,
        ep_size=args.ep,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        first_k_dense_replace=args.first_k_dense_replace,
        disable_mmap=args.disable_mmap,
        verbose=not args.quiet,
    )

    reader.save_json(args.output)
    return 0


if __name__ == '__main__':
    exit(main())
