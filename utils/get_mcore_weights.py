#!/usr/bin/env python3
"""
MCore (Megatron-Core) 格式权重读取与保存工具

用法：
  python get_mcore_weights.py /path/to/mcore/checkpoint \
    --tp 2 --pp 8 --ep 64 \
    --schedules-method dualpipev \
    --vpp-stage 2 \
    --num-layers 32 \
    --first-k-dense-replace 2 \
    --num-attention-heads 64 \
    --num-key-value-heads 32 \
    --output weights_info.json

功能：
1. 读取 MCore 格式 checkpoint 中的权重信息
2. 支持 VPP/DualPipe 调度算法
3. 提取权重名称、形状、数据类型
4. 合并所有 TP rank 的权重，恢复原始形状
5. 展开融合权重（QKV→q,k,v; gate_up→gate,up），完全匹配 HF 格式
6. 支持 MoE 模型（展开 128 个专家的权重）
7. 保存为 JSON 格式，与 model_weights_info.json 格式完全一致

输出格式与 model_weights_info.json 对齐:
{
  "metadata": {
    "total_size": 2061430584320,
    "num_layers": 32,
    "hidden_size": 7168,
    "vocab_size": 163840,
    "num_attention_heads": 64,
    "num_key_value_heads": 32,
    "n_routed_experts": 128,
    "n_shared_experts": 1
  },
  "weight_map": {
    "model.embed_tokens.weight": {"shape": [163840, 7168], "dtype": "bfloat16"},
    "model.layers.0.input_layernorm.weight": {"shape": [7168], "dtype": "bfloat16"},
    "model.layers.0.self_attn.q_proj.weight": {"shape": [7168, 7168], "dtype": "bfloat16"},
    ...
  }
}
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch

# 配置日志
logging.basicConfig(format='%(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int, ep: int) -> str:
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
    default_iters = ['iter_0000001', 'iter_0000000']
    for it_name in default_iters:
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


def _get_tensor_size(tensor: torch.Tensor) -> int:
    """计算张量的字节大小。"""
    return tensor.numel() * tensor.element_size()


@dataclass
class WeightInfo:
    """权重信息数据结构。"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    size_bytes: int


@dataclass
class ExtractResult:
    """提取结果数据结构。"""
    metadata: Dict[str, Any]
    weight_map: Dict[str, Dict[str, Any]]
    total_size: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


class CheckpointCache:
    """Checkpoint 文件缓存，避免重复加载。"""
    
    def __init__(self, max_size: int = 8):
        self._cache: Dict[str, dict] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
    
    def get(self, path: str) -> Optional[dict]:
        """获取缓存的 checkpoint。"""
        if path in self._cache:
            self._access_order.remove(path)
            self._access_order.append(path)
            return self._cache[path]
        return None
    
    def put(self, path: str, state: dict) -> None:
        """添加 checkpoint 到缓存。"""
        if path in self._cache:
            return
        
        while len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        
        self._cache[path] = state
        self._access_order.append(path)
    
    def clear(self) -> None:
        """清空缓存。"""
        self._cache.clear()
        self._access_order.clear()


class MCoreWeightExtractor:
    """MCore 权重提取器 - 读取并保存权重信息到 JSON"""

    def __init__(
        self,
        mcore_dir: str,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        schedules_method: Optional[str] = None,
        vpp_stage: Optional[int] = None,
        num_layers: Optional[int] = None,
        first_k_dense_replace: int = 2,
        num_experts: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_key_value_heads: Optional[int] = None,
        hidden_size: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        moe_ffn_hidden_size: Optional[int] = None,
        vocab_size: Optional[int] = None,
        io_threads: int = 4,
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
        
        # VPP/DualPipe 配置
        self.schedules_method = schedules_method
        self.dualpipe = schedules_method == 'dualpipev'
        self.vpp_stage = vpp_stage
        
        # 模型配置
        self.num_layers = num_layers or 32
        self.first_k_dense_replace = first_k_dense_replace
        self.num_experts = num_experts or 128
        self.num_attention_heads = num_attention_heads or 64
        self.num_key_value_heads = num_key_value_heads or 32
        self.hidden_size = hidden_size or 7168
        self.ffn_hidden_size = ffn_hidden_size or 18432
        self.moe_ffn_hidden_size = moe_ffn_hidden_size or 12288
        self.vocab_size = vocab_size or 163840
        self.n_shared_experts = 1
        
        # 计算 head_dim
        self.head_dim = self.hidden_size // self.num_attention_heads
        
        # 其他配置
        self.io_threads = max(1, int(io_threads))
        self.disable_mmap = bool(disable_mmap)
        
        # 存储结果
        self.weight_infos: Dict[str, WeightInfo] = {}
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Checkpoint 缓存
        self._cache = CheckpointCache(max_size=max(4, tp_size * 2))
        
        # 内部状态
        self._rank_dir_map: Dict[Tuple[int, int], List[int]] = {}
        self.vpp_size: Optional[int] = None
        self._vpp_model_keys: Optional[list[str]] = None
        self.vpprank_layer_idxs: Dict[int, Dict[int, List[int]]] = defaultdict(dict)
        self.layer2loc_vpp: Dict[int, Tuple[int, int, int]] = {}
        self._stage_local_to_global: Dict[Tuple[int, Optional[int]], Dict[int, int]] = {}
        
        if self.verbose:
            logger.info('Resolved iter_dir: %s', self.iter_dir)

    def _validate(self) -> None:
        """验证参数合法性。"""
        if not os.path.isdir(self.mcore_dir):
            raise FileNotFoundError(f'加载目录不存在: {self.mcore_dir}')
        if self.tp_size <= 0 or self.pp_size <= 0 or self.ep_size <= 0:
            raise ValueError('并行度必须 > 0')

    def _resolve_rank_ckpt_path(self, tp_rank: int, pp_rank: int, ep_rank: int | None) -> str:
        """解析 rank checkpoint 路径。"""
        candidates: list[str] = []
        if ep_rank is not None:
            candidates.append(_mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size, self.pp_size, self.ep_size))
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
            f'无法定位 rank 文件: tp={tp_rank}, pp={pp_rank}, ep={ep_rank}, iter_dir={self.iter_dir}\n'
            f'尝试的路径: {[os.path.join(self.iter_dir, p, "model_optim_rng.pt") for p in candidates]}'
        )

    def _build_rank_dir_map(self) -> None:
        """构建 rank 目录映射 - 参考 convert_ckpt_mcore2hf.py"""
        mp_dirs: list[str] = []
        try:
            for d in os.listdir(self.iter_dir):
                if d.startswith('mp_rank_'):
                    mp_dirs.append(d)
        except FileNotFoundError:
            mp_dirs = []
        
        rank_map: dict[tuple[int, int], set[int]] = defaultdict(set)
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

    def _load_checkpoint(self, path: str) -> dict:
        """加载 checkpoint，带缓存。"""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        
        state = _torch_load_compat(path, disable_mmap=self.disable_mmap)
        self._cache.put(path, state)
        return state

    def _detect_vpp(self) -> tuple[int | None, list[str] | None]:
        """检测 VPP 配置 - 参考 convert_ckpt_mcore2hf.py"""
        try:
            ckpt_path = self._resolve_rank_ckpt_path(0, 0, None)
        except FileNotFoundError:
            for tp in range(self.tp_size):
                for pp in range(self.pp_size):
                    try:
                        ckpt_path = self._resolve_rank_ckpt_path(tp, pp, 0 if self.ep_size > 1 else None)
                        break
                    except FileNotFoundError:
                        continue
                else:
                    continue
                break
            else:
                return None, None
        
        if self.verbose:
            logger.info('Detecting vpp from: %s', ckpt_path)
        
        state = self._load_checkpoint(ckpt_path)
        model_keys = sorted([
            k for k in state.keys() if k.startswith('model') and k != 'model'
        ])
        
        if 'model0' in state and 'model1' in state:
            if self.verbose:
                logger.info('Detected VPP: vpp_size=%d', len(model_keys))
            return len(model_keys), model_keys
        return None, None

    def _detect_num_layers(self, state: dict) -> int:
        """从 checkpoint 中检测层数。"""
        max_layer_idx = -1
        layer_pattern = re.compile(r'decoder\.layers\.(\d+)\.')
        
        if 'model0' in state:
            state_to_check = state['model0']
        elif 'model' in state:
            state_to_check = state['model']
        else:
            state_to_check = state
        
        for key in state_to_check.keys():
            match = layer_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                max_layer_idx = max(max_layer_idx, layer_idx)
        
        if self.num_layers:
            return self.num_layers
        
        return max_layer_idx + 1 if max_layer_idx >= 0 else 32

    def _build_vpprank_layer_map_dualpipe(self, num_layers: int) -> None:
        """构建 DualPipe 的 VPP layer 映射。"""
        layers_each_pp = num_layers // self.pp_size
        layer_pop_num = layers_each_pp // 2
        all_layers = list(range(num_layers))
        dualpipe_layers: list[int] = []
        
        while all_layers:
            dualpipe_layers.extend(all_layers[:layer_pop_num])
            dualpipe_layers.extend(all_layers[-layer_pop_num:])
            all_layers = all_layers[layer_pop_num:-layer_pop_num]

        pp_rank = 0
        vpp_rank = 0
        each_pp_layer = num_layers // self.pp_size
        
        if self.vpp_stage is None:
            self.vpp_stage = layers_each_pp // 2
        
        for idx, layer in enumerate(dualpipe_layers):
            if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = []
            
            self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
            
            if (idx + 1) % self.vpp_stage == 0:
                vpp_rank += 1
            if (idx + 1) % each_pp_layer == 0:
                pp_rank += 1
                vpp_rank = 0
        
        for pp_rank in range(self.pp_size):
            for vpp_rank in self.vpprank_layer_idxs[pp_rank]:
                stage_key = (pp_rank, vpp_rank)
                self._stage_local_to_global[stage_key] = {}
                for local_idx, hf_layer in enumerate(self.vpprank_layer_idxs[pp_rank][vpp_rank]):
                    self.layer2loc_vpp[hf_layer] = (pp_rank, vpp_rank, local_idx)
                    self._stage_local_to_global[stage_key][local_idx] = hf_layer

    def _build_vpprank_layer_map_standard(self, num_layers: int) -> None:
        """构建标准 VPP layer 映射。"""
        if self.vpp_size is None:
            raise ValueError("标准 VPP 模式需要 vpp_size")
        
        vpp_stage = self.vpp_stage or (num_layers // (self.pp_size * self.vpp_size))
        layers_each_vpp = [[vpp_stage] * self.vpp_size for _ in range(self.pp_size)]
        real_layers = list(range(num_layers))
        
        for vpp_rank in range(self.vpp_size):
            for pp_rank in range(self.pp_size):
                count = layers_each_vpp[pp_rank][vpp_rank]
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                    real_layers.pop(0) for _ in range(count)
                ]

        for pp_rank in range(self.pp_size):
            for vpp_rank in range(self.vpp_size):
                stage_key = (pp_rank, vpp_rank)
                self._stage_local_to_global[stage_key] = {}
                for local_idx, hf_layer in enumerate(self.vpprank_layer_idxs[pp_rank][vpp_rank]):
                    self.layer2loc_vpp[hf_layer] = (pp_rank, vpp_rank, local_idx)
                    self._stage_local_to_global[stage_key][local_idx] = hf_layer

    def _build_vpprank_layer_map(self, num_layers: int) -> None:
        """构建 VPP rank 到 layer 的映射。"""
        if self.dualpipe:
            self._build_vpprank_layer_map_dualpipe(num_layers)
        else:
            self._build_vpprank_layer_map_standard(num_layers)

    def _load_rank_state(self, tp_rank: int, pp_rank: int, ep_rank: int | None,
                         vpp_rank: int | None) -> dict:
        """加载单个 rank 的 state"""
        ckpt_path = self._resolve_rank_ckpt_path(tp_rank, pp_rank, ep_rank)
        state = self._load_checkpoint(ckpt_path)
        
        if vpp_rank is None:
            return state.get('model', state)
        return state.get(f'model{vpp_rank}', state.get('model', state))

    def _load_models_for_stage(self, pp_rank: int, vpp_rank: int | None) -> dict:
        """加载指定 stage 的所有 TP rank"""
        models: dict[tuple[int, int], dict] = {}
        
        if self.verbose:
            logger.info('Loading models for stage: pp_rank=%d vpp_rank=%s', pp_rank, vpp_rank)

        ranks_to_load: list[tuple[int, int]] = []
        for tp_rank in range(self.tp_size):
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if not eps:
                ranks_to_load.append((tp_rank, 0))
            else:
                for ep_rank in eps:
                    ranks_to_load.append((tp_rank, ep_rank))

        def load_one(tp_rank: int, ep_rank: int) -> tuple[int, int, dict | None]:
            try:
                st = self._load_rank_state(tp_rank, pp_rank, ep_rank, vpp_rank)
                return tp_rank, ep_rank, st
            except FileNotFoundError:
                return tp_rank, ep_rank, None

        max_workers = min(self.io_threads, len(ranks_to_load))
        if max_workers <= 1:
            for tp_rank, ep_rank in ranks_to_load:
                tp, ep, st = load_one(tp_rank, ep_rank)
                if st is not None:
                    models[(tp, ep)] = st
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = [ex.submit(load_one, tp, ep) for tp, ep in ranks_to_load]
                for fut in as_completed(futures):
                    tp, ep, st = fut.result()
                    if st is not None:
                        models[(tp, ep)] = st

        if self.verbose:
            logger.info('Loaded models for stage: pp_rank=%d vpp_rank=%s models=%d',
                       pp_rank, vpp_rank, len(models))
        
        return models

    def _get_global_layer_id(self, local_idx: int, pp_rank: Optional[int], 
                             vpp_rank: Optional[int]) -> int:
        """根据 local_idx 和 stage 信息获取全局 layer id。"""
        if self.vpp_size is not None and pp_rank is not None and vpp_rank is not None:
            stage_key = (pp_rank, vpp_rank)
            layer_id = self._stage_local_to_global.get(stage_key, {}).get(local_idx)
            if layer_id is not None:
                return layer_id
            for global_id, (pr, vr, li) in self.layer2loc_vpp.items():
                if pr == pp_rank and vr == vpp_rank and li == local_idx:
                    return global_id
            return local_idx
        else:
            if self.pp_size > 1 and pp_rank is not None and self.num_layers:
                layers_per_pp = self.num_layers // self.pp_size
                return pp_rank * layers_per_pp + local_idx
            return local_idx

    def _get_tp_parallel_dim(self, mcore_name: str) -> Optional[int]:
        """确定权重在哪个维度上进行张量并行切分。"""
        # Row-wise (dim=0)
        if 'embedding.word_embeddings.weight' in mcore_name:
            return 0
        if 'output_layer.weight' in mcore_name:
            return 0
        if 'self_attention.linear_qkv.weight' in mcore_name:
            return 0
        if 'self_attention.linear_qkv.bias' in mcore_name:
            return 0
        if 'mlp.linear_fc1.weight' in mcore_name:
            return 0
        if 'mlp.linear_fc1.bias' in mcore_name:
            return 0
        if 'shared_experts.linear_fc1.weight' in mcore_name:
            return 0
        if 'experts.local_experts' in mcore_name and 'linear_fc1.weight' in mcore_name:
            return 0
        if 'experts.local_experts' in mcore_name and 'linear_fc1.bias' in mcore_name:
            return 0
        
        # Column-wise (dim=1)
        if 'self_attention.linear_proj.weight' in mcore_name:
            return 1
        if 'mlp.linear_fc2.weight' in mcore_name:
            return 1
        if 'shared_experts.linear_fc2.weight' in mcore_name:
            return 1
        if 'experts.local_experts' in mcore_name and 'linear_fc2.weight' in mcore_name:
            return 1
        
        # 不切分
        if 'layernorm' in mcore_name.lower():
            return None
        if 'router.weight' in mcore_name:
            return None
        if 'router.expert_bias' in mcore_name:
            return None
        
        return None

    def _merge_tp_shapes(self, name: str, tp_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """合并所有 TP rank 的形状，恢复原始形状。"""
        if not tp_shapes:
            return ()
        
        if len(tp_shapes) == 1:
            return tp_shapes[0]
        
        parallel_dim = self._get_tp_parallel_dim(name)
        if parallel_dim is None:
            return tp_shapes[0]
        
        base_shape = list(tp_shapes[0])
        total_dim = sum(s[parallel_dim] for s in tp_shapes)
        base_shape[parallel_dim] = total_dim
        return tuple(base_shape)

    def _is_moe_layer(self, layer_id: int) -> bool:
        """判断指定层是否为 MoE 层。"""
        return layer_id >= self.first_k_dense_replace

    def extract_weights(self) -> ExtractResult:
        """提取所有权重信息。"""
        self._validate()
        self._build_rank_dir_map()
        
        result = ExtractResult(
            metadata={},
            weight_map={}
        )
        
        if self.verbose:
            print("=" * 80)
            print("MCore 权重提取")
            print("=" * 80)
            print(f"\nMCore 目录: {self.mcore_dir}")
            print(f"迭代目录: {self.iter_dir}")
            print(f"并行配置: TP={self.tp_size}, PP={self.pp_size}, EP={self.ep_size}")
            print(f"模型配置: layers={self.num_layers}, first_k_dense={self.first_k_dense_replace}")
        
        # 检测 VPP
        self.vpp_size, self._vpp_model_keys = self._detect_vpp()
        if self.vpp_size:
            if self.verbose:
                print(f"\n检测到 VPP: vpp_size={self.vpp_size}")
            if self.vpp_stage is None and self.dualpipe:
                self.vpp_stage = max(1, self.vpp_size // 2)
                if self.verbose:
                    print(f"自动设置 vpp_stage={self.vpp_stage} (dualpipe)")
            elif self.vpp_stage is None:
                self.vpp_stage = 1
                self.warnings.append("检测到 VPP 但未提供 --vpp-stage，使用默认值 1")
        
        # 加载 checkpoint 检测 num_layers
        try:
            sample_ckpt_path = self._resolve_rank_ckpt_path(0, 0, 0 if self.ep_size > 1 else None)
            sample_state = self._load_checkpoint(sample_ckpt_path)
            detected_num_layers = self._detect_num_layers(sample_state)
            if self.num_layers is None:
                self.num_layers = detected_num_layers
        except Exception as e:
            if self.num_layers is None:
                self.num_layers = 32
                self.warnings.append(f"无法检测 num_layers，使用默认值 {self.num_layers}")
        
        # 构建 VPP layer 映射
        if self.vpp_size and self.num_layers:
            self._build_vpprank_layer_map(self.num_layers)
        
        # 确定所有 stages
        if self.vpp_size is None:
            stages = [(pp, None) for pp in range(self.pp_size)]
        else:
            stages = [(pp, vpp) for pp in range(self.pp_size) for vpp in range(self.vpp_size)]
        
        if self.verbose:
            print(f"\n开始提取权重信息，共 {len(stages)} 个 stage(s)...")
        
        # 存储所有权重 (按 HF 名称去重)
        all_weights: Dict[str, Dict[str, Any]] = {}
        total_size = 0
        
        # 处理每个 stage
        for pp_rank, vpp_rank in stages:
            if self.verbose:
                print(f"\n处理 PP Rank {pp_rank}" + (f", VPP Rank {vpp_rank}" if vpp_rank is not None else "") + "...")
            
            try:
                models = self._load_models_for_stage(pp_rank, vpp_rank)
            except Exception as e:
                self.errors.append(f"无法加载 stage pp={pp_rank}, vpp={vpp_rank}: {e}")
                continue
            
            if not models:
                self.warnings.append(f"Stage pp={pp_rank}, vpp={vpp_rank} 没有模型")
                continue
            
            # 收集所有权重并合并 TP 形状
            all_mcore_weights: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = {}
            
            for (tp_rank, ep_rank), state in models.items():
                for name, tensor in state.items():
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    if name not in all_mcore_weights:
                        all_mcore_weights[name] = {}
                    all_mcore_weights[name][(tp_rank, ep_rank)] = tensor
            
            # 处理每个 MCore 权重
            for mcore_name, tp_ep_tensors in all_mcore_weights.items():
                tp_shapes = []
                dtype = None
                ep_rank_for_expert = None  # 用于计算全局专家 ID
                
                for (tp_rank, ep_rank), tensor in tp_ep_tensors.items():
                    tp_shapes.append(tuple(tensor.shape))
                    if dtype is None:
                        dtype = _get_torch_dtype_name(tensor.dtype)
                    # 记录 ep_rank (用于专家 ID 计算，假设同一个专家的权重来自同一个 EP rank)
                    if ep_rank_for_expert is None:
                        ep_rank_for_expert = ep_rank
                
                if not tp_shapes:
                    continue
                
                merged_shape = self._merge_tp_shapes(mcore_name, tp_shapes)
                
                # 提取 local layer id
                layer_match = re.match(r'decoder\.layers\.(\d+)\.', mcore_name)
                if layer_match:
                    local_idx = int(layer_match.group(1))
                    layer_id = self._get_global_layer_id(local_idx, pp_rank, vpp_rank)
                else:
                    layer_id = None
                
                # 转换为 HF 格式 (传入 ep_rank 用于计算全局专家 ID)
                hf_entries = self._convert_mcore_to_hf_entries(mcore_name, merged_shape, layer_id, ep_rank_for_expert)
                
                for hf_name, hf_shape in hf_entries:
                    if hf_name not in all_weights:
                        # 估算 size_bytes (shape 乘积 * 2 字节对于 bf16/fp16)
                        elem_size = 2 if dtype in ('bfloat16', 'float16') else 4
                        size_bytes = sum(hf_shape) * elem_size if len(hf_shape) == 1 else hf_shape[0] * hf_shape[1] * elem_size
                        
                        all_weights[hf_name] = {
                            'shape': list(hf_shape),
                            'dtype': dtype,
                        }
                        total_size += size_bytes
        
        # 构建 metadata (与 model_weights_info.json 对齐)
        result.weight_map = all_weights
        result.total_size = total_size
        result.metadata = {
            'total_size': total_size,
            'num_layers': self.num_layers,
            'hidden_size': self.hidden_size,
            'vocab_size': self.vocab_size,
            'num_attention_heads': self.num_attention_heads,
            'num_key_value_heads': self.num_key_value_heads,
            'n_routed_experts': self.num_experts,
            'n_shared_experts': self.n_shared_experts,
        }
        result.errors = self.errors
        result.warnings = self.warnings
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("提取完成")
            print("=" * 80)
            print(f"\n统计信息:")
            print(f"  总权重数量: {len(all_weights)}")
            print(f"  总大小: {total_size:,} bytes ({total_size / 1e9:.2f} GB)")
            print(f"  层数: {self.num_layers}")
            
            # 按类型统计
            layer_count = sum(1 for n in all_weights if 'model.layers.' in n)
            embed_count = sum(1 for n in all_weights if 'embed_tokens' in n)
            norm_count = sum(1 for n in all_weights if 'model.norm' in n)
            head_count = sum(1 for n in all_weights if 'lm_head' in n)
            gate_count = sum(1 for n in all_weights if '.gate.weight' in n)
            
            print(f"\n  权重分布:")
            print(f"    Embedding: {embed_count}")
            print(f"    Layers: {layer_count}")
            print(f"    Norm: {norm_count}")
            print(f"    LM Head: {head_count}")
            print(f"    Router Gates: {gate_count}")
            
            if self.errors:
                print(f"\n错误: {len(self.errors)} 个")
                for e in self.errors[:5]:
                    print(f"  - {e}")
            
            if self.warnings:
                print(f"\n警告: {len(self.warnings)} 个")
                for w in self.warnings[:5]:
                    print(f"  - {w}")
        
        return result

    def _convert_mcore_to_hf_entries(self, mcore_name: str, shape: Tuple[int, ...], 
                                      layer_id: Optional[int],
                                      ep_rank: Optional[int] = None) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        将 MCore 权重名称转换为 HF 格式的条目列表。
        对于融合权重，展开为多个条目。
        
        Args:
            mcore_name: MCore 权重名称
            shape: 合并后的形状
            layer_id: 全局层 ID
            ep_rank: Expert Parallel rank (用于计算全局专家 ID)
        
        Returns:
            [(hf_name, shape), ...]
        """
        results = []
        
        # Embedding
        if mcore_name == 'embedding.word_embeddings.weight':
            results.append(('model.embed_tokens.weight', shape))
            return results
        
        # Final LayerNorm
        if mcore_name == 'decoder.final_layernorm.weight':
            results.append(('model.norm.weight', shape))
            return results
        
        # Output layer
        if mcore_name == 'output_layer.weight':
            results.append(('lm_head.weight', shape))
            return results
        
        # 需要 layer_id 的权重
        if layer_id is None:
            return results
        
        prefix = f'model.layers.{layer_id}'
        
        # Self Attention
        if 'self_attention.linear_proj.weight' in mcore_name:
            results.append((f'{prefix}.self_attn.o_proj.weight', shape))
            return results
        
        if 'self_attention.q_layernorm.weight' in mcore_name:
            results.append((f'{prefix}.self_attn.q_layernorm.weight', shape))
            return results
        
        if 'self_attention.k_layernorm.weight' in mcore_name:
            results.append((f'{prefix}.self_attn.k_layernorm.weight', shape))
            return results
        
        # QKV 融合权重展开
        if 'self_attention.linear_qkv.weight' in mcore_name:
            qkv_out_dim, hidden = shape
            # 计算 Q, K, V 的输出维度
            q_out = self.num_attention_heads * self.head_dim
            k_out = self.num_key_value_heads * self.head_dim
            v_out = self.num_key_value_heads * self.head_dim
            
            # 如果形状不匹配，按比例分配
            expected_total = q_out + k_out + v_out
            if qkv_out_dim != expected_total:
                total_heads = self.num_attention_heads + 2 * self.num_key_value_heads
                q_out = qkv_out_dim * self.num_attention_heads // total_heads
                k_out = qkv_out_dim * self.num_key_value_heads // total_heads
                v_out = qkv_out_dim - q_out - k_out
            
            results.append((f'{prefix}.self_attn.q_proj.weight', (q_out, hidden)))
            results.append((f'{prefix}.self_attn.k_proj.weight', (k_out, hidden)))
            results.append((f'{prefix}.self_attn.v_proj.weight', (v_out, hidden)))
            return results
        
        # LayerNorm
        if 'input_layernorm.weight' in mcore_name:
            results.append((f'{prefix}.input_layernorm.weight', shape))
            return results
        
        if 'pre_mlp_layernorm.weight' in mcore_name or 'post_attention_layernorm.weight' in mcore_name:
            results.append((f'{prefix}.post_attention_layernorm.weight', shape))
            return results
        
        # 判断是否为 MoE 层
        is_moe = self._is_moe_layer(layer_id)
        
        # Dense MLP (前 first_k_dense_replace 层)
        if not is_moe and 'mlp.linear_fc1.weight' in mcore_name:
            fused_dim, hidden = shape
            single_dim = fused_dim // 2
            results.append((f'{prefix}.mlp.gate_proj.weight', (single_dim, hidden)))
            results.append((f'{prefix}.mlp.up_proj.weight', (single_dim, hidden)))
            return results
        
        if not is_moe and 'mlp.linear_fc2.weight' in mcore_name:
            results.append((f'{prefix}.mlp.down_proj.weight', shape))
            return results
        
        # MoE 层
        if is_moe:
            # Router (EP 间需要合并)
            if 'mlp.router.weight' in mcore_name:
                results.append((f'{prefix}.mlp.gate.weight', shape))
                return results
            
            if 'mlp.router.expert_bias' in mcore_name:
                results.append((f'{prefix}.mlp.gate.e_score_correction_bias', shape))
                return results
            
            # Shared Experts
            if 'shared_experts.linear_fc1.weight' in mcore_name:
                fused_dim, hidden = shape
                single_dim = fused_dim // 2
                results.append((f'{prefix}.mlp.shared_experts.gate_proj.weight', (single_dim, hidden)))
                results.append((f'{prefix}.mlp.shared_experts.up_proj.weight', (single_dim, hidden)))
                return results
            
            if 'shared_experts.linear_fc2.weight' in mcore_name:
                results.append((f'{prefix}.mlp.shared_experts.down_proj.weight', shape))
                return results
            
            # MoE Experts - local_experts 格式
            # 计算全局专家 ID: global_expert_id = ep_rank * num_local_experts + local_expert_id
            num_local_experts = self.num_experts // self.ep_size if self.ep_size > 1 else self.num_experts
            
            expert_fc1_match = re.match(r'.*mlp\.experts\.local_experts\.(\d+)\.linear_fc1\.weight', mcore_name)
            if expert_fc1_match:
                local_expert_id = int(expert_fc1_match.group(1))
                # 使用 ep_rank 计算全局专家 ID
                if ep_rank is not None:
                    global_expert_id = ep_rank * num_local_experts + local_expert_id
                else:
                    global_expert_id = local_expert_id
                fused_dim, hidden = shape
                single_dim = fused_dim // 2
                results.append((f'{prefix}.mlp.experts.{global_expert_id}.gate_proj.weight', (single_dim, hidden)))
                results.append((f'{prefix}.mlp.experts.{global_expert_id}.up_proj.weight', (single_dim, hidden)))
                return results
            
            expert_fc2_match = re.match(r'.*mlp\.experts\.local_experts\.(\d+)\.linear_fc2\.weight', mcore_name)
            if expert_fc2_match:
                local_expert_id = int(expert_fc2_match.group(1))
                # 使用 ep_rank 计算全局专家 ID
                if ep_rank is not None:
                    global_expert_id = ep_rank * num_local_experts + local_expert_id
                else:
                    global_expert_id = local_expert_id
                results.append((f'{prefix}.mlp.experts.{global_expert_id}.down_proj.weight', shape))
                return results
        
        return results

    def get_json_output(self) -> dict:
        """生成 JSON 格式的输出。"""
        result = self.extract_weights()
        
        return {
            'metadata': result.metadata,
            'weight_map': result.weight_map,
        }

    def save_json(self, output_path: str, indent: int = 2) -> None:
        """保存权重信息到 JSON 文件。"""
        output = self.get_json_output()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=indent, ensure_ascii=False)
        
        if self.verbose:
            print(f"\nJSON 输出已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='读取 MCore 格式权重并保存为 JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 (TP=1, PP=1, EP=1)
  python get_mcore_weights.py /path/to/mcore/ckpt --output weights_info.json

  # Kimi2-1T 默认配置
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --ep 64 \\
    --num-layers 32 --first-k-dense-replace 2 \\
    --num-attention-heads 64 --num-key-value-heads 32 \\
    --hidden-size 7168 --vocab-size 163840 --num-experts 128 \\
    --output weights_info.json

  # 使用 DualPipe
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --ep 64 \\
    --schedules-method dualpipev --vpp-stage 2 \\
    --num-layers 32 --first-k-dense-replace 2 \\
    --output weights_info.json

  # 静默模式
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --output weights_info.json --quiet
        """
    )

    parser.add_argument('mcore_dir', type=str, help='MCore checkpoint 目录')
    parser.add_argument('--tp', type=int, default=1, help='Tensor 并行大小 (默认: 1)')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline 并行大小 (默认: 1)')
    parser.add_argument('--ep', type=int, default=1, help='Expert 并行大小 (默认: 1)')
    parser.add_argument('--schedules-method', type=str, default=None, choices=['dualpipev'],
                        help='调度方法 (dualpipev 用于 DualPipe)')
    parser.add_argument('--vpp-stage', type=int, default=None, help='Virtual pipeline stage 大小')
    parser.add_argument('--num-layers', type=int, default=None, help='模型的总层数')
    parser.add_argument('--first-k-dense-replace', type=int, default=2,
                        help='前 K 层使用 Dense MLP 而不是 MoE (默认: 2)')
    parser.add_argument('--num-attention-heads', type=int, default=64, help='注意力头数 (默认: 64)')
    parser.add_argument('--num-key-value-heads', type=int, default=32, help='KV 头数 (默认: 32)')
    parser.add_argument('--hidden-size', type=int, default=7168, help='隐藏层维度 (默认: 7168)')
    parser.add_argument('--vocab-size', type=int, default=163840, help='词表大小 (默认: 163840)')
    parser.add_argument('--num-experts', type=int, default=128, help='MoE 专家总数 (默认: 128)')
    parser.add_argument('--io-threads', type=int, default=4, help='加载 checkpoint 的线程数 (默认: 4)')
    parser.add_argument('--disable-mmap', action='store_true', help='禁用 torch.load 的 mmap')
    parser.add_argument('--output', '-o', type=str, default=None, help='输出 JSON 文件路径')
    parser.add_argument('--quiet', action='store_true', help='静默模式 (减少日志输出)')

    args = parser.parse_args()
    
    extractor = MCoreWeightExtractor(
        mcore_dir=args.mcore_dir,
        tp_size=args.tp,
        pp_size=args.pp,
        ep_size=args.ep,
        schedules_method=args.schedules_method,
        vpp_stage=args.vpp_stage,
        num_layers=args.num_layers,
        first_k_dense_replace=args.first_k_dense_replace,
        num_experts=args.num_experts,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        hidden_size=args.hidden_size,
        vocab_size=args.vocab_size,
        io_threads=args.io_threads,
        disable_mmap=args.disable_mmap,
        verbose=not args.quiet,
    )

    if args.output:
        extractor.save_json(args.output)
    else:
        output = extractor.get_json_output()
        print(json.dumps(output, indent=2, ensure_ascii=False))
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
