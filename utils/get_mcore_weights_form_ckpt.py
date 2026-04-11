#!/usr/bin/env python3
"""
MCore 格式权重读取与保存工具

从 MCore checkpoint 中提取权重信息（名称、形状、数据类型、requires_grad），
输出格式与 model_param_mapping.json 的 megatron_params 部分对齐。

用法:
  python get_mcore_weights_form_ckpt.py /path/to/mcore/checkpoint \
    --tp 2 --pp 8 --ep 64 \
    --num-layers 32 \
    --output model_param_mapping.json
"""

import argparse
import inspect
import json
import logging
import os
import re
import sys
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Set, Callable

import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass(frozen=True)
class TensorInfo:
    """张量基本信息（不可变）"""
    shape: Tuple[int, ...]
    dtype: str
    requires_grad: bool
    
    @property
    def size_bytes(self) -> int:
        """计算张量字节大小"""
        numel = 1
        for dim in self.shape:
            numel *= dim
        return numel * _dtype_to_elem_size(self.dtype)


@dataclass
class WeightInfo:
    """权重信息数据结构"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    requires_grad: bool
    size_bytes: int
    source: str = ""


@dataclass
class ExtractResult:
    """提取结果数据结构"""
    megatron_params: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    total_params: int = 0
    total_size: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class ParallelConfig:
    """并行配置"""
    tp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    vpp_size: Optional[int] = None
    vpp_stage: Optional[int] = None
    schedules_method: Optional[str] = None
    
    @property
    def dualpipe(self) -> bool:
        return self.schedules_method == 'dualpipev'


@dataclass
class ModelConfig:
    """模型配置"""
    num_layers: int = 32
    num_experts: int = 128
    num_attention_heads: int = 64
    num_query_groups: int = 2
    hidden_size: int = 7168
    kv_channels: int = 128
    ffn_hidden_size: int = 18432
    moe_ffn_hidden_size: int = 12288
    first_k_dense_replace: int = 2
    vocab_size: int = 163840
    
    @property
    def qkv_dim(self) -> int:
        """QKV 总维度"""
        return (self.num_attention_heads + 2 * self.num_query_groups) * self.kv_channels
    
    @property
    def attention_proj_dim(self) -> int:
        """Attention 输出投影维度"""
        return self.num_attention_heads * self.kv_channels


# =============================================================================
# Utility Functions
# =============================================================================

def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int, ep: int) -> str:
    """生成 mp_rank 目录前缀"""
    if ep == 1 and pp == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


def _resolve_iter_dir(load_dir: str) -> str:
    """解析迭代目录，支持 latest 指针和默认迭代目录"""
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

    for it_name in ['iter_0000001', 'iter_0000000']:
        it_path = os.path.join(load_dir, it_name)
        if os.path.isdir(it_path):
            return it_path

    if os.path.basename(load_dir).startswith('iter_'):
        return load_dir

    raise FileNotFoundError(
        f'无法定位迭代目录: {load_dir}\n'
        f'请确保目录包含 latest_checkpointed_iteration.txt 或 iter_XXXXXXX 子目录'
    )


def _torch_load_compat(path: str, disable_mmap: bool = False) -> dict:
    """兼容不同 PyTorch 版本的 torch.load 调用"""
    base = {'map_location': 'cpu'}
    try:
        sig = inspect.signature(torch.load)
    except Exception:
        sig = None

    support_weights_only = bool(sig and 'weights_only' in sig.parameters)
    support_mmap = bool(sig and 'mmap' in sig.parameters and not disable_mmap)

    candidates: List[dict] = [
        {**base, 'weights_only': True, 'mmap': True} if support_weights_only and support_mmap else base,
        {**base, 'weights_only': False, 'mmap': True} if support_mmap else base,
        {**base, 'weights_only': False} if support_weights_only else base,
        base,
    ]

    for kw in candidates:
        try:
            return torch.load(path, **kw)
        except Exception:
            continue

    raise RuntimeError(f"无法加载 checkpoint: {path}")


def _get_torch_dtype_name(dtype: torch.dtype) -> str:
    """获取 PyTorch 数据类型的字符串名称"""
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


def _dtype_to_elem_size(dtype: str) -> int:
    """根据数据类型名称获取元素字节大小"""
    dtype_lower = dtype.lower()
    if any(x in dtype_lower for x in ['float64', 'fp64', 'int64']):
        return 8
    if any(x in dtype_lower for x in ['float32', 'fp32', 'int32']):
        return 4
    if any(x in dtype_lower for x in ['float16', 'fp16', 'bfloat16', 'bf16', 'int16']):
        return 2
    if any(x in dtype_lower for x in ['int8', 'uint8']):
        return 1
    return 2  # 默认 bf16/fp16


def _get_tensor_size(shape: Tuple[int, ...], dtype: str) -> int:
    """计算张量的字节大小"""
    numel = 1
    for dim in shape:
        numel *= dim
    return numel * _dtype_to_elem_size(dtype)


def _natural_sort_key(name: str) -> Tuple:
    """自然排序 key，确保 embedding -> layers -> final_norm -> output 顺序"""
    layer_match = re.search(r'layers\.(\d+)', name)
    if layer_match:
        layer_num = int(layer_match.group(1))
        suffix = name.split('layers.')[1].split('.', 1)[1] if '.' in name.split('layers.')[1] else ''
        return (1, layer_num, suffix)
    if 'embedding' in name:
        return (0, 0, name)
    if 'final_layernorm' in name:
        return (2, 0, name)
    if 'output_layer' in name:
        return (3, 0, name)
    return (4, 0, name)


# =============================================================================
# Checkpoint Cache
# =============================================================================

class CheckpointCache:
    """LRU 风格的 Checkpoint 文件缓存"""
    
    def __init__(self, max_size: int = 8):
        self._cache: Dict[str, dict] = {}
        self._access_order: List[str] = []
        self._max_size = max_size
    
    def get(self, path: str) -> Optional[dict]:
        if path not in self._cache:
            return None
        self._access_order.remove(path)
        self._access_order.append(path)
        return self._cache[path]
    
    def put(self, path: str, state: dict) -> None:
        if path in self._cache:
            return
        while len(self._cache) >= self._max_size:
            oldest = self._access_order.pop(0)
            del self._cache[oldest]
        self._cache[path] = state
        self._access_order.append(path)


# =============================================================================
# Layer Mapping Strategy
# =============================================================================

class LayerMapper(ABC):
    """Layer 映射抽象基类"""
    
    @abstractmethod
    def build_mapping(self, num_layers: int, pp_size: int, vpp_stage: int) -> Dict[int, Tuple[int, int, int]]:
        """
        构建 layer 到 (pp_rank, vpp_rank, local_idx) 的映射
        返回: {global_layer_id: (pp_rank, vpp_rank, local_idx)}
        """
        pass


class StandardVppMapper(LayerMapper):
    """标准 VPP Layer 映射"""
    
    def build_mapping(self, num_layers: int, pp_size: int, vpp_stage: int) -> Dict[int, Tuple[int, int, int]]:
        layers_per_vpp = vpp_stage
        mapping = {}
        
        for pp_rank in range(pp_size):
            for vpp_rank in range(num_layers // (pp_size * layers_per_vpp)):
                for local_idx in range(layers_per_vpp):
                    global_layer = pp_rank * (num_layers // pp_size) + vpp_rank * layers_per_vpp + local_idx
                    mapping[global_layer] = (pp_rank, vpp_rank, local_idx)
        
        return mapping


class DualPipeMapper(LayerMapper):
    """DualPipe Layer 映射"""
    
    def build_mapping(self, num_layers: int, pp_size: int, vpp_stage: int) -> Dict[int, Tuple[int, int, int]]:
        layers_per_pp = num_layers // pp_size
        layer_pop_num = layers_per_pp // 2
        all_layers = list(range(num_layers))
        dualpipe_layers: List[int] = []

        # DualPipe 交错分布
        while all_layers:
            dualpipe_layers.extend(all_layers[:layer_pop_num])
            dualpipe_layers.extend(all_layers[-layer_pop_num:])
            all_layers = all_layers[layer_pop_num:-layer_pop_num]

        mapping = {}
        pp_rank = 0
        vpp_rank = 0
        layers_per_stage = vpp_stage

        for idx, layer in enumerate(dualpipe_layers):
            mapping[layer] = (pp_rank, vpp_rank, idx % layers_per_stage)
            if (idx + 1) % layers_per_stage == 0:
                vpp_rank += 1
            if (idx + 1) % layers_per_pp == 0:
                pp_rank += 1
                vpp_rank = 0

        return mapping


class LayerMapperFactory:
    """Layer Mapper 工厂"""
    
    @staticmethod
    def create(dualpipe: bool) -> LayerMapper:
        return DualPipeMapper() if dualpipe else StandardVppMapper()


# =============================================================================
# Parallel Strategy
# =============================================================================

class ParallelStrategy(ABC):
    """并行切分策略抽象基类"""
    
    @abstractmethod
    def get_tp_parallel_dim(self, name: str) -> Optional[int]:
        """返回 TP 切分维度 (0=row-wise, 1=column-wise, None=不切分)"""
        pass
    
    @abstractmethod
    def is_ep_sharded(self, name: str) -> bool:
        """判断是否在 EP 维度切分"""
        pass
    
    @abstractmethod
    def get_expected_shape(self, name: str, config: ModelConfig) -> Optional[Tuple[int, ...]]:
        """获取期望的权重 shape"""
        pass


class MoeParallelStrategy(ParallelStrategy):
    """MoE 模型的并行切分策略"""
    
    # 预编译正则模式以提高性能
    PATTERNS = {
        'expert_weights': re.compile(r'\.experts\.(?:weight[12]|local_experts)'),
        'shared_experts_fc1': re.compile(r'shared_experts\.linear_fc1'),
        'shared_experts_fc2': re.compile(r'shared_experts\.linear_fc2'),
        'embedding': re.compile(r'embedding\.word_embeddings\.weight'),
        'output': re.compile(r'output_layer\.weight'),
        'qkv': re.compile(r'self_attention\.linear_qkv'),
        'proj': re.compile(r'self_attention\.linear_proj'),
        'dense_mlp_fc1': re.compile(r'\.mlp\.linear_fc1\.weight$'),
        'dense_mlp_fc2': re.compile(r'\.mlp\.linear_fc2\.weight$'),
        'layernorm': re.compile(r'layernorm', re.I),
        'router': re.compile(r'router\.(?:weight|expert_bias|score_bias)'),
        'qk_norm': re.compile(r'[qk]_layernorm'),
    }
    
    def get_tp_parallel_dim(self, name: str) -> Optional[int]:
        """确定权重在哪个维度上进行张量并行切分"""
        # 专家权重不参与 TP 切分
        if self.PATTERNS['expert_weights'].search(name):
            return None
        
        # Shared experts 参与 TP 切分
        if self.PATTERNS['shared_experts_fc1'].search(name):
            return 0
        if self.PATTERNS['shared_experts_fc2'].search(name):
            return 1
        
        # Row-wise (dim=0): 输出维度被切分
        if any(self.PATTERNS[p].search(name) for p in ['embedding', 'output', 'qkv']):
            return 0
        if 'mlp.linear_fc1' in name and 'shared_experts' not in name:
            return 0
        
        # Column-wise (dim=1): 输入维度被切分
        if self.PATTERNS['proj'].search(name):
            return 1
        if 'mlp.linear_fc2' in name and 'shared_experts' not in name:
            return 1
        
        # 不切分
        if self.PATTERNS['layernorm'].search(name):
            return None
        if self.PATTERNS['router'].search(name):
            return None
        
        return None
    
    def is_ep_sharded(self, name: str) -> bool:
        """判断权重是否在 EP 维度上切分"""
        return self.PATTERNS['expert_weights'].search(name) is not None
    
    def get_expected_shape(self, name: str, config: ModelConfig) -> Optional[Tuple[int, ...]]:
        """根据模型配置计算期望的 shape"""
        h = config.hidden_size
        
        # Embedding / Output
        if self.PATTERNS['embedding'].search(name) or self.PATTERNS['output'].search(name):
            return (config.vocab_size, h)
        
        # Q/K Layernorm (MLA)
        if self.PATTERNS['qk_norm'].search(name):
            return (config.kv_channels,)
        
        # LayerNorm
        if self.PATTERNS['layernorm'].search(name):
            return (h,)
        
        # Attention QKV
        if 'self_attention.linear_qkv.weight' in name:
            return (config.qkv_dim, h)
        if 'self_attention.linear_qkv.bias' in name:
            return (config.qkv_dim,)
        
        # Attention projection
        if 'self_attention.linear_proj.weight' in name:
            return (h, config.attention_proj_dim)
        
        # Dense MLP (first_k_dense_replace layers)
        if self.PATTERNS['dense_mlp_fc1'].search(name):
            return (config.ffn_hidden_size, h)
        if self.PATTERNS['dense_mlp_fc2'].search(name):
            return (h, config.ffn_hidden_size)
        
        # Shared experts
        if self.PATTERNS['shared_experts_fc1'].search(name) and '.weight' in name:
            return (config.moe_ffn_hidden_size, h)
        if self.PATTERNS['shared_experts_fc2'].search(name) and '.weight' in name:
            return (h, config.moe_ffn_hidden_size)
        
        # Router
        if 'router.weight' in name:
            return (config.num_experts, h)
        if 'router.expert_bias' in name:
            return (config.num_experts,)
        
        # Experts (grouped_gemm 格式)
        if '.experts.weight1' in name:
            return (h, config.num_experts * config.moe_ffn_hidden_size * 2)
        if '.experts.weight2' in name:
            return (config.num_experts * config.moe_ffn_hidden_size, h)
        
        return None


# =============================================================================
# Shape Merger
# =============================================================================

class ShapeMerger:
    """形状合并器"""
    
    def __init__(self, strategy: ParallelStrategy):
        self.strategy = strategy
    
    def merge_tp_shapes(self, name: str, tp_shapes: List[Tuple[int, ...]], 
                        debug: bool = False) -> Tuple[int, ...]:
        """合并所有 TP rank 的形状"""
        if not tp_shapes or len(tp_shapes) == 1:
            return tp_shapes[0] if tp_shapes else ()
        
        parallel_dim = self.strategy.get_tp_parallel_dim(name)
        if parallel_dim is None:
            return tp_shapes[0]
        
        base_shape = list(tp_shapes[0])
        base_shape[parallel_dim] = sum(s[parallel_dim] for s in tp_shapes)
        
        if debug:
            logger.info('  TP merge %s: dim=%d, shapes=%s -> %s',
                       name, parallel_dim, tp_shapes, tuple(base_shape))
        
        return tuple(base_shape)
    
    def merge_ep_shapes(self, name: str, ep_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """合并所有 EP rank 的形状"""
        if not ep_shapes or len(ep_shapes) == 1:
            return ep_shapes[0] if ep_shapes else ()
        
        # grouped_gemm 格式的 experts.weight1
        if '.experts.weight1' in name:
            base_shape = list(ep_shapes[0])
            base_shape[1] = sum(s[1] for s in ep_shapes)
            return tuple(base_shape)
        
        # grouped_gemm 格式的 experts.weight2
        if '.experts.weight2' in name:
            base_shape = list(ep_shapes[0])
            base_shape[0] = sum(s[0] for s in ep_shapes)
            return tuple(base_shape)
        
        return ep_shapes[0]
    
    def merge(self, name: str, tp_ep_shapes: Dict[Tuple[int, int], Tuple[int, ...]],
              is_ep_sharded: bool) -> Tuple[int, ...]:
        """
        合并所有 TP/EP rank 的形状
        
        Args:
            name: 权重名称
            tp_ep_shapes: {(tp_rank, ep_rank): shape}
            is_ep_sharded: 是否在 EP 维度切分
        """
        # 按 EP rank 分组
        ep_groups: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
        for (tp_rank, ep_rank), shape in tp_ep_shapes.items():
            ep_groups[ep_rank].append((tp_rank, shape))
        
        # 在每个 EP 组内合并 TP shapes
        ep_merged: Dict[int, Tuple[int, ...]] = {}
        for ep_rank, tp_list in ep_groups.items():
            tp_shapes = [s for _, s in sorted(tp_list)]
            ep_merged[ep_rank] = self.merge_tp_shapes(name, tp_shapes)
        
        # 合并 EP 维度
        if is_ep_sharded and len(ep_merged) > 1:
            sorted_shapes = [ep_merged[ep] for ep in sorted(ep_merged.keys())]
            return self.merge_ep_shapes(name, sorted_shapes)
        else:
            return ep_merged[sorted(ep_merged.keys())[0]]


# =============================================================================
# Checkpoint Reader
# =============================================================================

class CheckpointLoader:
    """Checkpoint 加载器"""
    
    def __init__(self, iter_dir: str, parallel: ParallelConfig, 
                 cache: CheckpointCache, disable_mmap: bool = False):
        self.iter_dir = iter_dir
        self.parallel = parallel
        self.cache = cache
        self.disable_mmap = disable_mmap
        self._rank_dir_map: Dict[Tuple[int, int], List[int]] = {}
    
    def build_rank_map(self) -> None:
        """构建 rank 目录映射"""
        mp_dirs = [d for d in os.listdir(self.iter_dir) if d.startswith('mp_rank_')]
        rank_map: Dict[Tuple[int, int], Set[int]] = defaultdict(set)
        
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
            
            pp, ep = self._parse_pp_ep(idxs)
            rank_map[(tp, pp)].add(ep)
        
        self._rank_dir_map = {k: sorted(list(v)) for k, v in rank_map.items()}
        
        # 验证完整性
        expected = self.parallel.tp_size * self.parallel.pp_size
        actual = len(self._rank_dir_map)
        if actual < expected:
            logger.warning('Rank map incomplete: %d/%d entries', actual, expected)
    
    def _parse_pp_ep(self, idxs: List) -> Tuple[int, int]:
        """解析 PP 和 EP 索引"""
        if not idxs or idxs[0] is None:
            return 0, 0
        
        if len(idxs) == 1:
            v = idxs[0]
            if self.parallel.pp_size > 1 and self.parallel.ep_size == 1:
                return v, 0
            elif self.parallel.pp_size == 1 and self.parallel.ep_size > 1:
                return 0, v
            else:
                return (v, 0) if v < self.parallel.pp_size else (0, v)
        
        return idxs[0], idxs[1]
    
    def get_ckpt_path(self, tp_rank: int, pp_rank: int, 
                      ep_rank: Optional[int]) -> str:
        """获取 checkpoint 文件路径"""
        candidates = self._build_candidates(tp_rank, pp_rank, ep_rank)
        
        for p in candidates:
            path = os.path.join(self.iter_dir, p, 'model_optim_rng.pt')
            if os.path.isfile(path):
                return path
        
        raise FileNotFoundError(
            f'无法定位 rank 文件: tp={tp_rank}, pp={pp_rank}, ep={ep_rank}'
        )
    
    def _build_candidates(self, tp_rank: int, pp_rank: int, 
                          ep_rank: Optional[int]) -> List[str]:
        """构建候选路径列表"""
        p, tp, pp, ep = self.parallel, self.parallel.tp_size, self.parallel.pp_size, self.parallel.ep_size
        
        if ep_rank is not None:
            return [
                _mp_prefix(tp_rank, pp_rank, ep_rank, tp, pp, ep),
                f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}',
                f'mp_rank_{tp_rank:02}_{ep_rank:03}_{pp_rank:03}',
                f'mp_rank_{tp_rank:02}_{ep_rank:03}',
            ]
        else:
            return [
                f'mp_rank_{tp_rank:02}_{pp_rank:03}_000',
                f'mp_rank_{tp_rank:02}_{pp_rank:03}_001',
                f'mp_rank_{tp_rank:02}_{pp_rank:03}',
                f'mp_rank_{tp_rank:02}',
            ]
    
    def load(self, tp_rank: int, pp_rank: int, ep_rank: Optional[int],
             vpp_rank: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """加载指定 rank 的 state"""
        path = self.get_ckpt_path(tp_rank, pp_rank, ep_rank)
        
        # 检查缓存
        cached = self.cache.get(path)
        if cached is not None:
            state = cached
        else:
            state = _torch_load_compat(path, disable_mmap=self.disable_mmap)
            self.cache.put(path, state)
        
        # 提取 model 部分
        key = f'model{vpp_rank}' if vpp_rank is not None else 'model'
        if key in state:
            return state[key]
        elif 'state_dict' in state:
            return state['state_dict']
        return state
    
    def load_stage(self, pp_rank: int, vpp_rank: Optional[int], 
                   io_threads: int) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """加载指定 stage 的所有 TP/EP rank"""
        models: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
        
        # 收集需要加载的 ranks
        ranks_to_load: List[Tuple[int, int]] = []
        for tp_rank in range(self.parallel.tp_size):
            ep_ranks = self._rank_dir_map.get((tp_rank, pp_rank), [0])
            for ep_rank in ep_ranks:
                ranks_to_load.append((tp_rank, ep_rank))
        
        def load_one(tp: int, ep: int) -> Optional[Tuple[int, int, Dict]]:
            try:
                return (tp, ep, self.load(tp, pp_rank, ep, vpp_rank))
            except FileNotFoundError:
                return None
        
        if io_threads <= 1 or len(ranks_to_load) <= 1:
            for tp, ep in ranks_to_load:
                result = load_one(tp, ep)
                if result:
                    models[(result[0], result[1])] = result[2]
        else:
            with ThreadPoolExecutor(max_workers=min(io_threads, len(ranks_to_load))) as ex:
                futures = [ex.submit(load_one, tp, ep) for tp, ep in ranks_to_load]
                for fut in as_completed(futures):
                    result = fut.result()
                    if result:
                        models[(result[0], result[1])] = result[2]
        
        return models


# =============================================================================
# Main Reader Class
# =============================================================================

class MCoreCheckpointReader:
    """
    MCore Checkpoint 读取器
    
    读取权重信息并保存为 JSON（输出格式与 model_param_mapping.json 对齐）
    """
    
    def __init__(
        self,
        mcore_dir: str,
        tp_size: int = 1,
        pp_size: int = 1,
        ep_size: int = 1,
        num_layers: Optional[int] = None,
        num_experts: Optional[int] = None,
        num_attention_heads: Optional[int] = None,
        num_query_groups: Optional[int] = None,
        hidden_size: Optional[int] = None,
        kv_channels: Optional[int] = None,
        ffn_hidden_size: Optional[int] = None,
        moe_ffn_hidden_size: Optional[int] = None,
        first_k_dense_replace: int = 2,
        vocab_size: Optional[int] = None,
        schedules_method: Optional[str] = None,
        vpp_stage: Optional[int] = None,
        io_threads: int = 16,
        disable_mmap: bool = False,
        verbose: bool = True,
        validate_shapes: bool = True,
    ):
        self.verbose = verbose
        self.mcore_dir = mcore_dir
        self.iter_dir = _resolve_iter_dir(mcore_dir)
        
        # 配置
        self.parallel = ParallelConfig(
            tp_size=tp_size,
            pp_size=pp_size,
            ep_size=ep_size,
            vpp_stage=vpp_stage,
            schedules_method=schedules_method,
        )
        self.model = ModelConfig(
            num_layers=num_layers or 32,
            num_experts=num_experts or 128,
            num_attention_heads=num_attention_heads or 64,
            num_query_groups=num_query_groups or 2,
            hidden_size=hidden_size or 7168,
            kv_channels=kv_channels or 128,
            ffn_hidden_size=ffn_hidden_size or 18432,
            moe_ffn_hidden_size=moe_ffn_hidden_size or 12288,
            first_k_dense_replace=first_k_dense_replace,
            vocab_size=vocab_size or 163840,
        )
        
        # 其他配置
        self.io_threads = max(1, io_threads)
        self.disable_mmap = disable_mmap
        self.validate_shapes = validate_shapes
        
        # 组件
        self.cache = CheckpointCache(max_size=max(4, tp_size * 2))
        self.strategy = MoeParallelStrategy()
        self.merger = ShapeMerger(self.strategy)
        self.loader = CheckpointLoader(self.iter_dir, self.parallel, self.cache, disable_mmap)
        
        # 状态
        self.layer_mapper: Optional[LayerMapper] = None
        self._layer2loc: Dict[int, Tuple[int, int, int]] = {}
    
    def _validate(self) -> None:
        """验证参数合法性"""
        if not os.path.isdir(self.mcore_dir):
            raise FileNotFoundError(f'加载目录不存在: {self.mcore_dir}')
        if self.parallel.tp_size <= 0 or self.parallel.pp_size <= 0 or self.parallel.ep_size <= 0:
            raise ValueError('并行度必须 > 0')
    
    def _detect_vpp(self) -> Tuple[Optional[int], List[str]]:
        """检测 VPP 配置"""
        try:
            # 尝试加载第一个可用的 checkpoint
            for tp in range(self.parallel.tp_size):
                for pp in range(self.parallel.pp_size):
                    try:
                        path = self.loader.get_ckpt_path(tp, pp, 0 if self.parallel.ep_size > 1 else None)
                        state = _torch_load_compat(path, self.disable_mmap)
                        model_keys = [k for k in state.keys() if k.startswith('model') and k != 'model']
                        if len(model_keys) > 1:
                            return len(model_keys), model_keys
                        return None, []
                    except FileNotFoundError:
                        continue
        except Exception:
            pass
        return None, []
    
    def _build_layer_mapping(self) -> None:
        """构建 layer 映射"""
        self.layer_mapper = LayerMapperFactory.create(self.parallel.dualpipe)
        
        vpp_stage = self.parallel.vpp_stage
        if vpp_stage is None:
            vpp_stage = max(1, self.model.num_layers // (self.parallel.pp_size * 2))
        
        self._layer2loc = self.layer_mapper.build_mapping(
            self.model.num_layers, self.parallel.pp_size, vpp_stage
        )
        
        if self.verbose:
            logger.info('Layer mapping built: dualpipe=%s, vpp_stage=%d', 
                       self.parallel.dualpipe, vpp_stage)
    
    def _get_global_layer_id(self, local_idx: int, pp_rank: int, 
                             vpp_rank: Optional[int]) -> int:
        """获取全局 layer id"""
        if self.parallel.vpp_size is None:
            return pp_rank * (self.model.num_layers // self.parallel.pp_size) + local_idx
        
        for global_id, (p, v, l) in self._layer2loc.items():
            if p == pp_rank and v == vpp_rank and l == local_idx:
                return global_id
        return local_idx
    
    def _convert_layer_index(self, name: str, pp_rank: int, 
                             vpp_rank: Optional[int]) -> str:
        """转换 layer index 为全局索引"""
        match = re.match(r'(decoder\.layers\.|model\.layers\.)(\d+)(.*)', name)
        if not match:
            return name
        
        prefix, local_idx, suffix = match.group(1), int(match.group(2)), match.group(3)
        global_idx = self._get_global_layer_id(local_idx, pp_rank, vpp_rank)
        return f'{prefix}{global_idx}{suffix}'
    
    def _validate_shape(self, name: str, shape: Tuple[int, ...]) -> List[str]:
        """验证 shape 是否正确"""
        warnings = []
        expected = self.strategy.get_expected_shape(name, self.model)
        
        if expected is None:
            return warnings
        
        if len(shape) != len(expected):
            warnings.append(f'{name}: 维度不匹配，期望 {len(expected)}D，实际 {len(shape)}D')
            return warnings
        
        for i, (actual, exp) in enumerate(zip(shape, expected)):
            if actual != exp:
                warnings.append(f'{name}: 维度 {i} 不匹配，期望 {exp}，实际 {actual}')
        
        return warnings
    
    def extract_weights(self) -> ExtractResult:
        """提取所有权重信息"""
        self._validate()
        self.loader.build_rank_map()
        
        # 检测并配置 VPP
        self.parallel.vpp_size, _ = self._detect_vpp()
        if self.parallel.vpp_size:
            if self.parallel.vpp_stage is None:
                self.parallel.vpp_stage = max(1, self.model.num_layers // (self.parallel.pp_size * 2))
            self._build_layer_mapping()
        
        megatron_params: Dict[str, Dict[str, Any]] = {}
        total_params = 0
        total_size = 0
        errors: List[str] = []
        warnings: List[str] = []
        
        if self.verbose:
            logger.info("=" * 80)
            logger.info("MCore 权重提取")
            logger.info("=" * 80)
            logger.info("目录: %s", self.mcore_dir)
            logger.info("并行: TP=%d, PP=%d, EP=%d, VPP=%s",
                       self.parallel.tp_size, self.parallel.pp_size, 
                       self.parallel.ep_size, self.parallel.vpp_size)
            logger.info("模型: layers=%d, hidden=%d, experts=%d",
                       self.model.num_layers, self.model.hidden_size, self.model.num_experts)
        
        # 确定所有 stages
        if self.parallel.vpp_size is None:
            stages = [(pp, None) for pp in range(self.parallel.pp_size)]
        else:
            stages = [(pp, vpp) for pp in range(self.parallel.pp_size) 
                     for vpp in range(self.parallel.vpp_size)]
        
        # 处理每个 stage
        for pp_rank, vpp_rank in stages:
            if self.verbose:
                logger.info("处理 PP=%d%s...", pp_rank, 
                           f" VPP={vpp_rank}" if vpp_rank is not None else "")
            
            try:
                models = self.loader.load_stage(pp_rank, vpp_rank, self.io_threads)
            except Exception as e:
                errors.append(f"Stage pp={pp_rank}, vpp={vpp_rank}: {e}")
                continue
            
            if not models:
                errors.append(f"Stage pp={pp_rank}, vpp={vpp_rank} 没有模型")
                continue
            
            # 收集所有权重
            weight_tensors: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = defaultdict(dict)
            for (tp_rank, ep_rank), state in models.items():
                for name, tensor in state.items():
                    if isinstance(tensor, torch.Tensor):
                        weight_tensors[name][(tp_rank, ep_rank)] = tensor
            
            # 处理每个权重
            for name, tp_ep_tensors in weight_tensors.items():
                final_name = self._convert_layer_index(name, pp_rank, vpp_rank)
                mcore_full_name = f'module.{final_name}'
                
                if mcore_full_name in megatron_params:
                    continue
                
                # 收集信息
                is_ep_sharded = self.strategy.is_ep_sharded(name)
                tp_ep_shapes = {
                    key: tuple(tensor.shape) 
                    for key, tensor in tp_ep_tensors.items()
                }
                
                # 合并 shape
                merged_shape = self.merger.merge(name, tp_ep_shapes, is_ep_sharded)
                
                # 验证
                if self.validate_shapes:
                    warnings.extend(self._validate_shape(mcore_full_name, merged_shape))
                
                # 获取第一个 tensor 的 dtype 和 requires_grad
                first_tensor = next(iter(tp_ep_tensors.values()))
                dtype = _get_torch_dtype_name(first_tensor.dtype)
                requires_grad = getattr(first_tensor, 'requires_grad', True)
                
                # 添加到结果
                megatron_params[mcore_full_name] = {
                    'shape': list(merged_shape),
                    'dtype': dtype,
                    'requires_grad': requires_grad,
                }
                
                # 统计
                numel = 1
                for dim in merged_shape:
                    numel *= dim
                total_params += numel
                total_size += _get_tensor_size(merged_shape, dtype)
        
        # 构建元数据
        metadata = {
            'total_params': total_params,
            'total_size_bytes': total_size,
            'total_size_gb': round(total_size / 1e9, 2),
            'num_weights': len(megatron_params),
            'model_config': {
                'num_layers': self.model.num_layers,
                'hidden_size': self.model.hidden_size,
                'vocab_size': self.model.vocab_size,
                'num_attention_heads': self.model.num_attention_heads,
                'num_query_groups': self.model.num_query_groups,
                'kv_channels': self.model.kv_channels,
                'num_experts': self.model.num_experts,
                'ffn_hidden_size': self.model.ffn_hidden_size,
                'moe_ffn_hidden_size': self.model.moe_ffn_hidden_size,
                'first_k_dense_replace': self.model.first_k_dense_replace,
            },
            'parallel_config': {
                'tp_size': self.parallel.tp_size,
                'pp_size': self.parallel.pp_size,
                'ep_size': self.parallel.ep_size,
                'vpp_size': self.parallel.vpp_size,
            },
            'source': {
                'mcore_dir': self.mcore_dir,
                'iter_dir': self.iter_dir,
            },
            'note': f'Weights merged from TP={self.parallel.tp_size}, EP={self.parallel.ep_size}',
        }
        
        if errors:
            metadata['errors'] = errors
        if warnings:
            metadata['warnings'] = warnings
        
        # 打印统计
        if self.verbose:
            self._print_stats(megatron_params, total_params, total_size, errors, warnings)
        
        return ExtractResult(
            megatron_params=megatron_params,
            metadata=metadata,
            total_params=total_params,
            total_size=total_size,
            errors=errors,
            warnings=warnings,
        )
    
    def _print_stats(self, params: Dict, total_params: int, total_size: int,
                     errors: List[str], warnings: List[str]) -> None:
        """打印统计信息"""
        logger.info("=" * 80)
        logger.info("提取完成")
        logger.info("=" * 80)
        logger.info("总权重数量: %d", len(params))
        logger.info("总参数量: %d (%.2f B)", total_params, total_params / 1e9)
        logger.info("总大小: %.2f GB", total_size / 1e9)
        
        # 按类型统计
        categories = {
            'Embedding': lambda n: 'embedding' in n,
            'Layers': lambda n: '.layers.' in n,
            'LayerNorm': lambda n: 'layernorm' in n.lower(),
            'Output Layer': lambda n: 'output_layer' in n,
            'Experts': lambda n: 'experts' in n,
        }
        
        logger.info("权重分布:")
        for cat, check in categories.items():
            count = sum(1 for n in params if check(n))
            logger.info("  %s: %d", cat, count)
        
        if warnings:
            logger.warning("Shape 警告: %d 个", len(warnings))
            for w in warnings[:10]:
                logger.warning("  - %s", w)
            if len(warnings) > 10:
                logger.warning("  ... 还有 %d 个", len(warnings) - 10)
        
        if errors:
            logger.error("错误: %d 个", len(errors))
            for e in errors[:5]:
                logger.error("  - %s", e)
    
    def get_json_output(self) -> dict:
        """生成 JSON 格式的输出"""
        result = self.extract_weights()
        sorted_params = dict(sorted(result.megatron_params.items(), key=lambda x: _natural_sort_key(x[0])))
        
        return {
            'megatron_params': sorted_params,
            'metadata': result.metadata,
        }
    
    def save_json(self, output_path: str, indent: int = 2) -> None:
        """保存权重信息到 JSON 文件"""
        output = self.get_json_output()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=indent, ensure_ascii=False)
        
        if self.verbose:
            logger.info("JSON 输出已保存到: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='读取 MCore 格式权重并保存为 JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt --output model.json

  # Kimi2-1T 配置 (TP=2, PP=8, EP=64)
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt \\
    --tp 2 --pp 8 --ep 64 --num-layers 32 --output model.json

  # DualPipe 模式
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt \\
    --tp 2 --pp 8 --ep 64 --schedules-method dualpipev --output model.json
        """
    )

    parser.add_argument('mcore_dir', help='MCore checkpoint 目录')
    parser.add_argument('--tp', type=int, default=1, help='Tensor 并行大小')
    parser.add_argument('--pp', type=int, default=1, help='Pipeline 并行大小')
    parser.add_argument('--ep', type=int, default=1, help='Expert 并行大小')
    parser.add_argument('--num-layers', type=int, default=32, help='模型层数')
    parser.add_argument('--num-attention-heads', type=int, default=64, help='注意力头数')
    parser.add_argument('--num-query-groups', type=int, default=2, help='KV Groups')
    parser.add_argument('--hidden-size', type=int, default=7168, help='隐藏层维度')
    parser.add_argument('--kv-channels', type=int, default=128, help='KV 通道数')
    parser.add_argument('--ffn-hidden-size', type=int, default=18432, help='FFN 隐藏层维度')
    parser.add_argument('--moe-ffn-hidden-size', type=int, default=12288, help='MoE FFN 维度')
    parser.add_argument('--num-experts', type=int, default=128, help='专家数')
    parser.add_argument('--vocab-size', type=int, default=163840, help='词表大小')
    parser.add_argument('--first-k-dense-replace', type=int, default=2, help='前 K 层 Dense MLP')
    parser.add_argument('--schedules-method', type=str, default=None, help='调度方法')
    parser.add_argument('--vpp-stage', type=int, default=None, help='VPP stage 数')
    parser.add_argument('--io-threads', type=int, default=4, help='IO 线程数')
    parser.add_argument('--disable-mmap', action='store_true', help='禁用 mmap')
    parser.add_argument('--no-validate', action='store_true', help='禁用 shape 验证')
    parser.add_argument('--output', '-o', type=str, required=True, help='输出 JSON 路径')
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
        num_query_groups=args.num_query_groups,
        hidden_size=args.hidden_size,
        kv_channels=args.kv_channels,
        ffn_hidden_size=args.ffn_hidden_size,
        moe_ffn_hidden_size=args.moe_ffn_hidden_size,
        first_k_dense_replace=args.first_k_dense_replace,
        vocab_size=args.vocab_size,
        schedules_method=args.schedules_method,
        vpp_stage=args.vpp_stage,
        io_threads=args.io_threads,
        disable_mmap=args.disable_mmap,
        verbose=not args.quiet,
        validate_shapes=not args.no_validate,
    )

    reader.save_json(args.output)
    return 0


if __name__ == '__main__':
    sys.exit(main())
