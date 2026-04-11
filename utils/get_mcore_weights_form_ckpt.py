#!/usr/bin/env python3
"""
MCore 格式权重读取与保存工具

从 MCore checkpoint 中提取权重信息（名称、形状、数据类型、requires_grad），
输出格式与 model_param_mapping_tp1.json 的 megatron_params 部分对齐。

修复内容：
1. 修复 TP 合并逻辑 - 对于非 EP 切分权重，正确合并所有 TP rank 的 shape
2. 添加 kv_channels 参数支持 - 正确计算 attention 维度
3. 修复 expert 权重的 TP 维度判断 - expert 不参与 TP 切分
4. 添加 shape 验证 - 自动检测合并是否正确

用法：
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
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Set

import torch

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
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


def _get_tensor_size(shape: Tuple[int, ...], dtype: str) -> int:
    """计算张量的字节大小。"""
    numel = 1
    for dim in shape:
        numel *= dim
    
    elem_size = 2  # 默认 bf16/fp16
    dtype_lower = dtype.lower()
    if 'float32' in dtype_lower or 'fp32' in dtype_lower:
        elem_size = 4
    elif 'float64' in dtype_lower or 'fp64' in dtype_lower:
        elem_size = 8
    elif 'int8' in dtype_lower or 'uint8' in dtype_lower:
        elem_size = 1
    elif 'int16' in dtype_lower:
        elem_size = 2
    elif 'int32' in dtype_lower:
        elem_size = 4
    elif 'int64' in dtype_lower:
        elem_size = 8
    
    return numel * elem_size


@dataclass
class WeightInfo:
    """权重信息数据结构。"""
    name: str
    shape: Tuple[int, ...]
    dtype: str
    requires_grad: bool
    size_bytes: int
    source: str = ""  # 来源信息，如 "pp=0,vpp=0,tp=0,ep=0"


@dataclass
class ExtractResult:
    """提取结果数据结构。"""
    megatron_params: Dict[str, Dict[str, Any]]
    metadata: Dict[str, Any]
    total_params: int = 0
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


class MCoreCheckpointReader:
    """
    MCore Checkpoint 读取器
    
    读取权重信息并保存为 JSON（输出格式与 model_param_mapping_tp1.json 对齐）
    保留 MCore 原始权重名称。
    
    修复：正确合并 TP 和 EP 维度的权重
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
        num_key_value_heads: Optional[int] = None,
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
        self.num_experts = num_experts or 128
        self.num_attention_heads = num_attention_heads or 64
        self.num_key_value_heads = num_key_value_heads or 2
        self.hidden_size = hidden_size or 7168
        self.kv_channels = kv_channels or 128  # 重要：从训练脚本获取
        self.ffn_hidden_size = ffn_hidden_size or 18432
        self.moe_ffn_hidden_size = moe_ffn_hidden_size or 12288
        self.first_k_dense_replace = first_k_dense_replace
        self.vocab_size = vocab_size or 163840
        self.validate_shapes = validate_shapes

        # 其他配置
        self.io_threads = max(1, int(io_threads))
        self.disable_mmap = disable_mmap

        # Checkpoint 缓存
        self._cache = CheckpointCache(max_size=max(4, tp_size * 2))

        # 内部状态
        self._rank_dir_map: Dict[Tuple[int, int], List[int]] = {}
        self.vpp_size: Optional[int] = None
        self._vpp_model_keys: Optional[List[str]] = None
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
            f'iter_dir={self.iter_dir}\n'
            f'尝试的路径: {[os.path.join(self.iter_dir, p, "model_optim_rng.pt") for p in candidates]}'
        )

    def _load_checkpoint(self, path: str) -> dict:
        """加载 checkpoint，带缓存。"""
        cached = self._cache.get(path)
        if cached is not None:
            return cached
        
        state = _torch_load_compat(path, disable_mmap=self.disable_mmap)
        self._cache.put(path, state)
        return state

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
            # 显示前几个映射
            for i, (k, v) in enumerate(list(self._rank_dir_map.items())[:5]):
                logger.info('  (tp=%d, pp=%d) -> ep_ranks=%s', k[0], k[1], v)

    def _detect_vpp(self) -> Tuple[Optional[int], Optional[List[str]]]:
        """检测 VPP 配置。"""
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

    def _build_vpprank_layer_map_dualpipe(self) -> None:
        """构建 DualPipe 的 VPP layer 映射。"""
        layers_each_pp = self.num_layers // self.pp_size
        layer_pop_num = layers_each_pp // 2
        all_layers = list(range(self.num_layers))
        dualpipe_layers: list[int] = []

        while all_layers:
            dualpipe_layers.extend(all_layers[:layer_pop_num])
            dualpipe_layers.extend(all_layers[-layer_pop_num:])
            all_layers = all_layers[layer_pop_num:-layer_pop_num]

        pp_rank = 0
        vpp_rank = 0
        each_pp_layer = self.num_layers // self.pp_size

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

    def _build_vpprank_layer_map_standard(self) -> None:
        """构建标准 VPP layer 映射。"""
        if self.vpp_size is None:
            raise ValueError("标准 VPP 模式需要 vpp_size")

        vpp_stage = self.vpp_stage or (self.num_layers // (self.pp_size * self.vpp_size))
        layers_each_vpp = [[vpp_stage] * self.vpp_size for _ in range(self.pp_size)]
        real_layers = list(range(self.num_layers))

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

    def _build_vpprank_layer_map(self) -> None:
        """构建 VPP rank 到 layer 的映射。"""
        if self.dualpipe:
            self._build_vpprank_layer_map_dualpipe()
        else:
            self._build_vpprank_layer_map_standard()

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
            if self.pp_size > 1 and pp_rank is not None:
                layers_per_pp = self.num_layers // self.pp_size
                return pp_rank * layers_per_pp + local_idx
            return local_idx

    def _load_rank_state(self, tp_rank: int, pp_rank: int,
                         ep_rank: Optional[int], vpp_rank: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """加载单个 rank 的 state。"""
        ckpt_path = self._resolve_rank_ckpt_path(tp_rank, pp_rank, ep_rank)
        state = self._load_checkpoint(ckpt_path)
        
        # 处理可能的嵌套结构
        if vpp_rank is not None and f'model{vpp_rank}' in state:
            state = state[f'model{vpp_rank}']
        elif 'model' in state:
            state = state['model']
        elif 'state_dict' in state:
            state = state['state_dict']
            
        return state

    def _load_models_for_stage(self, pp_rank: int, vpp_rank: Optional[int]) -> Dict[Tuple[int, int], Dict[str, torch.Tensor]]:
        """加载指定 stage 的所有 TP 和 EP rank。"""
        models: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}

        if self.verbose:
            logger.info('Loading models for stage: pp_rank=%d vpp_rank=%s', pp_rank, vpp_rank)

        ranks_to_load: List[Tuple[int, int]] = []
        for tp_rank in range(self.tp_size):
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if not eps:
                ranks_to_load.append((tp_rank, 0))
            else:
                for ep_rank in eps:
                    ranks_to_load.append((tp_rank, ep_rank))

        def load_one(tp_rank: int, ep_rank: int) -> Tuple[int, int, Optional[Dict[str, torch.Tensor]]]:
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

    def _get_tp_parallel_dim(self, mcore_name: str) -> Optional[int]:
        """
        确定权重在哪个维度上进行张量并行切分。
        
        重要：专家权重不参与 TP 切分（expert-tensor-parallel-size=1）
        
        返回:
            0: 在第0维切分 (row-wise)
            1: 在第1维切分 (column-wise)
            None: 不切分
        """
        # 专家权重不参与 TP 切分
        if 'experts.local_experts' in mcore_name:
            return None
        if '.experts.weight1' in mcore_name or '.experts.weight2' in mcore_name:
            return None
        
        # Shared experts 参与 TP 切分
        if 'shared_experts.linear_fc1.weight' in mcore_name:
            return 0
        if 'shared_experts.linear_fc1.bias' in mcore_name:
            return 0
        if 'shared_experts.linear_fc2.weight' in mcore_name:
            return 1

        # Row-wise (dim=0): 输出维度被切分
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

        # Column-wise (dim=1): 输入维度被切分
        if 'self_attention.linear_proj.weight' in mcore_name:
            return 1
        if 'mlp.linear_fc2.weight' in mcore_name:
            return 1

        # 不切分
        if 'layernorm' in mcore_name.lower():
            return None
        if 'router.weight' in mcore_name:
            return None
        if 'router.expert_bias' in mcore_name:
            return None
        if 'router.score_bias' in mcore_name:
            return None
        if '.bias' in mcore_name and 'fc' in mcore_name:
            # 某些 bias 可能被切分，需要进一步检查
            pass

        return None

    def _is_ep_sharded(self, mcore_name: str) -> bool:
        """判断权重是否在 EP 维度上切分。"""
        # 专家权重在 EP 维度上切分
        if 'experts.local_experts' in mcore_name:
            return True
        if '.experts.weight1' in mcore_name or '.experts.weight2' in mcore_name:
            return True
        return False

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

    def _merge_ep_shapes(self, name: str, ep_shapes: List[Tuple[int, ...]], 
                         ep_ranks: List[int]) -> Tuple[int, ...]:
        """
        合并所有 EP rank 的形状，恢复原始形状。
        
        对于 MoE 专家权重，需要合并不同 EP rank 的专家。
        """
        if not ep_shapes or len(ep_shapes) == 1:
            return ep_shapes[0] if ep_shapes else ()

        # 对于 grouped_gemm 格式的 experts.weight1/weight2
        if '.experts.weight1' in name:
            # shape: [hidden, num_local_experts*intermediate_size*2]
            # 在第1维按专家数合并
            base_shape = list(ep_shapes[0])
            total_experts_dim = sum(s[1] for s in ep_shapes)
            base_shape[1] = total_experts_dim
            return tuple(base_shape)
        
        if '.experts.weight2' in name:
            # shape: [num_local_experts*intermediate_size, hidden]
            # 在第0维按专家数合并
            base_shape = list(ep_shapes[0])
            total_experts_dim = sum(s[0] for s in ep_shapes)
            base_shape[0] = total_experts_dim
            return tuple(base_shape)

        # 对于 local_experts 格式，每个专家是独立的
        return ep_shapes[0]

    def _get_expected_shape(self, name: str) -> Optional[Tuple[int, ...]]:
        """
        获取权重的期望形状（用于验证）。
        
        根据模型配置计算期望的 shape。
        """
        # Embedding
        if 'embedding.word_embeddings.weight' in name:
            return (self.vocab_size, self.hidden_size)
        
        # Output layer
        if 'output_layer.weight' in name:
            return (self.vocab_size, self.hidden_size)
        
        # LayerNorm
        if 'layernorm.weight' in name.lower():
            return (self.hidden_size,)
        
        # Q/K Layernorm (for MLA)
        if 'q_layernorm.weight' in name or 'k_layernorm.weight' in name:
            return (self.kv_channels,)
        
        # Attention QKV
        if 'self_attention.linear_qkv.weight' in name:
            # QKV dim = num_heads * kv_channels + 2 * kv_heads * kv_channels
            qkv_dim = (self.num_attention_heads + 2 * self.num_key_value_heads) * self.kv_channels
            return (qkv_dim, self.hidden_size)
        
        if 'self_attention.linear_qkv.bias' in name:
            qkv_dim = (self.num_attention_heads + 2 * self.num_key_value_heads) * self.kv_channels
            return (qkv_dim,)
        
        # Attention projection
        if 'self_attention.linear_proj.weight' in name:
            return (self.hidden_size, self.num_attention_heads * self.kv_channels)
        
        # Dense MLP (first_k_dense_replace layers)
        if '.mlp.linear_fc1.weight' in name and 'shared_experts' not in name and 'experts' not in name:
            return (self.ffn_hidden_size, self.hidden_size)
        if '.mlp.linear_fc2.weight' in name and 'shared_experts' not in name and 'experts' not in name:
            return (self.hidden_size, self.ffn_hidden_size)
        
        # Shared experts
        if 'shared_experts.linear_fc1.weight' in name:
            return (self.moe_ffn_hidden_size, self.hidden_size)
        if 'shared_experts.linear_fc2.weight' in name:
            return (self.hidden_size, self.moe_ffn_hidden_size)
        
        # Router
        if 'router.weight' in name:
            return (self.num_experts, self.hidden_size)
        if 'router.expert_bias' in name:
            return (self.num_experts,)
        
        # Experts
        if '.experts.weight1' in name:
            # [hidden, num_experts * intermediate_size * 2]
            return (self.hidden_size, self.num_experts * self.moe_ffn_hidden_size * 2)
        if '.experts.weight2' in name:
            # [num_experts * intermediate_size, hidden]
            return (self.num_experts * self.moe_ffn_hidden_size, self.hidden_size)
        
        return None

    def _validate_merged_shape(self, name: str, merged_shape: Tuple[int, ...]) -> List[str]:
        """验证合并后的 shape 是否正确。"""
        warnings = []
        expected = self._get_expected_shape(name)
        
        if expected is None:
            return warnings
        
        if len(merged_shape) != len(expected):
            warnings.append(f'{name}: 维度不匹配，期望 {len(expected)}D，实际 {len(merged_shape)}D')
            return warnings
        
        for i, (actual, exp) in enumerate(zip(merged_shape, expected)):
            if actual != exp:
                warnings.append(f'{name}: 维度 {i} 不匹配，期望 {exp}，实际 {actual}')
        
        return warnings

    def extract_weights(self) -> ExtractResult:
        """
        提取所有权重信息。
        
        修复：正确合并 TP 维度的权重
        对于非 EP 切分权重，跨所有 EP rank 收集 TP shapes 并正确合并
        """
        self._validate()
        self._build_rank_dir_map()

        # 检测 VPP
        self.vpp_size, self._vpp_model_keys = self._detect_vpp()
        if self.vpp_size:
            if self.verbose:
                logger.info('Detected VPP: vpp_size=%d', self.vpp_size)
            if self.vpp_stage is None and self.dualpipe:
                self.vpp_stage = max(1, self.num_layers // (self.pp_size * 2))
                if self.verbose:
                    logger.info('Auto-set vpp_stage=%d (dualpipe)', self.vpp_stage)
            elif self.vpp_stage is None:
                self.vpp_stage = 1
                if self.verbose:
                    logger.info('Using default vpp_stage=1')
            self._build_vpprank_layer_map()

        megatron_params: Dict[str, Dict[str, Any]] = {}
        total_params = 0
        total_size = 0
        errors: List[str] = []
        warnings: List[str] = []

        if self.verbose:
            logger.info("=" * 80)
            logger.info("MCore 权重提取（原始格式）")
            logger.info("=" * 80)
            logger.info("MCore 目录: %s", self.mcore_dir)
            logger.info("迭代目录: %s", self.iter_dir)
            logger.info("并行配置: TP=%d, PP=%d, EP=%d, VPP=%s",
                       self.tp_size, self.pp_size, self.ep_size, self.vpp_size)
            logger.info("模型配置: layers=%d, hidden=%d, experts=%d, kv_channels=%d",
                       self.num_layers, self.hidden_size, self.num_experts, self.kv_channels)

        # 确定所有 stages
        if self.vpp_size is None:
            stages = [(pp, None) for pp in range(self.pp_size)]
        else:
            stages = [(pp, vpp) for pp in range(self.pp_size) for vpp in range(self.vpp_size)]

        if self.verbose:
            logger.info("开始提取权重信息，共 %d 个 stage(s)...", len(stages))

        # 处理每个 stage
        for pp_rank, vpp_rank in stages:
            if self.verbose:
                logger.info("处理 PP Rank %d%s...", 
                           pp_rank, 
                           f", VPP Rank {vpp_rank}" if vpp_rank is not None else "")

            try:
                models = self._load_models_for_stage(pp_rank, vpp_rank)
            except Exception as e:
                errors.append(f"无法加载 stage pp={pp_rank}, vpp={vpp_rank}: {e}")
                continue

            if not models:
                errors.append(f"Stage pp={pp_rank}, vpp={vpp_rank} 没有模型")
                continue

            # 按权重名称收集所有 TP/EP rank 的张量
            weight_tensors: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = defaultdict(dict)
            
            for (tp_rank, ep_rank), state in models.items():
                for name, tensor in state.items():
                    if not isinstance(tensor, torch.Tensor):
                        continue
                    weight_tensors[name][(tp_rank, ep_rank)] = tensor

            # 处理每个权重
            for name, tp_ep_tensors in weight_tensors.items():
                # 转换 layer index
                final_name = self._convert_layer_index(name, pp_rank, vpp_rank)
                mcore_full_name = f'module.{final_name}'
                
                # 如果已经处理过，跳过
                if mcore_full_name in megatron_params:
                    continue
                
                # 判断是否是 EP 切分权重
                is_ep_sharded = self._is_ep_sharded(name)
                tp_parallel_dim = self._get_tp_parallel_dim(name)
                
                # 收集所有 (tp_rank, ep_rank) 的 shapes
                all_shapes: Dict[Tuple[int, int], Tuple[int, ...]] = {}
                all_dtypes: Dict[Tuple[int, int], str] = {}
                all_requires_grad: Dict[Tuple[int, int], bool] = {}
                
                for (tp_rank, ep_rank), tensor in tp_ep_tensors.items():
                    all_shapes[(tp_rank, ep_rank)] = tuple(tensor.shape)
                    all_dtypes[(tp_rank, ep_rank)] = _get_torch_dtype_name(tensor.dtype)
                    all_requires_grad[(tp_rank, ep_rank)] = getattr(tensor, 'requires_grad', True)
                
                # 按 EP rank 分组
                ep_groups: Dict[int, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)
                for (tp_rank, ep_rank), shape in all_shapes.items():
                    ep_groups[ep_rank].append((tp_rank, shape))
                
                # 在每个 EP 组内合并 TP shapes
                ep_merged_shapes: Dict[int, Tuple[int, ...]] = {}
                for ep_rank, tp_shapes_list in ep_groups.items():
                    # 按 TP rank 排序
                    tp_shapes_list_sorted = sorted(tp_shapes_list, key=lambda x: x[0])
                    tp_shapes = [s for _, s in tp_shapes_list_sorted]
                    ep_merged_shapes[ep_rank] = self._merge_tp_shapes(name, tp_shapes)
                
                # 合并所有 EP rank 的 shapes
                if is_ep_sharded and len(ep_merged_shapes) > 1:
                    # EP 切分权重：在 EP 维度合并
                    sorted_ep_ranks = sorted(ep_merged_shapes.keys())
                    sorted_shapes = [ep_merged_shapes[ep] for ep in sorted_ep_ranks]
                    merged_shape = self._merge_ep_shapes(name, sorted_shapes, sorted_ep_ranks)
                else:
                    # 非 EP 切分权重：所有 EP 组的 shape 应该相同
                    # 但由于 TP 合并，我们需要重新检查
                    if tp_parallel_dim is not None and len(ep_merged_shapes) > 1:
                        # 这是一个 TP 切分权重，但在多个 EP rank 上都有相同的数据
                        # 我们需要确保 TP 合并在所有 EP rank 上进行
                        # 收集所有 EP rank 的所有 TP shapes
                        all_tp_shapes = []
                        for ep_rank in sorted(ep_groups.keys()):
                            tp_shapes_list = sorted(ep_groups[ep_rank], key=lambda x: x[0])
                            for tp_rank, shape in tp_shapes_list:
                                all_tp_shapes.append(shape)
                        # 重新合并所有 TP shapes
                        merged_shape = self._merge_tp_shapes(name, all_tp_shapes)
                    else:
                        # 不切分的权重，取第一个
                        merged_shape = list(ep_merged_shapes.values())[0]
                
                # 验证合并后的 shape
                if self.validate_shapes:
                    shape_warnings = self._validate_merged_shape(mcore_full_name, merged_shape)
                    warnings.extend(shape_warnings)
                
                # 获取 dtype 和 requires_grad（从第一个可用的）
                first_key = list(all_dtypes.keys())[0]
                dtype = all_dtypes[first_key]
                requires_grad = all_requires_grad[first_key]
                
                # 添加到 megatron_params
                size_bytes = _get_tensor_size(merged_shape, dtype)
                
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
                total_size += size_bytes

        if self.verbose:
            logger.info("=" * 80)
            logger.info("提取完成")
            logger.info("=" * 80)
            logger.info("统计信息:")
            logger.info("  总权重数量: %d", len(megatron_params))
            logger.info("  总参数量: %d (%.2f B)", total_params, total_params / 1e9)
            logger.info("  总大小: %.2f GB", total_size / 1e9)
            
            # 按类型统计
            layer_count = sum(1 for n in megatron_params if '.layers.' in n)
            embed_count = sum(1 for n in megatron_params if 'embedding' in n)
            norm_count = sum(1 for n in megatron_params if 'layernorm' in n.lower())
            output_count = sum(1 for n in megatron_params if 'output_layer' in n)
            expert_count = sum(1 for n in megatron_params if 'experts' in n)
            
            logger.info("权重分布:")
            logger.info("  Embedding: %d", embed_count)
            logger.info("  Layers: %d", layer_count)
            logger.info("  LayerNorm: %d", norm_count)
            logger.info("  Output Layer: %d", output_count)
            logger.info("  Experts: %d", expert_count)
            
            if warnings:
                logger.warning("Shape 警告: %d 个", len(warnings))
                for w in warnings[:10]:
                    logger.warning("  - %s", w)
                if len(warnings) > 10:
                    logger.warning("  ... 还有 %d 个警告", len(warnings) - 10)
            
            if errors:
                logger.error("错误: %d 个", len(errors))
                for e in errors[:5]:
                    logger.error("  - %s", e)

        # 构建输出
        metadata = {
            'total_params': total_params,
            'total_size_bytes': total_size,
            'total_size_gb': round(total_size / 1e9, 2),
            'num_weights': len(megatron_params),
            'model_config': {
                'num_layers': self.num_layers,
                'hidden_size': self.hidden_size,
                'vocab_size': self.vocab_size,
                'num_attention_heads': self.num_attention_heads,
                'num_key_value_heads': self.num_key_value_heads,
                'kv_channels': self.kv_channels,
                'num_experts': self.num_experts,
                'ffn_hidden_size': self.ffn_hidden_size,
                'moe_ffn_hidden_size': self.moe_ffn_hidden_size,
                'first_k_dense_replace': self.first_k_dense_replace,
            },
            'parallel_config': {
                'tp_size': self.tp_size,
                'pp_size': self.pp_size,
                'ep_size': self.ep_size,
                'vpp_size': self.vpp_size,
            },
            'source': {
                'mcore_dir': self.mcore_dir,
                'iter_dir': self.iter_dir,
            },
            'note': f'Weights merged from TP={self.tp_size}, EP={self.ep_size} ranks, shapes are original (not sharded)',
        }
        
        if errors:
            metadata['errors'] = errors
        if warnings:
            metadata['warnings'] = warnings

        return ExtractResult(
            megatron_params=megatron_params,
            metadata=metadata,
            total_params=total_params,
            total_size=total_size,
            errors=errors,
            warnings=warnings,
        )

    def _convert_layer_index(self, name: str, pp_rank: int, 
                             vpp_rank: Optional[int]) -> str:
        """
        将 checkpoint 中的 local layer index 转换为 global layer index。
        
        例如: decoder.layers.0.xxx -> decoder.layers.8.xxx (当 pp_rank=1, num_layers=32, pp_size=4)
        """
        layer_match = re.match(r'(decoder\.layers\.|model\.layers\.)(\d+)(.*)', name)
        if not layer_match:
            return name
        
        prefix = layer_match.group(1)
        local_idx = int(layer_match.group(2))
        suffix = layer_match.group(3)
        
        global_idx = self._get_global_layer_id(local_idx, pp_rank, vpp_rank)
        return f'{prefix}{global_idx}{suffix}'

    def _sort_key(self, name: str) -> tuple:
        """
        生成排序 key，确保按 layer 顺序排列。
        
        顺序：embedding -> layers 0,1,2... -> final_layernorm -> output_layer
        """
        # 提取 layer 编号
        layer_match = re.search(r'layers\.(\d+)', name)
        if layer_match:
            layer_num = int(layer_match.group(1))
            # 返回 (category_order, layer_num, rest_of_name)
            # 对于 layers，使用 layer_num 作为主要排序依据
            prefix = name.split('layers.')[0] + 'layers.'
            suffix = name.split('layers.')[1].split('.', 1)[1] if '.' in name.split('layers.')[1] else ''
            return (1, layer_num, suffix)
        elif 'embedding' in name:
            return (0, 0, name)
        elif 'final_layernorm' in name:
            return (2, 0, name)
        elif 'output_layer' in name:
            return (3, 0, name)
        else:
            return (4, 0, name)

    def get_json_output(self) -> dict:
        """生成 JSON 格式的输出，按 layer 顺序排序。"""
        result = self.extract_weights()
        
        # 对 megatron_params 按 layer 顺序排序
        sorted_params = dict(sorted(result.megatron_params.items(), key=lambda x: self._sort_key(x[0])))
        
        return {
            'megatron_params': sorted_params,
            'metadata': result.metadata,
        }

    def save_json(self, output_path: str, indent: int = 2) -> None:
        """保存权重信息到 JSON 文件。"""
        output = self.get_json_output()

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=indent, ensure_ascii=False)

        if self.verbose:
            logger.info("JSON 输出已保存到: %s", output_path)


def main():
    parser = argparse.ArgumentParser(
        description='读取 MCore 格式权重并保存为 JSON（与 model_param_mapping_tp1.json 对齐）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法 (TP=1, PP=1, EP=1)
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt --output model_param_mapping.json

  # 指定并行配置 (Kimi2-1T 默认配置: TP=2, PP=8, EP=64)
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt \\
    --tp 2 --pp 8 --ep 64 \\
    --num-layers 32 \\
    --num-attention-heads 64 --num-key-value-heads 2 \\
    --hidden-size 7168 --num-experts 128 \\
    --kv-channels 128 --ffn-hidden-size 18432 --moe-ffn-hidden-size 12288 \\
    --vocab-size 163840 \\
    --output model_param_mapping.json

  # DualPipe 模式
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt \\
    --tp 2 --pp 8 --ep 64 \\
    --schedules-method dualpipev --vpp-stage 2 \\
    --output model_param_mapping.json

  # 静默模式
  python get_mcore_weights_form_ckpt.py /path/to/mcore/ckpt --output model_param_mapping.json --quiet
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
    parser.add_argument('--kv-channels',
                        type=int,
                        default=128,
                        help='每个注意力头的 key/value 维度 (默认: 128)')
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
    parser.add_argument('--vocab-size',
                        type=int,
                        default=163840,
                        help='词表大小 (默认: 163840)')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=2,
                        help='前 K 层使用 Dense MLP (默认: 2)')
    parser.add_argument('--schedules-method',
                        type=str,
                        default=None,
                        help='调度方法 (如 dualpipev)')
    parser.add_argument('--vpp-stage',
                        type=int,
                        default=None,
                        help='VPP stage 数 (用于 DualPipe)')
    parser.add_argument('--io-threads',
                        type=int,
                        default=4,
                        help='IO 线程数 (默认: 4)')
    parser.add_argument('--disable-mmap',
                        action='store_true',
                        help='禁用 torch.load 的 mmap')
    parser.add_argument('--no-validate',
                        action='store_true',
                        help='禁用 shape 验证')
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
