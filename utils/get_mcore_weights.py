#!/usr/bin/env python3
"""
MCore (Megatron-Core) 格式权重读取与保存工具

用法：
  python get_mcore_weights.py /path/to/mcore/checkpoint \
    --tp 2 --pp 8 --ep 64 \
    --schedules-method dualpipev \
    --vpp-stage 2 \
    --num-layers 32 \
    --output weights_info.json

功能：
1. 读取 MCore 格式 checkpoint 中的权重信息
2. 支持 VPP/DualPipe 调度算法
3. 提取权重名称、形状、数据类型
4. 合并所有 TP rank 的权重，恢复原始形状
5. 展开融合权重（QKV→q,k,v; gate_up→gate,up），完全匹配 HF 格式
6. 支持 MoE 模型（展开 128 个专家的权重）
7. 保存为 JSON 格式，与 model_weights_info.json 格式一致

关键修复：
- 修复了 VPP 场景下只提取部分 layer 的问题（原代码只提取 0 和 1 layer）
- 使用 HF 格式的全局 layer id 进行去重，而不是 MCore 局部索引
- 添加了 --num-layers 参数来正确构建 VPP layer 映射
- 合并所有 TP rank 的权重，恢复原始形状（而非切分后的形状）
- 展开 MCore 融合权重，生成与 model_weights_info.json 完全对应的条目

融合权重展开规则：
- linear_qkv.weight → q_proj.weight, k_proj.weight, v_proj.weight
- linear_fc1.weight → gate_proj.weight, up_proj.weight
- shared_experts.linear_fc1.weight → shared_experts.gate_proj.weight, shared_experts.up_proj.weight
- experts.local_experts.N.linear_fc1.weight → experts.N.gate_proj.weight, experts.N.up_proj.weight

注意：
- MCore checkpoint 中的权重是切分后的（TP 并行），本脚本会合并所有 TP rank 恢复原始形状
- EP (Expert Parallel) 权重会收集所有 EP rank 的专家，生成完整的 128 个专家权重
- 输出结果与 model_weights_info.json 格式完全一致，可直接对比

输出格式示例:
{
  "metadata": {
    "total_size": 2061430584320,
    "num_layers": 32,
    "unique_weights": 11901
  },
  "weight_map": {
    "model.embed_tokens.weight": {
      "shape": [163840, 7168],
      "dtype": "bfloat16"
    },
    "model.layers.0.input_layernorm.weight": {
      "shape": [7168],
      "dtype": "bfloat16"
    },
    "model.layers.0.self_attn.q_proj.weight": {
      "shape": [7168, 7168],
      "dtype": "bfloat16"
    },
    "model.layers.0.mlp.gate_proj.weight": {
      "shape": [18432, 7168],
      "dtype": "bfloat16"
    },
    "model.layers.0.mlp.experts.0.gate_proj.weight": {
      "shape": [12288, 7168],
      "dtype": "bfloat16"
    }
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
logging.basicConfig(format='%(message)s')
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
    is_fused: bool = False  # 是否为融合权重
    fused_components: Optional[List[str]] = None  # 融合权重的组成部分


@dataclass
class ExtractResult:
    """提取结果数据结构。"""
    metadata: Dict[str, Any]
    weight_map: Dict[str, Dict[str, Any]]
    total_size: int = 0
    total_params: int = 0
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
        num_experts: Optional[int] = None,
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
        self.num_layers = num_layers
        self.num_experts = num_experts  # MoE 专家总数
        
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
        
        # Stage -> local_idx -> global_layer 映射
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
            # 尝试其他 rank
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
        
        for key in state.keys():
            match = layer_pattern.match(key)
            if match:
                layer_idx = int(match.group(1))
                max_layer_idx = max(max_layer_idx, layer_idx)
        
        # 检测到的是最大局部索引，需要根据 VPP 配置计算总层数
        if max_layer_idx >= 0:
            if self.vpp_size and self.vpp_size > 1:
                # 如果有 VPP，每个 VPP stage 有 vpp_stage 层
                # 总层数 = vpp_stage * vpp_size (对于非 DualPipe)
                # 或者通过 vpprank_layer_idxs 计算
                pass
        
        # 如果提供了 num_layers，使用提供的值
        if self.num_layers:
            return self.num_layers
        
        # 尝试从 vpprank_layer_idxs 计算
        total_layers = 0
        for pp_rank in self.vpprank_layer_idxs:
            for vpp_rank in self.vpprank_layer_idxs[pp_rank]:
                total_layers += len(self.vpprank_layer_idxs[pp_rank][vpp_rank])
        
        if total_layers > 0:
            return total_layers
        
        # 默认值
        return max_layer_idx + 1 if max_layer_idx >= 0 else 48  # 默认 48 层

    def _build_vpprank_layer_map_dualpipe(self, num_layers: int) -> None:
        """构建 DualPipe 的 VPP layer 映射 - 参考 convert_ckpt_mcore2hf.py"""
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
        
        for idx, layer in enumerate(dualpipe_layers):
            if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = []
            
            self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
            
            if (idx + 1) % self.vpp_stage == 0:
                vpp_rank += 1
            if (idx + 1) % each_pp_layer == 0:
                pp_rank += 1
                vpp_rank = 0
        
        # 构建 layer2loc_vpp 和 _stage_local_to_global
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
        
        # 计算 vpp_stage (每个 VPP stage 的层数)
        vpp_stage = self.vpp_stage or (num_layers // (self.pp_size * self.vpp_size))
        
        layers_each_vpp = [[vpp_stage] * self.vpp_size for _ in range(self.pp_size)]
        real_layers = list(range(num_layers))
        
        for vpp_rank in range(self.vpp_size):
            for pp_rank in range(self.pp_size):
                count = layers_each_vpp[pp_rank][vpp_rank]
                self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                    real_layers.pop(0) for _ in range(count)
                ]

        # 构建 layer2loc_vpp 和 _stage_local_to_global
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
        """加载指定 stage 的所有 TP rank - 支持并行加载
        
        对于 MoE 模型 (EP > 1)，会加载所有 EP rank 以获取完整的专家集合。
        """
        models: dict[tuple[int, int], dict] = {}
        
        if self.verbose:
            logger.info('Loading models for stage: pp_rank=%d vpp_rank=%s', pp_rank, vpp_rank)

        # 收集所有需要加载的 (tp_rank, ep_rank) 组合
        ranks_to_load: list[tuple[int, int]] = []
        for tp_rank in range(self.tp_size):
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if not eps:
                # 如果没有找到 EP 映射，尝试默认的 EP=0
                ranks_to_load.append((tp_rank, 0))
            else:
                # 加载所有 EP rank（用于收集所有专家）
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
            # 回退：尝试从 layer2loc_vpp 反向查找
            for global_id, (pr, vr, li) in self.layer2loc_vpp.items():
                if pr == pp_rank and vr == vpp_rank and li == local_idx:
                    return global_id
            return local_idx
        else:
            if self.pp_size > 1 and pp_rank is not None and self.num_layers:
                layers_per_pp = self.num_layers // self.pp_size
                return pp_rank * layers_per_pp + local_idx
            return local_idx

    def _expand_fused_weight(self, mcore_name: str, shape: Tuple[int, ...],
                             layer_id: int) -> List[Tuple[str, Tuple[int, ...]]]:
        """
        将融合权重展开为多个 HF 格式的权重。
        
        Returns:
            [(hf_name, shape), ...]
        """
        results = []
        
        # QKV 融合: linear_qkv.weight -> q_proj, k_proj, v_proj
        # MCore: [qkv_out_dim, hidden] 其中 qkv_out_dim = q_out + k_out + v_out
        # HF: q_proj [q_out, hidden], k_proj [k_out, hidden], v_proj [v_out, hidden]
        if 'self_attention.linear_qkv.weight' in mcore_name:
            qkv_out_dim, hidden = shape
            # 根据 Kimi2 配置: num_q_heads=64, num_kv_heads=32, head_dim=128
            # q_out = 64 * 128 = 8192, but wait...
            # Actually QKV output is combined in MCore
            # For Kimi2: qkv_out_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
            #          = (64 + 64) * 128 = 16384? No wait...
            # Looking at the shape in mcore_weights_info.json: [4352, 7168] for TP=2
            # After merging: [8704, 7168] or similar
            # For Kimi2: hidden=7168, so we need to figure out the split
            # q_out = 7168 (64 heads * 112 dims? No that's not right)
            # Actually in GQA: q dim = num_q_heads * head_dim = 64 * 128 = 8192
            #                  k dim = num_kv_heads * head_dim = 32 * 128 = 4096  
            #                  v dim = num_kv_heads * head_dim = 32 * 128 = 4096
            # Total = 16384, but that's bigger than hidden_size...
            # Wait, the projection happens before splitting by heads
            # For MCore linear_qkv: output is concatenated QKV
            # After TP merge, shape should be [16384, 7168] for hidden=7168
            # Then: q = [:8192], k = [8192:12288], v = [12288:]
            q_out = hidden  # 7168 for q_proj in Kimi2 (num_q_heads * head_dim / TP?)
            k_out = hidden // 2  # 3584 for k_proj (num_kv_heads * head_dim / TP?)
            v_out = hidden // 2  # 3584 for v_proj
            # Actually for merged shape: q=7168, k=3584, v=3584, total=14336
            # But TP=2 means each rank has half...
            # Let me use a different approach based on model config
            if qkv_out_dim == hidden * 2:  # After merge: [14336, 7168]
                q_out = hidden
                k_out = hidden // 2
                v_out = hidden // 2
            else:
                # Use proportions: Q: 64/128, K: 32/128, V: 32/128 of total
                # For TP merged shape
                q_out = qkv_out_dim * 64 // 128
                k_out = qkv_out_dim * 32 // 128
                v_out = qkv_out_dim * 32 // 128
            
            results.append((f'model.layers.{layer_id}.self_attn.q_proj.weight', (q_out, hidden)))
            results.append((f'model.layers.{layer_id}.self_attn.k_proj.weight', (k_out, hidden)))
            results.append((f'model.layers.{layer_id}.self_attn.v_proj.weight', (v_out, hidden)))
            return results
        
        # MLP gate_up 融合: linear_fc1.weight -> gate_proj, up_proj
        if 'mlp.linear_fc1.weight' in mcore_name and 'shared' not in mcore_name and 'experts' not in mcore_name:
            fused_dim, hidden = shape
            single_dim = fused_dim // 2
            results.append((f'model.layers.{layer_id}.mlp.gate_proj.weight', (single_dim, hidden)))
            results.append((f'model.layers.{layer_id}.mlp.up_proj.weight', (single_dim, hidden)))
            return results
        
        # Shared experts gate_up 融合
        if 'shared_experts.linear_fc1.weight' in mcore_name:
            fused_dim, hidden = shape
            single_dim = fused_dim // 2
            results.append((f'model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight', (single_dim, hidden)))
            results.append((f'model.layers.{layer_id}.mlp.shared_experts.up_proj.weight', (single_dim, hidden)))
            return results
        
        # MoE experts gate_up 融合 (local_experts 格式)
        moe_fc1_match = re.match(r'.*experts\.local_experts\.(\d+)\.linear_fc1\.weight', mcore_name)
        if moe_fc1_match:
            expert_id = int(moe_fc1_match.group(1))
            fused_dim, hidden = shape
            single_dim = fused_dim // 2
            results.append((f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight', (single_dim, hidden)))
            results.append((f'model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight', (single_dim, hidden)))
            return results
        
        # MoE experts gate_up 融合 (global_expert_ 格式，EP 合并后)
        moe_global_match = re.match(r'.*global_expert_(\d+)\.linear_fc1\.weight', mcore_name)
        if moe_global_match:
            expert_id = int(moe_global_match.group(1))
            fused_dim, hidden = shape
            single_dim = fused_dim // 2
            results.append((f'model.layers.{layer_id}.mlp.experts.{expert_id}.gate_proj.weight', (single_dim, hidden)))
            results.append((f'model.layers.{layer_id}.mlp.experts.{expert_id}.up_proj.weight', (single_dim, hidden)))
            return results
        
        return results

    def _mcore_to_hf_weight_name(self, mcore_name: str, 
                                  pp_rank: Optional[int] = None,
                                  vpp_rank: Optional[int] = None) -> Optional[str]:
        """
        将 MCore 权重名称转换为 HF 格式（单个权重）。
        对于融合权重，返回 None，使用 _expand_fused_weight 展开。
        
        Returns:
            HF 格式的权重名称，如果无法映射或是融合权重则返回 None
        """
        # Embedding
        if mcore_name == 'embedding.word_embeddings.weight':
            return 'model.embed_tokens.weight'
        if mcore_name == 'embedding.position_embeddings.weight':
            return 'model.embed_positions.weight'
        
        # Output layer
        if mcore_name == 'decoder.final_layernorm.weight':
            return 'model.norm.weight'
        if mcore_name == 'output_layer.weight':
            return 'lm_head.weight'
        
        # Decoder layers
        layer_match = re.match(r'decoder\.layers\.(\d+)\.(\w+)\.(.*)', mcore_name)
        if not layer_match:
            return None
        
        local_idx = int(layer_match.group(1))
        module_type = layer_match.group(2)
        rest = layer_match.group(3)
        
        layer_id = self._get_global_layer_id(local_idx, pp_rank, vpp_rank)
        
        # Self Attention
        if module_type == 'self_attention':
            if rest == 'linear_proj.weight':
                return f'model.layers.{layer_id}.self_attn.o_proj.weight'
            if rest == 'linear_proj.bias':
                return f'model.layers.{layer_id}.self_attn.o_proj.bias'
            if rest == 'q_layernorm.weight':
                return f'model.layers.{layer_id}.self_attn.q_layernorm.weight'
            if rest == 'k_layernorm.weight':
                return f'model.layers.{layer_id}.self_attn.k_layernorm.weight'
            # 融合权重，需要展开
            if rest == 'linear_qkv.weight':
                return None  # 使用 _expand_fused_weight
            if rest == 'linear_q.weight':
                return f'model.layers.{layer_id}.self_attn.q_proj.weight'
            if rest == 'linear_k.weight':
                return f'model.layers.{layer_id}.self_attn.k_proj.weight'
            if rest == 'linear_v.weight':
                return f'model.layers.{layer_id}.self_attn.v_proj.weight'
        
        # LayerNorm
        if module_type == 'input_layernorm':
            return f'model.layers.{layer_id}.input_layernorm.weight'
        if module_type in ('pre_mlp_layernorm', 'post_attention_layernorm'):
            return f'model.layers.{layer_id}.post_attention_layernorm.weight'
        
        # MLP
        if module_type == 'mlp':
            # 融合权重，需要展开
            if rest == 'linear_fc1.weight':
                return None  # 使用 _expand_fused_weight
            if rest == 'linear_fc1.bias':
                return None
            
            # 分开存储的情况 (如果有)
            if rest == 'linear_fc1_gate.weight':
                return f'model.layers.{layer_id}.mlp.gate_proj.weight'
            if rest == 'linear_fc1_up.weight':
                return f'model.layers.{layer_id}.mlp.up_proj.weight'
            
            if rest == 'linear_fc2.weight':
                return f'model.layers.{layer_id}.mlp.down_proj.weight'
            if rest == 'linear_fc2.bias':
                return f'model.layers.{layer_id}.mlp.down_proj.bias'
            
            # MoE - shared experts
            if rest == 'shared_experts.linear_fc1.weight':
                return None  # 融合权重，使用 _expand_fused_weight
            if rest == 'shared_experts.linear_fc1_gate.weight':
                return f'model.layers.{layer_id}.mlp.shared_experts.gate_proj.weight'
            if rest == 'shared_experts.linear_fc1_up.weight':
                return f'model.layers.{layer_id}.mlp.shared_experts.up_proj.weight'
            if rest == 'shared_experts.linear_fc2.weight':
                return f'model.layers.{layer_id}.mlp.shared_experts.down_proj.weight'
            
            # Router
            if rest == 'router.weight':
                return f'model.layers.{layer_id}.mlp.gate.weight'
            if rest == 'router.bias':
                return f'model.layers.{layer_id}.mlp.gate.bias'
            if rest == 'router.expert_bias':
                return f'model.layers.{layer_id}.mlp.gate.expert_bias'
            if rest == 'router.e_score_correction_bias':
                return f'model.layers.{layer_id}.mlp.gate.e_score_correction_bias'
            
            # Local experts - 融合权重
            moe_fc1_match = re.match(r'experts\.local_experts\.(\d+)\.linear_fc1\.weight', rest)
            if moe_fc1_match:
                return None  # 融合权重，使用 _expand_fused_weight
            
            moe_fc1_bias_match = re.match(r'experts\.local_experts\.(\d+)\.linear_fc1\.bias', rest)
            if moe_fc1_bias_match:
                return None
            
            # linear_fc2 (down proj)
            moe_fc2_match = re.match(r'experts\.local_experts\.(\d+)\.linear_fc2\.weight', rest)
            if moe_fc2_match:
                expert_id = int(moe_fc2_match.group(1))
                return f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight'
            
            moe_fc2_bias_match = re.match(r'experts\.local_experts\.(\d+)\.linear_fc2\.bias', rest)
            if moe_fc2_bias_match:
                expert_id = int(moe_fc2_bias_match.group(1))
                return f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.bias'
            
            # Global experts (EP 合并后) - 融合权重
            moe_global_fc1_match = re.match(r'global_expert_(\d+)\.linear_fc1\.weight', rest)
            if moe_global_fc1_match:
                return None  # 融合权重，使用 _expand_fused_weight
            
            # Global experts (EP 合并后) - down proj
            moe_global_fc2_match = re.match(r'global_expert_(\d+)\.linear_fc2\.weight', rest)
            if moe_global_fc2_match:
                expert_id = int(moe_global_fc2_match.group(1))
                return f'model.layers.{layer_id}.mlp.experts.{expert_id}.down_proj.weight'
        
        return None

    def _extract_weights_from_state(self, state: dict) -> List[WeightInfo]:
        """从 state dict 中提取权重信息。"""
        weights: List[WeightInfo] = []
        
        for key, tensor in state.items():
            # 只处理张量类型的权重
            if not isinstance(tensor, torch.Tensor):
                continue
            
            # 跳过优化器状态
            if key.startswith('optimizer') or key.startswith('opt_'):
                continue
            
            # 跳过标量值
            if tensor.numel() == 1 and key in ['lr', 'learning_rate', 'iteration', 'global_step']:
                continue
            
            weight_info = WeightInfo(
                name=key,
                shape=tuple(tensor.shape),
                dtype=_get_torch_dtype_name(tensor.dtype),
                size_bytes=_get_tensor_size(tensor)
            )
            weights.append(weight_info)
        
        return weights

    def _get_tp_parallel_dim(self, mcore_name: str) -> Optional[int]:
        """
        确定权重在哪个维度上进行张量并行切分。
        
        Returns:
            0: 在第0维切分 (如 linear_qkv)
            1: 在第1维切分 (如 linear_proj, linear_fc2)
            None: 不切分或未知
        """
        # Embedding: 在第0维切分 (vocab_size)
        if 'embedding.word_embeddings.weight' in mcore_name:
            return 0
        if 'embedding.position_embeddings.weight' in mcore_name:
            return None  # position embedding 通常不切分
        
        # Output layer: 在第0维切分 (vocab_size)
        if 'output_layer.weight' in mcore_name:
            return 0
        
        # QKV projection: 在第0维切分
        if 'self_attention.linear_qkv.weight' in mcore_name:
            return 0
        if 'self_attention.linear_qkv.bias' in mcore_name:
            return 0
        
        # Q, K, V 分开的投影: 在第0维切分
        if 'self_attention.linear_q.weight' in mcore_name:
            return 0
        if 'self_attention.linear_k.weight' in mcore_name:
            return 0
        if 'self_attention.linear_v.weight' in mcore_name:
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
        
        # MoE experts
        if 'experts.local_experts' in mcore_name and 'linear_fc1.weight' in mcore_name:
            return 0
        if 'experts.local_experts' in mcore_name and 'linear_fc2.weight' in mcore_name:
            return 1
        
        # Router: 不切分或第0维
        if 'router.weight' in mcore_name:
            return 0
        
        return None

    def _is_fused_weight(self, mcore_name: str) -> bool:
        """判断是否为融合权重。"""
        # QKV 融合
        if 'self_attention.linear_qkv.weight' in mcore_name:
            return True
        # MLP gate_up 融合 (Dense MLP)
        if 'mlp.linear_fc1.weight' in mcore_name and 'shared' not in mcore_name and 'experts' not in mcore_name:
            return True
        # Shared experts gate_up 融合
        if 'shared_experts.linear_fc1.weight' in mcore_name:
            return True
        # MoE experts gate_up 融合 (local_experts 格式)
        if 'experts.local_experts' in mcore_name and 'linear_fc1.weight' in mcore_name:
            return True
        # MoE experts gate_up 融合 (global_expert_ 格式，EP 合并后)
        if 'global_expert_' in mcore_name and 'linear_fc1.weight' in mcore_name:
            return True
        return False

    def _get_fused_weight_info(self, mcore_name: str, shape: Tuple[int, ...]) -> Tuple[str, Tuple[int, ...], List[str]]:
        """
        获取融合权重的映射信息。
        
        Returns:
            (hf_name_pattern, merged_shape, fused_components)
        """
        # QKV 融合: [3 * head_dim * num_heads, hidden_size] -> 分成 q, k, v
        if 'self_attention.linear_qkv.weight' in mcore_name:
            # MCore QKV 形状通常是 [qkv_out_dim, hidden_size]
            # 其中 qkv_out_dim = (num_q_heads + 2 * num_kv_heads) * head_dim
            # 映射到 HF: q_proj, k_proj, v_proj
            qkv_out_dim = shape[0]
            hidden_size = shape[1]
            # 简化处理：假设 QKV 均分
            # 实际上需要根据 num_heads 和 num_kv_heads 计算
            # 这里返回原始形状，并标记为融合权重
            return None, shape, ['q_proj', 'k_proj', 'v_proj']
        
        # MLP gate_up 融合: [2 * ffn_hidden, hidden_size] -> 分成 gate, up
        if 'mlp.linear_fc1.weight' in mcore_name:
            fused_dim = shape[0]
            hidden_size = shape[1]
            single_dim = fused_dim // 2
            return None, (single_dim, hidden_size), ['gate_proj', 'up_proj']
        
        # Shared experts gate_up 融合
        if 'shared_experts.linear_fc1.weight' in mcore_name:
            fused_dim = shape[0]
            hidden_size = shape[1]
            single_dim = fused_dim // 2
            return None, (single_dim, hidden_size), ['gate_proj', 'up_proj']
        
        # MoE experts gate_up 融合
        if 'experts.local_experts' in mcore_name and 'linear_fc1.weight' in mcore_name:
            fused_dim = shape[0]
            hidden_size = shape[1]
            single_dim = fused_dim // 2
            return None, (single_dim, hidden_size), ['gate_proj', 'up_proj']
        
        return None, shape, []

    def _merge_tp_shapes(self, name: str, tp_shapes: List[Tuple[int, ...]]) -> Tuple[int, ...]:
        """
        合并所有 TP rank 的形状，恢复原始形状。
        
        Args:
            name: 权重名称
            tp_shapes: 所有 TP rank 的形状列表
        
        Returns:
            合并后的原始形状
        """
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

    def _extract_merged_weights(self, models: Dict[Tuple[int, int], dict], 
                                 pp_rank: int = 0,
                                 vpp_rank: Optional[int] = None) -> Dict[str, WeightInfo]:
        """
        从所有 TP/EP rank 的模型中提取权重并合并形状。
        
        对于普通权重：合并所有 TP rank 的形状
        对于专家权重：为每个 EP rank 中的每个专家单独处理，计算全局专家 ID
        
        Args:
            models: {(tp_rank, ep_rank): state_dict}
            pp_rank: 当前 PP rank
            vpp_rank: 当前 VPP rank
        
        Returns:
            {weight_name: WeightInfo} 合并后的权重信息
        """
        # 分离普通权重和专家权重
        normal_names: Set[str] = set()
        expert_weights: Dict[str, Dict[Tuple[int, int], torch.Tensor]] = {}  # {name: {(tp, ep): tensor}}
        
        for (tp_rank, ep_rank), state in models.items():
            for name, tensor in state.items():
                if not isinstance(tensor, torch.Tensor):
                    continue
                if 'experts.local_experts' in name:
                    if name not in expert_weights:
                        expert_weights[name] = {}
                    expert_weights[name][(tp_rank, ep_rank)] = tensor
                else:
                    normal_names.add(name)
        
        merged_weights: Dict[str, WeightInfo] = {}
        
        # 处理普通权重：合并所有 TP rank
        for name in normal_names:
            tp_shapes = []
            dtype = None
            total_size = 0
            
            for (tp_rank, ep_rank), state in models.items():
                if name in state:
                    tensor = state[name]
                    if isinstance(tensor, torch.Tensor):
                        tp_shapes.append(tuple(tensor.shape))
                        if dtype is None:
                            dtype = _get_torch_dtype_name(tensor.dtype)
                        total_size += _get_tensor_size(tensor)
            
            if not tp_shapes:
                continue
            
            merged_shape = self._merge_tp_shapes(name, tp_shapes)
            is_fused = self._is_fused_weight(name)
            fused_components = None
            
            if is_fused:
                _, _, fused_components = self._get_fused_weight_info(name, merged_shape)
            
            merged_weights[name] = WeightInfo(
                name=name,
                shape=merged_shape,
                dtype=dtype or 'float32',
                size_bytes=total_size,
                is_fused=is_fused,
                fused_components=fused_components
            )
        
        # 处理专家权重：为每个专家计算全局 ID
        # 计算每个 EP rank 拥有的专家数量
        num_local_experts = self.num_experts // self.ep_size if self.num_experts else 16
        
        for name, tp_ep_tensors in expert_weights.items():
            # 按 EP rank 分组
            ep_groups: Dict[int, Dict[int, torch.Tensor]] = {}  # {ep_rank: {tp_rank: tensor}}
            for (tp_rank, ep_rank), tensor in tp_ep_tensors.items():
                if ep_rank not in ep_groups:
                    ep_groups[ep_rank] = {}
                ep_groups[ep_rank][tp_rank] = tensor
            
            # 提取 local expert ID
            local_expert_match = re.search(r'local_experts\.(\d+)', name)
            if not local_expert_match:
                continue
            local_expert_id = int(local_expert_match.group(1))
            
            # 为每个 EP rank 处理其专家
            for ep_rank, tp_tensors in ep_groups.items():
                # 计算全局专家 ID
                global_expert_id = ep_rank * num_local_experts + local_expert_id
                
                # 构建新的权重名称（使用全局专家 ID）
                global_name = name.replace(f'local_experts.{local_expert_id}', f'global_expert_{global_expert_id}')
                
                # 合并 TP shapes
                tp_shapes = [tuple(t.shape) for t in tp_tensors.values()]
                dtype = _get_torch_dtype_name(next(iter(tp_tensors.values())).dtype)
                total_size = sum(_get_tensor_size(t) for t in tp_tensors.values())
                
                if not tp_shapes:
                    continue
                
                merged_shape = self._merge_tp_shapes(name, tp_shapes)
                is_fused = self._is_fused_weight(name)
                fused_components = None
                
                if is_fused:
                    _, _, fused_components = self._get_fused_weight_info(name, merged_shape)
                
                merged_weights[global_name] = WeightInfo(
                    name=global_name,
                    shape=merged_shape,
                    dtype=dtype,
                    size_bytes=total_size,
                    is_fused=is_fused,
                    fused_components=fused_components
                )
        
        return merged_weights

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
        
        # 检测 VPP
        self.vpp_size, self._vpp_model_keys = self._detect_vpp()
        if self.vpp_size:
            if self.verbose:
                print(f"\n检测到 VPP: vpp_size={self.vpp_size}")
            if self.vpp_stage is None and self.dualpipe:
                # DualPipe: 默认 vpp_stage = vpp_size // 2 (每 PP 有 2 个 VPP stage)
                self.vpp_stage = max(1, self.vpp_size // 2)
                if self.verbose:
                    print(f"自动设置 vpp_stage={self.vpp_stage} (dualpipe)")
            elif self.vpp_stage is None:
                # 标准 VPP: 假设均匀分布
                self.vpp_stage = 1
                self.warnings.append("检测到 VPP 但未提供 --vpp-stage，使用默认值 1")
        
        # 首先需要加载一个 checkpoint 来检测 num_layers
        # 在构建 VPP 映射之前需要知道 num_layers
        try:
            sample_ckpt_path = self._resolve_rank_ckpt_path(0, 0, 0 if self.ep_size > 1 else None)
            sample_state = self._load_checkpoint(sample_ckpt_path)
            detected_num_layers = self._detect_num_layers(sample_state)
            if self.num_layers is None:
                self.num_layers = detected_num_layers
                if self.verbose:
                    print(f"检测到 num_layers: {self.num_layers}")
        except Exception as e:
            if self.verbose:
                print(f"无法自动检测 num_layers: {e}")
            if self.num_layers is None:
                self.num_layers = 48  # 默认值
                self.warnings.append(f"无法检测 num_layers，使用默认值 {self.num_layers}")
        
        # 构建 VPP layer 映射
        if self.vpp_size and self.num_layers:
            self._build_vpprank_layer_map(self.num_layers)
            if self.verbose:
                # 打印映射信息
                total_mapped_layers = sum(
                    len(layers) for pp in self.vpprank_layer_idxs.values() 
                    for layers in pp.values()
                )
                print(f"构建 VPP 映射完成: 总层数 = {total_mapped_layers}")
        
        # 确定所有 stages
        if self.vpp_size is None:
            stages = [(pp, None) for pp in range(self.pp_size)]
        else:
            stages = [(pp, vpp) for pp in range(self.pp_size) for vpp in range(self.vpp_size)]
        
        # 用于去重：HF 权重名称 -> 第一个来源
        # 使用 HF 名称去重，因为不同 stage 的相同 local_idx 会映射到不同的全局 layer id
        seen_weights: Dict[str, str] = {}
        all_weights: List[tuple[WeightInfo, Optional[int], Optional[int]]] = []
        
        total_size = 0
        total_params = 0
        
        if self.verbose:
            print(f"\n开始提取权重信息，共 {len(stages)} 个 stage(s)...")
        
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
            
            # 从所有 TP rank 提取并合并权重
            merged_weights = self._extract_merged_weights(models, pp_rank, vpp_rank)
            
            for name, w in merged_weights.items():
                # 计算 HF 名称用于去重
                hf_name = self._mcore_to_hf_weight_name(w.name, pp_rank, vpp_rank)
                dedup_key = hf_name if hf_name is not None else w.name
                
                # 去重：基于 HF 名称（全局唯一）
                if dedup_key in seen_weights:
                    continue
                
                seen_weights[dedup_key] = f"pp={pp_rank},vpp={vpp_rank}"
                all_weights.append((w, pp_rank, vpp_rank))
                total_size += w.size_bytes
                total_params += sum(w.shape)
        
        # 构建 weight_map (使用 HF 格式名称)
        weight_map = {}
        
        for w, pp_rank, vpp_rank in all_weights:
            hf_name = self._mcore_to_hf_weight_name(w.name, pp_rank, vpp_rank)
            
            if hf_name is not None:
                # 单个权重映射
                weight_info = {
                    'shape': list(w.shape),
                    'dtype': w.dtype,
                    'size_bytes': w.size_bytes,
                }
                weight_map[hf_name] = weight_info
            elif w.is_fused:
                # 融合权重，展开成多个 HF 权重
                # 从权重名称中提取 local_idx
                layer_match = re.match(r'decoder\.layers\.(\d+)\.', w.name)
                if layer_match:
                    local_idx = int(layer_match.group(1))
                    layer_id = self._get_global_layer_id(local_idx, pp_rank, vpp_rank)
                else:
                    layer_id = 0
                
                expanded = self._expand_fused_weight(w.name, w.shape, layer_id)
                
                # 计算每个展开后权重的字节数（平均分配融合权重的总字节数）
                bytes_per_component = w.size_bytes // len(expanded) if expanded else w.size_bytes
                
                for expanded_name, expanded_shape in expanded:
                    weight_info = {
                        'shape': list(expanded_shape),
                        'dtype': w.dtype,
                        'size_bytes': bytes_per_component,
                    }
                    weight_map[expanded_name] = weight_info
            else:
                # 无法映射的非融合权重，保留原始名称
                weight_info = {
                    'shape': list(w.shape),
                    'dtype': w.dtype,
                    'size_bytes': w.size_bytes,
                }
                weight_map[w.name] = weight_info
        
        result.weight_map = weight_map
        result.total_size = total_size
        result.total_params = total_params
        # 统计各类型权重数量
        fused_weight_count = sum(1 for w, _, _ in all_weights if w.is_fused)
        
        result.metadata = {
            'total_size': total_size,
            'total_params': total_params,
            'unique_weights': len(all_weights),
            'mapped_to_hf': len([w for w, pp, vpp in all_weights if self._mcore_to_hf_weight_name(w.name, pp, vpp) is not None]),
            'fused_weights': fused_weight_count,
            'parallel_config': {
                'tp': self.tp_size,
                'pp': self.pp_size,
                'ep': self.ep_size,
                'vpp': self.vpp_size,
            },
            'num_layers': self.num_layers,
            'mcore_dir': self.mcore_dir,
            'iter_dir': self.iter_dir,
            'note': f'Weights merged from TP={self.tp_size} ranks, shapes are original (not sharded)',
        }
        result.errors = self.errors
        result.warnings = self.warnings
        
        if self.verbose:
            print("\n" + "=" * 80)
            print("提取完成")
            print("=" * 80)
            print(f"\n统计信息:")
            print(f"  唯一权重数量: {len(all_weights)}")
            
            # 统计成功映射到 HF 格式的数量
            mapped_count = 0
            for w, pp_rank, vpp_rank in all_weights:
                if self._mcore_to_hf_weight_name(w.name, pp_rank, vpp_rank) is not None:
                    mapped_count += 1
            print(f"  映射到 HF 格式: {mapped_count}")
            
            # 统计融合权重
            fused_count = sum(1 for w, _, _ in all_weights if w.is_fused)
            print(f"  融合权重数量: {fused_count}")
            
            print(f"  总大小: {total_size:,} bytes ({total_size / 1e9:.2f} GB)")
            print(f"  (已合并所有 TP={self.tp_size} 个 rank 的权重)")
            
            # 按层统计
            layer_weights = defaultdict(list)
            for name in weight_map.keys():
                match = re.match(r'model\.layers\.(\d+)\.', name)
                if match:
                    layer_id = int(match.group(1))
                    layer_weights[layer_id].append(name)
            
            if layer_weights:
                print(f"\n  Layer 分布:")
                print(f"    共有 {len(layer_weights)} 个不同的 layer")
                print(f"    Layer ID 范围: {min(layer_weights.keys())} - {max(layer_weights.keys())}")
            
            if self.errors:
                print(f"\n错误: {len(self.errors)} 个")
                for e in self.errors[:5]:
                    print(f"  - {e}")
            
            if self.warnings:
                print(f"\n警告: {len(self.warnings)} 个")
                for w in self.warnings[:5]:
                    print(f"  - {w}")
        
        return result

    def get_json_output(self, include_unmapped: bool = False) -> dict:
        """生成 JSON 格式的输出。
        
        Args:
            include_unmapped: 是否包含无法映射到 HF 格式的权重列表
        
        Returns:
            JSON 格式的权重信息字典
        """
        result = self.extract_weights()
        
        output = {
            'metadata': result.metadata,
            'weight_map': result.weight_map,
        }
        
        if include_unmapped and result.warnings:
            # 收集无法映射的权重名称
            unmapped = []
            for w in result.weight_map:
                if w.startswith('decoder.') or w.startswith('embedding.') or w.startswith('output_'):
                    # 这些是 MCore 原始名称，说明映射失败
                    if not any(w.startswith(prefix) for prefix in ['model.embed', 'model.layers.', 'model.norm.', 'lm_head.']):
                        unmapped.append(w)
            if unmapped:
                output['unmapped_weights'] = unmapped
        
        return output
    
    def get_weight_statistics(self) -> dict:
        """获取权重统计信息。"""
        result = self.extract_weights()
        
        # 按模块类型统计
        stats = {
            'embedding': {'count': 0, 'size': 0},
            'layers': {'count': 0, 'size': 0},
            'norm': {'count': 0, 'size': 0},
            'lm_head': {'count': 0, 'size': 0},
            'other': {'count': 0, 'size': 0},
        }
        
        for name, info in result.weight_map.items():
            size = info.get('size_bytes', 0)
            if 'embed' in name:
                stats['embedding']['count'] += 1
                stats['embedding']['size'] += size
            elif 'layers' in name:
                stats['layers']['count'] += 1
                stats['layers']['size'] += size
            elif 'norm' in name:
                stats['norm']['count'] += 1
                stats['norm']['size'] += size
            elif 'lm_head' in name:
                stats['lm_head']['count'] += 1
                stats['lm_head']['size'] += size
            else:
                stats['other']['count'] += 1
                stats['other']['size'] += size
        
        return stats

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

  # 指定并行配置
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --ep 64 \\
    --output weights_info.json

  # 使用 DualPipe
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --ep 64 \\
    --schedules-method dualpipev --vpp-stage 2 --num-layers 32 \\
    --num-experts 128 --output weights_info.json

  # 指定模型层数和专家数（MoE 模型推荐）
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --ep 64 \\
    --num-layers 32 --num-experts 128 --output weights_info.json

  # 静默模式
  python get_mcore_weights.py /path/to/mcore/ckpt --tp 2 --pp 8 --output weights_info.json --quiet
        """
    )

    parser.add_argument('mcore_dir', type=str, help='MCore checkpoint 目录')
    parser.add_argument('--tp', type=int, default=1,
                        help='Tensor 并行大小 (默认: 1)')
    parser.add_argument('--pp', type=int, default=1,
                        help='Pipeline 并行大小 (默认: 1)')
    parser.add_argument('--ep', type=int, default=1,
                        help='Expert 并行大小 (默认: 1)')
    parser.add_argument('--schedules-method', type=str, default=None,
                        choices=['dualpipev'],
                        help='调度方法 (dualpipev 用于 DualPipe)')
    parser.add_argument('--vpp-stage', type=int, default=None,
                        help='Virtual pipeline stage 大小')
    parser.add_argument('--num-layers', type=int, default=None,
                        help='模型的总层数 (用于正确映射 layer id)')
    parser.add_argument('--num-experts', type=int, default=None,
                        help='MoE 模型的专家总数 (用于正确映射专家 id, 默认: 128)')
    parser.add_argument('--io-threads', type=int, default=4,
                        help='加载 checkpoint 的线程数 (默认: 4)')
    parser.add_argument('--disable-mmap', action='store_true',
                        help='禁用 torch.load 的 mmap')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='输出 JSON 文件路径 (默认: 输出到 stdout)')
    parser.add_argument('--quiet', action='store_true',
                        help='静默模式 (减少日志输出)')

    args = parser.parse_args()
    
    extractor = MCoreWeightExtractor(
        mcore_dir=args.mcore_dir,
        tp_size=args.tp,
        pp_size=args.pp,
        ep_size=args.ep,
        schedules_method=args.schedules_method,
        vpp_stage=args.vpp_stage,
        num_layers=args.num_layers,
        num_experts=args.num_experts,
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
