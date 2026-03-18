import argparse
import json
import logging as logger
import os
import time
from collections import defaultdict
from typing import Optional

import safetensors.torch
import torch

logger.basicConfig(format='')
logger.getLogger().setLevel(logger.INFO)

HIDDEN_SIZE = 7168
NUM_EXPERTS = 128
FIRST_K_DENSE_REPLACE = 3
NUM_ATTENTION_HEADS = 112
QK_HEAD_DIM = 128
QK_POS_EMB_HEAD_DIM = 64
V_HEAD_DIM = 128
Q_LORA_RANK = 1536


def _parse_int_list(value: Optional[str]) -> Optional[list[int]]:
    if value is None or value == '':
        return None
    return list(map(int, value.split(',')))


def _mp_prefix(tp_rank: int, pp_rank: int, ep_rank: int, tp: int, pp: int,
               ep: int) -> str:
    if ep == 1 and pp == 1:
        return f'mp_rank_{tp_rank:02}'
    if ep == 1:
        return f'mp_rank_{tp_rank:02}_{pp_rank:03}'
    if pp == 1:
        return f'mp_rank_{tp_rank:02}_{ep_rank:03}'
    return f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'


def _resolve_iter_dir(load_dir: str) -> str:
    if os.path.isdir(os.path.join(load_dir, 'iter_0000001')):
        return os.path.join(load_dir, 'iter_0000001')
    latest = os.path.join(load_dir, 'latest_checkpointed_iteration.txt')
    if os.path.isfile(latest):
        with open(latest) as f:
            it = f.read().strip()
        return os.path.join(load_dir, f'iter_{int(it):07d}')
    if os.path.basename(load_dir).startswith('iter_'):
        return load_dir
    raise FileNotFoundError(f'无法定位迭代目录: {load_dir}')


class MgCkptConvert:

    def __init__(
        self,
        mg_load_dir: str,
        hf_save_dir: str,
        num_layers: int,
        tp_size: int,
        pp_size: int,
        ep_size: int,
        first_k_dense_replace: int,
        hidden_size: int,
        num_experts: int,
        num_attention_heads: int,
        qk_head_dim: int,
        v_head_dim: int,
        qk_pos_emb_head_dim: int,
        moe_grouped_gemm: bool,
        moe_tp_extend_ep: bool,
        mla_mm_split: bool,
        schedules_method: str | None,
        vpp_stage: int | None,
        num_layer_list: str | None,
        noop_layers: str | None,
        rotary_base: float,
        q_lora_rank: int,
    ):
        self.verbose = os.environ.get('CKPT_CONVERT_VERBOSE', '1') != '0'
        self.log_rank_load = os.environ.get('CKPT_CONVERT_LOG_RANK',
                                            '1') != '0'
        self.mg_load_dir = mg_load_dir
        self.hf_save_dir = hf_save_dir
        self.iter_dir = _resolve_iter_dir(mg_load_dir)
        if self.verbose:
            logger.info('Resolved iter_dir: %s', self.iter_dir)

        self.num_layers = num_layers
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.ep_size = ep_size
        self.first_k_dense_replace = first_k_dense_replace
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.num_attention_heads = num_attention_heads
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.qk_pos_emb_head_dim = qk_pos_emb_head_dim
        self.moe_grouped_gemm = moe_grouped_gemm
        self.moe_tp_extend_ep = moe_tp_extend_ep
        self.mla_mm_split = mla_mm_split
        self.schedules_method = schedules_method
        self.dualpipe = schedules_method == 'dualpipev'
        self.vpp_stage = vpp_stage
        self.num_layer_list = num_layer_list
        self.noop_layers_list = sorted(_parse_int_list(noop_layers) or [])
        self.rotary_base = rotary_base
        self.q_lora_rank = q_lora_rank

        os.makedirs(self.hf_save_dir, exist_ok=True)

        self.vpp_size, self._vpp_model_keys = self._detect_vpp()
        if self.verbose:
            logger.info('Detected vpp_size=%s dualpipe=%s', self.vpp_size,
                        self.dualpipe)
        if self.vpp_size is not None and self.vpp_stage is None:
            if self.dualpipe:
                layers_each_pp = self.num_layers // self.pp_size
                self.vpp_stage = layers_each_pp // 2
            else:
                raise ValueError('检测到 vpp，但未提供 --vpp-stage 且非 dualpipev')

        self._validate()

        if self.vpp_size is None:
            self.pprank_layer_idxs: dict[int, list[int]] = defaultdict(list)
            self.layer2loc: dict[int, tuple[int, int]] = {}
            self._build_pprank_layer_map()
        else:
            self.vpprank_layer_idxs: dict[int,
                                          dict[int,
                                               list[int]]] = defaultdict(dict)
            self.layer2loc_vpp: dict[int, tuple[int, int, int]] = {}
            self._build_vpprank_layer_map()

        self.num_real_layers = self.num_layers - len(self.noop_layers_list)
        inv_dim = self.hidden_size // self.num_attention_heads
        inv_freq = 1.0 / (self.rotary_base**(
            torch.arange(0, inv_dim, 2, dtype=torch.float32) / inv_dim))
        self.inv_freq = inv_freq

        self.weight_map: dict[str, str] = {}
        self._rank_dir_map: dict[tuple[int, int], list[int]] = {}
        self._build_rank_dir_map()
        if self.verbose:
            logger.info('Rank dir map size=%d', len(self._rank_dir_map))
            sample_keys = sorted(
                self._rank_dir_map.keys())[:min(12, len(self._rank_dir_map))]
            for k in sample_keys:
                eps = self._rank_dir_map[k]
                logger.info('Rank dir map sample tp=%d pp=%d: ep_count=%d',
                            k[0], k[1], len(eps))
            missing = []
            for tp in range(self.tp_size):
                for pp in range(self.pp_size):
                    if (tp, pp) not in self._rank_dir_map:
                        missing.append((tp, pp))
            if missing:
                logger.info('Missing (tp,pp) in rank dirs: %s', missing[:20])

    def _validate(self) -> None:
        if self.num_layers <= 0:
            raise ValueError('num_layers 必须 > 0')
        if self.tp_size <= 0 or self.pp_size <= 0 or self.ep_size <= 0:
            raise ValueError('并行度必须 > 0')
        if self.num_experts % self.ep_size != 0:
            raise ValueError('num_experts 必须能整除 ep_size')
        if self.dualpipe and self.vpp_size is None:
            raise ValueError('dualpipev 需要 vpp checkpoint (model0/model1)')

    def _detect_vpp(self) -> tuple[int | None, list[str] | None]:
        ckpt_path = self._resolve_rank_ckpt_path(0, 0, None)
        if self.verbose:
            logger.info('Detecting vpp from: %s', ckpt_path)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        model_keys = sorted([
            k for k in state.keys() if k.startswith('model') and k != 'model'
        ])
        if 'model0' in state and 'model1' in state:
            return len(model_keys), model_keys
        return None, None

    def _build_rank_dir_map(self) -> None:
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

    def _build_pprank_layer_map(self) -> None:
        layers_each_pp = [self.num_layers // self.pp_size] * self.pp_size
        if self.num_layer_list is not None:
            layer_list = list(map(int, self.num_layer_list.split(',')))
            if len(layer_list) != self.pp_size or sum(
                    layer_list) != self.num_layers:
                raise ValueError('num-layer-list 非法')
            layers_each_pp = layer_list

        if self.noop_layers_list and self.num_layer_list is None:
            base = self.num_layers // self.pp_size
            for layer in self.noop_layers_list:
                pp_rank = layer // base
                layers_each_pp[pp_rank] -= 1

        real_layers = list(range(self.num_real_layers))
        for pp_rank in range(self.pp_size):
            self.pprank_layer_idxs[pp_rank] = [
                real_layers.pop(0) for _ in range(layers_each_pp[pp_rank])
            ]
            for local_idx, hf_layer in enumerate(
                    self.pprank_layer_idxs[pp_rank]):
                self.layer2loc[hf_layer] = (pp_rank, local_idx)

    def _build_vpprank_layer_map(self) -> None:
        if self.dualpipe:
            noop_list = self.noop_layers_list
            min_noop = noop_list[0] if noop_list else None

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
            for idx, layer in enumerate(dualpipe_layers):
                if vpp_rank not in self.vpprank_layer_idxs[pp_rank]:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = []

                if not noop_list:
                    self.vpprank_layer_idxs[pp_rank][vpp_rank].append(layer)
                else:
                    if layer in noop_list:
                        if (idx + 1) % self.vpp_stage == 0:
                            vpp_rank += 1
                        if (idx + 1) % each_pp_layer == 0:
                            pp_rank += 1
                            vpp_rank = 0
                        continue
                    if layer < min_noop:
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(
                            layer)
                    else:
                        before = 0
                        for n in noop_list:
                            if n < layer:
                                before += 1
                            else:
                                break
                        self.vpprank_layer_idxs[pp_rank][vpp_rank].append(
                            layer - before)

                if (idx + 1) % self.vpp_stage == 0:
                    vpp_rank += 1
                if (idx + 1) % each_pp_layer == 0:
                    pp_rank += 1
                    vpp_rank = 0
        else:
            layers_each_vpp = [[self.vpp_stage] * self.vpp_size
                               for _ in range(self.pp_size)]
            if self.noop_layers_list:
                for layer in self.noop_layers_list:
                    vpp_idx = layer // self.vpp_stage // self.pp_size
                    pp_idx = (
                        layer %
                        (self.pp_size * self.vpp_stage)) // self.vpp_stage
                    layers_each_vpp[pp_idx][vpp_idx] -= 1

            real_layers = list(range(self.num_real_layers))
            for vpp_rank in range(self.vpp_size):
                for pp_rank in range(self.pp_size):
                    self.vpprank_layer_idxs[pp_rank][vpp_rank] = [
                        real_layers.pop(0)
                        for _ in range(layers_each_vpp[pp_rank][vpp_rank])
                    ]

        for pp_rank in range(self.pp_size):
            for vpp_rank in range(self.vpp_size):
                for local_idx, hf_layer in enumerate(
                        self.vpprank_layer_idxs[pp_rank][vpp_rank]):
                    self.layer2loc_vpp[hf_layer] = (pp_rank, vpp_rank,
                                                    local_idx)

    def _load_rank_state(self, tp_rank: int, pp_rank: int, ep_rank: int | None,
                         vpp_rank: int | None) -> dict[str, torch.Tensor]:
        ckpt_path = self._resolve_rank_ckpt_path(tp_rank, pp_rank, ep_rank)
        t0 = time.perf_counter()
        if self.verbose and self.log_rank_load:
            logger.info('Loading rank ckpt: tp=%d pp=%d ep=%s vpp=%s path=%s',
                        tp_rank, pp_rank, ep_rank, vpp_rank, ckpt_path)
        state = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        if self.verbose and self.log_rank_load:
            dt = time.perf_counter() - t0
            logger.info('Loaded rank ckpt: tp=%d pp=%d ep=%s vpp=%s in %.2fs',
                        tp_rank, pp_rank, ep_rank, vpp_rank, dt)
        if vpp_rank is None:
            return state['model']
        return state[f'model{vpp_rank}']

    def _resolve_rank_ckpt_path(self, tp_rank: int, pp_rank: int,
                                ep_rank: int | None) -> str:
        candidates: list[str] = []
        if ep_rank is not None:
            candidates.append(
                _mp_prefix(tp_rank, pp_rank, ep_rank, self.tp_size,
                           self.pp_size, self.ep_size))
            candidates.append(
                f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}')
            candidates.append(
                f'mp_rank_{tp_rank:02}_{ep_rank:03}_{pp_rank:03}')
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
            f'无法定位 rank 文件: tp={tp_rank}, pp={pp_rank}, ep={ep_rank}, iter_dir={self.iter_dir}'
        )

    def _load_models_for_stage(
        self, pp_rank: int, vpp_rank: int | None
    ) -> dict[tuple[int, int], dict[str, torch.Tensor]]:
        models: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        if self.verbose:
            logger.info('Loading models for stage: pp_rank=%d vpp_rank=%s',
                        pp_rank, vpp_rank)

        for tp_rank in range(self.tp_size):
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if not eps:
                continue
            base_ep = eps[0]
            state = self._load_rank_state(tp_rank, pp_rank, base_ep, vpp_rank)
            models[(tp_rank, base_ep)] = state
            models[(tp_rank, 0)] = state

        if self.verbose:
            logger.info(
                'Loaded models for stage: pp_rank=%d vpp_rank=%s keys=%d',
                pp_rank, vpp_rank, len(models))
        return models

    def _tp_ranks_for_ep(self, pp_rank: int, ep_rank: int) -> list[int]:
        tps: list[int] = []
        for tp_rank in range(self.tp_size):
            eps = self._rank_dir_map.get((tp_rank, pp_rank), [])
            if ep_rank in eps:
                tps.append(tp_rank)
        return tps

    def _load_sparse_ep_state(self, tp_rank: int, pp_rank: int, ep_rank: int,
                              vpp_rank: int | None) -> dict[str, torch.Tensor]:
        state = self._load_rank_state(tp_rank, pp_rank, ep_rank, vpp_rank)
        keep: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if '.mlp.experts.' in k or '.mlp.router.' in k:
                keep[k] = v
        return keep

    def _reconstruct_router_lazy(self,
                                 base_models: dict[tuple[int, int],
                                                   dict[str, torch.Tensor]],
                                 pp_rank: int,
                                 vpp_rank: int | None,
                                 key: str) -> torch.Tensor:
        for tp_rank in range(self.tp_size):
            d = base_models.get((tp_rank, 0))
            if d is None:
                continue
            t = d.get(key)
            if t is not None and t.shape[0] == self.num_experts:
                return t.clone()

        sample = None
        for tp_rank in range(self.tp_size):
            d = base_models.get((tp_rank, 0))
            if d is None:
                continue
            if key in d:
                sample = d[key]
                break
        if sample is None:
            tps = self._tp_ranks_for_ep(pp_rank, 0)
            if not tps:
                raise ValueError(f'找不到 ep=0 的权重: pp={pp_rank} key={key}')
            d0 = self._load_sparse_ep_state(tps[0], pp_rank, 0, vpp_rank)
            if key not in d0:
                raise ValueError(f'找不到 router 权重: pp={pp_rank} key={key}')
            sample = d0[key]
            del d0

        out = torch.empty((self.num_experts, ) + sample.shape[1:],
                          dtype=sample.dtype)
        num_local = self.num_experts // self.ep_size

        import gc

        for ep in range(self.ep_size):
            owners = self._tp_ranks_for_ep(pp_rank, ep)
            if not owners:
                raise ValueError(
                    f'找不到 ep={ep} 的权重目录: pp={pp_rank} key={key}')
            if len(owners) != 1:
                raise ValueError(
                    f'router 不支持跨 TP 分片重建: pp={pp_rank} ep={ep} owners={owners}'
                )
            tp = owners[0]
            d = self._load_sparse_ep_state(tp, pp_rank, ep, vpp_rank)
            if key in d:
                part = d[key]
            else:
                d_full = self._load_rank_state(tp, pp_rank, ep, vpp_rank)
                if key not in d_full:
                    raise ValueError(
                        f'找不到 router 权重: pp={pp_rank} ep={ep} key={key}')
                part = d_full[key]
                del d_full
            if part.shape[0] == self.num_experts:
                out = part.clone()
                del d
                gc.collect()
                break
            if part.shape[0] != num_local:
                raise ValueError(
                    f'router 分片形状异常: pp={pp_rank} ep={ep} key={key} shape={part.shape}'
                )
            out[ep * num_local:(ep + 1) * num_local] = part
            del d
            gc.collect()
        return out

    def _gather_tp_row(self,
                       models: dict[tuple[int, int], dict[str, torch.Tensor]],
                       key: str,
                       ep_rank: int = 0) -> torch.Tensor:
        parts = [
            models[(tp_rank, ep_rank)].pop(key)
            for tp_rank in range(self.tp_size)
        ]
        return torch.cat(parts, dim=0)

    def _gather_tp_col(self,
                       models: dict[tuple[int, int], dict[str, torch.Tensor]],
                       key: str,
                       ep_rank: int = 0) -> torch.Tensor:
        parts = [
            models[(tp_rank, ep_rank)].pop(key)
            for tp_rank in range(self.tp_size)
        ]
        return torch.cat(parts, dim=1)

    def _set_preprocess(
            self, hf: dict[str, torch.Tensor],
            models: dict[tuple[int, int], dict[str, torch.Tensor]]) -> None:
        parts = [
            models[(tp_rank, 0)]['embedding.word_embeddings.weight']
            for tp_rank in range(self.tp_size)
        ]
        hf['model.embed_tokens.weight'] = torch.cat(parts, dim=0).clone()

    def _set_postprocess(
            self, hf: dict[str, torch.Tensor],
            models: dict[tuple[int, int], dict[str, torch.Tensor]]) -> None:
        hf['model.norm.weight'] = models[(
            0, 0)]['decoder.final_layernorm.weight'].clone()
        parts = [
            models[(tp_rank, 0)]['output_layer.weight']
            for tp_rank in range(self.tp_size)
        ]
        hf['lm_head.weight'] = torch.cat(parts, dim=0).clone()

    def _set_layer_norm(self, hf: dict[str, torch.Tensor],
                        models: dict[tuple[int, int], dict[str, torch.Tensor]],
                        hf_layer: int, local_idx: int) -> None:
        in_key = f'decoder.layers.{local_idx}.input_layernorm.weight'
        mlp_key = f'decoder.layers.{local_idx}.pre_mlp_layernorm.weight'
        hf[f'model.layers.{hf_layer}.input_layernorm.weight'] = models[(
            0, 0)].pop(in_key).clone()
        hf[f'model.layers.{hf_layer}.post_attention_layernorm.weight'] = models[
            (0, 0)].pop(mlp_key).clone()

    def _set_layer_attn(self, hf: dict[str, torch.Tensor],
                        models: dict[tuple[int, int], dict[str, torch.Tensor]],
                        hf_layer: int, local_idx: int) -> None:
        prefix = f'decoder.layers.{local_idx}.self_attention'
        qkv_key = f'{prefix}.linear_qkv.weight'
        proj_key = f'{prefix}.linear_proj.weight'
        q_norm_key = f'{prefix}.q_layernorm.weight'
        kv_norm_key = f'{prefix}.kv_layernorm.weight'
        q_up_key = f'{prefix}.linear_q_up_proj.weight'
        kv_up_key = f'{prefix}.linear_kv_up_proj.weight'

        if kv_norm_key not in models[(0, 0)]:
            k_norm_key = f'{prefix}.k_layernorm.weight'
            linear_proj_list: list[torch.Tensor] = []
            q_parts: list[torch.Tensor] = []
            k_parts: list[torch.Tensor] = []
            v_parts: list[torch.Tensor] = []

            head_dim = self.hidden_size // self.num_attention_heads
            if self.num_attention_heads % self.tp_size != 0:
                raise ValueError(
                    f'num_attention_heads={self.num_attention_heads} 不能整除 tp_size={self.tp_size}'
                )
            q_per_tp = (self.num_attention_heads // self.tp_size) * head_dim

            for tp_rank in range(self.tp_size):
                linear_proj_list.append(models[(tp_rank,
                                                0)].pop(proj_key).clone())
                qkv_shard = models[(tp_rank, 0)].pop(qkv_key)
                rem = qkv_shard.shape[0] - q_per_tp
                if rem < 0 or rem % 2 != 0:
                    raise ValueError(
                        f'{qkv_key} 分片形状异常: {qkv_shard.shape}, q_per_tp={q_per_tp}'
                    )
                kv_per_tp = rem // 2
                q_r, k_r, v_r = torch.split(qkv_shard,
                                            [q_per_tp, kv_per_tp, kv_per_tp],
                                            dim=0)
                q_parts.append(q_r.clone())
                k_parts.append(k_r.clone())
                v_parts.append(v_r.clone())

            o_proj = torch.cat(linear_proj_list, dim=1)
            hf[f'model.layers.{hf_layer}.self_attn.q_proj.weight'] = torch.cat(
                q_parts, dim=0).clone()
            hf[f'model.layers.{hf_layer}.self_attn.k_proj.weight'] = torch.cat(
                k_parts, dim=0).clone()
            hf[f'model.layers.{hf_layer}.self_attn.v_proj.weight'] = torch.cat(
                v_parts, dim=0).clone()
            hf[f'model.layers.{hf_layer}.self_attn.o_proj.weight'] = o_proj.clone(
            )
            hf[f'model.layers.{hf_layer}.self_attn.q_layernorm.weight'] = models[
                (0, 0)].pop(q_norm_key).clone()
            hf[f'model.layers.{hf_layer}.self_attn.k_layernorm.weight'] = models[
                (0, 0)].pop(k_norm_key).clone()
            hf[f'model.layers.{hf_layer}.self_attn.rotary_emb.inv_freq'] = self.inv_freq.clone(
            )
            return

        linear_proj_list: list[torch.Tensor] = []
        linear_qb_list: list[torch.Tensor] = []
        linear_kvb_list: list[torch.Tensor] = []
        qk_nope_list: list[torch.Tensor] = []
        qk_rope_list: list[torch.Tensor] = []
        kv_nope_list: list[torch.Tensor] = []
        linear_v_list: list[torch.Tensor] = []

        for tp_rank in range(self.tp_size):
            linear_proj_list.append(models[(tp_rank, 0)].pop(proj_key).clone())
            if self.mla_mm_split:
                qk_nope_list.append(
                    models[(tp_rank,
                            0)].pop(f'{prefix}.linear_qk_nope.weight'))
                qk_rope_list.append(
                    models[(tp_rank,
                            0)].pop(f'{prefix}.linear_qk_rope.weight'))
                kv_nope_list.append(
                    models[(tp_rank,
                            0)].pop(f'{prefix}.linear_kv_nope.weight'))
                linear_v_list.append(
                    models[(tp_rank, 0)].pop(f'{prefix}.linear_v.weight'))
            else:
                linear_qb_list.append(models[(tp_rank,
                                              0)].pop(q_up_key).clone())
                linear_kvb_list.append(models[(tp_rank,
                                               0)].pop(kv_up_key).clone())

        o_proj = torch.cat(linear_proj_list, dim=1)

        if self.mla_mm_split:
            qk_nope_weight = torch.cat(qk_nope_list,
                                       dim=0).reshape(self.num_attention_heads,
                                                      self.qk_head_dim, -1)
            qk_rope_weight = torch.cat(qk_rope_list,
                                       dim=0).reshape(self.num_attention_heads,
                                                      self.qk_pos_emb_head_dim,
                                                      -1)
            kv_nope_weight = torch.cat(kv_nope_list,
                                       dim=0).reshape(self.num_attention_heads,
                                                      self.qk_head_dim, -1)
            linear_v_weight = torch.cat(linear_v_list, dim=0).reshape(
                self.num_attention_heads, self.v_head_dim, -1)
            q_b_proj = torch.cat(
                [qk_nope_weight, qk_rope_weight], dim=1).reshape(
                    self.num_attention_heads *
                    (self.qk_head_dim + self.qk_pos_emb_head_dim), -1)
            kv_b_proj = torch.cat([kv_nope_weight, linear_v_weight],
                                  dim=1).reshape(
                                      self.num_attention_heads *
                                      (self.qk_head_dim + self.v_head_dim), -1)
        else:
            q_b_proj = torch.cat(linear_qb_list, dim=0)
            kv_b_proj = torch.cat(linear_kvb_list, dim=0)

        qkv = models[(0, 0)].pop(qkv_key)
        if qkv.shape[0] < self.q_lora_rank:
            raise ValueError(
                f'linear_qkv.weight 行数 {qkv.shape[0]} 小于 q_lora_rank={self.q_lora_rank}'
            )
        q_a_proj = qkv[:self.q_lora_rank, :].clone()
        kv_a_proj = qkv[self.q_lora_rank:, :].clone()
        q_a_ln = models[(0, 0)].pop(q_norm_key)
        kv_a_ln = models[(0, 0)].pop(kv_norm_key)

        hf[f'model.layers.{hf_layer}.self_attn.q_a_proj.weight'] = q_a_proj
        hf[f'model.layers.{hf_layer}.self_attn.kv_a_proj_with_mqa.weight'] = kv_a_proj
        hf[f'model.layers.{hf_layer}.self_attn.o_proj.weight'] = o_proj
        hf[f'model.layers.{hf_layer}.self_attn.q_a_layernorm.weight'] = q_a_ln
        hf[f'model.layers.{hf_layer}.self_attn.kv_a_layernorm.weight'] = kv_a_ln
        hf[f'model.layers.{hf_layer}.self_attn.q_b_proj.weight'] = q_b_proj
        hf[f'model.layers.{hf_layer}.self_attn.kv_b_proj.weight'] = kv_b_proj
        hf[f'model.layers.{hf_layer}.self_attn.rotary_emb.inv_freq'] = self.inv_freq.clone(
        )

    def _reconstruct_router(self, models: dict[tuple[int, int],
                                               dict[str, torch.Tensor]],
                            key: str) -> torch.Tensor:
        t = models.get((0, 0), {}).get(key)
        if t is not None and t.shape[0] == self.num_experts:
            return models[(0, 0)].pop(key).clone()

        for _, v in models.items():
            if key in v:
                if v[key].shape[0] == self.num_experts:
                    return v.pop(key).clone()

        sample = None
        for _, v in models.items():
            if key in v:
                sample = v[key]
                break

        if sample is None:
            raise ValueError(
                f'Router weight {key} not found in any loaded model')

        out = torch.empty((self.num_experts, ) + sample.shape[1:],
                          dtype=sample.dtype)

        num_local = self.num_experts // self.ep_size
        for ep_rank in range(self.ep_size):
            target_tp = -1
            for tp in range(self.tp_size):
                if (tp, ep_rank) in models and key in models[(tp, ep_rank)]:
                    target_tp = tp
                    break

            if target_tp != -1:
                part = models[(target_tp, ep_rank)].pop(key)
                if part.shape[0] == num_local:
                    out[ep_rank * num_local:(ep_rank + 1) * num_local] = part
                else:
                    raise ValueError(
                        f'{key} rank {ep_rank} shape mismatch: {part.shape}')

        return out

    def _set_layer_mlp(self, hf: dict[str, torch.Tensor],
                       models: dict[tuple[int, int], dict[str, torch.Tensor]],
                       hf_layer: int, local_idx: int) -> None:
        prefix = f'decoder.layers.{local_idx}.mlp'
        if hf_layer < self.first_k_dense_replace:
            fc1 = self._gather_tp_row(models, f'{prefix}.linear_fc1.weight')
            fc2 = self._gather_tp_col(models, f'{prefix}.linear_fc2.weight')
            gate, up = torch.chunk(fc1, 2, dim=0)
            hf[f'model.layers.{hf_layer}.mlp.gate_proj.weight'] = gate.clone()
            hf[f'model.layers.{hf_layer}.mlp.up_proj.weight'] = up.clone()
            hf[f'model.layers.{hf_layer}.mlp.down_proj.weight'] = fc2.clone()
            return

        pp_rank = self.layer2loc[hf_layer][0] if self.vpp_size is None else self.layer2loc_vpp[
            hf_layer][0]
        vpp_rank = None if self.vpp_size is None else self.layer2loc_vpp[
            hf_layer][1]
        router = self._reconstruct_router_lazy(models, pp_rank, vpp_rank,
                                               f'{prefix}.router.weight')
        router_bias = self._reconstruct_router_lazy(
            models, pp_rank, vpp_rank, f'{prefix}.router.expert_bias')

        hf[f'model.layers.{hf_layer}.mlp.gate.weight'] = router.clone()
        hf[f'model.layers.{hf_layer}.mlp.gate.e_score_correction_bias'] = router_bias.clone(
        )

        shared_fc1 = self._gather_tp_row(
            models, f'{prefix}.shared_experts.linear_fc1.weight')
        shared_fc2 = self._gather_tp_col(
            models, f'{prefix}.shared_experts.linear_fc2.weight')
        shared_gate, shared_up = torch.chunk(shared_fc1, 2, dim=0)
        hf[f'model.layers.{hf_layer}.mlp.shared_experts.gate_proj.weight'] = shared_gate.clone(
        )
        hf[f'model.layers.{hf_layer}.mlp.shared_experts.up_proj.weight'] = shared_up.clone(
        )
        hf[f'model.layers.{hf_layer}.mlp.shared_experts.down_proj.weight'] = shared_fc2.clone(
        )

        import gc

        if self.moe_grouped_gemm:
            w1_key = f'{prefix}.experts.weight1'
            w2_key = f'{prefix}.experts.weight2'
            if self.moe_tp_extend_ep:
                raise ValueError('moe_tp_extend_ep 暂不支持稀疏专家目录格式')
            else:
                num_local = self.num_experts // self.ep_size
                for ep_rank in range(self.ep_size):
                    owners = self._tp_ranks_for_ep(pp_rank, ep_rank)
                    if not owners:
                        raise ValueError(
                            f'找不到 ep={ep_rank} 的权重目录: pp={pp_rank}')
                    shards_w1: list[torch.Tensor] = []
                    shards_w2: list[torch.Tensor] = []
                    for tp_rank in owners:
                        d = self._load_sparse_ep_state(tp_rank, pp_rank,
                                                       ep_rank, vpp_rank)
                        if w1_key not in d or w2_key not in d:
                            raise ValueError(
                                f'找不到 moe 权重: pp={pp_rank} ep={ep_rank} tp={tp_rank}'
                            )
                        shards_w1.append(d[w1_key])
                        shards_w2.append(d[w2_key])
                        del d
                        gc.collect()

                    local_w1 = shards_w1[0] if len(
                        shards_w1) == 1 else torch.cat(shards_w1, dim=1)
                    local_w2 = shards_w2[0] if len(
                        shards_w2) == 1 else torch.cat(shards_w2, dim=0)
                    w1_3d = local_w1.view(self.hidden_size, num_local,
                                          -1).permute(1, 0, 2).contiguous()
                    w2_3d = local_w2.view(num_local, -1, self.hidden_size)
                    for li in range(num_local):
                        expert = ep_rank * num_local + li
                        fc1 = w1_3d[li].t()
                        gate, up = torch.chunk(fc1, 2, dim=0)
                        down = w2_3d[li].t()
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'] = gate.clone(
                        )
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight'] = up.clone(
                        )
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'] = down.clone(
                        )
        else:
            num_local = self.num_experts // self.ep_size
            for ep_rank in range(self.ep_size):
                owners = self._tp_ranks_for_ep(pp_rank, ep_rank)
                if not owners:
                    raise ValueError(
                        f'找不到 ep={ep_rank} 的权重目录: pp={pp_rank}')
                local_states: dict[int, dict[str, torch.Tensor]] = {}
                for tp_rank in owners:
                    local_states[tp_rank] = self._load_sparse_ep_state(
                        tp_rank, pp_rank, ep_rank, vpp_rank)

                for li in range(num_local):
                    expert = ep_rank * num_local + li
                    local_prefix = f'{prefix}.experts.local_experts.{li}'
                    fc1_key = f'{local_prefix}.linear_fc1.weight'
                    fc2_key = f'{local_prefix}.linear_fc2.weight'

                    if len(owners) == 1:
                        tp_rank = owners[0]
                        fc1 = local_states[tp_rank].pop(fc1_key)
                        fc2 = local_states[tp_rank].pop(fc2_key)
                        gate, up = torch.chunk(fc1, 2, dim=0)
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'] = gate.clone(
                        )
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight'] = up.clone(
                        )
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'] = fc2.clone(
                        )
                    else:
                        fc1_parts = [
                            local_states[tp].pop(fc1_key) for tp in owners
                        ]
                        fc2_parts = [
                            local_states[tp].pop(fc2_key) for tp in owners
                        ]
                        fc1 = torch.cat(fc1_parts, dim=0)
                        fc2 = torch.cat(fc2_parts, dim=1)
                        gate, up = torch.chunk(fc1, 2, dim=0)
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.gate_proj.weight'] = gate.clone(
                        )
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.up_proj.weight'] = up.clone(
                        )
                        hf[f'model.layers.{hf_layer}.mlp.experts.{expert}.down_proj.weight'] = fc2.clone(
                        )

                for _, st in local_states.items():
                    del st
                gc.collect()

    def _save_shard(self, tensors: dict[str, torch.Tensor], shard_idx: int,
                    total_shards: int) -> None:
        name = f'model-{shard_idx:05d}-of-{total_shards:06d}.safetensors'
        path = os.path.join(self.hf_save_dir, name)
        for k in tensors.keys():
            self.weight_map[k] = name
        packed: dict[str, torch.Tensor] = {}
        for k, v in tensors.items():
            if not v.is_contiguous():
                packed[k] = v.contiguous()
            else:
                packed[k] = v
        safetensors.torch.save_file(packed, path, metadata={'format': 'pt'})
        if self.verbose:
            logger.info('Saved shard %d/%d: %s (tensors=%d)', shard_idx,
                        total_shards, path, len(tensors))

    def run(self) -> None:
        total_shards = self.num_real_layers + 2

        if self.verbose:
            logger.info(
                'Start converting: layers=%d real_layers=%d tp=%d pp=%d ep=%d vpp=%s dualpipe=%s save_dir=%s',
                self.num_layers, self.num_real_layers, self.tp_size,
                self.pp_size, self.ep_size, self.vpp_size, self.dualpipe,
                self.hf_save_dir)

        import gc

        if self.vpp_size is None:
            for pp_rank in range(self.pp_size):
                models = self._load_models_for_stage(pp_rank=pp_rank,
                                                     vpp_rank=None)

                if pp_rank == 0:
                    base_tensors: dict[str, torch.Tensor] = {}
                    self._set_preprocess(base_tensors, models)
                    self._save_shard(base_tensors, 1, total_shards)

                layers = self.pprank_layer_idxs[pp_rank]
                for hf_layer in layers:
                    local_idx = self.layer2loc[hf_layer][1]
                    if self.verbose:
                        logger.info(
                            'Converting layer %d/%d: pp_rank=%d local_idx=%d',
                            hf_layer, self.num_real_layers - 1, pp_rank,
                            local_idx)

                    layer_tensors: dict[str, torch.Tensor] = {}
                    self._set_layer_norm(layer_tensors, models, hf_layer,
                                         local_idx)
                    self._set_layer_attn(layer_tensors, models, hf_layer,
                                         local_idx)
                    self._set_layer_mlp(layer_tensors, models, hf_layer,
                                        local_idx)
                    self._save_shard(layer_tensors, hf_layer + 2, total_shards)

                if pp_rank == self.pp_size - 1:
                    tail_tensors: dict[str, torch.Tensor] = {}
                    self._set_postprocess(tail_tensors, models)
                    self._save_shard(tail_tensors, total_shards, total_shards)

                del models
                gc.collect()

        else:
            for pp_rank in range(self.pp_size):
                for vpp_rank in range(self.vpp_size):
                    models = self._load_models_for_stage(pp_rank=pp_rank,
                                                         vpp_rank=vpp_rank)

                    if pp_rank == 0 and vpp_rank == 0:
                        base_tensors: dict[str, torch.Tensor] = {}
                        self._set_preprocess(base_tensors, models)
                        self._save_shard(base_tensors, 1, total_shards)

                    if pp_rank in self.vpprank_layer_idxs and vpp_rank in self.vpprank_layer_idxs[
                            pp_rank]:
                        layers = self.vpprank_layer_idxs[pp_rank][vpp_rank]
                        for hf_layer in layers:
                            local_idx = self.layer2loc_vpp[hf_layer][2]
                            if self.verbose:
                                logger.info(
                                    'Converting layer %d/%d: pp_rank=%d vpp_rank=%d local_idx=%d',
                                    hf_layer, self.num_real_layers - 1,
                                    pp_rank, vpp_rank, local_idx)
                            layer_tensors: dict[str, torch.Tensor] = {}
                            self._set_layer_norm(layer_tensors, models,
                                                 hf_layer, local_idx)
                            self._set_layer_attn(layer_tensors, models,
                                                 hf_layer, local_idx)
                            self._set_layer_mlp(layer_tensors, models,
                                                hf_layer, local_idx)
                            self._save_shard(layer_tensors, hf_layer + 2,
                                             total_shards)

                    is_post = False
                    if self.dualpipe:
                        if pp_rank == 0 and vpp_rank == self.vpp_size - 1:
                            is_post = True
                    else:
                        if pp_rank == self.pp_size - 1 and vpp_rank == self.vpp_size - 1:
                            is_post = True

                    if is_post:
                        tail_tensors: dict[str, torch.Tensor] = {}
                        self._set_postprocess(tail_tensors, models)
                        self._save_shard(tail_tensors, total_shards,
                                         total_shards)

                    del models
                    gc.collect()

        index_path = os.path.join(self.hf_save_dir,
                                  'model.safetensors.index.json')
        with open(index_path, 'w') as f:
            json.dump({'metadata': {}, 'weight_map': self.weight_map}, f)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load-dir',
                        type=str,
                        required=True,
                        help='Directory to load model checkpoint from')
    parser.add_argument('--save-dir',
                        type=str,
                        required=True,
                        help='Directory to save model checkpoint to')
    parser.add_argument(
        '--source-tensor-parallel-size',
        type=int,
        default=1,
        help='Source tensor model parallel size, defaults to 1')
    parser.add_argument(
        '--source-pipeline-parallel-size',
        type=int,
        default=1,
        help='Source pipeline model parallel size, default to 1')
    parser.add_argument('--source-expert-parallel-size',
                        type=int,
                        default=1,
                        help='Source expert model parallel size, default to 1')
    parser.add_argument('--num-layers-per-virtual-pipeline-stage',
                        '--vpp-stage',
                        dest='num_layers_per_virtual_pipeline_stage',
                        type=int,
                        default=None,
                        help='Number of layers per virtual pipeline stage')
    parser.add_argument('--moe-grouped-gemm',
                        action='store_true',
                        help='Use moe grouped gemm.')
    parser.add_argument('--noop-layers',
                        type=str,
                        default='47',
                        help='Specity the noop layers.')
    parser.add_argument('--mtp-num-layers',
                        type=int,
                        default=0,
                        help='Multi-Token prediction layer num')
    parser.add_argument(
        '--num-layer-list',
        type=str,
        help='a list of number of layers, separated by comma; e.g., 4,4,4,4')
    parser.add_argument(
        '--moe-tp-extend-ep',
        action='store_true',
        help=
        'use tp group to extend experts parallism instead of sharding weight tensor of experts in tp group'
    )
    parser.add_argument('--mla-mm-split',
                        action='store_true',
                        default=False,
                        help='Split 2 up-proj matmul into 4 in MLA')
    parser.add_argument(
        '--schedules-method',
        type=str,
        default=None,
        choices=['dualpipev'],
        help='An innovative bidirectional pipeline parallelism algorithm.')
    parser.add_argument('--num-layers',
                        type=int,
                        default=48,
                        help='Number of transformer layers.')
    parser.add_argument('--first-k-dense-replace',
                        type=int,
                        default=3,
                        help='Customizing the number of dense layers.')
    parser.add_argument('--rotary-base',
                        type=float,
                        default=50000.0,
                        help='Rotary base for RoPE')
    parser.add_argument('--hidden-size',
                        type=int,
                        default=None,
                        help='Override hidden size.')
    parser.add_argument('--num-experts',
                        type=int,
                        default=None,
                        help='Override num experts.')
    parser.add_argument('--num-attention-heads',
                        type=int,
                        default=None,
                        help='Override attention heads.')
    parser.add_argument('--qk-head-dim',
                        type=int,
                        default=None,
                        help='Override qk head dim (MLA).')
    parser.add_argument('--v-head-dim',
                        type=int,
                        default=None,
                        help='Override v head dim (MLA).')
    parser.add_argument('--qk-pos-emb-head-dim',
                        type=int,
                        default=None,
                        help='Override qk pos emb head dim (MLA).')
    parser.add_argument('--q-lora-rank',
                        type=int,
                        default=Q_LORA_RANK,
                        help='q LoRA rank used by MLA.')

    args = parser.parse_args()
    return args


def main() -> None:
    args = get_args()
    logger.info('Arguments: %s', args)
    hidden_size = args.hidden_size or HIDDEN_SIZE
    num_experts = args.num_experts or NUM_EXPERTS
    num_attention_heads = args.num_attention_heads or NUM_ATTENTION_HEADS
    qk_head_dim = args.qk_head_dim or QK_HEAD_DIM
    v_head_dim = args.v_head_dim or V_HEAD_DIM
    qk_pos_emb_head_dim = args.qk_pos_emb_head_dim or QK_POS_EMB_HEAD_DIM
    converter = MgCkptConvert(
        mg_load_dir=args.load_dir,
        hf_save_dir=args.save_dir,
        num_layers=args.num_layers,
        tp_size=args.source_tensor_parallel_size,
        pp_size=args.source_pipeline_parallel_size,
        ep_size=args.source_expert_parallel_size,
        first_k_dense_replace=args.first_k_dense_replace,
        hidden_size=hidden_size,
        num_experts=num_experts,
        num_attention_heads=num_attention_heads,
        qk_head_dim=qk_head_dim,
        v_head_dim=v_head_dim,
        qk_pos_emb_head_dim=qk_pos_emb_head_dim,
        moe_grouped_gemm=args.moe_grouped_gemm,
        moe_tp_extend_ep=args.moe_tp_extend_ep,
        mla_mm_split=args.mla_mm_split,
        schedules_method=args.schedules_method,
        vpp_stage=args.num_layers_per_virtual_pipeline_stage,
        num_layer_list=args.num_layer_list,
        noop_layers=args.noop_layers,
        rotary_base=args.rotary_base,
        q_lora_rank=args.q_lora_rank,
    )
    converter.run()


if __name__ == '__main__':
    main()
