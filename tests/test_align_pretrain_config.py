import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

import torch


REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
UTILS_DIR = os.path.join(REPO_ROOT, 'utils')
if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)


def _sha256_tensor(t: torch.Tensor) -> str:
    t = t.detach().cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    es = int(t.element_size())
    vd = {1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[es]
    b = t.view(vd).numpy().tobytes()
    return hashlib.sha256(b).hexdigest()


def _dualpipe_layers(num_layers: int, pp_size: int) -> list[int]:
    layers_each_pp = num_layers // pp_size
    layer_pop_num = layers_each_pp // 2
    all_layers = list(range(num_layers))
    out: list[int] = []
    while all_layers:
        out.extend(all_layers[:layer_pop_num])
        out.extend(all_layers[-layer_pop_num:])
        all_layers = all_layers[layer_pop_num:-layer_pop_num]
    return out


def _stage_layers_dualpipe(num_layers: int, pp_size: int,
                           vpp_stage: int) -> dict[tuple[int, int], list[int]]:
    dualpipe_layers = _dualpipe_layers(num_layers, pp_size)
    out: dict[tuple[int, int], list[int]] = {}
    pp_rank = 0
    vpp_rank = 0
    each_pp_layer = num_layers // pp_size
    for idx, layer in enumerate(dualpipe_layers):
        out.setdefault((pp_rank, vpp_rank), []).append(layer)
        if (idx + 1) % vpp_stage == 0:
            vpp_rank += 1
        if (idx + 1) % each_pp_layer == 0:
            pp_rank += 1
            vpp_rank = 0
    return out


def _write_rank_ckpt(base_dir: str, tp_rank: int, pp_rank: int, ep_rank: int,
                     model0: dict[str, torch.Tensor],
                     model1: dict[str, torch.Tensor]) -> None:
    iter_dir = os.path.join(base_dir, 'iter_0000001')
    os.makedirs(iter_dir, exist_ok=True)
    mp_dir = os.path.join(iter_dir,
                          f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}')
    os.makedirs(mp_dir, exist_ok=True)
    path = os.path.join(mp_dir, 'model_optim_rng.pt')
    payload = {
        'model0': model0,
        'model1': model1,
        'checkpoint_version': 3.0,
        'iteration': 1,
        'args': {
            'rotary_base': 50000.0
        },
    }
    torch.save(payload,
               path,
               pickle_protocol=4,
               _use_new_zipfile_serialization=True)


def _build_dummy_mcore_ckpt(base_dir: str) -> dict[str, int]:
    tp_size = 2
    pp_size = 2
    ep_size = 2
    num_layers = 4
    first_k_dense_replace = 2

    hidden_size = 16
    vocab_size = 32
    ffn_hidden = 32
    moe_ffn_hidden = 24
    num_experts = 4
    num_attention_heads = 4
    num_query_groups = 2
    qk_head_dim = 4
    qk_pos_emb_head_dim = 2
    v_head_dim = 4

    vpp_stage = (num_layers // pp_size) // 2
    stage_map = _stage_layers_dualpipe(num_layers, pp_size, vpp_stage)

    g = torch.Generator()
    g.manual_seed(1234)

    def rand(shape: tuple[int, ...]) -> torch.Tensor:
        return torch.randn(shape, generator=g, dtype=torch.bfloat16)

    def attn_qkv_rows() -> int:
        heads_per_tp = num_attention_heads // tp_size
        q_head_dim = qk_head_dim + qk_pos_emb_head_dim
        kv_heads = num_query_groups
        kv_heads_per_tp = kv_heads // tp_size
        return heads_per_tp * q_head_dim + kv_heads_per_tp * (q_head_dim +
                                                              v_head_dim)

    qkv_rows = attn_qkv_rows()

    for pp_rank in range(pp_size):
        layers_vpp0 = stage_map[(pp_rank, 0)]
        layers_vpp1 = stage_map[(pp_rank, 1)]
        if len(layers_vpp0) != 1 or len(layers_vpp1) != 1:
            raise RuntimeError('dummy config expects 1 layer per stage')
        layer0 = layers_vpp0[0]
        layer1 = layers_vpp1[0]

        stage_shared: dict[tuple[int, int], dict[str, torch.Tensor]] = {}
        for vpp_rank, hf_layer in [(0, layer0), (1, layer1)]:
            stage_shared[(vpp_rank, hf_layer)] = {
                'input_ln': rand((hidden_size, )),
                'pre_mlp_ln': rand((hidden_size, )),
                'q_ln': rand((qk_head_dim + qk_pos_emb_head_dim, )),
                'k_ln': rand((qk_head_dim + qk_pos_emb_head_dim, )),
            }
        final_ln_shared = rand((hidden_size, ))

        for ep_rank in range(ep_size):
            tp_rank = ep_rank % tp_size

            model0: dict[str, torch.Tensor] = {}
            model1: dict[str, torch.Tensor] = {}

            if pp_rank == 0:
                model0['embedding.word_embeddings.weight'] = rand(
                    (vocab_size // tp_size, hidden_size))
                model1['decoder.final_layernorm.weight'] = final_ln_shared
                model1['output_layer.weight'] = rand(
                    (vocab_size // tp_size, hidden_size))

            for vpp_rank, hf_layer in [(0, layer0), (1, layer1)]:
                local_idx = 0
                dst = model0 if vpp_rank == 0 else model1
                prefix = f'decoder.layers.{local_idx}'
                sh = stage_shared[(vpp_rank, hf_layer)]

                dst[f'{prefix}.input_layernorm.weight'] = sh['input_ln']
                dst[f'{prefix}.pre_mlp_layernorm.weight'] = sh['pre_mlp_ln']
                dst[f'{prefix}.self_attention.linear_qkv.weight'] = rand(
                    (qkv_rows, hidden_size))
                dst[f'{prefix}.self_attention.linear_proj.weight'] = rand(
                    (hidden_size, hidden_size // tp_size))
                dst[f'{prefix}.self_attention.q_layernorm.weight'] = sh['q_ln']
                dst[f'{prefix}.self_attention.k_layernorm.weight'] = sh['k_ln']

                mlp_prefix = f'{prefix}.mlp'
                if hf_layer < first_k_dense_replace:
                    dst[f'{mlp_prefix}.linear_fc1.weight'] = rand(
                        (2 * ffn_hidden // tp_size, hidden_size))
                    dst[f'{mlp_prefix}.linear_fc2.weight'] = rand(
                        (hidden_size, ffn_hidden // tp_size))
                else:
                    num_local = num_experts // ep_size
                    dst[f'{mlp_prefix}.router.weight'] = rand(
                        (num_local, hidden_size))
                    dst[f'{mlp_prefix}.router.expert_bias'] = rand((num_local, ))
                    dst[f'{mlp_prefix}.shared_experts.linear_fc1.weight'] = rand(
                        (2 * moe_ffn_hidden // tp_size, hidden_size))
                    dst[f'{mlp_prefix}.shared_experts.linear_fc2.weight'] = rand(
                        (hidden_size, moe_ffn_hidden // tp_size))

                    dst[f'{mlp_prefix}.experts.weight1'] = rand(
                        (hidden_size, num_local * 2 * moe_ffn_hidden))
                    dst[f'{mlp_prefix}.experts.weight2'] = rand(
                        (num_local * moe_ffn_hidden, hidden_size))

            _write_rank_ckpt(base_dir, tp_rank, pp_rank, ep_rank, model0, model1)

    latest = os.path.join(base_dir, 'latest_checkpointed_iteration.txt')
    with open(latest, 'w') as f:
        f.write('1')

    return {
        'tp_size': tp_size,
        'pp_size': pp_size,
        'ep_size': ep_size,
        'num_layers': num_layers,
        'first_k_dense_replace': first_k_dense_replace,
        'hidden_size': hidden_size,
        'vocab_size': vocab_size,
        'ffn_hidden': ffn_hidden,
        'moe_ffn_hidden': moe_ffn_hidden,
        'num_experts': num_experts,
        'num_attention_heads': num_attention_heads,
        'num_query_groups': num_query_groups,
        'qk_head_dim': qk_head_dim,
        'qk_pos_emb_head_dim': qk_pos_emb_head_dim,
        'v_head_dim': v_head_dim,
        'rotary_base': 50000.0,
    }


class TestAlignPretrainConfig(unittest.TestCase):

    def test_pretrain_script_matches_config_1t(self) -> None:
        script = os.path.join(REPO_ROOT, 'scripts', 'pretrain_kimi2_1t_4k.sh')
        cfg_path = os.path.join(REPO_ROOT, 'models', 'config_1t.json')

        import pretrain_config

        cfg = pretrain_config.parse_pretrain_script(script)
        with open(cfg_path) as f:
            hf_cfg = json.load(f)

        self.assertEqual(int(pretrain_config.get_int(cfg, '--num-layers')),
                         int(hf_cfg['num_hidden_layers']))
        self.assertEqual(int(pretrain_config.get_int(cfg, '--hidden-size')),
                         int(hf_cfg['hidden_size']))
        self.assertEqual(
            int(pretrain_config.get_int(cfg, '--num-attention-heads')),
            int(hf_cfg['num_attention_heads']))
        self.assertEqual(int(pretrain_config.get_int(cfg, '--vocab-size')),
                         int(hf_cfg['vocab_size']))
        self.assertEqual(float(pretrain_config.get_float(cfg, '--rotary-base')),
                         float(hf_cfg['rope_theta']))
        self.assertTrue(pretrain_config.get_bool(cfg, '--use-flash-attn'))
        self.assertTrue(pretrain_config.get_bool(cfg, '--bf16'))
        self.assertTrue(pretrain_config.get_bool(cfg, '--moe-grouped-gemm'))
        self.assertEqual(pretrain_config.get_flag(cfg, '--schedules-method'),
                         'dualpipev')

        self.assertFalse(bool(hf_cfg['tie_word_embeddings']))
        rope = hf_cfg['rope_scaling']
        self.assertEqual(str(rope['type']), 'yarn')
        self.assertEqual(float(rope['factor']), 32.0)
        self.assertEqual(int(rope['original_max_position_embeddings']), 4096)

    def test_dummy_roundtrip_mcore_hf_mcore(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            mcore_in = os.path.join(td, 'mcore_in')
            hf_mid = os.path.join(td, 'hf_mid')
            mcore_out = os.path.join(td, 'mcore_out')
            os.makedirs(mcore_in, exist_ok=True)

            p = _build_dummy_mcore_ckpt(mcore_in)

            cmd_m2h = [
                sys.executable,
                os.path.join(REPO_ROOT, 'utils', 'convert_ckpt_mcore2hf.py'),
                '--load-dir',
                mcore_in,
                '--save-dir',
                hf_mid,
                '--num-layers',
                str(p['num_layers']),
                '--first-k-dense-replace',
                str(p['first_k_dense_replace']),
                '--source-tensor-parallel-size',
                str(p['tp_size']),
                '--source-pipeline-parallel-size',
                str(p['pp_size']),
                '--source-expert-parallel-size',
                str(p['ep_size']),
                '--moe-grouped-gemm',
                '--moe-tp-extend-ep',
                '--schedules-method',
                'dualpipev',
                '--hidden-size',
                str(p['hidden_size']),
                '--ffn-hidden-size',
                str(p['ffn_hidden']),
                '--moe-ffn-hidden-size',
                str(p['moe_ffn_hidden']),
                '--vocab-size',
                str(p['vocab_size']),
                '--num-experts',
                str(p['num_experts']),
                '--num-attention-heads',
                str(p['num_attention_heads']),
                '--num-key-value-heads',
                str(p['num_attention_heads'] // p['num_query_groups']),
                '--qk-head-dim',
                str(p['qk_head_dim']),
                '--v-head-dim',
                str(p['v_head_dim']),
                '--qk-pos-emb-head-dim',
                str(p['qk_pos_emb_head_dim']),
                '--rotary-base',
                str(p['rotary_base']),
                '--pp-workers',
                '1',
                '--io-threads',
                '2',
            ]
            subprocess.check_call(cmd_m2h, cwd=REPO_ROOT)

            self.assertTrue(os.path.isfile(os.path.join(hf_mid, 'config.json')))
            self.assertTrue(
                os.path.isfile(os.path.join(hf_mid,
                                            'model.safetensors.index.json')))

            cmd_h2m = [
                sys.executable,
                os.path.join(REPO_ROOT, 'utils', 'convert_ckpt_hf2mcore.py'),
                '--load-dir',
                hf_mid,
                '--save-dir',
                mcore_out,
                '--num-layers',
                str(p['num_layers']),
                '--first-k-dense-replace',
                str(p['first_k_dense_replace']),
                '--target-tensor-parallel-size',
                str(p['tp_size']),
                '--target-pipeline-parallel-size',
                str(p['pp_size']),
                '--target-expert-parallel-size',
                str(p['ep_size']),
                '--moe-grouped-gemm',
                '--moe-tp-extend-ep',
                '--schedules-method',
                'dualpipev',
                '--pp-workers',
                '1',
            ]
            subprocess.check_call(cmd_h2m, cwd=REPO_ROOT)

            iter_in = os.path.join(mcore_in, 'iter_0000001')
            iter_out = os.path.join(mcore_out, 'iter_0000001')
            self.assertTrue(os.path.isdir(iter_out))

            for pp_rank in range(p['pp_size']):
                for ep_rank in range(p['ep_size']):
                    tp_rank = ep_rank % p['tp_size']
                    mp = f'mp_rank_{tp_rank:02}_{pp_rank:03}_{ep_rank:03}'
                    pin = os.path.join(iter_in, mp, 'model_optim_rng.pt')
                    pout = os.path.join(iter_out, mp, 'model_optim_rng.pt')
                    self.assertTrue(os.path.isfile(pout))

                    sin = torch.load(pin, map_location='cpu', weights_only=False)
                    sout = torch.load(pout,
                                      map_location='cpu',
                                      weights_only=False)

                    for k in ['model0', 'model1']:
                        din = sin[k]
                        dout = sout[k]
                        self.assertEqual(set(din.keys()), set(dout.keys()))
                        for tk in din.keys():
                            self.assertEqual(tuple(din[tk].shape),
                                             tuple(dout[tk].shape))
                            self.assertEqual(din[tk].dtype, dout[tk].dtype)
                            hin = _sha256_tensor(din[tk])
                            hout = _sha256_tensor(dout[tk])
                            self.assertEqual(hin,
                                             hout,
                                             msg=f'{mp} {k} {tk}')


if __name__ == '__main__':
    unittest.main()
