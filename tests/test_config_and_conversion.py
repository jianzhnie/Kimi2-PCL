from unittest.mock import patch

import torch

from models.configuration_deepseek_1t import DeepseekV3Config
from utils.convert_ckpt_hf2mcore import CkptConvert
from utils.convert_ckpt_mcore2hf import MgCkptConvert


def test_config_default_q_head_dims():
    cfg = DeepseekV3Config()
    assert cfg.qk_nope_head_dim == 128
    assert cfg.qk_rope_head_dim == 64
    assert cfg.v_head_dim == 128
    assert cfg.num_attention_heads == 64


def _make_converter_small_mcore2hf():
    with patch('utils.convert_ckpt_mcore2hf._resolve_iter_dir',
               return_value='/tmp/unused'):
        with patch.object(MgCkptConvert,
                          '_detect_vpp',
                          return_value=(None, ['model'])):
            return MgCkptConvert(
                mg_load_dir='/tmp/unused',
                hf_save_dir='/tmp/unused',
                num_layers=1,
                tp_size=2,
                pp_size=1,
                ep_size=1,
                first_k_dense_replace=0,
                hidden_size=16,
                num_experts=1,
                num_attention_heads=4,
                qk_head_dim=8,
                v_head_dim=8,
                qk_pos_emb_head_dim=4,
                moe_grouped_gemm=False,
                moe_tp_extend_ep=False,
                schedules_method=None,
                vpp_stage=None,
                num_layer_list=None,
                noop_layers='',
                rotary_base=50000.0,
                num_key_value_heads=32,
                vocab_size=128,
                max_position_embeddings=1024,
                tie_word_embeddings=False,
                ffn_hidden_size=32,
                moe_ffn_hidden_size=16,
                hf_config_template=None,
                cast_dtype=None,
                io_threads=1,
                disable_mmap=True,
                extra_config_kwargs={},
            )


def test_infer_qkv_layout_small():
    conv = _make_converter_small_mcore2hf()
    inferred = conv._infer_qkv_layout(44)
    assert inferred == (12, 1)


def test_mcore2hf_set_layer_attn_shapes():
    conv = _make_converter_small_mcore2hf()
    hf = {}
    models = {}
    rows = 24 + 12 + 8
    hidden = 16
    for tp in range(conv.tp_size):
        models[(tp, 0)] = {
            'decoder.layers.0.self_attention.linear_qkv.weight':
            torch.zeros(rows, hidden),
            'decoder.layers.0.self_attention.linear_proj.weight':
            torch.zeros(hidden, 16),
        }
    conv._set_layer_attn(hf, models, hf_layer=0, local_idx=0)
    assert hf['model.layers.0.self_attn.q_proj.weight'].shape == (48, hidden)
    assert hf['model.layers.0.self_attn.k_proj.weight'].shape == (24, hidden)
    assert hf['model.layers.0.self_attn.v_proj.weight'].shape == (16, hidden)
    assert hf['model.layers.0.self_attn.o_proj.weight'].shape == (hidden, 32)


def _make_converter_small_hf2mcore():
    with patch('utils.convert_ckpt_hf2mcore.CkptConvert._validate'), \
         patch('utils.convert_ckpt_hf2mcore.CkptConvert._read_weight_map', return_value={}):
        return CkptConvert(
            hf_model_path='/tmp/unused',
            mg_save_path='/tmp/unused',
            num_layers=1,
            tp_size=2,
            pp_size=1,
            ep_size=1,
            first_k_dense_replace=0,
            hidden_size=16,
            ffn_hidden_size=32,
            moe_ffn_hidden_size=16,
            vocab_size=128,
            num_key_value_heads=32,
            num_experts=1,
            num_attention_heads=4,
            qk_head_dim=8,
            v_head_dim=8,
            qk_pos_emb_head_dim=4,
            moe_grouped_gemm=False,
            moe_tp_extend_ep=False,
            schedules_method=None,
            vpp_stage=None,
            num_layer_list=None,
            noop_layers='',
            qlora_nf4=False,
            rotary_base=50000.0,
            print_init_summary=False,
            pp_workers=1,
            cast_dtype=None,
            tie_word_embeddings=False,
            hf_io_threads=1,
            qk_layernorm=True,
        )


def test_hf2mcore_set_layer_attn_shapes():
    conv = _make_converter_small_hf2mcore()
    weights = {}
    mg_model = {0: {0: {}, 1: {}}}
    hidden = 16
    weights['model.layers.0.self_attn.q_proj.weight'] = torch.zeros(48, hidden)
    weights['model.layers.0.self_attn.k_proj.weight'] = torch.zeros(24, hidden)
    weights['model.layers.0.self_attn.v_proj.weight'] = torch.zeros(16, hidden)
    weights['model.layers.0.self_attn.o_proj.weight'] = torch.zeros(hidden, 32)
    weights['model.layers.0.self_attn.q_layernorm.weight'] = torch.zeros(12)
    weights['model.layers.0.self_attn.k_layernorm.weight'] = torch.zeros(12)
    conv._set_layer_attn(hf_layer=0,
                         local_layer_idx=0,
                         weights=weights,
                         mg_model=mg_model)
    qkv_0 = mg_model[0][0]['decoder.layers.0.self_attention.linear_qkv.weight']
    qkv_1 = mg_model[0][1]['decoder.layers.0.self_attention.linear_qkv.weight']
    proj_0 = mg_model[0][0][
        'decoder.layers.0.self_attention.linear_proj.weight']
    proj_1 = mg_model[0][1][
        'decoder.layers.0.self_attention.linear_proj.weight']
    assert qkv_0.shape == (44, hidden)
    assert qkv_1.shape == (44, hidden)
    assert proj_0.shape == (hidden, 16)
    assert proj_1.shape == (hidden, 16)
