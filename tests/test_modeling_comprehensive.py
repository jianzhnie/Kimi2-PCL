"""
Comprehensive unit tests for modeling_deepseek.py (GQA + MoE architecture)

Coverage targets:
- GQA Attention with QK-norm (RMSNorm, matching Megatron --qk-layernorm --normalization RMSNorm)
- MoE with training/eval modes, aux_loss, router bias/dtype
- YaRN RoPE at model level
- Full forward/backward for CausalLM
- Edge cases, boundary testing, parameterized testing
"""

import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from models.configuration_deepseek import DeepseekV3Config
from models.modeling_deepseek import (
    DeepseekV3Attention, DeepseekV3DecoderLayer, DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification, DeepseekV3RMSNorm, DeepseekV3MLP,
    DeepseekV3Model, DeepseekV3MoE, DeepseekV3PreTrainedModel,
    DeepseekV3RotaryEmbedding,
    DeepseekV3LinearScalingRotaryEmbedding,
    DeepseekV3DynamicNTKScalingRotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding, MoEGate, _get_unpad_data,
    apply_rotary_pos_emb, repeat_kv, rotate_half, yarn_find_correction_dim,
    yarn_find_correction_range, yarn_get_mscale, yarn_linear_ramp_mask)

# =============================================================================
# Fixtures
# =============================================================================


def _make_gqa_config(**overrides):
    """Minimal GQA config for fast tests.

    GQA: 4 attention heads, 2 query groups, head_dim=16
    MoE: 4 routed experts, top-2, 1 shared expert
    """
    defaults = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_query_groups=2,
        kv_channels=16,
        max_position_embeddings=256,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act='silu',
        n_routed_experts=4,
        num_experts_per_tok=2,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        n_shared_experts=1,
        ep_size=1,
        n_group=2,
        topk_group=1,
        scoring_func='sigmoid',
        topk_method='noaux_tc',
        routed_scaling_factor=1.0,
        norm_topk_prob=True,
        seq_aux=True,
        moe_aux_loss_coeff=0.01,
        moe_z_loss_coeff=0.0,
        moe_router_enable_expert_bias=False,
        moe_router_dtype='fp32',
        attention_bias=False,
        attention_dropout=0.0,
        qk_layernorm=True,
        pad_token_id=0,
        tie_word_embeddings=False,
    )
    defaults.update(overrides)
    config = DeepseekV3Config(**defaults)
    config._attn_implementation = 'eager'
    return config


@pytest.fixture
def base_config():
    return _make_gqa_config()


@pytest.fixture
def moe_config(base_config):
    base_config.n_routed_experts = 4
    base_config.n_shared_experts = 1
    base_config.num_experts_per_tok = 2
    base_config.moe_aux_loss_coeff = 0.01
    base_config.moe_z_loss_coeff = 0.01
    return base_config


@pytest.fixture
def yarn_config(base_config):
    base_config.rope_scaling = {
        'type': 'yarn',
        'factor': 4.0,
        'original_max_position_embeddings': 64,
        'beta_fast': 1.0,
        'beta_slow': 1.0,
        'mscale': 1.0,
        'mscale_all_dim': 1.0,
    }
    return base_config


# =============================================================================
# Normalization Tests
# =============================================================================


class TestRMSNorm:

    @pytest.mark.parametrize('hidden_size', [1, 16, 64, 512])
    def test_various_sizes(self, hidden_size):
        norm = DeepseekV3RMSNorm(hidden_size)
        x = torch.randn(2, 8, hidden_size)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    @pytest.mark.parametrize('eps', [1e-12, 1e-6, 1e-3])
    def test_eps_values(self, eps):
        norm = DeepseekV3RMSNorm(64, eps=eps)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_zero_input(self):
        norm = DeepseekV3RMSNorm(64)
        x = torch.zeros(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_dtype_preservation(self):
        norm = DeepseekV3RMSNorm(64)
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 8, 64, dtype=dtype)
            out = norm(x)
            assert out.dtype == dtype


class TestQKNorm:
    """Test q_layernorm/k_layernorm use DeepseekV3RMSNorm, matching Megatron."""

    @pytest.mark.parametrize('hidden_size', [1, 16, 128, 512])
    def test_various_sizes(self, hidden_size):
        norm = DeepseekV3RMSNorm(hidden_size)
        x = torch.randn(2, 8, hidden_size)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_rmsnorm_behavior(self):
        """QKNorm uses RMSNorm (no mean subtraction), matching Megatron."""
        norm = DeepseekV3RMSNorm(64)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        rms = out.to(torch.float32).pow(2).mean(-1).sqrt()
        assert rms.mean().item() > 0.1

    def test_weight_only(self):
        norm = DeepseekV3RMSNorm(128)
        assert hasattr(norm, 'weight')
        assert not hasattr(norm, 'bias') or norm.bias is None

    def test_dtype_preservation(self):
        norm = DeepseekV3RMSNorm(64)
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(2, 8, 64, dtype=dtype)
            out = norm(x)
            assert out.dtype == dtype


# =============================================================================
# Rotary Embedding Tests
# =============================================================================


class TestRotaryEmbedding:

    @pytest.mark.parametrize('dim', [4, 16, 64, 128])
    @pytest.mark.parametrize('seq_len', [1, 16, 128])
    def test_shapes(self, dim, seq_len):
        rope = DeepseekV3RotaryEmbedding(dim, max_position_embeddings=seq_len * 2)
        x = torch.randn(2, 8, 4, dim)
        cos, sin = rope(x, seq_len=seq_len)
        assert cos.shape == (seq_len, dim)
        assert sin.shape == (seq_len, dim)

    def test_cache_expansion(self):
        rope = DeepseekV3RotaryEmbedding(64, max_position_embeddings=64)
        x = torch.randn(2, 8, 4, 64)
        rope(x, seq_len=32)
        rope(x, seq_len=256)
        assert rope.max_seq_len_cached >= 256

    def test_cache_no_rebuild_on_smaller(self):
        rope = DeepseekV3RotaryEmbedding(64, max_position_embeddings=512)
        x = torch.randn(2, 8, 4, 64)
        rope(x, seq_len=256)
        ptr = rope.cos_cached.data_ptr()
        rope(x, seq_len=128)
        assert rope.cos_cached.data_ptr() == ptr


class TestLinearScalingRotaryEmbedding:

    def test_factor1_matches_base(self):
        base = DeepseekV3RotaryEmbedding(64, max_position_embeddings=128)
        linear = DeepseekV3LinearScalingRotaryEmbedding(
            64, max_position_embeddings=128, scaling_factor=1.0)
        x = torch.randn(2, 8, 4, 64)
        c1, s1 = base(x, seq_len=64)
        c2, s2 = linear(x, seq_len=64)
        assert torch.allclose(c1, c2, atol=1e-5)
        assert torch.allclose(s1, s2, atol=1e-5)

    @pytest.mark.parametrize('factor', [2.0, 4.0, 32.0])
    def test_various_factors(self, factor):
        rope = DeepseekV3LinearScalingRotaryEmbedding(
            64, max_position_embeddings=128, scaling_factor=factor)
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=64)
        assert not torch.isnan(cos).any()


class TestDynamicNTKScalingRotaryEmbedding:

    def test_base_change_long_seq(self):
        rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(
            64, max_position_embeddings=128, scaling_factor=2.0)
        x = torch.randn(2, 8, 4, 64)
        rope(x, seq_len=64)
        inv1 = rope.inv_freq.clone()
        rope(x, seq_len=256)
        assert not torch.allclose(rope.inv_freq, inv1)


class TestYarnRotaryEmbedding:

    @pytest.mark.parametrize('factor', [1.0, 4.0, 32.0])
    def test_scaling(self, factor):
        rope = DeepseekV3YarnRotaryEmbedding(
            64, max_position_embeddings=4096, scaling_factor=factor,
            original_max_position_embeddings=4096)
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=128)
        assert not torch.isnan(cos).any()

    def test_mscale_with_all_dim(self):
        """mscale_all_dim > 0 should scale cos/sin values."""
        rope = DeepseekV3YarnRotaryEmbedding(
            64, max_position_embeddings=4096, scaling_factor=32.0,
            original_max_position_embeddings=4096, mscale=1.0, mscale_all_dim=1.0)
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=64)
        assert not torch.isnan(cos).any()


class TestYarnHelperFunctions:

    def test_find_correction_range_ordering(self):
        """low should always be <= high."""
        for low_rot, high_rot in [(1, 1), (1, 32), (32, 1)]:
            low, high = yarn_find_correction_range(low_rot, high_rot, 64)
            assert 0 <= low <= high < 64

    @pytest.mark.parametrize('scale,mscale,expected', [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 0.1 * math.log(2.0) + 1.0),
        (0.5, 1.0, 1.0),  # scale <= 1 returns 1
    ])
    def test_get_mscale(self, scale, mscale, expected):
        assert pytest.approx(yarn_get_mscale(scale, mscale), abs=1e-6) == expected

    def test_linear_ramp_mask(self):
        mask = yarn_linear_ramp_mask(0, 10, 64)
        assert mask.shape == (64,)
        assert mask.min() >= 0
        assert mask.max() <= 1

    def test_linear_ramp_mask_singularity(self):
        """min == max should not produce NaN."""
        mask = yarn_linear_ramp_mask(5, 5, 64)
        assert not torch.isnan(mask).any()


# =============================================================================
# Utility Functions
# =============================================================================


class TestRotateHalf:

    def test_values(self):
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        out = rotate_half(x)
        expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
        assert torch.allclose(out, expected)

    def test_shape_preservation(self):
        x = torch.randn(2, 4, 8, 64)
        assert rotate_half(x).shape == x.shape


class TestApplyRotaryPosEmb:

    def test_basic(self):
        b, h, s, d = 2, 4, 8, 64
        q = torch.randn(b, h, s, d)
        k = torch.randn(b, h, s, d)
        cos = torch.randn(s, d)
        sin = torch.randn(s, d)
        pos_ids = torch.arange(s).unsqueeze(0).expand(b, s)
        q_out, k_out = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        assert q_out.shape == q.shape
        assert k_out.shape == k.shape


class TestRepeatKV:

    @pytest.mark.parametrize('n_rep', [1, 2, 4, 32])
    def test_repeat(self, n_rep):
        x = torch.randn(2, 2, 8, 64)
        out = repeat_kv(x, n_rep)
        assert out.shape == (2, 2 * n_rep, 8, 64)

    def test_no_op(self):
        x = torch.randn(2, 2, 8, 64)
        assert repeat_kv(x, 1) is x


class TestGetUnpadData:

    def test_basic(self):
        mask = torch.ones(2, 8)
        mask[0, 6:] = 0
        mask[1, 7:] = 0
        indices, cu_seqlens, max_seqlen = _get_unpad_data(mask)
        assert indices.shape[0] == 13
        assert cu_seqlens.shape[0] == 3
        assert max_seqlen == 7


# =============================================================================
# MLP Tests
# =============================================================================


class TestMLP:

    def test_forward(self, base_config):
        mlp = DeepseekV3MLP(base_config)
        x = torch.randn(2, 8, base_config.hidden_size)
        assert mlp(x).shape == x.shape

    def test_custom_sizes(self, base_config):
        mlp = DeepseekV3MLP(base_config, hidden_size=32, intermediate_size=64)
        x = torch.randn(2, 8, 32)
        assert mlp(x).shape == (2, 8, 32)

    @pytest.mark.parametrize('activation', ['silu', 'gelu', 'relu'])
    def test_activations(self, base_config, activation):
        base_config.hidden_act = activation
        mlp = DeepseekV3MLP(base_config)
        x = torch.randn(2, 8, base_config.hidden_size)
        assert mlp(x).shape == x.shape


# =============================================================================
# MoE Gate Tests
# =============================================================================


class TestMoEGate:

    def test_eval_basic(self, moe_config):
        gate = MoEGate(moe_config)
        gate.eval()
        h = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(h)
        assert topk_idx.shape == (16, moe_config.num_experts_per_tok)
        assert aux_loss is None

    def test_training_aux_loss(self, moe_config):
        gate = MoEGate(moe_config)
        gate.train()
        h = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(h)
        assert aux_loss is not None
        assert aux_loss.requires_grad

    def test_invalid_scoring(self, moe_config):
        moe_config.scoring_func = 'invalid'
        gate = MoEGate(moe_config)
        with pytest.raises(NotImplementedError):
            gate(torch.randn(2, 8, moe_config.hidden_size))

    def test_invalid_topk_method(self, moe_config):
        moe_config.topk_method = 'invalid'
        gate = MoEGate(moe_config)
        with pytest.raises(NotImplementedError):
            gate(torch.randn(2, 8, moe_config.hidden_size))

    @pytest.mark.parametrize('dtype_str', ['fp32', 'bf16', 'fp16'])
    def test_router_dtypes(self, moe_config, dtype_str):
        moe_config.moe_router_dtype = dtype_str
        gate = MoEGate(moe_config)
        topk_idx, _, _ = gate(torch.randn(2, 8, moe_config.hidden_size))
        assert topk_idx.shape[0] == 16

    def test_invalid_router_dtype(self, moe_config):
        moe_config.moe_router_dtype = 'invalid'
        gate = MoEGate(moe_config)
        with pytest.raises(ValueError, match='Unsupported moe_router_dtype'):
            gate(torch.randn(2, 8, moe_config.hidden_size))

    def test_norm_topk_prob(self, moe_config):
        gate = MoEGate(moe_config)
        h = torch.randn(2, 8, moe_config.hidden_size)
        _, topk_weight, _ = gate(h)
        # After norm + scaling, weights sum to routed_scaling_factor
        weight_sums = topk_weight.sum(dim=-1)
        assert torch.allclose(weight_sums,
                              torch.full_like(weight_sums, gate.routed_scaling_factor),
                              atol=1e-4)

    def test_expert_bias(self, moe_config):
        moe_config.moe_router_enable_expert_bias = True
        gate = MoEGate(moe_config)
        assert gate.bias is not None
        gate.eval()
        topk_idx, _, _ = gate(torch.randn(2, 8, moe_config.hidden_size))
        assert topk_idx.shape[0] == 16

    def test_z_loss(self, moe_config):
        moe_config.moe_z_loss_coeff = 0.01
        gate = MoEGate(moe_config)
        gate.train()
        _, _, aux_loss = gate(torch.randn(2, 8, moe_config.hidden_size))
        assert aux_loss is not None


# =============================================================================
# MoE Tests
# =============================================================================


class TestMoE:

    def test_eval_forward(self, moe_config):
        moe = DeepseekV3MoE(moe_config)
        moe.eval()
        x = torch.randn(2, 8, moe_config.hidden_size)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss is None

    def test_train_forward(self, moe_config):
        moe = DeepseekV3MoE(moe_config)
        moe.train()
        x = torch.randn(2, 8, moe_config.hidden_size)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss is not None

    def test_no_shared_experts(self, moe_config):
        moe_config.n_shared_experts = None
        moe = DeepseekV3MoE(moe_config)
        assert not hasattr(moe, 'shared_experts')
        x = torch.randn(2, 8, moe_config.hidden_size)
        out, _ = moe(x)
        assert out.shape == x.shape

    def test_ep_init_without_dist(self, moe_config):
        """EP initialization should work gracefully without distributed."""
        moe_config.ep_size = 2
        with patch('torch.distributed.is_available', return_value=False):
            moe = DeepseekV3MoE(moe_config)
            assert moe.ep_size == 2
            assert moe.ep_rank == 0

    def test_moe_forward_method(self, moe_config):
        moe = DeepseekV3MoE(moe_config)
        moe.eval()
        x = torch.randn(16, moe_config.hidden_size)
        topk_idx = torch.randint(0, moe_config.n_routed_experts,
                                 (16, moe_config.num_experts_per_tok))
        topk_weight = torch.ones(16, moe_config.num_experts_per_tok) / moe_config.num_experts_per_tok
        out = moe.moe_forward(x, topk_idx, topk_weight)
        assert out.shape == x.shape


# =============================================================================
# Attention Tests
# =============================================================================


class TestAttention:

    def _make_position_embeddings(self, config, seq_len):
        """Create position embeddings using the model's RoPE."""
        rope = DeepseekV3YarnRotaryEmbedding(
            config.kv_channels,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        ) if config.rope_scaling and config.rope_scaling.get('type') == 'yarn' else \
            DeepseekV3RotaryEmbedding(
            config.kv_channels,
            max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta,
        )
        x_dummy = torch.randn(1, 1, 1, config.kv_channels)
        return rope(x_dummy, seq_len=seq_len)

    def test_forward_basic(self, base_config):
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        cos, sin = self._make_position_embeddings(base_config, 8)
        # Attention returns (output, weights, past_kv)
        out, attn_weights, _ = attn(x, position_ids=pos_ids,
                                    position_embeddings=(cos, sin))
        assert out.shape == x.shape
        assert attn_weights is None  # output_attentions=False by default

    def test_forward_with_output_attentions(self, base_config):
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        cos, sin = self._make_position_embeddings(base_config, 8)
        out, attn_weights, _ = attn(x, position_ids=pos_ids,
                                    position_embeddings=(cos, sin),
                                    output_attentions=True)
        assert out.shape == x.shape
        assert attn_weights is not None
        assert attn_weights.shape == (2, base_config.num_attention_heads, 8, 8)

    def test_forward_with_cache(self, base_config):
        from transformers.cache_utils import DynamicCache
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        cos, sin = self._make_position_embeddings(base_config, 8)

        cache = DynamicCache()
        x1 = torch.randn(2, 4, base_config.hidden_size)
        pos_ids1 = torch.arange(4).unsqueeze(0).expand(2, 4)
        out1, _, cache = attn(x1, position_ids=pos_ids1, past_key_value=cache,
                              use_cache=True, position_embeddings=(cos, sin))

        x2 = torch.randn(2, 1, base_config.hidden_size)
        pos_ids2 = torch.tensor([[4], [4]])
        out2, _, cache = attn(x2, position_ids=pos_ids2, past_key_value=cache,
                              use_cache=True, position_embeddings=(cos, sin))
        assert out2.shape == (2, 1, base_config.hidden_size)

    def test_attention_mask(self, base_config):
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        cos, sin = self._make_position_embeddings(base_config, 8)
        mask = torch.triu(torch.ones(8, 8) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(2, 1, 8, 8)
        out, _, _ = attn(x, attention_mask=mask, position_ids=pos_ids,
                         position_embeddings=(cos, sin))
        assert out.shape == x.shape

    def test_qk_layernorm_exists(self, base_config):
        """GQA attention should have q_layernorm and k_layernorm."""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        assert hasattr(attn, 'q_layernorm')
        assert hasattr(attn, 'k_layernorm')
        assert isinstance(attn.q_layernorm, DeepseekV3RMSNorm)
        assert isinstance(attn.k_layernorm, DeepseekV3RMSNorm)

    def test_gqa_projection_shapes(self, base_config):
        """Verify projection weight shapes match GQA architecture."""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        h = base_config.hidden_size
        nh = base_config.num_attention_heads
        ng = base_config.num_query_groups
        hd = base_config.kv_channels

        assert attn.q_proj.weight.shape == (nh * hd, h)
        assert attn.k_proj.weight.shape == (ng * hd, h)
        assert attn.v_proj.weight.shape == (ng * hd, h)
        assert attn.o_proj.weight.shape == (h, nh * hd)
        assert attn.q_layernorm.weight.shape == (hd,)
        assert attn.k_layernorm.weight.shape == (hd,)

    def test_invalid_head_config(self):
        """num_heads not divisible by num_query_groups should raise."""
        config = _make_gqa_config(num_attention_heads=4, num_query_groups=3)
        with pytest.raises(ValueError, match='must be divisible by'):
            DeepseekV3Attention(config, layer_idx=0)

    def test_mscale_scaling(self, yarn_config):
        """YaRN config with mscale_all_dim > 0 should increase scaling."""
        attn = DeepseekV3Attention(yarn_config, layer_idx=0)
        # With mscale_all_dim > 0, scaling should be > head_dim^(-0.5)
        base_scaling = yarn_config.kv_channels ** -0.5
        assert attn.scaling > base_scaling

    def test_no_mscale_without_yarn(self, base_config):
        """Without YaRN mscale, scaling should be head_dim^(-0.5)."""
        base_config.rope_scaling = None
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        expected = base_config.kv_channels ** -0.5
        assert pytest.approx(attn.scaling, abs=1e-6) == expected


# =============================================================================
# Decoder Layer Tests
# =============================================================================


class TestDecoderLayer:

    def _make_pos_emb(self, config, seq_len):
        rope = DeepseekV3RotaryEmbedding(
            config.kv_channels, max_position_embeddings=config.max_position_embeddings,
            base=config.rope_theta)
        return rope(torch.randn(1, 1, 1, config.kv_channels), seq_len=seq_len)

    def test_forward(self, base_config):
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        cos, sin = self._make_pos_emb(base_config, 8)
        outputs = layer(x, position_ids=pos_ids, position_embeddings=(cos, sin))
        assert outputs[0].shape == x.shape

    def test_dense_vs_moe(self, base_config):
        base_config.first_k_dense_replace = 1
        base_config.moe_layer_freq = 1

        layer0 = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        assert isinstance(layer0.mlp, DeepseekV3MLP)

        layer1 = DeepseekV3DecoderLayer(base_config, layer_idx=1)
        assert isinstance(layer1.mlp, DeepseekV3MoE)

    def test_moe_aux_loss_propagation(self, base_config):
        """MoE layer should return aux_loss."""
        base_config.first_k_dense_replace = 0  # All MoE
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        assert isinstance(layer.mlp, DeepseekV3MoE)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        cos, sin = self._make_pos_emb(base_config, 8)
        layer.train()
        outputs = layer(x, position_ids=pos_ids, position_embeddings=(cos, sin))
        # Last element should be aux_loss from MoE
        assert outputs[-1] is not None

    def test_padding_mask_warning(self, base_config):
        import warnings
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        cos, sin = self._make_pos_emb(base_config, 8)
        with pytest.warns(UserWarning, match='padding_mask.*deprecated'):
            layer(x, position_ids=pos_ids, position_embeddings=(cos, sin),
                  padding_mask=torch.ones(2, 8))


# =============================================================================
# Model Tests
# =============================================================================


class TestDeepseekV3Model:

    def test_forward(self, base_config):
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.last_hidden_state.shape == (2, 8, base_config.hidden_size)

    def test_forward_with_inputs_embeds(self, base_config):
        model = DeepseekV3Model(base_config)
        inputs_embeds = torch.randn(2, 8, base_config.hidden_size)
        out = model(inputs_embeds=inputs_embeds)
        assert out.last_hidden_state.shape == (2, 8, base_config.hidden_size)

    def test_both_inputs_error(self, base_config):
        model = DeepseekV3Model(base_config)
        with pytest.raises(ValueError, match='cannot specify both'):
            model(input_ids=torch.randint(0, 128, (2, 8)),
                  inputs_embeds=torch.randn(2, 8, base_config.hidden_size))

    def test_no_inputs_error(self, base_config):
        model = DeepseekV3Model(base_config)
        with pytest.raises(ValueError, match='specify either'):
            model()

    def test_output_hidden_states(self, base_config):
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, output_hidden_states=True)
        assert out.hidden_states is not None
        assert len(out.hidden_states) == base_config.num_hidden_layers + 1

    def test_output_attentions(self, base_config):
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, output_attentions=True)
        assert out.attentions is not None
        assert len(out.attentions) == base_config.num_hidden_layers

    def test_use_cache(self, base_config):
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, use_cache=True)
        assert out.past_key_values is not None

    def test_autoregressive_generation(self, base_config):
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 4))
        out1 = model(input_ids, use_cache=True)

        new_ids = torch.randint(0, base_config.vocab_size, (2, 1))
        out2 = model(new_ids, past_key_values=out1.past_key_values, use_cache=True)
        assert out2.last_hidden_state.shape == (2, 1, base_config.hidden_size)

    def test_get_set_embeddings(self, base_config):
        model = DeepseekV3Model(base_config)
        embeds = model.get_input_embeddings()
        assert embeds is model.embed_tokens

        new_embeds = torch.nn.Embedding(base_config.vocab_size, base_config.hidden_size)
        model.set_input_embeddings(new_embeds)
        assert model.get_input_embeddings() is new_embeds

    def test_return_dict_false(self, base_config):
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, return_dict=False)
        assert isinstance(out, tuple)

    def test_rope_embedding_created(self, base_config):
        """Model should create rotary embedding."""
        model = DeepseekV3Model(base_config)
        assert hasattr(model, 'rotary_emb')

    def test_yarn_rope_embedding(self, yarn_config):
        """Model with YaRN config should use YarnRotaryEmbedding."""
        model = DeepseekV3Model(yarn_config)
        assert isinstance(model.rotary_emb, DeepseekV3YarnRotaryEmbedding)


# =============================================================================
# CausalLM Tests
# =============================================================================


class TestDeepseekV3ForCausalLM:

    def test_forward(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, 8, base_config.vocab_size)
        assert out.logits.dtype == torch.float32

    def test_forward_with_labels(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert out.loss is not None
        assert out.loss.requires_grad

    def test_backward(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        model.train()
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        out.loss.backward()
        assert model.lm_head.weight.grad is not None

    def test_prepare_inputs_for_generation(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        model_inputs = model.prepare_inputs_for_generation(input_ids)
        assert 'input_ids' in model_inputs
        assert 'position_ids' in model_inputs

    def test_prepare_inputs_with_cache(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        past_kv = tuple(
            (torch.randn(2, 4, 4, base_config.kv_channels),
             torch.randn(2, 4, 4, base_config.kv_channels))
            for _ in range(base_config.num_hidden_layers))
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, past_key_values=past_kv)
        # past_length=4, input_ids length=8, should keep last 4 tokens
        assert model_inputs['input_ids'].shape[1] == 4

    def test_get_set_output_embeddings(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        assert model.get_output_embeddings() is model.lm_head
        new_head = torch.nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)
        model.set_output_embeddings(new_head)
        assert model.get_output_embeddings() is new_head

    def test_reorder_cache(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        cache = tuple(
            (torch.randn(2, 4, 8, base_config.kv_channels),
             torch.randn(2, 4, 8, base_config.kv_channels))
            for _ in range(base_config.num_hidden_layers))
        beam_idx = torch.tensor([1, 0])
        reordered = model._reorder_cache(cache, beam_idx)
        assert len(reordered) == len(cache)

    def test_reorder_cache_dynamic(self, base_config):
        """Test _reorder_cache with DynamicCache."""
        from transformers.cache_utils import DynamicCache
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 4))
        out = model(input_ids, use_cache=True)
        beam_idx = torch.tensor([1, 0])
        result = model._reorder_cache(out.past_key_values, beam_idx)
        # DynamicCache.reorder_cache may return None (in-place) or the cache itself
        assert result is None or isinstance(result, DynamicCache)

    def test_aux_loss_in_training(self, moe_config):
        """MoE aux_loss should be added to training loss."""
        moe_config.first_k_dense_replace = 0
        model = DeepseekV3ForCausalLM(moe_config)
        model.train()
        input_ids = torch.randint(0, moe_config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert out.loss is not None


# =============================================================================
# Sequence Classification Tests
# =============================================================================


class TestDeepseekV3ForSequenceClassification:

    @pytest.fixture
    def cls_config(self, base_config):
        base_config.num_labels = 3
        return base_config

    def test_forward(self, cls_config):
        model = DeepseekV3ForSequenceClassification(cls_config)
        input_ids = torch.randint(0, cls_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, cls_config.num_labels)

    def test_with_labels(self, cls_config):
        model = DeepseekV3ForSequenceClassification(cls_config)
        input_ids = torch.randint(0, cls_config.vocab_size, (2, 8))
        out = model(input_ids, labels=torch.tensor([0, 2]))
        assert out.loss is not None

    @pytest.mark.parametrize('num_labels', [1, 2, 5])
    def test_various_labels(self, base_config, num_labels):
        base_config.num_labels = num_labels
        model = DeepseekV3ForSequenceClassification(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        labels = torch.randn(2) if num_labels == 1 else torch.randint(0, num_labels, (2,))
        out = model(input_ids, labels=labels)
        assert out.loss is not None

    def test_no_padding_error(self, base_config):
        base_config.num_labels = 3
        base_config.pad_token_id = None
        model = DeepseekV3ForSequenceClassification(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        with pytest.raises(ValueError, match='batch sizes > 1'):
            model(input_ids)


# =============================================================================
# Integration / Edge Cases
# =============================================================================


class TestEdgeCases:

    def test_single_token(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (1, 1))
        out = model(input_ids)
        assert out.logits.shape == (1, 1, base_config.vocab_size)

    def test_large_batch(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (16, 8))
        out = model(input_ids)
        assert out.logits.shape == (16, 8, base_config.vocab_size)

    def test_long_sequence(self, base_config):
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (1, 256))
        out = model(input_ids)
        assert out.logits.shape == (1, 256, base_config.vocab_size)

    def test_bfloat16(self, base_config):
        model = DeepseekV3ForCausalLM(base_config).to(torch.bfloat16)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.logits.dtype == torch.float32

    def test_tie_word_embeddings_false(self, base_config):
        base_config.tie_word_embeddings = False
        model = DeepseekV3ForCausalLM(base_config)
        # lm_head and embed_tokens should be separate
        assert not torch.equal(model.lm_head.weight.data,
                               model.model.embed_tokens.weight.data)

    def test_all_moe_layers(self):
        """Test model with all MoE layers (first_k_dense_replace=0)."""
        config = _make_gqa_config(first_k_dense_replace=0)
        model = DeepseekV3ForCausalLM(config)
        model.train()
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids, labels=input_ids.clone())
        assert out.loss is not None
        out.loss.backward()

    def test_all_dense_layers(self):
        """Test model with all dense layers (no MoE)."""
        config = _make_gqa_config(n_routed_experts=None, first_k_dense_replace=999)
        model = DeepseekV3ForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, 8, config.vocab_size)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
