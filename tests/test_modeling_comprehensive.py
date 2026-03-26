"""
Comprehensive unit tests for modeling_deepseek.py

Coverage targets:
- Branch coverage >= 90%
- Critical path coverage 100%
- Boundary testing, exception testing, parameterized testing
"""

import math
import pytest
import torch
import torch.nn.functional as F
from unittest.mock import Mock, patch, MagicMock

from models.configuration_deepseek_1t import DeepseekV3Config
from models.modeling_deepseek import (
    DeepseekV3RMSNorm,
    DeepseekV3RotaryEmbedding,
    DeepseekV3LinearScalingRotaryEmbedding,
    DeepseekV3DynamicNTKScalingRotaryEmbedding,
    DeepseekV3YarnRotaryEmbedding,
    DeepseekV3MLP,
    MoEGate,
    DeepseekV3MoE,
    DeepseekV3Attention,
    DeepseekV3FlashAttention2,
    DeepseekV3DecoderLayer,
    DeepseekV3PreTrainedModel,
    DeepseekV3Model,
    DeepseekV3ForCausalLM,
    DeepseekV3ForSequenceClassification,
    rotate_half,
    apply_rotary_pos_emb,
    repeat_kv,
    yarn_find_correction_dim,
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
    _get_unpad_data,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def base_config():
    """Minimal config for fast tests"""
    config = DeepseekV3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_position_embeddings=128,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        hidden_act="silu",
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=64,
        first_k_dense_replace=0,
        ep_size=1,
        n_group=1,
        topk_group=1,
        attention_dropout=0.0,
        pad_token_id=0,  # For SequenceClassification tests
    )
    # Set attention implementation for DecoderLayer
    config._attn_implementation = "eager"
    return config


@pytest.fixture
def moe_config(base_config):
    """Config with MoE enabled"""
    base_config.n_routed_experts = 8
    base_config.n_shared_experts = 2
    base_config.num_experts_per_tok = 2
    base_config.moe_aux_loss_coeff = 0.01
    base_config.moe_z_loss_coeff = 0.01
    return base_config


@pytest.fixture
def yarn_config(base_config):
    """Config with YaRN RoPE scaling"""
    base_config.rope_scaling = {
        "type": "yarn",
        "factor": 32.0,
        "original_max_position_embeddings": 4096,
        "beta_fast": 32,
        "beta_slow": 1,
        "mscale": 1,
        "mscale_all_dim": 0,
    }
    return base_config


# =============================================================================
# RMSNorm Tests
# =============================================================================

class TestRMSNorm:
    """Test DeepseekV3RMSNorm with boundary and edge cases"""

    @pytest.mark.parametrize("hidden_size", [1, 16, 64, 512, 8192])
    def test_rmsnorm_various_sizes(self, hidden_size):
        """Test RMSNorm with various hidden sizes including edge cases"""
        norm = DeepseekV3RMSNorm(hidden_size)
        x = torch.randn(2, 8, hidden_size)
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    @pytest.mark.parametrize("eps", [1e-12, 1e-6, 1e-3, 1.0])
    def test_rmsnorm_eps_values(self, eps):
        """Test with various epsilon values"""
        norm = DeepseekV3RMSNorm(64, eps=eps)
        x = torch.randn(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape

    def test_rmsnorm_zero_input(self):
        """Test with zero input - should handle gracefully"""
        norm = DeepseekV3RMSNorm(64)
        x = torch.zeros(2, 8, 64)
        out = norm(x)
        assert out.shape == x.shape
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-6)

    def test_rmsnorm_very_large_input(self):
        """Test with very large input values"""
        norm = DeepseekV3RMSNorm(64)
        x = torch.randn(2, 8, 64) * 1000
        out = norm(x)
        assert out.shape == x.shape
        assert not torch.isnan(out).any()

    def test_rmsnorm_dtype_preservation(self):
        """Test that output dtype is preserved (RMSNorm casts to float32 internally then back)"""
        norm = DeepseekV3RMSNorm(64)
        # RMSNorm converts to float32 for numerical stability, then back to input dtype
        for dtype in [torch.float32, torch.float16, torch.bfloat16]:
            x = torch.randn(2, 8, 64, dtype=dtype)
            out = norm(x)
            # Output dtype should match input dtype (conversion back happens via .to(input_dtype))
            assert out.dtype == dtype, f"Expected {dtype}, got {out.dtype}"

    def test_rmsnorm_single_element(self):
        """Test with batch_size=1, seq_len=1"""
        norm = DeepseekV3RMSNorm(64)
        x = torch.randn(1, 1, 64)
        out = norm(x)
        assert out.shape == (1, 1, 64)


# =============================================================================
# Rotary Embedding Tests
# =============================================================================

class TestRotaryEmbedding:
    """Test various Rotary Embedding implementations"""

    @pytest.mark.parametrize("dim", [4, 8, 16, 64, 128])
    @pytest.mark.parametrize("seq_len", [1, 16, 128, 2048])
    def test_rotary_embedding_shapes(self, dim, seq_len):
        """Test RoPE produces correct output shapes"""
        rope = DeepseekV3RotaryEmbedding(dim, max_position_embeddings=seq_len * 2)
        x = torch.randn(2, 8, 4, dim)
        cos, sin = rope(x, seq_len=seq_len)
        assert cos.shape[0] == seq_len
        assert sin.shape[0] == seq_len
        assert cos.shape[-1] == dim
        assert sin.shape[-1] == dim

    @pytest.mark.parametrize("base", [500, 10000, 100000, 1000000])
    def test_rotary_embedding_different_bases(self, base):
        """Test with different RoPE theta bases"""
        rope = DeepseekV3RotaryEmbedding(64, base=base)
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=128)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

    def test_rotary_embedding_cache_expansion(self):
        """Test cache expands when seq_len exceeds max_position_embeddings"""
        rope = DeepseekV3RotaryEmbedding(64, max_position_embeddings=128)
        x = torch.randn(2, 8, 4, 64)
        # First call with small seq_len
        cos1, sin1 = rope(x, seq_len=64)
        # Second call with larger seq_len should expand cache
        cos2, sin2 = rope(x, seq_len=256)
        assert rope.max_seq_len_cached >= 256

    def test_rotary_embedding_no_cache_rebuild(self):
        """Test that cache is not rebuilt when seq_len decreases"""
        rope = DeepseekV3RotaryEmbedding(64, max_position_embeddings=512)
        x = torch.randn(2, 8, 4, 64)
        cos1, sin1 = rope(x, seq_len=256)
        cache_ptr = rope.cos_cached.data_ptr()
        cos2, sin2 = rope(x, seq_len=128)  # Smaller seq_len
        assert rope.cos_cached.data_ptr() == cache_ptr  # Same cache


class TestLinearScalingRotaryEmbedding:
    """Test Linear Scaling RoPE"""

    @pytest.mark.parametrize("scaling_factor", [1.0, 2.0, 4.0, 8.0, 32.0])
    def test_linear_scaling(self, scaling_factor):
        """Test linear scaling produces correct results"""
        rope = DeepseekV3LinearScalingRotaryEmbedding(
            64, max_position_embeddings=128, scaling_factor=scaling_factor
        )
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=128)
        assert cos.shape[0] == 128
        assert not torch.isnan(cos).any()

    def test_linear_scaling_vs_base(self):
        """Compare linear scaling with scaling_factor=1 to base RoPE"""
        base_rope = DeepseekV3RotaryEmbedding(64, max_position_embeddings=128)
        linear_rope = DeepseekV3LinearScalingRotaryEmbedding(
            64, max_position_embeddings=128, scaling_factor=1.0
        )
        x = torch.randn(2, 8, 4, 64)
        base_cos, base_sin = base_rope(x, seq_len=128)
        linear_cos, linear_sin = linear_rope(x, seq_len=128)
        assert torch.allclose(base_cos, linear_cos, atol=1e-5)
        assert torch.allclose(base_sin, linear_sin, atol=1e-5)


class TestDynamicNTKScalingRotaryEmbedding:
    """Test Dynamic NTK Scaling RoPE"""

    @pytest.mark.parametrize("scaling_factor", [1.0, 2.0, 4.0, 8.0])
    def test_dynamic_ntk_scaling(self, scaling_factor):
        """Test dynamic NTK scaling"""
        rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(
            64, max_position_embeddings=128, scaling_factor=scaling_factor
        )
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=256)  # Longer than max_position
        assert not torch.isnan(cos).any()

    def test_dynamic_ntk_base_change(self):
        """Test that base changes for long sequences"""
        max_pos = 128
        rope = DeepseekV3DynamicNTKScalingRotaryEmbedding(
            64, max_position_embeddings=max_pos, base=10000, scaling_factor=2.0
        )
        x = torch.randn(2, 8, 4, 64)
        # Short sequence - base should not change
        rope(x, seq_len=64)
        original_inv_freq = rope.inv_freq.clone()
        # Long sequence - base should change
        rope(x, seq_len=256)
        assert not torch.allclose(rope.inv_freq, original_inv_freq)


class TestYarnRotaryEmbedding:
    """Test YaRN RoPE"""

    @pytest.mark.parametrize("factor", [1.0, 4.0, 8.0, 32.0])
    def test_yarn_scaling(self, factor):
        """Test YaRN scaling with various factors"""
        rope = DeepseekV3YarnRotaryEmbedding(
            64,
            max_position_embeddings=4096,
            scaling_factor=factor,
            original_max_position_embeddings=4096,
        )
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=512)
        assert not torch.isnan(cos).any()
        assert not torch.isnan(sin).any()

    def test_yarn_mscale_application(self):
        """Test that mscale is applied to YaRN embeddings"""
        rope = DeepseekV3YarnRotaryEmbedding(
            64,
            max_position_embeddings=4096,
            scaling_factor=32.0,
            original_max_position_embeddings=4096,
            mscale=1.0,
        )
        x = torch.randn(2, 8, 4, 64)
        cos, sin = rope(x, seq_len=512)
        # mscale should be applied - with scaling_factor=32, mscale is ~1.35
        # So values can be up to ~1.35 (before clipping to [-1, 1] range)
        assert cos.abs().max() <= 1.5  # Allow for mscale > 1


class TestYarnHelperFunctions:
    """Test YaRN helper functions"""

    @pytest.mark.parametrize("num_rotations", [1, 10, 100])
    @pytest.mark.parametrize("dim", [32, 64, 128])
    def test_yarn_find_correction_dim(self, num_rotations, dim):
        """Test correction dimension calculation"""
        result = yarn_find_correction_dim(num_rotations, dim)
        assert isinstance(result, float)
        assert not math.isnan(result)

    def test_yarn_find_correction_range(self):
        """Test correction range calculation"""
        low, high = yarn_find_correction_range(1, 32, 64)
        assert 0 <= low <= high < 64

    @pytest.mark.parametrize("scale,mscale,expected", [
        (1.0, 1.0, 1.0),
        (2.0, 1.0, 0.1 * 1.0 * math.log(2.0) + 1.0),
        (4.0, 0.5, 0.1 * 0.5 * math.log(4.0) + 1.0),
    ])
    def test_yarn_get_mscale(self, scale, mscale, expected):
        """Test mscale calculation"""
        result = yarn_get_mscale(scale, mscale)
        assert pytest.approx(result, abs=1e-6) == expected

    @pytest.mark.parametrize("min_val,max_val,dim", [
        (0, 10, 64),
        (5, 5, 32),  # Edge case: min == max
        (0, 32, 128),
    ])
    def test_yarn_linear_ramp_mask(self, min_val, max_val, dim):
        """Test linear ramp mask generation"""
        mask = yarn_linear_ramp_mask(min_val, max_val, dim)
        assert mask.shape == (dim,)
        assert mask.min() >= 0
        assert mask.max() <= 1

    def test_yarn_linear_ramp_mask_singularity(self):
        """Test mask handles min == max (singularity case)"""
        mask = yarn_linear_ramp_mask(5, 5, 64)
        assert mask.shape == (64,)
        assert not torch.isnan(mask).any()


# =============================================================================
# Utility Functions Tests
# =============================================================================

class TestRotateHalf:
    """Test rotate_half function"""

    def test_rotate_half_shape_preservation(self):
        """Test output shape matches input"""
        x = torch.randn(2, 8, 4, 64)
        out = rotate_half(x)
        assert out.shape == x.shape

    def test_rotate_half_values(self):
        """Test rotation produces expected values"""
        x = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])
        out = rotate_half(x)
        # First half becomes -second half, second half becomes first half
        expected = torch.tensor([[[[-3.0, -4.0, 1.0, 2.0]]]])
        assert torch.allclose(out, expected)


class TestApplyRotaryPosEmb:
    """Test apply_rotary_pos_emb function"""

    def test_apply_rotary_basic(self):
        """Test basic rotary position embedding application"""
        batch, heads, seq_len, head_dim = 2, 4, 8, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, pos_ids)
        assert q_embed.shape == q.shape
        assert k_embed.shape == k.shape

    def test_apply_rotary_unsqueeze_dim1(self):
        """Test with unsqueeze_dim=1 (standard case for batch, heads, seq, head_dim)"""
        batch, heads, seq_len, head_dim = 2, 4, 8, 64
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        cos = torch.randn(seq_len, head_dim)
        sin = torch.randn(seq_len, head_dim)
        pos_ids = torch.arange(seq_len).unsqueeze(0).expand(batch, seq_len)

        q_embed, k_embed = apply_rotary_pos_emb(q, k, cos, sin, pos_ids, unsqueeze_dim=1)
        assert q_embed.shape == q.shape


class TestRepeatKV:
    """Test repeat_kv function (GQA)"""

    @pytest.mark.parametrize("n_rep", [1, 2, 4, 8])
    def test_repeat_kv(self, n_rep):
        """Test key/value repetition for GQA"""
        batch, num_kv_heads, seq_len, head_dim = 2, 2, 8, 64
        hidden = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        out = repeat_kv(hidden, n_rep)
        expected_heads = num_kv_heads * n_rep
        assert out.shape == (batch, expected_heads, seq_len, head_dim)

    def test_repeat_kv_no_op(self):
        """Test n_rep=1 returns same tensor"""
        batch, num_kv_heads, seq_len, head_dim = 2, 2, 8, 64
        hidden = torch.randn(batch, num_kv_heads, seq_len, head_dim)
        out = repeat_kv(hidden, 1)
        assert out is hidden  # Should return same reference


class TestGetUnpadData:
    """Test _get_unpad_data function"""

    def test_get_unpad_data(self):
        """Test unpadding data extraction"""
        batch_size, seq_len = 2, 8
        attention_mask = torch.ones(batch_size, seq_len)
        attention_mask[0, 6:] = 0  # First sequence has 6 valid tokens
        attention_mask[1, 7:] = 0  # Second sequence has 7 valid tokens

        indices, cu_seqlens, max_seqlen = _get_unpad_data(attention_mask)
        assert indices.shape[0] == 13  # 6 + 7 = 13 valid tokens
        assert cu_seqlens.shape[0] == batch_size + 1
        assert max_seqlen == 7


# =============================================================================
# MLP Tests
# =============================================================================

class TestMLP:
    """Test DeepseekV3MLP"""

    def test_mlp_forward(self, base_config):
        """Test basic MLP forward pass"""
        mlp = DeepseekV3MLP(base_config)
        x = torch.randn(2, 8, base_config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape

    def test_mlp_custom_sizes(self, base_config):
        """Test MLP with custom hidden and intermediate sizes"""
        mlp = DeepseekV3MLP(base_config, hidden_size=32, intermediate_size=64)
        x = torch.randn(2, 8, 32)
        out = mlp(x)
        assert out.shape == (2, 8, 32)

    def test_mlp_various_batch_seq(self, base_config):
        """Test with various batch and sequence sizes"""
        mlp = DeepseekV3MLP(base_config)
        for batch in [1, 4, 8]:
            for seq_len in [1, 16, 128]:
                x = torch.randn(batch, seq_len, base_config.hidden_size)
                out = mlp(x)
                assert out.shape == (batch, seq_len, base_config.hidden_size)

    @pytest.mark.parametrize("activation", ["silu", "gelu", "relu"])
    def test_mlp_activations(self, base_config, activation):
        """Test MLP with different activations"""
        base_config.hidden_act = activation
        mlp = DeepseekV3MLP(base_config)
        x = torch.randn(2, 8, base_config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape


# =============================================================================
# MoE Gate Tests
# =============================================================================

class TestMoEGate:
    """Test MoE Gate with various configurations"""

    def test_moe_gate_basic(self, moe_config):
        """Test basic gating"""
        gate = MoEGate(moe_config)
        gate.eval()  # Explicitly set to eval mode
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(hidden)
        assert topk_idx.shape == (2 * 8, moe_config.num_experts_per_tok)
        assert topk_weight.shape == topk_idx.shape
        assert aux_loss is None  # Not in training mode

    def test_moe_gate_training(self, moe_config):
        """Test gating in training mode with aux loss"""
        gate = MoEGate(moe_config)
        gate.train()
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(hidden)
        assert aux_loss is not None
        assert aux_loss.requires_grad

    @pytest.mark.parametrize("scoring_func", ["sigmoid"])
    def test_moe_gate_scoring(self, moe_config, scoring_func):
        """Test different scoring functions"""
        moe_config.scoring_func = scoring_func
        gate = MoEGate(moe_config)
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(hidden)
        assert topk_weight.min() >= 0  # Sigmoid produces non-negative

    def test_moe_gate_invalid_scoring(self, moe_config):
        """Test invalid scoring function raises error"""
        moe_config.scoring_func = "invalid"
        gate = MoEGate(moe_config)
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        with pytest.raises(NotImplementedError):
            gate(hidden)

    def test_moe_gate_invalid_topk_method(self, moe_config):
        """Test invalid topk method raises error"""
        moe_config.topk_method = "invalid"
        gate = MoEGate(moe_config)
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        with pytest.raises(NotImplementedError):
            gate(hidden)

    def test_moe_gate_router_dtypes(self, moe_config):
        """Test router with different dtypes"""
        for dtype_str, dtype in [("fp32", torch.float32), ("bf16", torch.bfloat16), ("fp16", torch.float16)]:
            moe_config.moe_router_dtype = dtype_str
            gate = MoEGate(moe_config)
            hidden = torch.randn(2, 8, moe_config.hidden_size)
            topk_idx, topk_weight, aux_loss = gate(hidden)
            assert topk_idx.shape[0] == 16

    def test_moe_gate_invalid_router_dtype(self, moe_config):
        """Test invalid router dtype raises error"""
        moe_config.moe_router_dtype = "invalid"
        gate = MoEGate(moe_config)
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        with pytest.raises(ValueError, match="Unsupported moe_router_dtype"):
            gate(hidden)

    def test_moe_gate_norm_topk_prob(self, moe_config):
        """Test normalized topk probabilities"""
        moe_config.norm_topk_prob = True
        gate = MoEGate(moe_config)
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(hidden)
        # After normalization, weights sum to 1, then scaled by routed_scaling_factor
        # So weights should sum to approximately routed_scaling_factor per token
        weight_sums = topk_weight.sum(dim=-1)
        expected_sum = gate.routed_scaling_factor
        assert torch.allclose(weight_sums, torch.full_like(weight_sums, expected_sum), atol=1e-4)

    def test_moe_gate_expert_bias(self, moe_config):
        """Test gating with expert bias enabled"""
        moe_config.moe_router_enable_expert_bias = True
        gate = MoEGate(moe_config)
        assert gate.bias is not None
        hidden = torch.randn(2, 8, moe_config.hidden_size)
        topk_idx, topk_weight, aux_loss = gate(hidden)
        assert topk_idx.shape[0] == 16


# =============================================================================
# MoE Tests
# =============================================================================

class TestMoE:
    """Test DeepseekV3MoE"""

    def test_moe_forward_eval(self, moe_config):
        """Test MoE forward in eval mode"""
        moe = DeepseekV3MoE(moe_config)
        moe.eval()
        x = torch.randn(2, 8, moe_config.hidden_size)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss is None  # No aux loss in eval

    def test_moe_forward_train(self, moe_config):
        """Test MoE forward in train mode"""
        moe = DeepseekV3MoE(moe_config)
        moe.train()
        x = torch.randn(2, 8, moe_config.hidden_size)
        out, aux_loss = moe(x)
        assert out.shape == x.shape
        assert aux_loss is not None

    def test_moe_ep_size_greater_than_one(self, moe_config):
        """Test MoE with expert parallelism (would need distributed)"""
        moe_config.ep_size = 2
        # Note: Full EP testing requires distributed setup
        # This test just verifies initialization
        with patch('torch.distributed.is_available', return_value=False):
            moe = DeepseekV3MoE(moe_config)
            assert moe.ep_size == 2

    def test_moe_moe_forward_method(self, moe_config):
        """Test moe_forward method directly"""
        moe = DeepseekV3MoE(moe_config)
        moe.eval()
        x = torch.randn(16, moe_config.hidden_size)
        # Create dummy topk indices and weights
        topk_idx = torch.randint(0, moe_config.n_routed_experts, (16, moe_config.num_experts_per_tok))
        topk_weight = torch.ones(16, moe_config.num_experts_per_tok) / moe_config.num_experts_per_tok
        out = moe.moe_forward(x, topk_idx, topk_weight)
        assert out.shape == x.shape

    def test_moe_no_shared_experts(self, moe_config):
        """Test MoE without shared experts"""
        moe_config.n_shared_experts = None
        moe = DeepseekV3MoE(moe_config)
        assert not hasattr(moe, 'shared_experts') or moe.shared_experts is None
        x = torch.randn(2, 8, moe_config.hidden_size)
        out, aux_loss = moe(x)
        assert out.shape == x.shape


# =============================================================================
# Attention Tests
# =============================================================================

class TestAttention:
    """Test DeepseekV3Attention"""

    def test_attention_forward(self, base_config):
        """Test basic attention forward"""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        out, attn_weights, past_kv = attn(x, position_ids=pos_ids)
        assert out.shape == x.shape

    def test_attention_with_cache(self, base_config):
        """Test attention with KV cache"""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()

        # First forward pass
        x1 = torch.randn(2, 4, base_config.hidden_size)
        pos_ids1 = torch.arange(4).unsqueeze(0).expand(2, 4)
        out1, _, cache = attn(x1, position_ids=pos_ids1, past_key_value=cache, use_cache=True)

        # Second forward pass (autoregressive)
        x2 = torch.randn(2, 1, base_config.hidden_size)
        pos_ids2 = torch.tensor([[4], [4]])
        out2, _, cache = attn(x2, position_ids=pos_ids2, past_key_value=cache, use_cache=True)

        assert out2.shape == (2, 1, base_config.hidden_size)

    def test_attention_output_attentions(self, base_config):
        """Test attention with output_attentions=True"""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        out, attn_weights, _ = attn(x, position_ids=pos_ids, output_attentions=True)
        assert attn_weights is not None
        expected_shape = (2, base_config.num_attention_heads, 8, 8)
        assert attn_weights.shape == expected_shape

    def test_attention_attention_mask(self, base_config):
        """Test attention with attention mask"""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        # Create causal mask
        mask = torch.triu(torch.ones(8, 8) * float('-inf'), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0).expand(2, 1, 8, 8)
        out, _, _ = attn(x, attention_mask=mask, position_ids=pos_ids)
        assert out.shape == x.shape

    def test_attention_invalid_head_config(self, base_config):
        """Test invalid head configuration raises error"""
        base_config.num_key_value_heads = 3  # Not divisible by num_heads (4)
        with pytest.raises(ValueError, match="must be divisible by"):
            DeepseekV3Attention(base_config, layer_idx=0)

    def test_attention_without_layer_idx_warning(self, base_config, caplog):
        """Test warning when layer_idx not provided"""
        import logging
        with caplog.at_level(logging.WARNING):
            _ = DeepseekV3Attention(base_config, layer_idx=None)
            assert "without passing `layer_idx`" in caplog.text

    def test_attention_qk_layernorm(self, base_config):
        """Test attention with Q/K layernorm enabled"""
        base_config.qk_layernorm = True
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        assert attn.q_layernorm is not None
        assert attn.k_layernorm is not None
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        out, _, _ = attn(x, position_ids=pos_ids)
        assert out.shape == x.shape

    @pytest.mark.parametrize("scaling_type", ["linear", "dynamic", "yarn"])
    def test_attention_rope_scaling(self, base_config, scaling_type):
        """Test attention with different RoPE scaling"""
        if scaling_type == "linear":
            base_config.rope_scaling = {"type": "linear", "factor": 2.0}
        elif scaling_type == "dynamic":
            base_config.rope_scaling = {"type": "dynamic", "factor": 2.0}
        elif scaling_type == "yarn":
            base_config.rope_scaling = {
                "type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 4096,
            }

        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        out, _, _ = attn(x, position_ids=pos_ids)
        assert out.shape == x.shape

    def test_attention_invalid_rope_scaling(self, base_config):
        """Test invalid RoPE scaling type raises error"""
        base_config.rope_scaling = {"type": "invalid", "factor": 2.0}
        with pytest.raises(ValueError, match="Unknown RoPE scaling type"):
            DeepseekV3Attention(base_config, layer_idx=0)


# =============================================================================
# Decoder Layer Tests
# =============================================================================

class TestDecoderLayer:
    """Test DeepseekV3DecoderLayer"""

    def test_decoder_layer_forward(self, base_config):
        """Test decoder layer forward pass"""
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        outputs = layer(x, position_ids=pos_ids)
        # First element is always hidden_states
        assert outputs[0].shape == x.shape

    def test_decoder_layer_with_cache(self, base_config):
        """Test decoder layer with cache"""
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()

        x = torch.randn(2, 4, base_config.hidden_size)
        pos_ids = torch.arange(4).unsqueeze(0).expand(2, 4)
        out, _, cache = layer(x, position_ids=pos_ids, past_key_value=cache, use_cache=True)
        assert cache is not None

    def test_decoder_layer_output_attentions(self, base_config):
        """Test decoder layer with output_attentions"""
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)
        outputs = layer(x, position_ids=pos_ids, output_attentions=True)
        assert len(outputs) >= 2  # hidden_states + attentions

    def test_decoder_layer_dense_vs_moe(self, base_config):
        """Test that early layers use dense MLP, later layers use MoE"""
        base_config.first_k_dense_replace = 1
        base_config.moe_layer_freq = 1

        # First layer should be dense
        layer0 = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        assert isinstance(layer0.mlp, DeepseekV3MLP)

        # Second layer should be MoE
        layer1 = DeepseekV3DecoderLayer(base_config, layer_idx=1)
        assert isinstance(layer1.mlp, DeepseekV3MoE)

    def test_decoder_layer_padding_mask_deprecated(self, base_config):
        """Test deprecation warning for padding_mask"""
        import warnings
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        x = torch.randn(2, 8, base_config.hidden_size)
        pos_ids = torch.arange(8).unsqueeze(0).expand(2, 8)

        with pytest.warns(UserWarning, match="padding_mask.*deprecated"):
            layer(x, position_ids=pos_ids, padding_mask=torch.ones(2, 8))


# =============================================================================
# Model Tests
# =============================================================================

class TestDeepseekV3Model:
    """Test DeepseekV3Model"""

    def test_model_forward(self, base_config):
        """Test basic model forward pass"""
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.last_hidden_state.shape == (2, 8, base_config.hidden_size)

    def test_model_forward_with_inputs_embeds(self, base_config):
        """Test model with inputs_embeds instead of input_ids"""
        model = DeepseekV3Model(base_config)
        inputs_embeds = torch.randn(2, 8, base_config.hidden_size)
        out = model(inputs_embeds=inputs_embeds)
        assert out.last_hidden_state.shape == (2, 8, base_config.hidden_size)

    def test_model_forward_both_inputs_error(self, base_config):
        """Test error when both input_ids and inputs_embeds provided"""
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        inputs_embeds = torch.randn(2, 8, base_config.hidden_size)
        with pytest.raises(ValueError, match="cannot specify both"):
            model(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def test_model_forward_no_inputs_error(self, base_config):
        """Test error when neither input_ids nor inputs_embeds provided"""
        model = DeepseekV3Model(base_config)
        with pytest.raises(ValueError, match="specify either"):
            model()

    def test_model_output_hidden_states(self, base_config):
        """Test model with output_hidden_states=True"""
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, output_hidden_states=True)
        assert out.hidden_states is not None
        # hidden_states includes: embedding + each layer's output
        # The actual count depends on implementation
        assert len(out.hidden_states) >= base_config.num_hidden_layers + 1

    def test_model_output_attentions(self, base_config):
        """Test model with output_attentions=True"""
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, output_attentions=True)
        assert out.attentions is not None
        assert len(out.attentions) == base_config.num_hidden_layers

    def test_model_use_cache(self, base_config):
        """Test model with use_cache=True"""
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, use_cache=True)
        assert out.past_key_values is not None

    def test_model_with_past_key_values(self, base_config):
        """Test model generation with past_key_values"""
        from transformers.cache_utils import DynamicCache
        model = DeepseekV3Model(base_config)

        # First call
        input_ids = torch.randint(0, base_config.vocab_size, (2, 4))
        out1 = model(input_ids, use_cache=True)

        # Second call with single token
        new_ids = torch.randint(0, base_config.vocab_size, (2, 1))
        out2 = model(new_ids, past_key_values=out1.past_key_values, use_cache=True)
        assert out2.last_hidden_state.shape == (2, 1, base_config.hidden_size)

    def test_model_get_set_embeddings(self, base_config):
        """Test get_input_embeddings and set_input_embeddings"""
        model = DeepseekV3Model(base_config)
        embeds = model.get_input_embeddings()
        assert embeds is model.embed_tokens

        new_embeds = torch.nn.Embedding(base_config.vocab_size, base_config.hidden_size)
        model.set_input_embeddings(new_embeds)
        assert model.get_input_embeddings() is new_embeds

    def test_model_return_dict_false(self, base_config):
        """Test model with return_dict=False"""
        model = DeepseekV3Model(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids, return_dict=False)
        assert isinstance(out, tuple)


class TestDeepseekV3ForCausalLM:
    """Test DeepseekV3ForCausalLM"""

    def test_causallm_forward(self, base_config):
        """Test causal LM forward pass"""
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, 8, base_config.vocab_size)

    def test_causallm_forward_with_labels(self, base_config):
        """Test causal LM with labels for loss computation"""
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.requires_grad

    def test_causallm_backward(self, base_config):
        """Test backward pass through causal LM"""
        model = DeepseekV3ForCausalLM(base_config)
        model.train()
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        out.loss.backward()
        # Check that gradients exist
        assert model.lm_head.weight.grad is not None

    def test_causallm_prepare_inputs_for_generation(self, base_config):
        """Test prepare_inputs_for_generation"""
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        model_inputs = model.prepare_inputs_for_generation(input_ids)
        assert "input_ids" in model_inputs

    def test_causallm_prepare_inputs_with_cache(self, base_config):
        """Test prepare_inputs_for_generation with past_key_values"""
        model = DeepseekV3ForCausalLM(base_config)

        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        # Use tuple format for past_key_values instead of DynamicCache
        # Format: tuple of (key, value) pairs for each layer
        past_key_values = tuple(
            (torch.randn(2, 4, 4, 16), torch.randn(2, 4, 4, 16))
            for _ in range(base_config.num_hidden_layers)
        )

        model_inputs = model.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        # With cache containing 4 tokens and input_ids of length 8,
        # it should slice to keep only unprocessed tokens (8-4=4)
        assert model_inputs["input_ids"].shape[1] == 4

    def test_causallm_get_set_output_embeddings(self, base_config):
        """Test get_output_embeddings and set_output_embeddings"""
        model = DeepseekV3ForCausalLM(base_config)
        out_embeds = model.get_output_embeddings()
        assert out_embeds is model.lm_head

        new_out_embeds = torch.nn.Linear(base_config.hidden_size, base_config.vocab_size, bias=False)
        model.set_output_embeddings(new_out_embeds)
        assert model.get_output_embeddings() is new_out_embeds

    def test_causallm_reorder_cache(self, base_config):
        """Test _reorder_cache for beam search"""
        model = DeepseekV3ForCausalLM(base_config)

        # Create dummy cache
        cache = tuple(
            (torch.randn(2, 4, 8, 16), torch.randn(2, 4, 8, 16))
            for _ in range(base_config.num_hidden_layers)
        )
        beam_idx = torch.tensor([1, 0])

        reordered = model._reorder_cache(cache, beam_idx)
        assert len(reordered) == len(cache)


class TestDeepseekV3ForSequenceClassification:
    """Test DeepseekV3ForSequenceClassification"""

    @pytest.fixture
    def classification_config(self, base_config):
        base_config.num_labels = 3
        return base_config

    def test_sequence_classification_forward(self, classification_config):
        """Test sequence classification forward"""
        model = DeepseekV3ForSequenceClassification(classification_config)
        input_ids = torch.randint(0, classification_config.vocab_size, (2, 8))
        out = model(input_ids)
        assert out.logits.shape == (2, classification_config.num_labels)

    def test_sequence_classification_with_labels(self, classification_config):
        """Test sequence classification with labels"""
        model = DeepseekV3ForSequenceClassification(classification_config)
        input_ids = torch.randint(0, classification_config.vocab_size, (2, 8))
        labels = torch.tensor([0, 2])
        out = model(input_ids, labels=labels)
        assert out.loss is not None

    @pytest.mark.parametrize("num_labels", [1, 2, 5])
    def test_sequence_classification_various_labels(self, base_config, num_labels):
        """Test with various number of labels"""
        base_config.num_labels = num_labels
        model = DeepseekV3ForSequenceClassification(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        if num_labels == 1:
            labels = torch.randn(2)  # Regression
        else:
            labels = torch.randint(0, num_labels, (2,))
        out = model(input_ids, labels=labels)
        assert out.loss is not None

    def test_sequence_classification_no_padding_token(self, base_config):
        """Test error when batch > 1 and no padding token"""
        base_config.num_labels = 3
        base_config.pad_token_id = None
        model = DeepseekV3ForSequenceClassification(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
        with pytest.raises(ValueError, match="batch sizes > 1"):
            model(input_ids)


# =============================================================================
# PreTrainedModel Tests
# =============================================================================

class TestPreTrainedModel:
    """Test DeepseekV3PreTrainedModel base class"""

    def test_init_weights(self, base_config):
        """Test weight initialization"""
        model = DeepseekV3Model(base_config)
        # Check that weights were initialized
        for name, param in model.named_parameters():
            if 'weight' in name:
                assert not torch.all(param == 0)


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.benchmark
class TestPerformanceBenchmarks:
    """Performance benchmark tests"""

    def test_rmsnorm_benchmark(self, benchmark, base_config):
        """Benchmark RMSNorm performance"""
        norm = DeepseekV3RMSNorm(base_config.hidden_size)
        x = torch.randn(8, 128, base_config.hidden_size)
        benchmark(norm, x)

    def test_mlp_benchmark(self, benchmark, base_config):
        """Benchmark MLP performance"""
        mlp = DeepseekV3MLP(base_config)
        x = torch.randn(8, 128, base_config.hidden_size)
        benchmark(mlp, x)

    def test_attention_benchmark(self, benchmark, base_config):
        """Benchmark Attention performance"""
        attn = DeepseekV3Attention(base_config, layer_idx=0)
        x = torch.randn(2, 128, base_config.hidden_size)
        pos_ids = torch.arange(128).unsqueeze(0).expand(2, 128)
        benchmark(attn, x, position_ids=pos_ids)

    def test_decoder_layer_benchmark(self, benchmark, base_config):
        """Benchmark Decoder Layer performance"""
        layer = DeepseekV3DecoderLayer(base_config, layer_idx=0)
        x = torch.randn(2, 128, base_config.hidden_size)
        pos_ids = torch.arange(128).unsqueeze(0).expand(2, 128)
        benchmark(layer, x, position_ids=pos_ids)


# =============================================================================
# Edge Cases and Robustness Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and robustness"""

    def test_very_small_hidden_size(self):
        """Test with very small hidden size"""
        config = DeepseekV3Config(
            vocab_size=64,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            qk_nope_head_dim=2,
            qk_rope_head_dim=2,
            v_head_dim=2,
        )
        model = DeepseekV3ForCausalLM(config)
        input_ids = torch.randint(0, 64, (1, 4))
        out = model(input_ids)
        assert out.logits.shape == (1, 4, 64)

    def test_very_long_sequence(self, base_config):
        """Test with very long sequence"""
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (1, 2048))
        out = model(input_ids)
        assert out.logits.shape == (1, 2048, base_config.vocab_size)

    def test_single_token_input(self, base_config):
        """Test with single token input"""
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (1, 1))
        out = model(input_ids)
        assert out.logits.shape == (1, 1, base_config.vocab_size)

    def test_large_batch_size(self, base_config):
        """Test with large batch size"""
        model = DeepseekV3ForCausalLM(base_config)
        input_ids = torch.randint(0, base_config.vocab_size, (32, 8))
        out = model(input_ids)
        assert out.logits.shape == (32, 8, base_config.vocab_size)

    def test_various_dtypes(self, base_config):
        """Test with various dtypes"""
        for dtype in [torch.float32, torch.float16]:
            model = DeepseekV3ForCausalLM(base_config)
            model = model.to(dtype)
            input_ids = torch.randint(0, base_config.vocab_size, (2, 8))
            out = model(input_ids)
            assert out.logits.dtype == torch.float32  # Logits always float32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
