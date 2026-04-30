"""
Numerical alignment test: DeepseekV3Attention (eager) vs DeepseekV3FlashAttention.

Verifies that both implementations produce identical outputs for:
  1. Prefill without padding
  2. Prefill with padding (varlen path)
  3. Decode with KV cache

Run on a machine with CUDA and flash_attn installed:
    python -m tests.test_flash_attn_alignment
"""
import sys
import os
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from models.modeling_deepseek import (
    DeepseekV3Attention,
    DeepseekV3FlashAttention,
    DeepseekV3RMSNorm,
    apply_rotary_pos_emb,
    _get_unpad_data,
)
from models.configuration_deepseek import DeepseekV3Config


def make_config(**overrides):
    defaults = dict(
        hidden_size=256,
        num_attention_heads=8,
        num_query_groups=2,
        kv_channels=32,
        attention_bias=False,
        attention_dropout=0.0,
        rms_norm_eps=1e-6,
        rope_scaling=None,
        max_position_embeddings=512,
        rope_theta=10000.0,
        # MoE fields (not used by attention, but required by config)
        n_routed_experts=None,
    )
    defaults.update(overrides)
    return DeepseekV3Config(**defaults)


def copy_weights(eager: DeepseekV3Attention, flash: DeepseekV3FlashAttention):
    """Copy all weights from eager to flash attention module."""
    flash.load_state_dict(eager.state_dict(), strict=True)


@torch.no_grad()
def test_prefill_no_padding():
    """Test 1: Prefill without padding — standard flash_attn_func path."""
    config = make_config()
    layer_idx = 0
    eager = DeepseekV3Attention(config, layer_idx).cuda().eval()
    flash = DeepseekV3FlashAttention(config, layer_idx).cuda().eval()
    copy_weights(eager, flash)

    bsz, seq_len = 2, 64
    hidden = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.float16)
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

    # Build RoPE cos/sin (simplified — just use a basic rotary embedding)
    dim = config.kv_channels
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    t = torch.arange(seq_len, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos, sin = emb.cos().to(torch.float16), emb.sin().to(torch.float16)

    # Eager: needs 4D causal mask
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        None, (bsz, seq_len), hidden, 0,
    )

    eager_out, _, _ = eager(
        hidden,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )

    # Flash: attention_mask=None (no padding), causal handled by kernel
    flash_out, _, _ = flash(
        hidden,
        attention_mask=None,
        position_ids=position_ids,
        position_embeddings=(cos, sin),
    )

    max_diff = (eager_out - flash_out).abs().max().item()
    mean_diff = (eager_out - flash_out).abs().mean().item()
    passed = torch.allclose(eager_out, flash_out, atol=1e-2, rtol=1e-2)
    print(f"[Prefill no-padding]  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  passed={passed}")
    return passed


@torch.no_grad()
def test_prefill_with_padding():
    """Test 2: Prefill with right-padded sequences — varlen path."""
    config = make_config()
    layer_idx = 0
    eager = DeepseekV3Attention(config, layer_idx).cuda().eval()
    flash = DeepseekV3FlashAttention(config, layer_idx).cuda().eval()
    copy_weights(eager, flash)

    bsz, seq_len = 4, 32
    hidden = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.float16)
    position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

    # Create attention mask: right-padded sequences of different lengths
    attention_mask = torch.ones(bsz, seq_len, device="cuda", dtype=torch.long)
    attention_mask[0, 20:] = 0  # sequence 0 has 20 valid tokens
    attention_mask[1, 28:] = 0  # sequence 1 has 28 valid tokens
    attention_mask[2, 15:] = 0  # sequence 2 has 15 valid tokens
    # sequence 3 is fully valid (32 tokens)

    # Build RoPE
    dim = config.kv_channels
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
    t = torch.arange(seq_len, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_cached = emb.cos().to(torch.float16)
    sin_cached = emb.sin().to(torch.float16)

    # Eager: 4D causal + padding mask
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        attention_mask, (bsz, seq_len), hidden, 0,
    )
    eager_out, _, _ = eager(
        hidden,
        attention_mask=causal_mask,
        position_ids=position_ids,
        position_embeddings=(cos_cached, sin_cached),
    )

    # Flash: 2D mask with 0s → varlen path
    flash_out, _, _ = flash(
        hidden,
        attention_mask=attention_mask,
        position_ids=position_ids,
        position_embeddings=(cos_cached, sin_cached),
    )

    # Compare only at valid (non-padded) positions
    valid_mask = attention_mask.bool().unsqueeze(-1).expand_as(eager_out)
    eager_valid = eager_out[valid_mask]
    flash_valid = flash_out[valid_mask]

    max_diff = (eager_valid - flash_valid).abs().max().item()
    mean_diff = (eager_valid - flash_valid).abs().mean().item()
    passed = torch.allclose(eager_valid, flash_valid, atol=1e-2, rtol=1e-2)
    print(f"[Prefill w/ padding]  max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  passed={passed}")
    return passed


@torch.no_grad()
def test_decode_with_cache():
    """Test 3: Decode step with KV cache — single token query."""
    config = make_config()
    layer_idx = 0
    eager = DeepseekV3Attention(config, layer_idx).cuda().eval()
    flash = DeepseekV3FlashAttention(config, layer_idx).cuda().eval()
    copy_weights(eager, flash)

    bsz = 2
    prefill_len = 16
    decode_len = 1

    dim = config.kv_channels
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))

    from transformers.cache_utils import DynamicCache

    # --- Prefill both ---
    hidden_prefill = torch.randn(bsz, prefill_len, config.hidden_size, device="cuda", dtype=torch.float16)
    pos_prefill = torch.arange(prefill_len, device="cuda").unsqueeze(0).expand(bsz, -1)
    t = torch.arange(prefill_len, device="cuda", dtype=inv_freq.dtype)
    freqs = torch.outer(t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    cos_pf, sin_pf = emb.cos().to(torch.float16), emb.sin().to(torch.float16)

    eager_cache = DynamicCache()
    flash_cache = DynamicCache()

    _, _, eager_cache = eager(
        hidden_prefill,
        attention_mask=None,
        position_ids=pos_prefill,
        past_key_value=eager_cache,
        use_cache=True,
        position_embeddings=(cos_pf, sin_pf),
    )
    _, _, flash_cache = flash(
        hidden_prefill,
        attention_mask=None,
        position_ids=pos_prefill,
        past_key_value=flash_cache,
        use_cache=True,
        position_embeddings=(cos_pf, sin_pf),
    )

    # --- Decode step ---
    hidden_decode = torch.randn(bsz, decode_len, config.hidden_size, device="cuda", dtype=torch.float16)
    pos_decode = torch.tensor([[prefill_len]], device="cuda").expand(bsz, -1)
    total_len = prefill_len + decode_len
    t2 = torch.arange(total_len, device="cuda", dtype=inv_freq.dtype)
    freqs2 = torch.outer(t2, inv_freq)
    emb2 = torch.cat((freqs2, freqs2), dim=-1)
    cos_dec, sin_dec = emb2.cos().to(torch.float16), emb2.sin().to(torch.float16)

    # Eager decode
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    causal_mask = _prepare_4d_causal_attention_mask(
        None, (bsz, decode_len), hidden_decode, prefill_len,
    )
    eager_out, _, _ = eager(
        hidden_decode,
        attention_mask=causal_mask,
        position_ids=pos_decode,
        past_key_value=eager_cache,
        position_embeddings=(cos_dec, sin_dec),
    )

    # Flash decode
    flash_out, _, _ = flash(
        hidden_decode,
        attention_mask=None,
        position_ids=pos_decode,
        past_key_value=flash_cache,
        position_embeddings=(cos_dec, sin_dec),
    )

    max_diff = (eager_out - flash_out).abs().max().item()
    mean_diff = (eager_out - flash_out).abs().mean().item()
    passed = torch.allclose(eager_out, flash_out, atol=1e-2, rtol=1e-2)
    print(f"[Decode w/ cache]     max_diff={max_diff:.6f}  mean_diff={mean_diff:.6f}  passed={passed}")
    return passed


@torch.no_grad()
def test_different_gqa_ratios():
    """Test 4: Various GQA head ratios."""
    results = []
    for num_heads, kv_groups in [(8, 2), (8, 4), (16, 2), (16, 4), (16, 8)]:
        config = make_config(
            num_attention_heads=num_heads,
            num_query_groups=kv_groups,
            kv_channels=64,
            hidden_size=num_heads * 64,
        )
        layer_idx = 0
        eager = DeepseekV3Attention(config, layer_idx).cuda().eval()
        flash = DeepseekV3FlashAttention(config, layer_idx).cuda().eval()
        copy_weights(eager, flash)

        bsz, seq_len = 2, 32
        hidden = torch.randn(bsz, seq_len, config.hidden_size, device="cuda", dtype=torch.float16)
        position_ids = torch.arange(seq_len, device="cuda").unsqueeze(0).expand(bsz, -1)

        dim = config.kv_channels
        inv_freq = 1.0 / (10000.0 ** (torch.arange(0, dim, 2, device="cuda").float() / dim))
        t = torch.arange(seq_len, device="cuda", dtype=inv_freq.dtype)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(torch.float16), emb.sin().to(torch.float16)

        from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
        causal_mask = _prepare_4d_causal_attention_mask(
            None, (bsz, seq_len), hidden, 0,
        )
        eager_out, _, _ = eager(hidden, attention_mask=causal_mask,
                                position_ids=position_ids,
                                position_embeddings=(cos, sin))
        flash_out, _, _ = flash(hidden, attention_mask=None,
                                position_ids=position_ids,
                                position_embeddings=(cos, sin))

        max_diff = (eager_out - flash_out).abs().max().item()
        passed = torch.allclose(eager_out, flash_out, atol=1e-2, rtol=1e-2)
        results.append((num_heads, kv_groups, max_diff, passed))
        print(f"  GQA {num_heads}/{kv_groups}: max_diff={max_diff:.6f}  passed={passed}")

    all_passed = all(r[3] for r in results)
    print(f"[GQA ratios]          all_passed={all_passed}")
    return all_passed


def main():
    print("=" * 60)
    print("DeepseekV3 Flash Attention Alignment Test")
    print("=" * 60)
    results = []
    results.append(test_prefill_no_padding())
    results.append(test_prefill_with_padding())
    results.append(test_decode_with_cache())
    results.append(test_different_gqa_ratios())

    print("=" * 60)
    if all(results):
        print("ALL TESTS PASSED")
    else:
        print(f"SOME TESTS FAILED: {sum(results)}/{len(results)} passed")
    print("=" * 60)


if __name__ == "__main__":
    main()
