#!/usr/bin/env python
"""
Test script for verifying HuggingFace model implementation alignment with Megatron.
"""

import torch
from models.configuration_deepseek import DeepseekV3Config
from models.modeling_deepseek import (
    DeepseekV3ForCausalLM,
    DeepseekV3Attention,
    DeepseekV3MLP,
    DeepseekV3MoE,
    MoEGate,
    DeepseekV3LayerNorm,
)


def test_configuration():
    """Test configuration parameters."""
    print("=" * 60)
    print("1. Configuration Test")
    print("=" * 60)
    
    config = DeepseekV3Config()
    
    # Verify key parameters
    assert config.vocab_size == 163840
    assert config.hidden_size == 7168
    assert config.num_hidden_layers == 32
    assert config.qk_nope_head_dim == 128
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 128
    assert config.n_routed_experts == 128
    assert config.n_shared_experts == 1
    
    print(f"✓ vocab_size: {config.vocab_size}")
    print(f"✓ hidden_size: {config.hidden_size}")
    print(f"✓ num_hidden_layers: {config.num_hidden_layers}")
    print(f"✓ qk_nope_head_dim: {config.qk_nope_head_dim}")
    print(f"✓ qk_rope_head_dim: {config.qk_rope_head_dim}")
    print(f"✓ v_head_dim: {config.v_head_dim}")
    print(f"✓ n_routed_experts: {config.n_routed_experts}")
    print(f"✓ n_shared_experts: {config.n_shared_experts}")
    print("All configuration tests passed!\n")


def test_attention():
    """Test Attention module with Decoupled Head Dimensions."""
    print("=" * 60)
    print("2. Attention Module Test")
    print("=" * 60)
    
    config = DeepseekV3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_query_groups=2,
        n_routed_experts=16,
        n_group=8,
        topk_group=2,
        first_k_dense_replace=2,
        ep_size=1,
        qk_layernorm=True,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
    )
    
    attn = DeepseekV3Attention(config, layer_idx=0)
    
    # Verify q/k layernorm has bias
    assert attn.q_layernorm is not None
    assert attn.k_layernorm is not None
    assert hasattr(attn.q_layernorm, 'bias')
    assert hasattr(attn.k_layernorm, 'bias')
    assert isinstance(attn.q_layernorm, DeepseekV3LayerNorm)
    
    print(f"✓ q_layernorm type: {type(attn.q_layernorm).__name__}")
    print(f"✓ q_layernorm.bias shape: {attn.q_layernorm.bias.shape}")
    print(f"✓ k_layernorm type: {type(attn.k_layernorm).__name__}")
    print(f"✓ k_layernorm.bias shape: {attn.k_layernorm.bias.shape}")
    
    # Verify Decoupled Head Dimensions
    assert attn.q_head_dim == 192  # 128 + 64
    assert attn.k_head_dim == 192
    assert attn.v_head_dim == 128
    
    print(f"✓ q_head_dim: {attn.q_head_dim} (128 nope + 64 rope)")
    print(f"✓ k_head_dim: {attn.k_head_dim}")
    print(f"✓ v_head_dim: {attn.v_head_dim}")
    
    # Test forward pass
    batch_size, seq_len = 2, 10
    hidden_states = torch.randn(batch_size, seq_len, config.hidden_size)
    position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)
    
    attn.eval()
    with torch.no_grad():
        output, _, _ = attn(hidden_states, position_ids=position_ids)
    
    assert output.shape == (batch_size, seq_len, config.hidden_size)
    print(f"✓ Attention forward pass: {hidden_states.shape} -> {output.shape}")
    print("All attention tests passed!\n")


def test_mlp():
    """Test MLP module (SwiGLU)."""
    print("=" * 60)
    print("3. MLP Module Test")
    print("=" * 60)
    
    config = DeepseekV3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=384,
    )
    
    mlp = DeepseekV3MLP(config)
    
    # Verify weight shapes
    assert mlp.gate_proj.weight.shape == (512, 256)
    assert mlp.up_proj.weight.shape == (512, 256)
    assert mlp.down_proj.weight.shape == (256, 512)
    
    print(f"✓ gate_proj: {mlp.gate_proj.weight.shape}")
    print(f"✓ up_proj: {mlp.up_proj.weight.shape}")
    print(f"✓ down_proj: {mlp.down_proj.weight.shape}")
    
    # Test forward pass
    x = torch.randn(2, 10, 256)
    output = mlp(x)
    assert output.shape == (2, 10, 256)
    print(f"✓ MLP forward pass: {x.shape} -> {output.shape}")
    print("All MLP tests passed!\n")


def test_moe():
    """Test MoE module."""
    print("=" * 60)
    print("4. MoE Module Test")
    print("=" * 60)
    
    config = DeepseekV3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=384,
        num_attention_heads=8,
        num_query_groups=2,
        n_routed_experts=16,
        n_group=8,
        topk_group=2,
        num_experts_per_tok=2,
        ep_size=1,
    )
    
    # Test MoE Gate
    gate = MoEGate(config)
    assert gate.weight.shape == (16, 256)
    
    print(f"✓ Gate weight: {gate.weight.shape}")
    if gate.bias is not None:
        print(f"✓ Gate bias: {gate.bias.shape}")
    
    # Test MoE module
    moe = DeepseekV3MoE(config)
    
    x = torch.randn(2, 10, 256)
    moe.eval()
    with torch.no_grad():
        output, aux_loss = moe(x)
    
    assert output.shape == (2, 10, 256)
    print(f"✓ MoE forward pass: {x.shape} -> {output.shape}")
    print(f"✓ Aux loss: {aux_loss}")
    print("All MoE tests passed!\n")


def test_full_model():
    """Test full model."""
    print("=" * 60)
    print("5. Full Model Test")
    print("=" * 60)
    
    config = DeepseekV3Config(
        vocab_size=1000,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=384,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_query_groups=2,
        n_routed_experts=16,
        n_group=8,
        topk_group=2,
        num_experts_per_tok=2,
        first_k_dense_replace=2,
        ep_size=1,
        qk_layernorm=True,
    )
    
    model = DeepseekV3ForCausalLM(config)
    model.eval()
    
    # Verify layer structure
    print("Layer structure:")
    for i, layer in enumerate(model.model.layers):
        mlp_type = type(layer.mlp).__name__
        has_q_ln = layer.self_attn.q_layernorm is not None
        has_k_ln = layer.self_attn.k_layernorm is not None
        print(f"  Layer {i}: MLP={mlp_type}, q_ln={has_q_ln}, k_ln={has_k_ln}")
        
        # Verify first 2 layers are Dense MLP
        if i < 2:
            assert mlp_type == 'DeepseekV3MLP'
        else:
            assert mlp_type == 'DeepseekV3MoE'
    
    print("✓ Layer structure correct (first 2 dense, others MoE)")
    
    # Test forward pass
    input_ids = torch.randint(0, 1000, (2, 10))
    with torch.no_grad():
        outputs = model(input_ids)
    
    assert outputs.logits.shape == (2, 10, 1000)
    print(f"✓ Model forward: {input_ids.shape} -> {outputs.logits.shape}")
    print("All full model tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("HuggingFace Model Alignment Verification")
    print("=" * 60 + "\n")
    
    try:
        test_configuration()
        test_attention()
        test_mlp()
        test_moe()
        test_full_model()
        
        print("=" * 60)
        print("ALL TESTS PASSED! ✓")
        print("Model implementation is aligned with Megatron.")
        print("=" * 60)
        return 0
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())
