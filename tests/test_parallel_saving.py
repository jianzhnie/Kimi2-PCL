#!/usr/bin/env python3
"""
测试脚本，用于验证 convert_ckpt_hf2mcore.py 的并行保存优化
"""

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from safetensors.torch import save_file

# Default test configuration
DEFAULT_TEST_CONFIG = {
    'hidden_size': 128,
    'num_attention_heads': 4,
    'num_key_value_heads': 2,
    'qk_head_dim': 128,
    'v_head_dim': 128,
    'vocab_size': 1000,
}


def get_model_dimensions(config: dict = None) -> dict:
    """Calculate model weight dimensions based on config.
    
    Args:
        config: Configuration dictionary, uses DEFAULT_TEST_CONFIG if None
        
    Returns:
        Dictionary with calculated dimensions
    """
    cfg = config or DEFAULT_TEST_CONFIG
    q_head_dim = cfg['qk_head_dim']
    
    return {
        'hidden_size': cfg['hidden_size'],
        'q_head_dim': q_head_dim,
        'expected_q_rows': cfg['num_attention_heads'] * q_head_dim,
        'expected_k_rows': cfg['num_key_value_heads'] * q_head_dim,
        'expected_v_rows': cfg['num_key_value_heads'] * cfg['v_head_dim'],
        'vocab_size': cfg['vocab_size'],
    }


def create_attention_weights(
    layer_idx: int,
    dims: dict,
    hidden_size: int = None,
) -> dict:
    """Create attention weights for a layer.
    
    Args:
        layer_idx: Layer index
        dims: Dimensions dictionary from get_model_dimensions
        hidden_size: Optional override for hidden_size
        
    Returns:
        Dictionary of weight tensors
    """
    hs = hidden_size or dims['hidden_size']
    prefix = f"model.layers.{layer_idx}.self_attn"
    
    return {
        f"{prefix}.q_proj.weight": torch.randn(dims['expected_q_rows'], hs),
        f"{prefix}.k_proj.weight": torch.randn(dims['expected_k_rows'], hs),
        f"{prefix}.v_proj.weight": torch.randn(dims['expected_v_rows'], hs),
        f"{prefix}.o_proj.weight": torch.randn(hs, hs),
    }


def create_dense_mlp_weights(
    layer_idx: int,
    hidden_size: int,
    intermediate_size: int,
) -> dict:
    """Create dense MLP weights for a layer.
    
    Args:
        layer_idx: Layer index
        hidden_size: Hidden size
        intermediate_size: Intermediate size
        
    Returns:
        Dictionary of weight tensors
    """
    prefix = f"model.layers.{layer_idx}.mlp"
    return {
        f"{prefix}.gate_proj.weight": torch.randn(intermediate_size, hidden_size),
        f"{prefix}.up_proj.weight": torch.randn(intermediate_size, hidden_size),
        f"{prefix}.down_proj.weight": torch.randn(hidden_size, intermediate_size),
    }


def create_moe_weights(
    layer_idx: int,
    hidden_size: int,
    num_experts: int,
    moe_ffn_hidden_size: int,
) -> dict:
    """Create MoE weights for a layer.
    
    Args:
        layer_idx: Layer index
        hidden_size: Hidden size
        num_experts: Number of experts
        moe_ffn_hidden_size: MoE FFN hidden size
        
    Returns:
        Dictionary of weight tensors
    """
    weights = {}
    prefix = f"model.layers.{layer_idx}.mlp"
    
    # Router
    weights[f"{prefix}.gate.weight"] = torch.randn(num_experts, hidden_size)
    
    # Shared experts
    weights[f"{prefix}.shared_experts.gate_proj.weight"] = torch.randn(
        moe_ffn_hidden_size, hidden_size)
    weights[f"{prefix}.shared_experts.up_proj.weight"] = torch.randn(
        moe_ffn_hidden_size, hidden_size)
    weights[f"{prefix}.shared_experts.down_proj.weight"] = torch.randn(
        hidden_size, moe_ffn_hidden_size)
    
    # Individual experts
    for expert_idx in range(num_experts):
        expert_prefix = f"{prefix}.experts.{expert_idx}"
        weights[f"{expert_prefix}.gate_proj.weight"] = torch.randn(
            moe_ffn_hidden_size, hidden_size)
        weights[f"{expert_prefix}.up_proj.weight"] = torch.randn(
            moe_ffn_hidden_size, hidden_size)
        weights[f"{expert_prefix}.down_proj.weight"] = torch.randn(
            hidden_size, moe_ffn_hidden_size)
    
    return weights


def create_layer_norm_weights(layer_idx: int, hidden_size: int) -> dict:
    """Create layer norm weights for a layer."""
    return {
        f"model.layers.{layer_idx}.input_layernorm.weight": torch.randn(hidden_size),
        f"model.layers.{layer_idx}.post_attention_layernorm.weight": torch.randn(hidden_size),
    }


def create_common_weights(vocab_size: int, hidden_size: int) -> dict:
    """Create common weights (embeddings, head, norm)."""
    return {
        "model.embed_tokens.weight": torch.randn(vocab_size, hidden_size),
        "model.norm.weight": torch.randn(hidden_size),
        "lm_head.weight": torch.randn(vocab_size, hidden_size),
    }


def save_hf_model(
    model_dir: str,
    weights: dict,
    config: dict,
) -> None:
    """Save weights and config as HF model format.
    
    Args:
        model_dir: Directory to save the model
        weights: Dictionary of weight tensors
        config: Model configuration dictionary
    """
    os.makedirs(model_dir, exist_ok=True)
    
    # Save config
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        json.dump(config, f)
    
    # Save weights
    save_file(weights, os.path.join(model_dir, "model-00001-of-00001.safetensors"))
    
    # Create index
    weight_map = {k: "model-00001-of-00001.safetensors" for k in weights.keys()}
    index_content = {
        "metadata": {"total_size": 1000000},
        "weight_map": weight_map
    }
    with open(os.path.join(model_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index_content, f)


def create_dummy_hf_model(model_dir: str) -> None:
    """Create a simple dense HF model for testing."""
    dims = get_model_dimensions()
    hidden_size = dims['hidden_size']
    
    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "kimi_k2",
        "hidden_size": hidden_size,
        "num_hidden_layers": 4,
        "num_attention_heads": DEFAULT_TEST_CONFIG['num_attention_heads'],
        "num_key_value_heads": DEFAULT_TEST_CONFIG['num_key_value_heads'],
        "intermediate_size": 256,
        "vocab_size": dims['vocab_size'],
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16"
    }
    
    # Create weights for all 4 layers
    weights = {}
    for layer_idx in range(4):
        weights.update(create_layer_norm_weights(layer_idx, hidden_size))
        weights.update(create_attention_weights(layer_idx, dims))
        weights.update(create_dense_mlp_weights(layer_idx, hidden_size, 256))
    
    weights.update(create_common_weights(dims['vocab_size'], hidden_size))
    save_hf_model(model_dir, weights, config)


def create_moe_hf_model(model_dir: str) -> None:
    """Create an MoE HF model for testing."""
    dims = get_model_dimensions()
    hidden_size = dims['hidden_size']
    num_experts = 8
    moe_ffn_hidden_size = 512
    first_k_dense_replace = 2
    
    config = {
        "architectures": ["DeepseekV3ForCausalLM"],
        "model_type": "kimi_k2",
        "hidden_size": hidden_size,
        "num_hidden_layers": 4,
        "num_attention_heads": DEFAULT_TEST_CONFIG['num_attention_heads'],
        "num_key_value_heads": DEFAULT_TEST_CONFIG['num_key_value_heads'],
        "intermediate_size": 256,
        "moe_intermediate_size": moe_ffn_hidden_size,
        "num_experts": num_experts,
        "vocab_size": dims['vocab_size'],
        "max_position_embeddings": 2048,
        "torch_dtype": "bfloat16"
    }
    
    weights = {}
    
    # Layer 0 and 1: dense
    for layer_idx in range(first_k_dense_replace):
        weights.update(create_layer_norm_weights(layer_idx, hidden_size))
        weights.update(create_attention_weights(layer_idx, dims))
        weights.update(create_dense_mlp_weights(layer_idx, hidden_size, 256))
    
    # Layer 2 and 3: MoE
    for layer_idx in range(first_k_dense_replace, 4):
        weights.update(create_layer_norm_weights(layer_idx, hidden_size))
        weights.update(create_attention_weights(layer_idx, dims))
        weights.update(create_moe_weights(
            layer_idx, hidden_size, num_experts, moe_ffn_hidden_size))
    
    weights.update(create_common_weights(dims['vocab_size'], hidden_size))
    save_hf_model(model_dir, weights, config)


def get_convert_script_path() -> str:
    """Get the path to the convert script."""
    # Script is in utils directory, not tests directory
    repo_root = os.path.dirname(os.path.dirname(__file__))
    script_path = os.path.join(repo_root, "utils", "convert_ckpt_hf2mcore.py")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Convert script not found: {script_path}")
    return script_path


def run_convert_command(cmd: list, timeout: int = 300) -> tuple[bool, subprocess.CompletedProcess]:
    """Run convert command and return success status and result.
    
    Args:
        cmd: Command list
        timeout: Timeout in seconds
        
    Returns:
        Tuple of (success, result)
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.returncode == 0, result
    except subprocess.TimeoutExpired:
        return False, None
    except Exception as e:
        print(f"Error running command: {e}")
        return False, None


def count_output_files(save_dir: str) -> int:
    """Count model_optim_rng.pt files in output directory."""
    count = 0
    for root, dirs, files in os.walk(save_dir):
        for f in files:
            if f == "model_optim_rng.pt":
                count += 1
    return count


def print_directory_tree(directory: str) -> None:
    """Print directory tree structure."""
    for root, dirs, files in os.walk(directory):
        level = root.replace(directory, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for f in files:
            print(f"{subindent}{f}")


def test_parallel_saving() -> bool:
    """Test parallel saving functionality for dense model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        hf_model_dir = os.path.join(temp_dir, "hf_model")
        save_dir = os.path.join(temp_dir, "mcore_output")
        
        print(f"Creating test HF model at: {hf_model_dir}")
        create_dummy_hf_model(hf_model_dir)
        
        script_path = get_convert_script_path()
        
        cmd = [
            sys.executable, script_path,
            "--load-dir", hf_model_dir,
            "--save-dir", save_dir,
            "--num-layers", "4",
            "--target-tensor-parallel-size", "2",
            "--target-pipeline-parallel-size", "2",
            "--target-expert-parallel-size", "1",
            "--num-experts", "1",
            "--num-attention-heads", "4",
            "--num-query-groups", "2",
            "--qk-head-dim", "128",
            "--v-head-dim", "128",
            "--hidden-size", "128",
            "--ffn-hidden-size", "256",
            "--vocab-size", "1000",
            "--save-workers", "4",
            "--cast-dtype", "fp32",
            "--first-k-dense-replace", "4",
        ]
        
        print("Running conversion command:")
        print(" ".join(cmd))
        
        start_time = time.time()
        success, result = run_convert_command(cmd)
        end_time = time.time()
        
        print(f"\nConversion took: {end_time - start_time:.2f} seconds")
        
        if result:
            print(f"Return code: {result.returncode}")
            if result.stdout:
                print("STDOUT:")
                print(result.stdout)
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
        
        if not success:
            print("Conversion failed!")
            return False
        
        if os.path.exists(save_dir):
            print(f"\nOutput directory contents:")
            print_directory_tree(save_dir)
            
            file_count = count_output_files(save_dir)
            print(f"\nFound {file_count} weight files")
            return file_count > 0
        else:
            print("Output directory does not exist!")
            return False


def test_moe_parallel_saving() -> bool:
    """Test parallel saving functionality for MoE model."""
    with tempfile.TemporaryDirectory() as temp_dir:
        hf_model_dir = os.path.join(temp_dir, "hf_model")
        save_dir = os.path.join(temp_dir, "mcore_output")
        
        print(f"\n=== Testing MoE Parallel Saving ===")
        print(f"Creating test HF model at: {hf_model_dir}")
        
        create_moe_hf_model(hf_model_dir)
        script_path = get_convert_script_path()
        
        cmd = [
            sys.executable, script_path,
            "--load-dir", hf_model_dir,
            "--save-dir", save_dir,
            "--num-layers", "4",
            "--target-tensor-parallel-size", "2",
            "--target-pipeline-parallel-size", "2",
            "--target-expert-parallel-size", "2",
            "--num-experts", "8",
            "--num-attention-heads", "4",
            "--num-query-groups", "2",
            "--qk-head-dim", "128",
            "--v-head-dim", "128",
            "--hidden-size", "128",
            "--ffn-hidden-size", "256",
            "--moe-ffn-hidden-size", "512",
            "--vocab-size", "1000",
            "--save-workers", "4",
            "--cast-dtype", "fp32",
            "--first-k-dense-replace", "2",
        ]
        
        print("Running MoE conversion command:")
        print(" ".join(cmd))
        
        start_time = time.time()
        success, result = run_convert_command(cmd)
        end_time = time.time()
        
        print(f"\nMoE conversion took: {end_time - start_time:.2f} seconds")
        
        if result:
            print(f"Return code: {result.returncode}")
            if result.stderr:
                print("STDERR (last 1500 chars):")
                print(result.stderr[-1500:])
        
        if success and os.path.exists(save_dir):
            file_count = count_output_files(save_dir)
            print(f"MoE test: Generated {file_count} weight files")
            return file_count > 0
        else:
            print("MoE test failed")
            return False


if __name__ == "__main__":
    print("Testing convert_ckpt_hf2mcore.py parallel saving optimization...")
    
    success1 = test_parallel_saving()
    success2 = test_moe_parallel_saving()
    
    if success1 and success2:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed!")
        sys.exit(1)
