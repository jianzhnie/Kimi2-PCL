#!/usr/bin/env python3
"""
Extract weight information from DeepseekV3 model definition.
Generates a JSON file containing weight names, shapes, and dtypes.

This script analyzes the model architecture definition to compute weight shapes
without instantiating the full model (which would be memory-intensive for large models).
"""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def load_config_module():
    """Load configuration module."""
    models_dir = Path(__file__).parent.parent / 'models'

    # Create a fake models package
    if 'models' not in sys.modules:
        models_pkg = type(sys)('models')
        models_pkg.__path__ = [str(models_dir)]
        sys.modules['models'] = models_pkg

    # Load configuration_deepseek module
    spec_config = importlib.util.spec_from_file_location(
        'models.configuration_deepseek',
        models_dir / 'configuration_deepseek.py')
    configuration_deepseek = importlib.util.module_from_spec(spec_config)
    sys.modules['models.configuration_deepseek'] = configuration_deepseek
    spec_config.loader.exec_module(configuration_deepseek)

    return configuration_deepseek.DeepseekV3Config


DeepseekV3Config = load_config_module()


def compute_weight_shape(name: str,
                         config: DeepseekV3Config) -> Tuple[List[int], str]:
    """
    Compute the shape of a weight based on its name and config.

    Args:
        name: Weight name (e.g., "model.embed_tokens.weight")
        config: Model configuration

    Returns:
        Tuple of (shape list, dtype string)
    """
    dtype = 'float16'  # Default dtype

    # Embedding weights
    if name == 'model.embed_tokens.weight':
        return [config.vocab_size, config.hidden_size], dtype

    # Final norm weight
    if name == 'model.norm.weight':
        return [config.hidden_size], dtype

    # LM head weight
    if name == 'lm_head.weight':
        return [config.vocab_size, config.hidden_size], dtype

    # Parse layer-specific weights
    if name.startswith('model.layers.'):
        parts = name.split('.')
        layer_idx = int(parts[2])

        # Determine if this layer uses MoE
        is_moe_layer = (config.n_routed_experts is not None
                        and layer_idx >= config.first_k_dense_replace
                        and layer_idx % config.moe_layer_freq == 0)

        # Input layer norm
        if 'input_layernorm.weight' in name:
            return [config.hidden_size], dtype

        # Post attention layer norm
        if 'post_attention_layernorm.weight' in name:
            return [config.hidden_size], dtype

        # Attention weights
        if 'self_attn.q_proj.weight' in name:
            q_head_dim = getattr(config, 'qk_nope_head_dim', 128) + getattr(
                config, 'qk_rope_head_dim', 64)
            return [
                config.num_attention_heads * q_head_dim, config.hidden_size
            ], dtype

        if 'self_attn.k_proj.weight' in name:
            k_head_dim = getattr(config, 'qk_nope_head_dim', 128) + getattr(
                config, 'qk_rope_head_dim', 64)
            return [
                config.num_key_value_heads * k_head_dim, config.hidden_size
            ], dtype

        if 'self_attn.v_proj.weight' in name:
            v_head_dim = getattr(config, 'v_head_dim', 128)
            return [
                config.num_key_value_heads * v_head_dim, config.hidden_size
            ], dtype

        if 'self_attn.o_proj.weight' in name:
            v_head_dim = getattr(config, 'v_head_dim', 128)
            return [
                config.hidden_size, config.num_attention_heads * v_head_dim
            ], dtype

        # Q/K layer norms (optional, based on qk_layernorm config)
        if 'self_attn.q_layernorm.weight' in name:
            q_head_dim = getattr(config, 'qk_nope_head_dim', 128) + getattr(
                config, 'qk_rope_head_dim', 64)
            return [q_head_dim], dtype

        if 'self_attn.k_layernorm.weight' in name:
            k_head_dim = getattr(config, 'qk_nope_head_dim', 128) + getattr(
                config, 'qk_rope_head_dim', 64)
            return [k_head_dim], dtype

        # MoE weights
        if is_moe_layer:
            # Gate weights
            if 'mlp.gate.weight' in name:
                return [config.n_routed_experts, config.hidden_size], dtype

            if 'mlp.gate.bias' in name:
                return [config.n_routed_experts], dtype

            if 'mlp.gate.e_score_correction_bias' in name:
                return [config.n_routed_experts], dtype

            # Expert weights (n_routed_experts experts)
            if 'mlp.experts.' in name and '.gate_proj.weight' in name:
                return [config.moe_intermediate_size,
                        config.hidden_size], dtype

            if 'mlp.experts.' in name and '.up_proj.weight' in name:
                return [config.moe_intermediate_size,
                        config.hidden_size], dtype

            if 'mlp.experts.' in name and '.down_proj.weight' in name:
                return [config.hidden_size,
                        config.moe_intermediate_size], dtype

            # Shared experts
            if 'mlp.shared_experts.gate_proj.weight' in name:
                intermediate_size = config.moe_intermediate_size * config.n_shared_experts
                return [intermediate_size, config.hidden_size], dtype

            if 'mlp.shared_experts.up_proj.weight' in name:
                intermediate_size = config.moe_intermediate_size * config.n_shared_experts
                return [intermediate_size, config.hidden_size], dtype

            if 'mlp.shared_experts.down_proj.weight' in name:
                intermediate_size = config.moe_intermediate_size * config.n_shared_experts
                return [config.hidden_size, intermediate_size], dtype

        # Dense MLP weights (for non-MoE layers)
        else:
            if 'mlp.gate_proj.weight' in name:
                return [config.intermediate_size, config.hidden_size], dtype

            if 'mlp.up_proj.weight' in name:
                return [config.intermediate_size, config.hidden_size], dtype

            if 'mlp.down_proj.weight' in name:
                return [config.hidden_size, config.intermediate_size], dtype

    # Unknown weight
    return None, dtype


def generate_weight_map(config: DeepseekV3Config) -> Dict:
    """
    Generate weight map from model configuration.

    Args:
        config: Model configuration

    Returns:
        Dictionary containing metadata and weight_map
    """
    weight_map = {}
    total_size = 0

    # Helper function to add weight
    def add_weight(name: str, shape: List[int], dtype: str = 'float16'):
        nonlocal total_size

        # Calculate size (2 bytes for float16, 4 for float32/bfloat16)
        if dtype == 'float16':
            element_size = 2
        elif dtype in ['float32', 'bfloat16']:
            element_size = 4
        else:
            element_size = 2

        numel = 1
        for dim in shape:
            numel *= dim

        size_bytes = numel * element_size
        total_size += size_bytes

        weight_map[name] = {
            'shape': shape,
            'dtype': dtype,
        }

    # Embedding weights
    add_weight('model.embed_tokens.weight',
               [config.vocab_size, config.hidden_size])

    # Process each layer
    for layer_idx in range(config.num_hidden_layers):
        prefix = f"model.layers.{layer_idx}"

        # Determine if this layer uses MoE
        is_moe_layer = (config.n_routed_experts is not None
                        and layer_idx >= config.first_k_dense_replace
                        and layer_idx % config.moe_layer_freq == 0)

        # Layer norms
        add_weight(f"{prefix}.input_layernorm.weight", [config.hidden_size])
        add_weight(f"{prefix}.post_attention_layernorm.weight",
                   [config.hidden_size])

        # Attention weights (standard GQA)
        head_dim = config.hidden_size // config.num_attention_heads

        # Q, K, V projections
        add_weight(f"{prefix}.self_attn.q_proj.weight",
                   [config.num_attention_heads * head_dim, config.hidden_size])
        add_weight(f"{prefix}.self_attn.k_proj.weight",
                   [config.num_key_value_heads * head_dim, config.hidden_size])
        add_weight(f"{prefix}.self_attn.v_proj.weight",
                   [config.num_key_value_heads * head_dim, config.hidden_size])
        add_weight(f"{prefix}.self_attn.o_proj.weight",
                   [config.hidden_size, config.num_attention_heads * head_dim])

        # Optional Q/K layer norms
        if getattr(config, 'qk_layernorm', False):
            add_weight(f"{prefix}.self_attn.q_layernorm.weight", [head_dim])
            add_weight(f"{prefix}.self_attn.k_layernorm.weight", [head_dim])

        # MLP / MoE weights
        if is_moe_layer:
            # Gate weights
            add_weight(f"{prefix}.mlp.gate.weight",
                       [config.n_routed_experts, config.hidden_size])

            if getattr(config, 'moe_router_enable_expert_bias', False):
                add_weight(f"{prefix}.mlp.gate.bias",
                           [config.n_routed_experts])

            if config.topk_method == 'noaux_tc':
                add_weight(f"{prefix}.mlp.gate.e_score_correction_bias",
                           [config.n_routed_experts])

            # Expert weights
            for expert_idx in range(config.n_routed_experts):
                expert_prefix = f"{prefix}.mlp.experts.{expert_idx}"
                add_weight(f"{expert_prefix}.gate_proj.weight",
                           [config.moe_intermediate_size, config.hidden_size])
                add_weight(f"{expert_prefix}.up_proj.weight",
                           [config.moe_intermediate_size, config.hidden_size])
                add_weight(f"{expert_prefix}.down_proj.weight",
                           [config.hidden_size, config.moe_intermediate_size])

            # Shared experts
            if config.n_shared_experts is not None and config.n_shared_experts > 0:
                shared_intermediate = config.moe_intermediate_size * config.n_shared_experts
                shared_prefix = f"{prefix}.mlp.shared_experts"
                add_weight(f"{shared_prefix}.gate_proj.weight",
                           [shared_intermediate, config.hidden_size])
                add_weight(f"{shared_prefix}.up_proj.weight",
                           [shared_intermediate, config.hidden_size])
                add_weight(f"{shared_prefix}.down_proj.weight",
                           [config.hidden_size, shared_intermediate])
        else:
            # Dense MLP
            add_weight(f"{prefix}.mlp.gate_proj.weight",
                       [config.intermediate_size, config.hidden_size])
            add_weight(f"{prefix}.mlp.up_proj.weight",
                       [config.intermediate_size, config.hidden_size])
            add_weight(f"{prefix}.mlp.down_proj.weight",
                       [config.hidden_size, config.intermediate_size])

    # Final norm
    add_weight('model.norm.weight', [config.hidden_size])

    # LM head
    add_weight('lm_head.weight', [config.vocab_size, config.hidden_size])

    return {
        'metadata': {
            'total_size': total_size,
            'num_layers': config.num_hidden_layers,
            'hidden_size': config.hidden_size,
            'vocab_size': config.vocab_size,
            'num_attention_heads': config.num_attention_heads,
            'num_key_value_heads': config.num_key_value_heads,
            'n_routed_experts': config.n_routed_experts,
            'n_shared_experts': config.n_shared_experts,
        },
        'weight_map': weight_map,
    }


def main():
    parser = argparse.ArgumentParser(
        description=
        'Extract weight information from DeepseekV3 model definition')
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default='weight_map.json',
        help='Output JSON file path (default: weight_map.json)',
    )
    parser.add_argument(
        '--dtype',
        type=str,
        default='float16',
        choices=['float32', 'float16', 'bfloat16'],
        help='Data type for model weights (default: float16)',
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to JSON config file. If not provided, uses default config.',
    )
    parser.add_argument(
        '--pretty',
        action='store_true',
        help='Pretty print JSON output',
    )

    args = parser.parse_args()

    # Load config
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = DeepseekV3Config(**config_dict)
    else:
        config = DeepseekV3Config()

    print('Configuration:')
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  hidden_size: {config.hidden_size}")
    print(f"  num_hidden_layers: {config.num_hidden_layers}")
    print(f"  num_attention_heads: {config.num_attention_heads}")
    print(f"  num_key_value_heads: {config.num_key_value_heads}")
    print(f"  intermediate_size: {config.intermediate_size}")
    print(f"  moe_intermediate_size: {config.moe_intermediate_size}")
    print(f"  n_routed_experts: {config.n_routed_experts}")
    print(f"  n_shared_experts: {config.n_shared_experts}")
    print(f"  first_k_dense_replace: {config.first_k_dense_replace}")
    print(f"  moe_layer_freq: {config.moe_layer_freq}")
    print(f"  dtype: {args.dtype}")
    print()

    # Generate weight map
    print('Generating weight map...')
    result = generate_weight_map(config)

    # Update dtype in weight map
    for weight_info in result['weight_map'].values():
        weight_info['dtype'] = args.dtype

    # Print summary
    total_params = len(result['weight_map'])
    total_size_gb = result['metadata']['total_size'] / (1024**3)
    print('\nSummary:')
    print(f"  Total parameters: {total_params}")
    print(
        f"  Total size: {total_size_gb:.2f} GB ({result['metadata']['total_size']:,} bytes)"
    )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        if args.pretty:
            json.dump(result, f, indent=2)
        else:
            json.dump(result, f)

    print(f"\nWeight map saved to: {output_path}")

    # Print some example weights
    print('\nExample weights:')
    example_weights = [
        'model.embed_tokens.weight',
        'model.layers.0.input_layernorm.weight',
        'model.layers.0.self_attn.q_proj.weight',
        'model.layers.0.self_attn.k_proj.weight',
        'model.layers.0.mlp.gate.weight',
        'model.layers.0.mlp.experts.0.gate_proj.weight',
        'model.norm.weight',
        'lm_head.weight',
    ]

    for name in example_weights:
        if name in result['weight_map']:
            info = result['weight_map'][name]
            print(f"  {name}: shape={info['shape']}, dtype={info['dtype']}")


if __name__ == '__main__':
    main()
