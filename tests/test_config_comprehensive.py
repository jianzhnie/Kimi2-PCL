"""
Comprehensive unit tests for configuration modules

Coverage targets:
- Branch coverage >= 90%
- Critical path coverage 100%
"""

import json
import pytest
from pathlib import Path

from models.configuration_deepseek import DeepseekV3Config


# =============================================================================
# Configuration Fixtures
# =============================================================================

@pytest.fixture
def minimal_config_kwargs():
    """Minimal configuration for fast tests"""
    return {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "max_position_embeddings": 128,
    }


@pytest.fixture
def full_config_kwargs():
    """Full configuration with all optional parameters"""
    return {
        "vocab_size": 1024,
        "hidden_size": 256,
        "intermediate_size": 512,
        "num_hidden_layers": 4,
        "num_attention_heads": 8,
        "num_key_value_heads": 4,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "rms_norm_eps": 1e-6,
        "hidden_act": "swiglu",
        "attention_dropout": 0.0,
        "rope_scaling": None,
        "qk_layernorm": True,
        "tie_word_embeddings": False,
    }


# =============================================================================
# 1T Config Tests
# =============================================================================

class TestDeepseekV3Config:
    """Test DeepseekV3Config from 1T configuration"""

    def test_default_initialization(self):
        """Test config with default values from actual config file"""
        config = DeepseekV3Config()

        # Check default values (from actual config_1t.json)
        assert config.vocab_size == 163840
        assert config.hidden_size == 7168
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 64
        assert config.intermediate_size == 18432
        assert config.max_position_embeddings == 131072
        assert config.num_key_value_heads == 32

    def test_custom_initialization(self, minimal_config_kwargs):
        """Test config with custom values"""
        config = DeepseekV3Config(**minimal_config_kwargs)

        assert config.vocab_size == 128
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2

    def test_full_initialization(self, full_config_kwargs):
        """Test config with all parameters"""
        config = DeepseekV3Config(**full_config_kwargs)

        for key, value in full_config_kwargs.items():
            assert getattr(config, key) == value

    def test_rope_scaling_initialization(self):
        """Test config with RoPE scaling"""
        rope_scaling = {
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
        }
        config = DeepseekV3Config(rope_scaling=rope_scaling)

        assert config.rope_scaling == rope_scaling

    @pytest.mark.parametrize("hidden_act", ["swiglu", "gelu", "relu", "silu"])
    def test_activation_functions(self, hidden_act):
        """Test various activation functions"""
        config = DeepseekV3Config(hidden_act=hidden_act)
        assert config.hidden_act == hidden_act

    @pytest.mark.parametrize("dropout", [0.0, 0.1, 0.5])
    def test_dropout_values(self, dropout):
        """Test various dropout values"""
        config = DeepseekV3Config(attention_dropout=dropout)
        assert config.attention_dropout == dropout

    def test_num_key_value_heads_gqa(self):
        """Test grouped query attention configuration"""
        # When group_query_attention=True, num_key_value_heads is calculated from num_query_groups
        config = DeepseekV3Config(
            num_attention_heads=64,
            num_query_groups=4,  # This will set num_key_value_heads = 64 // 4 = 16
        )
        assert config.num_attention_heads == 64
        assert config.num_key_value_heads == 16

    def test_gqa_configuration(self):
        """Test grouped query attention with custom values"""
        # When group_query_attention=True, num_key_value_heads is calculated from num_query_groups
        config = DeepseekV3Config(
            num_attention_heads=64,
            num_query_groups=8,  # This will set num_key_value_heads = 64 // 8 = 8
        )
        assert config.num_attention_heads == 64
        assert config.num_key_value_heads == 8

    def test_config_to_dict(self):
        """Test converting config to dictionary"""
        config = DeepseekV3Config()
        config_dict = config.to_dict()

        assert isinstance(config_dict, dict)
        assert "vocab_size" in config_dict
        assert "hidden_size" in config_dict
        assert config_dict["vocab_size"] == config.vocab_size

    def test_config_to_json_string(self):
        """Test converting config to JSON string"""
        config = DeepseekV3Config()
        json_str = config.to_json_string()

        assert isinstance(json_str, str)
        # Verify it's valid JSON
        parsed = json.loads(json_str)
        assert "vocab_size" in parsed

    def test_config_save_and_load(self, tmp_path):
        """Test saving and loading config"""
        config = DeepseekV3Config(
            vocab_size=256,
            hidden_size=128,
            num_hidden_layers=4,
        )

        # Save
        config_path = tmp_path / "config.json"
        config.save_pretrained(tmp_path)

        # Load
        loaded_config = DeepseekV3Config.from_pretrained(tmp_path)

        assert loaded_config.vocab_size == config.vocab_size
        assert loaded_config.hidden_size == config.hidden_size
        assert loaded_config.num_hidden_layers == config.num_hidden_layers

    def test_config_equality(self):
        """Test config equality comparison"""
        config1 = DeepseekV3Config(vocab_size=128, hidden_size=64)
        config2 = DeepseekV3Config(vocab_size=128, hidden_size=64)
        config3 = DeepseekV3Config(vocab_size=256, hidden_size=64)

        assert config1 == config2
        assert config1 != config3

    def test_config_repr(self):
        """Test config string representation"""
        config = DeepseekV3Config()
        repr_str = repr(config)

        assert "DeepseekV3Config" in repr_str
        assert "vocab_size" in repr_str

    def test_config_update(self):
        """Test updating config attributes"""
        config = DeepseekV3Config()
        config.update({"vocab_size": 256, "hidden_size": 128})

        assert config.vocab_size == 256
        assert config.hidden_size == 128

    def test_config_copy(self):
        """Test copying config via to_dict() and from_dict()"""
        config = DeepseekV3Config(vocab_size=128, hidden_size=64)
        config_dict = config.to_dict()
        config_copy = DeepseekV3Config(**config_dict)

        assert config.vocab_size == config_copy.vocab_size
        assert config.hidden_size == config_copy.hidden_size
        assert config is not config_copy  # Different objects


# =============================================================================
# 100B Config Tests
# =============================================================================

class TestDeepseekV3Config:
    """Test DeepseekV3Config from 100B configuration"""

    def test_default_initialization(self):
        """Test config with default values (1T model)"""
        config = DeepseekV3Config()

        # Check default values for 1T model (from actual config)
        assert config.vocab_size == 163840
        assert config.hidden_size == 7168
        assert config.num_hidden_layers == 32
        assert config.num_attention_heads == 64
        assert config.intermediate_size == 18432

    def test_custom_initialization(self, minimal_config_kwargs):
        """Test config with custom values"""
        config = DeepseekV3Config(**minimal_config_kwargs)

        assert config.vocab_size == 128
        assert config.hidden_size == 64
        assert config.num_hidden_layers == 2

    def test_custom_vs_default_differences(self):
        """Test differences between custom and default configs"""
        config_default = DeepseekV3Config()
        config_custom = DeepseekV3Config(hidden_size=4096, num_attention_heads=32)

        # Custom config has smaller hidden size than default
        assert config_custom.hidden_size < config_default.hidden_size
        # Custom config has fewer attention heads
        assert config_custom.num_attention_heads < config_default.num_attention_heads


# =============================================================================
# MoE Config Tests
# =============================================================================

class TestMoEConfiguration:
    """Test MoE-specific configuration"""

    def test_moe_default_config(self):
        """Test default MoE configuration"""
        config = DeepseekV3Config()

        assert config.n_routed_experts == 128
        assert config.n_shared_experts == 1
        assert config.num_experts_per_tok == 2
        assert config.moe_intermediate_size == 12288

    def test_moe_custom_config(self):
        """Test custom MoE configuration"""
        config = DeepseekV3Config(
            n_routed_experts=64,
            n_shared_experts=4,
            num_experts_per_tok=4,
            moe_intermediate_size=12288,
        )

        assert config.n_routed_experts == 64
        assert config.n_shared_experts == 4
        assert config.num_experts_per_tok == 4
        assert config.moe_intermediate_size == 12288

    def test_first_k_dense_replace(self):
        """Test first_k_dense_replace configuration"""
        config = DeepseekV3Config(first_k_dense_replace=2)

        assert config.first_k_dense_replace == 2

    def test_moe_layer_freq(self):
        """Test moe_layer_freq configuration"""
        config = DeepseekV3Config(moe_layer_freq=1)

        assert config.moe_layer_freq == 1

    def test_moe_aux_loss_coeff(self):
        """Test auxiliary loss coefficient"""
        config = DeepseekV3Config(moe_aux_loss_coeff=0.001)

        assert config.moe_aux_loss_coeff == 0.001

    def test_moe_z_loss_coeff(self):
        """Test z-loss coefficient"""
        config = DeepseekV3Config(moe_z_loss_coeff=0.0001)

        assert config.moe_z_loss_coeff == 0.0001


# =============================================================================
# Attention Config Tests
# =============================================================================

class TestAttentionConfiguration:
    """Test attention-specific configuration"""

    def test_gqa_configuration(self):
        """Test grouped query attention configuration"""
        # When group_query_attention=True, num_key_value_heads is calculated from num_query_groups
        config = DeepseekV3Config(
            num_attention_heads=64,
            num_query_groups=8,  # This will set num_key_value_heads = 64 // 8 = 8
        )

        assert config.num_attention_heads == 64
        assert config.num_key_value_heads == 8

    def test_qk_head_dims(self):
        """Test Q/K head dimension configuration"""
        config = DeepseekV3Config(
            qk_nope_head_dim=128,
            qk_rope_head_dim=64,
            v_head_dim=128,
        )

        assert config.qk_nope_head_dim == 128
        assert config.qk_rope_head_dim == 64
        assert config.v_head_dim == 128

    def test_rope_theta(self):
        """Test RoPE theta (base frequency)"""
        config = DeepseekV3Config(rope_theta=1000000.0)

        assert config.rope_theta == 1000000.0

    def test_attention_bias(self):
        """Test attention bias configuration"""
        config = DeepseekV3Config(attention_bias=False)

        assert config.attention_bias == False

    def test_qk_layernorm(self):
        """Test Q/K layernorm configuration"""
        config = DeepseekV3Config(qk_layernorm=True)

        assert config.qk_layernorm == True


# =============================================================================
# RoPE Scaling Config Tests
# =============================================================================

class TestRoPEScalingConfiguration:
    """Test RoPE scaling configuration"""

    def test_no_scaling(self):
        """Test configuration without RoPE scaling"""
        config = DeepseekV3Config(rope_scaling=None)

        assert config.rope_scaling is None

    def test_yarn_scaling(self):
        """Test YaRN RoPE scaling configuration"""
        rope_scaling = {
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
            "beta_fast": 32,
            "beta_slow": 1,
            "mscale": 1,
            "mscale_all_dim": 0,
        }
        config = DeepseekV3Config(rope_scaling=rope_scaling)

        assert config.rope_scaling["type"] == "yarn"
        assert config.rope_scaling["factor"] == 32.0

    def test_linear_scaling(self):
        """Test linear RoPE scaling configuration"""
        rope_scaling = {
            "type": "linear",
            "factor": 2.0,
        }
        config = DeepseekV3Config(rope_scaling=rope_scaling)

        assert config.rope_scaling["type"] == "linear"
        assert config.rope_scaling["factor"] == 2.0

    def test_dynamic_scaling(self):
        """Test dynamic NTK RoPE scaling configuration"""
        rope_scaling = {
            "type": "dynamic",
            "factor": 4.0,
        }
        config = DeepseekV3Config(rope_scaling=rope_scaling)

        assert config.rope_scaling["type"] == "dynamic"
        assert config.rope_scaling["factor"] == 4.0


# =============================================================================
# Edge Cases and Validation Tests
# =============================================================================

class TestConfigEdgeCases:
    """Test configuration edge cases"""

    def test_very_small_config(self):
        """Test with very small configuration"""
        config = DeepseekV3Config(
            vocab_size=16,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=2,
        )

        assert config.vocab_size == 16
        assert config.hidden_size == 4

    def test_very_large_config(self):
        """Test with very large configuration"""
        config = DeepseekV3Config(
            vocab_size=500000,
            hidden_size=16384,
            intermediate_size=65536,
            num_hidden_layers=100,
            num_attention_heads=128,
        )

        assert config.vocab_size == 500000
        assert config.hidden_size == 16384

    def test_zero_dropout(self):
        """Test with zero dropout"""
        config = DeepseekV3Config(attention_dropout=0.0)

        assert config.attention_dropout == 0.0

    def test_high_dropout(self):
        """Test with high dropout"""
        config = DeepseekV3Config(attention_dropout=0.9)

        assert config.attention_dropout == 0.9

    def test_tie_word_embeddings(self):
        """Test tie_word_embeddings configuration"""
        config = DeepseekV3Config(tie_word_embeddings=True)

        assert config.tie_word_embeddings == True

    def test_pad_token_id(self):
        """Test pad_token_id configuration"""
        config = DeepseekV3Config(pad_token_id=0)

        assert config.pad_token_id == 0

    def test_bos_token_id(self):
        """Test bos_token_id configuration"""
        config = DeepseekV3Config(bos_token_id=1)

        assert config.bos_token_id == 1

    def test_eos_token_id(self):
        """Test eos_token_id configuration"""
        config = DeepseekV3Config(eos_token_id=2)

        assert config.eos_token_id == 2


# =============================================================================
# Config JSON Serialization Tests
# =============================================================================

class TestConfigSerialization:
    """Test configuration serialization"""

    def test_config_to_json_file(self, tmp_path):
        """Test saving config to JSON file"""
        config = DeepseekV3Config(vocab_size=128, hidden_size=64)

        json_path = tmp_path / "config.json"
        config.to_json_file(json_path)

        assert json_path.exists()

        with open(json_path) as f:
            data = json.load(f)
        assert data["vocab_size"] == 128
        assert data["hidden_size"] == 64

    def test_config_from_json_file(self, tmp_path):
        """Test loading config from JSON file"""
        data = {
            "vocab_size": 256,
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 256,
            "max_position_embeddings": 512,
            "model_type": "kimi_k2",
        }

        json_path = tmp_path / "config.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        config = DeepseekV3Config.from_json_file(json_path)

        assert config.vocab_size == 256
        assert config.hidden_size == 128

    def test_rope_scaling_serialization(self, tmp_path):
        """Test RoPE scaling serialization"""
        rope_scaling = {
            "type": "yarn",
            "factor": 32.0,
            "original_max_position_embeddings": 4096,
        }
        config = DeepseekV3Config(rope_scaling=rope_scaling)

        json_str = config.to_json_string()
        parsed = json.loads(json_str)

        assert parsed["rope_scaling"] == rope_scaling


# =============================================================================
# Integration with Repository Config Files
# =============================================================================

class TestRepositoryConfigFiles:
    """Test with actual repository config files"""

    def test_load_1t_config_file(self):
        """Test loading actual 1T config file"""
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "models" / "config_1t.json"

        if config_path.exists():
            config = DeepseekV3Config.from_json_file(config_path)
            assert config.vocab_size == 163840
            assert config.hidden_size == 7168  # Actual value from config_1t.json
            assert config.num_hidden_layers == 32  # Actual value from config_1t.json

    def test_load_100b_config_file(self):
        """Test loading actual 100B config file"""
        repo_root = Path(__file__).parent.parent
        config_path = repo_root / "models" / "config_100b.json"

        if config_path.exists():
            config = DeepseekV3Config.from_json_file(config_path)
            assert config.vocab_size == 163840
            assert config.hidden_size == 4096  # Actual value from config_100b.json
            assert config.num_hidden_layers == 28  # Actual value from config_100b.json

    def test_config_consistency_with_repo(self):
        """Test that configs are consistent with repository"""
        repo_root = Path(__file__).parent.parent

        config_1t_path = repo_root / "models" / "config_1t.json"
        config_100b_path = repo_root / "models" / "config_100b.json"

        if config_1t_path.exists() and config_100b_path.exists():
            config_1t = DeepseekV3Config.from_json_file(config_1t_path)
            config_100b = DeepseekV3Config.from_json_file(config_100b_path)

            # Both should have same vocab size
            assert config_1t.vocab_size == config_100b.vocab_size

            # 1T has larger hidden size than 100B
            assert config_1t.hidden_size > config_100b.hidden_size


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
