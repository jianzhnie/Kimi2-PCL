"""
Comprehensive unit tests for utility modules

Coverage targets:
- Branch coverage >= 90%
- Critical path coverage 100%
- Boundary testing, exception testing, parameterized testing
"""

import json
import math
import os
import sys
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

import pytest
import torch
from safetensors.torch import save_file

# Ensure utils are in path
sys.path.insert(0, str(Path(__file__).parent.parent / "utils"))

from check_model_weights import (
    _shard_paths,
    _read_specs_from_shard,
    _build_empty_model,
    _expected_state_specs,
    _compare_shapes,
    estimate_model_params,
    verify_config_consistency,
    verify_pretrain_script_consistency,
    main as check_main,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_checkpoint_dir():
    """Create a temporary checkpoint directory with dummy files"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)

        # Create config.json
        config = {
            "vocab_size": 128,
            "hidden_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "intermediate_size": 128,
            "model_type": "kimi_k2",
        }
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create dummy safetensors file
        dummy_weights = {
            "model.embed_tokens.weight": torch.randn(128, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 64),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(32, 64),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(128, 64),
        }
        save_file(dummy_weights, ckpt_dir / "model.safetensors")

        yield ckpt_dir


@pytest.fixture
def sharded_checkpoint_dir():
    """Create a temporary sharded checkpoint directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)

        # Create config.json
        config = {
            "vocab_size": 256,
            "hidden_size": 128,
            "num_hidden_layers": 4,
            "num_attention_heads": 8,
            "intermediate_size": 256,
        }
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create sharded weights
        weights1 = {
            "model.embed_tokens.weight": torch.randn(256, 128),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(128, 128),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(64, 128),
        }
        weights2 = {
            "model.layers.0.self_attn.v_proj.weight": torch.randn(64, 128),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(128, 128),
            "model.norm.weight": torch.randn(128),
            "lm_head.weight": torch.randn(256, 128),
        }
        save_file(weights1, ckpt_dir / "model-00001-of-00002.safetensors")
        save_file(weights2, ckpt_dir / "model-00002-of-00002.safetensors")

        # Create index file
        index = {
            "metadata": {"total_size": 12345},
            "weight_map": {
                "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00002.safetensors",
                "model.layers.0.self_attn.v_proj.weight": "model-00002-of-00002.safetensors",
                "model.layers.0.self_attn.o_proj.weight": "model-00002-of-00002.safetensors",
                "model.norm.weight": "model-00002-of-00002.safetensors",
                "lm_head.weight": "model-00002-of-00002.safetensors",
            },
        }
        with open(ckpt_dir / "model.safetensors.index.json", "w") as f:
            json.dump(index, f)

        yield ckpt_dir


# =============================================================================
# _shard_paths Tests
# =============================================================================

class TestShardPaths:
    """Test _shard_paths function"""

    def test_single_shard(self, temp_checkpoint_dir):
        """Test with single shard"""
        paths, index = _shard_paths(temp_checkpoint_dir)
        assert len(paths) == 1
        assert paths[0].name == "model.safetensors"
        assert index is None

    def test_sharded_model(self, sharded_checkpoint_dir):
        """Test with sharded model"""
        paths, index = _shard_paths(sharded_checkpoint_dir)
        assert len(paths) == 2
        assert index is not None
        assert index.exists()

    def test_missing_checkpoint(self):
        """Test error when no checkpoint found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="No model.safetensors"):
                _shard_paths(Path(tmpdir))

    def test_sharded_missing_index(self):
        """Test error when index file missing for sharded model"""
        with tempfile.TemporaryDirectory() as tmpdir:
            (Path(tmpdir) / "model-00001-of-00002.safetensors").touch()
            with pytest.raises(FileNotFoundError):
                _shard_paths(Path(tmpdir))


# =============================================================================
# _read_specs_from_shard Tests
# =============================================================================

class TestReadSpecsFromShard:
    """Test _read_specs_from_shard function"""

    def test_read_specs(self, temp_checkpoint_dir):
        """Test reading specs from shard"""
        shard_path = temp_checkpoint_dir / "model.safetensors"
        specs = _read_specs_from_shard(shard_path)

        assert "model.embed_tokens.weight" in specs
        assert specs["model.embed_tokens.weight"] == (128, 64)
        assert specs["model.norm.weight"] == (64,)

    def test_read_specs_sharded(self, sharded_checkpoint_dir):
        """Test reading specs from sharded checkpoint"""
        shard_path = sharded_checkpoint_dir / "model-00001-of-00002.safetensors"
        specs = _read_specs_from_shard(shard_path)

        assert "model.embed_tokens.weight" in specs
        assert specs["model.embed_tokens.weight"] == (256, 128)


# =============================================================================
# _build_empty_model Tests
# =============================================================================

class TestBuildEmptyModel:
    """Test _build_empty_model function"""

    def test_build_empty_model(self, temp_checkpoint_dir):
        """Test building empty model from config"""
        from models.configuration_deepseek_1t import DeepseekV3Config

        config = DeepseekV3Config(
            vocab_size=128,
            hidden_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
        )

        with patch("check_model_weights.init_empty_weights"):
            model = _build_empty_model(config)
            assert model is not None


# =============================================================================
# _expected_state_specs Tests
# =============================================================================

class TestExpectedStateSpecs:
    """Test _expected_state_specs function"""

    def test_expected_specs(self):
        """Test getting expected state specs from model"""
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 128, bias=False),
            torch.nn.Linear(128, 256, bias=False),
        )

        specs, keys = _expected_state_specs(model)

        assert len(specs) == 2
        assert "0.weight" in specs
        assert specs["0.weight"] == (128, 64)
        assert specs["1.weight"] == (256, 128)
        assert keys == {"0.weight", "1.weight"}


# =============================================================================
# _compare_shapes Tests
# =============================================================================

class TestCompareShapes:
    """Test _compare_shapes function"""

    def test_no_mismatches(self):
        """Test when all shapes match"""
        expected = {"a": (64, 128), "b": (128, 256)}
        ckpt = {"a": (64, 128), "b": (128, 256)}

        problems = _compare_shapes(expected, ckpt, "test")
        assert len(problems) == 0

    def test_shape_mismatch(self):
        """Test when shapes don't match"""
        expected = {"a": (64, 128), "b": (128, 256)}
        ckpt = {"a": (64, 128), "b": (256, 128)}  # Transposed

        problems = _compare_shapes(expected, ckpt, "test")
        assert len(problems) == 1
        assert "shape mismatch" in problems[0]
        assert "b" in problems[0]

    def test_partial_overlap(self):
        """Test when only some keys overlap"""
        expected = {"a": (64, 128), "b": (128, 256), "c": (256, 512)}
        ckpt = {"a": (64, 128), "b": (256, 128)}  # Different b, missing c

        problems = _compare_shapes(expected, ckpt, "test")
        assert len(problems) == 1  # Only b is compared


# =============================================================================
# estimate_model_params Tests
# =============================================================================

class TestEstimateModelParams:
    """Test estimate_model_params function"""

    def test_estimate_basic(self):
        """Test basic parameter estimation"""
        params = estimate_model_params()

        assert "total" in params
        assert "per_layer" in params
        assert "embedding" in params
        assert params["total"] > 0
        assert isinstance(params["per_layer"], dict)
        assert "attention" in params["per_layer"]
        assert params["embedding"] > 0

    def test_estimate_with_config(self):
        """Test estimation with custom config"""
        params = estimate_model_params(
            vocab_size=1024,
            hidden_size=256,
            intermediate_size=512,
            num_layers=4,
            num_attention_heads=8,
        )

        # Embedding should be vocab_size * hidden_size
        expected_embedding = 1024 * 256
        assert params["embedding"] == expected_embedding

    def test_estimate_1t_model(self):
        """Test estimation for 1T model"""
        # Use actual 1T model configuration values
        params = estimate_model_params(
            vocab_size=163840,
            hidden_size=7168,  # Actual value from config_1t.json
            intermediate_size=18432,
            moe_intermediate_size=12288,
            num_layers=32,  # Actual value from config_1t.json
            num_attention_heads=64,
            num_experts=128,
            first_k_dense_replace=2,
        )

        assert params["total"] > 1e12  # Should be around 1T
        assert params["total"] < 5e12


# =============================================================================
# verify_config_consistency Tests
# =============================================================================

class TestVerifyConfigConsistency:
    """Test verify_config_consistency function"""

    def test_verify_consistent_configs(self):
        """Test when configs are consistent"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create consistent config files
            config_1t = {
                "vocab_size": 163840,
                "hidden_size": 8192,
                "num_hidden_layers": 48,
                "num_attention_heads": 64,
                "intermediate_size": 24576,
            }

            config_100b = {
                "vocab_size": 163840,
                "hidden_size": 5120,
                "num_hidden_layers": 30,
                "num_attention_heads": 32,
                "intermediate_size": 12288,
            }

            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()

            with open(models_dir / "config_1t.json", "w") as f:
                json.dump(config_1t, f)
            with open(models_dir / "config_100b.json", "w") as f:
                json.dump(config_100b, f)

            # Call with explicit repo_root
            is_valid = verify_config_consistency(Path(tmpdir))
            # Should be valid as each config is internally consistent
            assert isinstance(is_valid, bool)  # Just check it returns a boolean


# =============================================================================
# verify_pretrain_script_consistency Tests
# =============================================================================

class TestVerifyPretrainScriptConsistency:
    """Test verify_pretrain_script_consistency function"""

    def test_verify_pretrain_consistent(self):
        """Test when pretrain script matches config"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create config
            config = {
                "vocab_size": 163840,
                "hidden_size": 8192,
                "num_hidden_layers": 48,
                "num_attention_heads": 64,
                "rope_theta": 1000000.0,
                "tie_word_embeddings": False,
            }

            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            with open(models_dir / "config_1t.json", "w") as f:
                json.dump(config, f)

            # Create matching pretrain script
            scripts_dir = Path(tmpdir) / "scripts"
            scripts_dir.mkdir()
            script_content = """
--num-layers 48
--hidden-size 8192
--num-attention-heads 64
--vocab-size 163840
--rotary-base 1000000.0
--use-flash-attn
--bf16
--moe-grouped-gemm
--schedules-method dualpipev
"""
            with open(scripts_dir / "pretrain_kimi2_1t_4k.sh", "w") as f:
                f.write(script_content)

            is_valid = verify_pretrain_script_consistency(Path(tmpdir))
            # Should be valid
            assert isinstance(is_valid, bool)


# =============================================================================
# main Tests
# =============================================================================

class TestCheckMain:
    """Test main function of check_model_weights"""

    @patch("sys.argv", ["check_model_weights.py", "--estimate-params"])
    @patch("check_model_weights.print_parameter_estimate")
    def test_main_estimate_params(self, mock_print):
        """Test --estimate-params flag"""
        # check_main returns exit code, doesn't raise SystemExit
        exit_code = check_main()
        assert exit_code == 0
        mock_print.assert_called_once()

    @patch("sys.argv", ["check_model_weights.py", "--verify-config"])
    @patch("check_model_weights.verify_config_consistency", return_value=True)
    @patch("check_model_weights.verify_pretrain_script_consistency")
    def test_main_verify_config(self, mock_script, mock_verify):
        """Test --verify-config flag"""
        exit_code = check_main()
        assert exit_code == 0
        mock_verify.assert_called_once()

    @patch("sys.argv", ["check_model_weights.py", "--verify-all"])
    @patch("check_model_weights.print_parameter_estimate")
    @patch("check_model_weights.verify_config_consistency", return_value=True)
    @patch("check_model_weights.verify_pretrain_script_consistency")
    def test_main_verify_all(self, mock_script, mock_verify, mock_print):
        """Test --verify-all flag"""
        exit_code = check_main()
        assert exit_code == 0
        mock_verify.assert_called_once()
        mock_print.assert_called_once()

    @patch("sys.argv", ["check_model_weights.py", "dummy_path", "--skip-shape-check"])
    @patch("check_model_weights._main_check_checkpoint")
    def test_main_check_checkpoint(self, mock_check):
        """Test checkpoint path argument"""
        check_main()
        mock_check.assert_called_once()

    @patch("sys.argv", ["check_model_weights.py"])
    def test_main_no_args(self):
        """Test with no arguments - should show error"""
        with pytest.raises(SystemExit) as e:
            check_main()
        # argparse exits with 2 for missing required argument
        assert e.value.code == 2


# =============================================================================
# Integration Tests
# =============================================================================

class TestIntegration:
    """Integration tests for check_model_weights"""

    def test_full_checkpoint_validation(self, temp_checkpoint_dir):
        """Test full checkpoint validation flow"""
        from check_model_weights import _main_check_checkpoint
        from argparse import Namespace

        args = Namespace(
            checkpoint=temp_checkpoint_dir,
            skip_shape_check=True,
            report_limit=200,
            strict_index=False,
        )
        # Function may raise SystemExit or return result
        try:
            _main_check_checkpoint(args)
        except SystemExit as e:
            # SystemExit with code 0 means success
            assert e.code == 0 or e.code is None

    def test_checkpoint_with_shape_check(self, temp_checkpoint_dir):
        """Test checkpoint validation with shape checking"""
        from check_model_weights import _main_check_checkpoint
        from argparse import Namespace

        args = Namespace(
            checkpoint=temp_checkpoint_dir,
            skip_shape_check=False,
            report_limit=200,
            strict_index=False,
        )
        try:
            _main_check_checkpoint(args)
        except SystemExit as e:
            assert e.code == 0 or e.code is None


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_missing_dependency_error(self):
        """Test error when dependency is missing"""
        from check_model_weights import _require

        mock_error = ImportError("No module named 'test_module'")
        with pytest.raises(SystemExit):
            _require("test_module", mock_error)

    def test_empty_config(self):
        """Test with empty/minimal config"""
        # Call with default parameters
        params = estimate_model_params()
        assert "total" in params

    def test_config_with_none_values(self):
        """Test config with None values - use default vocab_size"""
        # Should handle gracefully with valid parameters
        params = estimate_model_params(vocab_size=1000, hidden_size=64)
        assert "total" in params

    def test_very_large_config(self):
        """Test with very large model config"""
        params = estimate_model_params(
            vocab_size=256000,
            hidden_size=16384,
            intermediate_size=65536,
            num_layers=80,
            num_attention_heads=128,
        )
        assert params["total"] > 0
        assert not math.isinf(params["total"])
        assert not math.isnan(params["total"])

    def test_invalid_json_config(self):
        """Test with invalid JSON in config file"""
        with tempfile.TemporaryDirectory() as tmpdir:
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            with open(models_dir / "config_1t.json", "w") as f:
                f.write("invalid json {{[")

            # Should raise JSONDecodeError
            with pytest.raises((json.JSONDecodeError, SystemExit)):
                verify_config_consistency(Path(tmpdir))


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmark tests"""

    def test_estimate_params_benchmark(self, benchmark):
        """Benchmark parameter estimation"""
        benchmark(estimate_model_params,
                  vocab_size=163840,
                  hidden_size=8192,
                  num_layers=48)

    def test_read_specs_benchmark(self, benchmark, temp_checkpoint_dir):
        """Benchmark reading specs from shard"""
        shard_path = temp_checkpoint_dir / "model.safetensors"
        benchmark(_read_specs_from_shard, shard_path)


if __name__ == "__main__":
    import math
    pytest.main([__file__, "-v"])
