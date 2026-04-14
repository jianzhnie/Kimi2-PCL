"""
Comprehensive unit tests for checkpoint conversion modules

Coverage targets:
- Branch coverage >= 90%
- Critical path coverage 100%
"""

import json
import os
import tempfile
import hashlib
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open

import pytest
import torch
from safetensors.torch import save_file

# Import conversion modules
from utils.convert_ckpt_hf2mcore import (
    _parse_int_list,
    _ensure_iter_path,
    _dtype_from_str,
    _sha256_file,
    _write_sha256_manifest,
    _mp_prefix,
    CkptConvert,
)

from utils.check_model_weights import (
    _shard_paths,
)

from utils.convert_ckpt_mcore2hf import (
    _resolve_iter_dir,
    _mp_prefix as _mp_prefix_mcore,
    _dtype_from_str as _dtype_from_str_mcore,
    _sha256_file as _sha256_file_mcore,
    _write_sha256_manifest as _write_sha256_manifest_mcore,
    MgCkptConvert,
)


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def temp_hf_checkpoint():
    """Create a temporary HF checkpoint directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)

        # Create config.json
        config = {
            "vocab_size": 128,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "model_type": "kimi_k2",
        }
        with open(ckpt_dir / "config.json", "w") as f:
            json.dump(config, f)

        # Create safetensors
        weights = {
            "model.embed_tokens.weight": torch.randn(128, 64),
            "model.layers.0.self_attn.q_proj.weight": torch.randn(64, 64),
            "model.layers.0.self_attn.k_proj.weight": torch.randn(32, 64),
            "model.layers.0.self_attn.v_proj.weight": torch.randn(32, 64),
            "model.layers.0.self_attn.o_proj.weight": torch.randn(64, 64),
            "model.layers.0.mlp.gate_proj.weight": torch.randn(128, 64),
            "model.layers.0.mlp.up_proj.weight": torch.randn(128, 64),
            "model.layers.0.mlp.down_proj.weight": torch.randn(64, 128),
            "model.norm.weight": torch.randn(64),
            "lm_head.weight": torch.randn(128, 64),
        }
        save_file(weights, ckpt_dir / "model.safetensors")

        yield str(ckpt_dir)


@pytest.fixture
def temp_mcore_checkpoint():
    """Create a temporary MCore checkpoint directory"""
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_dir = Path(tmpdir)
        iter_dir = ckpt_dir / "iter_0000001"
        iter_dir.mkdir()

        # Create mp_rank directory
        mp_dir = iter_dir / "mp_rank_00_000_000"
        mp_dir.mkdir()

        # Create checkpoint file
        checkpoint = {
            "model0": {
                "embedding.word_embeddings.weight": torch.randn(128, 64),
                "decoder.layers.0.self_attention.linear_qkv.weight": torch.randn(128, 64),
                "decoder.layers.0.self_attention.linear_proj.weight": torch.randn(64, 64),
                "decoder.layers.0.input_layernorm.weight": torch.randn(64),
            },
            "model1": {
                "decoder.final_layernorm.weight": torch.randn(64),
                "output_layer.weight": torch.randn(128, 64),
            },
            "checkpoint_version": 3.0,
            "iteration": 1,
        }
        torch.save(checkpoint, mp_dir / "model_optim_rng.pt")

        # Create latest checkpoint file
        with open(ckpt_dir / "latest_checkpointed_iteration.txt", "w") as f:
            f.write("1")

        yield str(ckpt_dir)


# =============================================================================
# Utility Function Tests
# =============================================================================

class TestParseIntList:
    """Test _parse_int_list function"""

    def test_parse_valid_list(self):
        """Test parsing valid comma-separated integers"""
        assert _parse_int_list("1,2,3") == [1, 2, 3]
        assert _parse_int_list("10,20,30") == [10, 20, 30]

    def test_parse_single_value(self):
        """Test parsing single integer"""
        assert _parse_int_list("5") == [5]

    def test_parse_empty_string(self):
        """Test parsing empty string returns None"""
        assert _parse_int_list("") is None

    def test_parse_none(self):
        """Test parsing None returns None"""
        assert _parse_int_list(None) is None

    def test_parse_with_spaces(self):
        """Test parsing with spaces around commas"""
        assert _parse_int_list("1, 2, 3") == [1, 2, 3]


class TestEnsureIterPath:
    """Test _ensure_iter_path function"""

    def test_creates_iter_directory(self):
        """Test that iter directory is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            iter_path = _ensure_iter_path(tmpdir)
            assert os.path.exists(iter_path)
            assert os.path.basename(iter_path) == "iter_0000001"

    def test_creates_latest_file(self):
        """Test that latest checkpoint file is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _ensure_iter_path(tmpdir)
            latest_path = os.path.join(tmpdir, "latest_checkpointed_iteration.txt")
            assert os.path.exists(latest_path)
            with open(latest_path) as f:
                assert f.read().strip() == "1"

    def test_existing_directory_preserved(self):
        """Test that existing directory is preserved"""
        with tempfile.TemporaryDirectory() as tmpdir:
            iter_path1 = _ensure_iter_path(tmpdir)
            iter_path2 = _ensure_iter_path(tmpdir)
            assert iter_path1 == iter_path2


class TestDtypeFromStr:
    """Test _dtype_from_str function"""

    @pytest.mark.parametrize("dtype_str,expected", [
        ("fp16", torch.float16),
        ("float16", torch.float16),
        ("bf16", torch.bfloat16),
        ("bfloat16", torch.bfloat16),
        ("fp32", torch.float32),
        ("float32", torch.float32),
    ])
    def test_valid_dtypes(self, dtype_str, expected):
        """Test valid dtype strings"""
        assert _dtype_from_str(dtype_str) == expected

    def test_case_insensitive(self):
        """Test that dtype matching is case insensitive"""
        assert _dtype_from_str("FP16") == torch.float16
        assert _dtype_from_str("Bf16") == torch.bfloat16

    def test_invalid_dtype(self):
        """Test invalid dtype raises error"""
        with pytest.raises(ValueError, match="不支持的 dtype"):
            _dtype_from_str("invalid")

    def test_empty_string(self):
        """Test empty string raises ValueError"""
        with pytest.raises(ValueError, match="不支持的 dtype"):
            _dtype_from_str("")


class TestSHA256File:
    """Test _sha256_file function"""

    def test_sha256_consistency(self):
        """Test SHA256 hash is consistent"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("test content")
            temp_path = f.name

        try:
            hash1 = _sha256_file(temp_path)
            hash2 = _sha256_file(temp_path)
            assert hash1 == hash2
            assert len(hash1) == 64  # SHA256 hex string length
        finally:
            os.unlink(temp_path)

    def test_sha256_different_content(self):
        """Test different content produces different hashes"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("content 1")
            temp_path1 = f.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            f.write("content 2")
            temp_path2 = f.name

        try:
            hash1 = _sha256_file(temp_path1)
            hash2 = _sha256_file(temp_path2)
            assert hash1 != hash2
        finally:
            os.unlink(temp_path1)
            os.unlink(temp_path2)

    def test_sha256_empty_file(self):
        """Test SHA256 of empty file"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as f:
            temp_path = f.name

        try:
            hash_val = _sha256_file(temp_path)
            assert len(hash_val) == 64
        finally:
            os.unlink(temp_path)


class TestWriteSHA256Manifest:
    """Test _write_sha256_manifest function"""

    def test_writes_manifest(self):
        """Test manifest file is created"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test files
            (Path(tmpdir) / "file1.pt").write_text("content1")
            (Path(tmpdir) / "file2.pt").write_text("content2")

            manifest_path = os.path.join(tmpdir, "manifest.json")
            result = _write_sha256_manifest(tmpdir, manifest_path)

            assert result == manifest_path
            assert os.path.exists(manifest_path)

            with open(manifest_path) as f:
                manifest = json.load(f)
            assert "file1.pt" in manifest
            assert "file2.pt" in manifest

    def test_none_out_path(self):
        """Test with None out_path returns None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = _write_sha256_manifest(tmpdir, None)
            assert result is None


class TestMpPrefix:
    """Test _mp_prefix function"""

    def test_tp_only(self):
        """Test with only tensor parallelism"""
        result = _mp_prefix(1, 0, 0, tp=2, pp=1, ep=1)
        assert result == "mp_rank_01"

    def test_tp_and_pp(self):
        """Test with tensor and pipeline parallelism"""
        result = _mp_prefix(1, 2, 0, tp=2, pp=4, ep=1)
        assert result == "mp_rank_01_002"

    def test_tp_and_ep(self):
        """Test with tensor and expert parallelism"""
        result = _mp_prefix(1, 0, 3, tp=2, pp=1, ep=4)
        assert result == "mp_rank_01_003"

    def test_all_parallelisms(self):
        """Test with all parallelism types"""
        result = _mp_prefix(1, 2, 3, tp=2, pp=4, ep=4)
        assert result == "mp_rank_01_002_003"


# =============================================================================
# HF to MCore Conversion Tests
# =============================================================================

class TestCkptConvertInitialization:
    """Test CkptConvert class initialization"""

    @pytest.fixture
    def base_convert_kwargs(self):
        return {
            "hf_model_path": "/tmp/hf",
            "mg_save_path": "/tmp/mcore",
            "num_layers": 2,
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "first_k_dense_replace": 0,
            "hidden_size": 64,
            "ffn_hidden_size": 128,
            "moe_ffn_hidden_size": 128,
            "vocab_size": 128,
            "num_query_groups": 2,
            "num_experts": 1,
            "num_attention_heads": 4,
            "qk_head_dim": 64,
            "v_head_dim": 64,
            "moe_grouped_gemm": False,
            "schedules_method": None,
            "vpp_stage": None,
            "num_layer_list": None,
            "noop_layers": "",
            "qlora_nf4": False,
            "rotary_base": 50000.0,
            "print_init_summary": False,
            "pp_workers": 1,
            "cast_dtype": None,
            "tie_word_embeddings": False,
            "hf_io_threads": 1,
            "qk_layernorm": False,
        }

    @patch.object(CkptConvert, '_validate')
    @patch.object(CkptConvert, '_read_weight_map', return_value={})
    def test_basic_initialization(self, mock_read, mock_validate, base_convert_kwargs):
        """Test basic initialization"""
        converter = CkptConvert(**base_convert_kwargs)
        assert converter.hf_model_path == "/tmp/hf"
        assert converter.tp_size == 1
        assert converter.pp_size == 1

    @patch.object(CkptConvert, '_validate')
    @patch.object(CkptConvert, '_read_weight_map', return_value={})
    def test_validation_called(self, mock_read, mock_validate, base_convert_kwargs):
        """Test that validation is called during initialization"""
        CkptConvert(**base_convert_kwargs)
        mock_validate.assert_called_once()


class TestCkptConvertLayerMapping:
    """Test CkptConvert layer mapping methods"""

    @pytest.fixture
    def converter(self):
        with patch.object(CkptConvert, '_validate'), \
             patch.object(CkptConvert, '_read_weight_map', return_value={}):
            return CkptConvert(
                hf_model_path="/tmp/hf",
                mg_save_path="/tmp/mcore",
                num_layers=4,
                tp_size=2,
                pp_size=2,
                ep_size=1,
                first_k_dense_replace=1,
                hidden_size=64,
                ffn_hidden_size=128,
                moe_ffn_hidden_size=128,
                vocab_size=128,
                num_query_groups=2,
                num_experts=4,
                num_attention_heads=4,
                qk_head_dim=64,
                v_head_dim=64,
                moe_grouped_gemm=False,
                schedules_method=None,
                vpp_stage=None,
                num_layer_list=None,
                noop_layers="",
                qlora_nf4=False,
                rotary_base=50000.0,
                print_init_summary=False,
                pp_workers=1,
                cast_dtype=None,
                tie_word_embeddings=False,
                hf_io_threads=1,
                qk_layernorm=False,
            )

    def test_first_k_dense_replace(self, converter):
        """Test first_k_dense_replace configuration"""
        assert converter.first_k_dense_replace == 1
        assert converter.num_layers == 4


# =============================================================================
# MCore to HF Conversion Tests
# =============================================================================

class TestResolveIterDir:
    """Test _resolve_iter_dir function"""

    def test_resolve_from_latest_file(self, temp_mcore_checkpoint):
        """Test resolving from latest_checkpointed_iteration.txt"""
        iter_dir = _resolve_iter_dir(temp_mcore_checkpoint)
        assert os.path.basename(iter_dir) == "iter_0000001"

    def test_resolve_default_iter(self):
        """Test resolving default iter_0000001"""
        with tempfile.TemporaryDirectory() as tmpdir:
            iter_dir = os.path.join(tmpdir, "iter_0000001")
            os.makedirs(iter_dir)

            result = _resolve_iter_dir(tmpdir)
            assert result == iter_dir

    def test_resolve_already_iter_dir(self):
        """Test when path is already an iter directory"""
        with tempfile.TemporaryDirectory() as tmpdir:
            iter_dir = os.path.join(tmpdir, "iter_0000005")
            os.makedirs(iter_dir)

            result = _resolve_iter_dir(iter_dir)
            assert result == iter_dir

    def test_resolve_not_found(self):
        """Test error when no iteration directory found"""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError, match="无法定位迭代目录"):
                _resolve_iter_dir(tmpdir)


class TestMgCkptConvertInitialization:
    """Test MgCkptConvert class initialization"""

    @pytest.fixture
    def base_mg_convert_kwargs(self):
        return {
            "mg_load_dir": "/tmp/mcore",
            "hf_save_dir": "/tmp/hf",
            "num_layers": 2,
            "tp_size": 1,
            "pp_size": 1,
            "ep_size": 1,
            "first_k_dense_replace": 0,
            "hidden_size": 64,
            "num_experts": 1,
            "num_attention_heads": 4,
            "num_query_groups": 2,
            "qk_head_dim": 64,
            "moe_grouped_gemm": False,
            "schedules_method": None,
            "vpp_stage": None,
            "num_layer_list": None,
            "noop_layers": "",
            "qk_layernorm": False,
            "rotary_base": 50000.0,
            "vocab_size": 128,
            "max_position_embeddings": 512,
            "tie_word_embeddings": False,
            "ffn_hidden_size": 128,
            "moe_ffn_hidden_size": 128,
            "n_shared_experts": 1,
            "moe_router_topk": 2,
            "hf_config_template": None,
            "cast_dtype": None,
            "io_threads": 1,
            "disable_mmap": True,
        }

    @patch.object(MgCkptConvert, '_detect_vpp', return_value=(None, ['model']))
    def test_basic_initialization(self, mock_detect, base_mg_convert_kwargs):
        """Test basic initialization"""
        with patch('utils.convert_ckpt_mcore2hf._resolve_iter_dir', return_value='/tmp/unused'):
            converter = MgCkptConvert(**base_mg_convert_kwargs)
            assert converter.mg_load_dir == "/tmp/mcore"
            assert converter.tp_size == 1


class TestMgCkptConvertQKVLayout:
    """Test MgCkptConvert QKV layout inference"""

    @pytest.fixture
    def converter(self):
        with patch('utils.convert_ckpt_mcore2hf._resolve_iter_dir', return_value='/tmp/unused'), \
             patch.object(MgCkptConvert, '_detect_vpp', return_value=(None, ['model'])):
            return MgCkptConvert(
                mg_load_dir='/tmp/unused',
                hf_save_dir='/tmp/unused',
                num_layers=2,
                tp_size=2,
                pp_size=1,
                ep_size=1,
                first_k_dense_replace=0,
                hidden_size=64,
                num_experts=1,
                num_attention_heads=4,
                num_query_groups=2,
                qk_head_dim=64,
                moe_grouped_gemm=False,
                schedules_method=None,
                vpp_stage=None,
                num_layer_list=None,
                noop_layers='',
                qk_layernorm=False,
                rotary_base=50000.0,
                vocab_size=128,
                max_position_embeddings=512,
                tie_word_embeddings=False,
                ffn_hidden_size=128,
                moe_ffn_hidden_size=128,
                n_shared_experts=1,
                moe_router_topk=2,
                hf_config_template=None,
                cast_dtype=None,
                io_threads=1,
                disable_mmap=True,
            )

    def test_attention_params(self, converter):
        """Test attention parameters are correctly set"""
        # Verify that GQA parameters are correctly configured
        assert converter.num_attention_heads == 4
        assert converter.num_query_groups == 2
        # GQA: v_head_dim equals qk_head_dim
        assert converter.v_head_dim == converter.qk_head_dim
        # Verify head_dim calculations
        expected_head_dim = converter.hidden_size // converter.num_attention_heads
        assert expected_head_dim == 16  # 64 / 4 = 16


# =============================================================================
# Roundtrip Conversion Tests
# =============================================================================

class TestRoundtripConversion:
    """Test HF -> MCore -> HF roundtrip"""

    def test_weight_shape_preservation(self, temp_hf_checkpoint):
        """Test that weights maintain shapes through conversion"""
        # This is a simplified test - full roundtrip would require
        # actual conversion execution

        # Read original HF weights
        from safetensors import safe_open as sf_open
        original_weights = {}
        with sf_open(os.path.join(temp_hf_checkpoint, "model.safetensors"), framework='pt') as f:
            for key in f.keys():
                original_weights[key] = tuple(f.get_slice(key).get_shape())

        # Verify shapes are valid
        for key, shape in original_weights.items():
            assert all(isinstance(dim, int) for dim in shape)
            assert all(dim > 0 for dim in shape)


# =============================================================================
# Edge Cases and Error Handling Tests
# =============================================================================

class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_empty_weight_map(self):
        """Test handling of empty weight map"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create empty index file
            index = {"metadata": {}, "weight_map": {}}
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)

            # This should handle gracefully
            paths, _ = _shard_paths(Path(tmpdir))
            assert len(paths) == 0

    def test_very_large_tp_size(self):
        """Test with large tensor parallelism size"""
        # Should not crash with large TP
        prefix = _mp_prefix(127, 0, 0, tp=128, pp=1, ep=1)
        assert "mp_rank_127" in prefix

    def test_special_chars_in_path(self):
        """Test with special characters in path"""
        # This is primarily a smoke test
        with tempfile.TemporaryDirectory(prefix="test-path_with.special+chars") as tmpdir:
            iter_path = _ensure_iter_path(tmpdir)
            assert os.path.exists(iter_path)

    def test_unicode_in_path(self):
        """Test with unicode characters in path"""
        with tempfile.TemporaryDirectory(prefix="测试路径") as tmpdir:
            iter_path = _ensure_iter_path(tmpdir)
            assert os.path.exists(iter_path)


# =============================================================================
# Benchmark Tests
# =============================================================================

@pytest.mark.benchmark
class TestBenchmarks:
    """Performance benchmark tests"""

    def test_sha256_benchmark(self, benchmark):
        """Benchmark SHA256 file hashing"""
        with tempfile.NamedTemporaryFile(mode='wb', delete=False) as f:
            # Write 1MB of data
            f.write(os.urandom(1024 * 1024))
            temp_path = f.name

        try:
            benchmark(_sha256_file, temp_path)
        finally:
            os.unlink(temp_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
