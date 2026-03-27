"""Unit tests for pretrain script parsing utilities.

These tests focus on parser robustness and offline determinism using local temp files.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from utils.pretrain_config import (
    _expand_vars,
    _extract_torchrun_section,
    _normalize_arg_text,
    _parse_assignments,
    _parse_quoted_blocks,
    _strip_quotes,
    get_bool,
    get_float,
    get_int,
    parse_pretrain_script,
)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('"value"', "value"),
        ("'value'", "value"),
        ("  plain  ", "plain"),
        ("'',", "'',"),
    ],
)
def test_pretrain_config_strip_quotes_mixed_inputs_returns_expected(raw: str, expected: str) -> None:
    """Test purpose: verify `_strip_quotes` handles quoted and non-quoted values.

    Inputs: mixed string values with spaces and quotes.
    Expected behavior: only balanced outer quotes are stripped.
    """
    # Arrange
    value = raw

    # Act
    result = _strip_quotes(value)

    # Assert
    assert result == expected


def test_pretrain_config_parse_assignments_with_export_and_comments_returns_valid_map() -> None:
    """Test purpose: parse shell-style assignments while skipping comments/invalid keys.

    Inputs: assignment lines including `export`, comments and invalid identifiers.
    Expected behavior: only valid keys are preserved with normalized values.
    """
    # Arrange
    lines = [
        "# comment",
        "export NUM_LAYERS=32",
        "HIDDEN_SIZE='7168'",
        "INVALID-KEY=100",
        "ROPE=50000;",
    ]

    # Act
    parsed = _parse_assignments(lines)

    # Assert
    assert parsed == {
        "NUM_LAYERS": "32",
        "HIDDEN_SIZE": "7168",
        "ROPE": "50000",
    }


def test_pretrain_config_parse_quoted_blocks_multiline_returns_named_block() -> None:
    """Test purpose: parse multi-line quoted block variables.

    Inputs: script lines containing a named quoted block.
    Expected behavior: block content is captured as a newline-joined string.
    """
    # Arrange
    lines = [
        "MODEL_ARGS=\"",
        "--num-layers 32",
        "--hidden-size 7168",
        "\"",
    ]

    # Act
    blocks = _parse_quoted_blocks(lines)

    # Assert
    assert "MODEL_ARGS" in blocks
    assert "--num-layers 32" in blocks["MODEL_ARGS"]


def test_pretrain_config_expand_vars_unknown_variable_kept_literal() -> None:
    """Test purpose: verify unknown variables are left unchanged.

    Inputs: text with both known and unknown variable placeholders.
    Expected behavior: known vars are expanded; unknown vars remain literal placeholders.
    """
    # Arrange
    text = "--hidden-size ${HS} --vocab-size $VS --missing $MISSING"
    variables = {"HS": "7168", "VS": "163840"}

    # Act
    expanded = _expand_vars(text, variables)

    # Assert
    assert expanded == "--hidden-size 7168 --vocab-size 163840 --missing $MISSING"


def test_pretrain_config_extract_torchrun_section_stops_at_tee_line() -> None:
    """Test purpose: extract only torchrun command section from a shell script.

    Inputs: lines before, during and after torchrun invocation.
    Expected behavior: extraction starts at torchrun and ends at tee TRAIN_LOG_PATH line.
    """
    # Arrange
    lines = [
        "echo start",
        "torchrun pretrain_gpt.py \\",
        "  --num-layers 32 \\",
        "2>&1 | tee ${TRAIN_LOG_PATH}",
        "echo done",
    ]

    # Act
    section = _extract_torchrun_section(lines)

    # Assert
    assert section[0].startswith("torchrun")
    assert "tee ${TRAIN_LOG_PATH}" in section[-1]
    assert all("echo done" not in line for line in section)


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("a  b   c", "a b c"),
        ("a \\\n b", "a b"),
        ("\\ta\\nb\\n", "a b"),
    ],
)
def test_pretrain_config_normalize_arg_text_whitespace_patterns_returns_single_spaced(raw: str, expected: str) -> None:
    """Test purpose: normalize escaped/newline-heavy argument strings.

    Inputs: raw argument text with backslashes, tabs and repeated spaces.
    Expected behavior: output is single-spaced and shell-parse friendly.
    """
    # Arrange
    text = raw

    # Act
    normalized = _normalize_arg_text(text)

    # Assert
    assert normalized == expected


def test_pretrain_config_parse_pretrain_script_minimal_script_returns_flags(tmp_path: Path) -> None:
    """Test purpose: ensure main parser extracts CLI flags from a minimal local script.

    Inputs: a temporary script containing variable assignment, block args and torchrun command.
    Expected behavior: parsed config exposes expected scalar and boolean flags.
    """
    # Arrange
    script_path = tmp_path / "pretrain.sh"
    script_path.write_text(
        "\n".join(
            [
                "NUM_LAYERS=32",
                "MODEL_ARGS=\"",
                "--hidden-size 7168",
                "--use-flash-attn",
                "\"",
                "torchrun pretrain_gpt.py $MODEL_ARGS --num-layers $NUM_LAYERS 2>&1 | tee ${TRAIN_LOG_PATH}",
            ]
        ),
        encoding="utf-8",
    )

    # Act
    cfg = parse_pretrain_script(str(script_path))

    # Assert
    assert get_int(cfg, "--num-layers") == 32
    assert get_int(cfg, "--hidden-size") == 7168
    assert get_bool(cfg, "--use-flash-attn") is True


@pytest.mark.parametrize(
    "raw,expected_int,expected_float,expected_bool",
    [
        ("1", 1, 1.0, True),
        ("0", 0, 0.0, False),
        ("true", None, None, True),
        ("off", None, None, False),
        ("invalid", None, None, None),
    ],
)
def test_pretrain_config_typed_getters_value_variants_return_expected(
    raw: str,
    expected_int: int | None,
    expected_float: float | None,
    expected_bool: bool | None,
) -> None:
    """Test purpose: validate typed getter behavior on numeric and boolean-like strings.

    Inputs: a parsed config containing a single `--x` flag with varying raw values.
    Expected behavior: getters coerce recognized values and fall back to defaults for invalid ones.
    """
    # Arrange
    script_cfg = type("Cfg", (), {"flags": {"--x": raw}})()

    # Act
    got_int = get_int(script_cfg, "--x")
    got_float = get_float(script_cfg, "--x")
    got_bool = get_bool(script_cfg, "--x")

    # Assert
    assert got_int == expected_int
    assert got_float == expected_float
    assert got_bool == expected_bool


@pytest.mark.benchmark
def test_pretrain_config_parse_pretrain_script_small_input_benchmark(benchmark, tmp_path: Path) -> None:
    """Test purpose: benchmark parser latency on a representative small script.

    Inputs: local generated script with common training flags.
    Expected behavior: parser runs quickly and returns a valid flag map.
    """
    # Arrange
    script_path = tmp_path / "bench_pretrain.sh"
    script_path.write_text(
        "\n".join(
            [
                "NUM_LAYERS=32",
                "HIDDEN_SIZE=7168",
                "torchrun pretrain_gpt.py --num-layers $NUM_LAYERS --hidden-size $HIDDEN_SIZE --bf16 2>&1 | tee ${TRAIN_LOG_PATH}",
            ]
        ),
        encoding="utf-8",
    )

    # Act
    cfg = benchmark(parse_pretrain_script, str(script_path))

    # Assert
    assert get_int(cfg, "--num-layers") == 32
