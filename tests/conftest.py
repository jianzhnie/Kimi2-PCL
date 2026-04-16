"""Shared pytest fixtures and test-environment setup."""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
UTILS_DIR = REPO_ROOT / 'utils'
MODELS_DIR = REPO_ROOT / 'models'

# Keep imports stable and isolated from host PYTHONPATH.
for path in (REPO_ROOT, UTILS_DIR, MODELS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


@pytest.fixture(scope='session', autouse=True)
def test_env_defaults() -> None:
    """Set deterministic, offline-friendly defaults for all test sessions."""
    os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
    os.environ.setdefault('PYTHONHASHSEED', '0')


@pytest.fixture(scope='session', autouse=True)
def deterministic_seed() -> None:
    """Seed Python RNG to reduce flaky behavior in randomized tests."""
    random.seed(0)


@pytest.fixture
def repo_root() -> Path:
    """Return repository root path."""
    return REPO_ROOT


@pytest.fixture
def utils_dir() -> Path:
    """Return utilities directory path."""
    return UTILS_DIR


@pytest.fixture
def models_dir() -> Path:
    """Return models directory path."""
    return MODELS_DIR
