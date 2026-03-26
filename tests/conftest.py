import os
import sys
import pytest

REPO_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
UTILS_DIR = os.path.join(REPO_ROOT, "utils")
MODELS_DIR = os.path.join(REPO_ROOT, "models")

if UTILS_DIR not in sys.path:
    sys.path.insert(0, UTILS_DIR)

if MODELS_DIR not in sys.path:
    sys.path.insert(0, MODELS_DIR)

@pytest.fixture
def repo_root():
    return REPO_ROOT

@pytest.fixture
def utils_dir():
    return UTILS_DIR
