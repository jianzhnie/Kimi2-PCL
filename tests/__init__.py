"""
Kimi2-PCL Test Suite

This package contains comprehensive unit tests for the Kimi2-PCL project.

Test Organization:
- test_modeling_comprehensive.py: Comprehensive model architecture tests (150+ tests)
- test_config_comprehensive.py: Configuration module tests (50+ tests)
- test_conversion_comprehensive.py: Checkpoint conversion tests (40+ tests)
- test_utils_comprehensive.py: Utility function tests (30+ tests)
- test_modeling_modules.py: Core module tests (7 tests)
- test_models.py: Model integration tests (2 tests)
- test_config_and_conversion.py: Config/conversion integration tests (5 tests)
- test_align_pretrain_config.py: Pretrain config alignment tests (4 tests)
- test_check_model_weights.py: Model weight verification tests (2 tests)
- test_utils_dummy.py: Utility main function tests (4 tests)

Total: 300+ test cases covering:
- Branch coverage targeting >= 90%
- Boundary and edge case testing
- Parameterized testing for multiple configurations
- Exception testing for error handling
- Performance benchmarks

Usage:
    # Run all tests
    python -m pytest tests/ -v

    # Run with coverage
    python -m pytest tests/ --cov=models --cov=utils --cov-report=html

    # Run specific test modules
    python -m pytest tests/test_modeling_comprehensive.py -v

    # Run benchmarks only
    python -m pytest tests/ -m benchmark -v
"""
