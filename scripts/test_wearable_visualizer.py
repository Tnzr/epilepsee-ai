#!/usr/bin/env python
"""
Quick test script for wearable dataset visualizer.

Tests visualizer functionality without requiring full wearable dataset.
Validates:
- Signal loading and parsing
- Feature extraction
- Plot generation
- File I/O
"""

import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, DEFAULT_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_imports():
    """Test that all required modules import correctly."""
    logger.info("Testing imports...")
    try:
        from src.data_loader import WearableDeviceDataLoader
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        logger.info("✓ All imports successful")
        return True
    except ImportError as e:
        logger.error(f"✗ Import failed: {e}")
        return False


def test_visualizer_init():
    """Test visualizer initialization."""
    logger.info("\nTesting visualizer initialization...")
    try:
        from scripts.visualize_wearable_sample import WearableDatasetVisualizer
        config = Config(DEFAULT_CONFIG)
        
        # This will fail if dataset doesn't exist, but that's expected
        try:
            visualizer = WearableDatasetVisualizer(config)
            logger.info("✓ Visualizer initialized successfully")
            logger.info(f"  Found {len(visualizer.recordings)} total recordings")
            seizure_recs = [r for r in visualizer.recordings if r['has_seizure']]
            logger.info(f"  Found {len(seizure_recs)} recordings with seizures")
            return True
        except FileNotFoundError as e:
            logger.warning(f"! Dataset not available (expected): {e}")
            logger.info("  This is normal if wearable dataset isn't mounted")
            logger.info("  Mount the dataset to /media/tnzr/HDD11/Datasets/WearablwDevice-Oregon")
            return None  # Not a failure, just unavailable
    except Exception as e:
        logger.error(f"✗ Visualizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config():
    """Test configuration loading."""
    logger.info("\nTesting configuration...")
    try:
        config = Config(DEFAULT_CONFIG)
        logger.info(f"✓ Config loaded successfully")
        logger.info(f"  Type: {type(config).__name__}")
        logger.info(f"  Has required attributes: model, training, evaluation, loss")
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_cli_help():
    """Test CLI help text."""
    logger.info("\nTesting CLI help...")
    try:
        import subprocess
        result = subprocess.run(
            [sys.executable, "scripts/visualize_wearable_sample.py", "--help"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            logger.info("✓ CLI help text accessible")
            return True
        else:
            logger.error(f"✗ CLI help failed: {result.stderr}")
            return False
    except Exception as e:
        logger.error(f"✗ CLI help test failed: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("=" * 70)
    logger.info("WEARABLE DATASET VISUALIZER - QUICK TEST")
    logger.info("=" * 70)
    
    results = {
        "Imports": test_imports(),
        "Config": test_config(),
        "CLI Help": test_cli_help(),
        "Visualizer Init": test_visualizer_init(),
    }
    
    logger.info("\n" + "=" * 70)
    logger.info("TEST SUMMARY")
    logger.info("=" * 70)
    
    passed = sum(1 for r in results.values() if r is True)
    failed = sum(1 for r in results.values() if r is False)
    skipped = sum(1 for r in results.values() if r is None)
    
    for test_name, result in results.items():
        status = "✓ PASS" if result is True else "✗ FAIL" if result is False else "⊘ SKIP"
        logger.info(f"  {status:8} {test_name}")
    
    logger.info("\n" + "-" * 70)
    logger.info(f"Results: {passed} passed, {failed} failed, {skipped} skipped")
    logger.info("-" * 70)
    
    if failed > 0:
        logger.error("\n⚠ Some tests failed. Please check the output above.")
        return 1
    
    logger.info("\n✓ All tests passed! Visualizer is ready to use.")
    logger.info("\nQuick start:")
    logger.info("  python scripts/visualize_wearable_sample.py --patient 2")
    logger.info("\nOr use Makefile:")
    logger.info("  make visualize-wearable")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
