#!/bin/bash
# Activation helper for Epilepsee-AI environment
# This script should be sourced after activating the conda environment

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Add project root to PYTHONPATH
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH}"

# Verify GPU availability
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s)"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
else
    echo "⚠ No GPUs detected (CPU mode)"
fi

echo ""
echo "Environment ready!"
echo "Project root: $PROJECT_ROOT"
echo "PYTHONPATH set"
echo ""
echo "Quick commands:"
echo "  python scripts/test_training.py          # Verify pipeline"
echo "  python scripts/train.py --help           # See training options"
echo "  python src/data_loader.py                # Test data loading"
