#!/bin/bash
# Quick activation script - run with: source quick_activate.sh

# Activate the epilepsee-ai environment
if command -v mamba &> /dev/null; then
    eval "$(conda shell.bash hook)"
    mamba activate epilepsee-ai
else
    conda activate epilepsee-ai
fi

# Set PYTHONPATH to repo root (works from any checkout location)
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):$PYTHONPATH"

# Show status
echo ""
echo "✓ Environment activated: epilepsee-ai"
echo "✓ PYTHONPATH set"
echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ GPUs available: $GPU_COUNT"
    nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | sed 's/^/  GPU /'
else
    echo "⚠  No GPUs detected (CPU mode)"
fi

echo ""
echo "Ready to train! Try:"
echo "  python scripts/test_training.py  # Verify everything works"
echo "  python scripts/train.py --help   # See training options"
echo ""
