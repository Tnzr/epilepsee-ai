#!/bin/bash
# Setup script for Epilepsee-AI conda environment

set -e  # Exit on error

PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
ENV_NAME="epilepsee-ai"

echo "============================================================"
echo "  Epilepsee-AI Environment Setup"
echo "============================================================"
echo ""

# Check if mamba is available, otherwise use conda
if command -v mamba &> /dev/null; then
    CONDA_CMD="mamba"
    echo "✓ Using mamba (faster)"
else
    CONDA_CMD="conda"
    echo "✓ Using conda"
fi

echo ""
echo "Creating conda environment: $ENV_NAME"
echo "This may take several minutes..."
echo ""

# Create environment
$CONDA_CMD env create -f environment.yml

echo ""
echo "============================================================"
echo "  Environment created successfully!"
echo "============================================================"
echo ""
echo "To activate the environment, run:"
echo ""
echo "  conda activate $ENV_NAME"
echo ""
echo "Or if using mamba:"
echo ""
echo "  mamba activate $ENV_NAME"
echo ""
echo "To add project to PYTHONPATH automatically, add this to ~/.bashrc:"
echo ""
echo "  export PYTHONPATH=\"${PROJECT_ROOT}:\$PYTHONPATH\""
echo ""
echo "Or run this command now (temporary for current session):"
echo ""
echo "  export PYTHONPATH=\"${PROJECT_ROOT}:\$PYTHONPATH\""
echo ""
echo "============================================================"
echo "  Quick Start"
echo "============================================================"
echo ""
echo "1. Activate environment:"
echo "   conda activate $ENV_NAME"
echo ""
echo "2. Set PYTHONPATH:"
echo "   export PYTHONPATH=\"${PROJECT_ROOT}:\$PYTHONPATH\""
echo ""
echo "3. Test data loader:"
echo "   python scripts/test_training.py"
echo ""
echo "4. Start training:"
echo "   python scripts/train.py --model-type ecg_lstm --epochs 10"
echo ""
echo "5. Multi-GPU training (2 GPUs):"
echo "   python -m torch.distributed.launch --nproc_per_node=2 scripts/train.py"
echo ""
