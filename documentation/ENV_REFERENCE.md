# Epilepsee-AI Environment Quick Reference

## Environment Setup

### Initial Setup (One Time)

```bash
# Navigate to project
cd $PROJECT_DIR

# The environment has already been created!
# Location: $HOME/.local/share/mamba/envs/epilepsee-ai
```

## Daily Usage

### Activate Environment

```bash
# Using mamba (recommended - faster)
mamba activate epilepsee-ai

# OR using conda
conda activate epilepsee-ai

# Set PYTHONPATH for project imports
source activate.sh
```

**Or do both in one command:**

```bash
mamba activate epilepsee-ai && source activate.sh
```

### Verify Installation

```bash
# Check PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}'); print(f'GPU count: {torch.cuda.device_count()}')"

# Test data loader
python src/data_loader.py

# Run full verification suite
python scripts/test_training.py
```

### Training Commands

```bash
# Single GPU training (testing with 10 epochs)
python scripts/train.py --model-type ecg_lstm --epochs 10 --batch-size 32

# Multi-GPU training (2 GPUs)
python -m torch.distributed.launch --nproc_per_node=2 scripts/train.py --model-type ecg_lstm --epochs 100

# With custom config
python scripts/train.py --config config/custom_config.yaml --model-type cnn_lstm --learning-rate 0.0001

# Different models
python scripts/train.py --model-type ecg_lstm      # LSTM + Attention (baseline)
python scripts/train.py --model-type cnn_lstm      # CNN-LSTM hybrid
python scripts/train.py --model-type multimodal    # Multi-stream fusion
```

## Environment Management

### Update Environment

```bash
# After modifying environment.yml
mamba env update -f environment.yml --prune
```

### Export Environment

```bash
# Export exact versions (for reproducibility)
conda env export > environment_frozen.yml
```

### Deactivate

```bash
conda deactivate
```

### Remove Environment (if needed)

```bash
conda env remove -n epilepsee-ai
```

## Package Information

### Installed Packages

- **Python**: 3.10
- **PyTorch**: 2.5.1 with CUDA 12.4 support
- **MNE-Python**: For EEG/ECG data processing
- **MNE-BIDS**: For BIDS format handling
- **PyEDF**: For EDF file reading
- **Weights & Biases**: For experiment tracking
- **TensorBoard**: For training visualization
- **JupyterLab**: For interactive development

### GPU Info

```bash
# Check GPU availability
nvidia-smi

# Check from Python
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

## Troubleshooting

### PYTHONPATH Issues

If you get `ModuleNotFoundError: No module named 'config'`:

```bash
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
```

Or add to `~/.bashrc` for permanent fix:

```bash
echo 'export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"' >> ~/.bashrc
source ~/.bashrc
```

### CUDA Issues

If CUDA is not detected:

```bash
# Check CUDA version
nvidia-smi

# Reinstall PyTorch with correct CUDA
mamba install pytorch torchvision torchaudio pytorch-cuda -c pytorch -c nvidia
```

### Environment Activation Not Working

```bash
# Initialize conda for your shell (one time)
conda init bash

# Restart terminal
```

## Quick Start Workflow

```bash
# 1. Activate environment
mamba activate epilepsee-ai

# 2. Set PYTHONPATH
source activate.sh

# 3. Verify setup
python scripts/test_training.py

# 4. Start training
python scripts/train.py --model-type ecg_lstm --epochs 10
```

## Additional Resources

- **Documentation**: See [README.md](README.md)
- **Implementation Guide**: See [IMPLEMENTATION.md](IMPLEMENTATION.md)
- **Development Plan**: See [Project_Development_Plan.md](Project_Development_Plan.md)
