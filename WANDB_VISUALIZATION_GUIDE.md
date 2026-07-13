# Wandb Logging & Signal Visualization - Integration Summary

## What Was Added

### 1. **Weights & Biases (wandb) Integration** ✅
- Installed `wandb 0.25.0` in the epilepsee-ai environment
- Integrated wandb logging into `src/training.py`:
  - Automatic wandb run initialization with experiment metadata
  - Hyperparameter logging on run start
  - Real-time metric logging during training (loss, MAE, RMSE, etc.)
  - Final summary logging on training completion
  - Works automatically with multi-GPU DDP training

### 2. **Signal Visualization Module** ✅
New file: `src/visualization.py` (450+ lines)

**SignalVisualizer class** provides:
- **plot_signal_comparison()** - Side-by-side normal vs seizure signals
  - Shows amplitude statistics
  - Interactive time axis
  - Color-coded waveforms (steelblue for normal, crimson for seizure)

- **plot_countdown_prediction()** - Predicted vs true countdown over time
  - Scatter + line plot showing detection accuracy
  - Pre-ictal classification probabilities
  - Marks inter-ictal (gray) vs pre-ictal (green) regions
  - Optional decision threshold visualization

- **plot_error_distribution()** - Prediction error analysis
  - Histogram of percent errors
  - Scatter plot (predicted vs true) with identity line
  - Statistical summary (MAE, RMSE, mean error)

- **plot_feature_importance()** - Feature heatmap visualization
  - Normalized feature values over time
  - Useful for understanding what features the model learns

- **Automatic wandb upload** - All figures auto-uploaded to wandb if enabled

### 3. **Visualization Notebook** ✅
New file: `notebooks/signal_visualization_analysis.ipynb`

**Interactive Jupyter notebook** featuring:
1. Environment setup with wandb integration
2. SeizeIT2 dataset loading and inspection
3. Load real normal vs seizure signal examples
4. Signal comparison visualizations
5. Example model predictions (simulated)
6. Error analysis and performance metrics
7. Sensitivity calculations (detection @ 3/5/10 min lead times)
8. wandb metric logging for experiment tracking

### 4. **Integration Test Script** ✅
New file: `scripts/test_wandb_integration.py`

**Standalone test script** that:
- Creates synthetic datasets
- Trains model with wandb logging enabled
- Generates all visualization types
- Verifies wandb integration works end-to-end
- Can be run without actual dataset to test pipeline

---

## Usage

### Enable wandb Logging During Training

```bash
# Single GPU
python scripts/train.py --model-type ecg_lstm --epochs 10

# Multi-GPU (wandb works automatically)
python -m torch.distributed.launch --nproc_per_node=2 \
    scripts/train.py --model-type ecg_lstm --epochs 100
```

**First time only:** wandb will prompt you to create account or paste API key
```
wandb: Logging into wandb.ai. (Learn how to obtain an API key: https://wandb.me/go_login)
wandb: You can find your API key in your browser here: https://wandb.ai/authorize
```

### View Training Metrics on wandb

1. Navigate to https://wandb.ai/home
2. Find your "epilepsee-ai" project
3. Click the latest run
4. View real-time metrics:
   - train/loss, val/loss
   - train/regression, train/classification
   - val/mae, val/medae, val/rmse
   - learning_rate tracking
   - Final training summary

### Create Visualizations in Notebook

```bash
# Activate environment
mamba activate epilepsee-ai
export PYTHONPATH="/home/tnzr/Documents/FIU/Research/Epilepsee-AI:$PYTHONPATH"

# Start Jupyter and open notebook
jupyter lab notebooks/signal_visualization_analysis.ipynb
```

Run cells in order to:
1. Load real SeizeIT2 data
2. Create normal vs seizure signal comparisons
3. Simulate model predictions
4. Analyze detection errors
5. Log results to wandb

### Run Integration Test

```bash
python scripts/test_wandb_integration.py
```

Generates test visualizations in `test_visualizations/` folder without needing full training.

---

## Metrics Logged to wandb

### Per-Epoch Metrics
```
epoch                          # Epoch number
train/loss                    # Total training loss
train/classification          # Classification loss
train/regression              # Regression loss
val/loss                      # Validation total loss
val/classification            # Val classification loss
val/regression                # Val regression loss
val/mae                       # Mean absolute error (minutes)
val/medae                     # Median absolute error
val/rmse                      # Root mean squared error
learning_rate                 # Current learning rate
```

### Final Summary
```
final/training_time_hours     # Total training duration
final/best_val_mae            # Best validation MAE
```

### Dataset Info (logged once at start)
```
model_type                    # e.g., "ecg_lstm"
hidden_dim, dropout           # Architecture params
learning_rate, batch_size     # Training hyperparams
dataset_size                  # "883 seizures"
num_gpus                      # Number of GPUs used
```

---

## Key Features

✅ **Distributed Training Compatible** - Works with DDP on 2+ GPUs
✅ **Non-Blocking** - wandb logging doesn't slow down training
✅ **Automatic Initialization** - wandb.init() called once per run
✅ **Graceful Degradation** - Works even if wandb unavailable or offline
✅ **Production Ready** - Only main rank (rank=0) logs to avoid conflicts

---

## Files Modified/Created

### Modified
- `src/training.py` - Added wandb initialization and metric logging

### Created
- `src/visualization.py` - Signal visualization module (450+ lines)
- `notebooks/signal_visualization_analysis.ipynb` - Interactive analysis notebook
- `scripts/test_wandb_integration.py` - Integration test script

### Dependencies Added
- `wandb==0.25.0` (already installed)

---

## Next Steps for Full Training

1. **Start training with wandb logging:**
   ```bash
   python scripts/train.py --model-type ecg_lstm --epochs 50
   ```

2. **Monitor on wandb dashboard:**
   Visit https://wandb.ai/home during training

3. **Analyze signals in notebook:**
   Run `notebooks/signal_visualization_analysis.ipynb` to create custom visualizations

4. **For multi-GPU training:**
   ```bash
   python -m torch.distributed.launch --nproc_per_node=2 \
       scripts/train.py --model-type multimodal --epochs 200
   ```

---

## Example Output

When you run training with wandb enabled:
```
INFO:src.training:wandb initialized: https://wandb.ai/username/epilepsee-ai/runs/abc123xyz
Epoch 1/50
  Train Loss: 42.3921
  Val Loss: 35.2104
  Val MAE: 2.1234 min
Epoch 2/50
  Train Loss: 38.1023
  Val Loss: 32.4567
  Val MAE: 1.8901 min
...
Training completed in 0.34 hours
Best validation MAE: 1.8234
```

All metrics automatically logged to wandb!

---

**Status**: ✅ All integration tests passed  
**Ready for**: Production training with full experiment tracking
