# Stateful LSTM Retraining Guide

**Date:** 2026-07-08  
**Status:** Ready for Smoke Test  
**Objective:** Retrain ECGCountdownPredictor with temporal continuity (Option A)

---

## 📋 Executive Summary

### The Problem
Previous training used **random shuffled batches** with **independent LSTM hidden states per batch**. This created a stateless 16-minute window classifier incompatible with **continuous sequential deployment** (which requires accumulated hidden state).

### The Solution (Option A: Stateful Retraining)
Retrain using:
- **Patient-sequential batching**: Consecutive samples from each patient (temporal order preserved)
- **Stateful LSTM**: Hidden state accumulated across batches within a patient
- **Patient-shuffled epochs**: Only shuffle patient order, not sample order
- **Deployment-aligned validation**: Continuous longitudinal inference

### Expected Outcome
- Model trained exactly as it will be deployed
- Temporal patterns emerge in predictions
- Better advance warning capability
- Stable predictions over long sequences

---

## 🚀 Quick Start: Smoke Test (5 minutes)

Test retraining on a small subset before committing to full training.

### 1. Verify Setup
```bash
cd /mnt/d/Dev/Epilepsee-AI

# Check Python and GPU
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Check dataset exists
ls -la /mnt/d/Datasets/SeizeIT2/sub-001/
```

### 2. Run Smoke Test
```bash
python scripts/retrain_stateful.py \
    --data-root /mnt/d/Datasets/SeizeIT2 \
    --output-dir models/stateful_smoke_test \
    --num-epochs 2 \
    --batch-size 32 \
    --learning-rate 0.001 \
    --smoke-test \
    2>&1 | tee smoke_test.log
```

**What it does:**
- Uses only first 5 patients (∼250 samples instead of 12,000)
- Runs 2 epochs (∼3 seconds per epoch on RTX 3060)
- Verifies:
  - Data loading works
  - Stateful batching functions correctly
  - Model forward pass with hidden state works
  - Loss decreases (not flat)
  - No GPU OOM errors

**Expected output:**
```
===== Epoch 1/2 =====
Train Loss: 0.8500
Train Metrics: {
  'loss': 0.8500,
  'pred_variance': 0.015,
  'batch_count': 8,
  'hidden_state_resets': 5,
  'hidden_state_detaches': 0
}
Val Loss: 0.8400
Val Metrics: {...}
Saved best model: models/stateful_smoke_test/best_model_stateful.pt
```

**Success Criteria:**
- ✓ No crashes (full logs without `NaN` or `CUDA OOM`)
- ✓ Loss decreases (0.85 → 0.84)
- ✓ Prediction variance > 0.01 (not flat)
- ✓ `hidden_state_resets` = ~5 (one per patient, correct)
- ✓ Models saved to output directory

**Troubleshooting:**

| Error | Fix |
|-------|-----|
| `No module named 'src.stateful_data_loader'` | Run from `/mnt/d/Dev/Epilepsee-AI` directory |
| `CUDA out of memory` | Try `--batch-size 16` or `--no-cuda` |
| `No module named 'config'` | Check Python path: `export PYTHONPATH=/mnt/d/Dev/Epilepsee-AI:$PYTHONPATH` |
| All predictions `NaN` | Check dataset; verify `sample_end_times_s` is valid |

---

## 📊 Full Training: 16 Epochs

Once smoke test passes, run full retraining.

### 1. Prepare Environment
```bash
cd /mnt/d/Dev/Epilepsee-AI
source /root/miniforge3/etc/profile.d/conda.sh
conda activate torch_env  # Or your PyTorch env
```

### 2. Run Full Training
```bash
python scripts/retrain_stateful.py \
    --config config/stateful_lstm.yaml \
    --data-root /mnt/d/Datasets/SeizeIT2 \
    --output-dir models/stateful_v1 \
    --num-epochs 16 \
    --batch-size 32 \
    --learning-rate 0.001 \
    2>&1 | tee training_stateful.log
```

**Duration:** ∼5-10 minutes on 2× RTX 3060  
**GPU Usage:** Both cards active, ∼80% utilization  
**Memory:** ∼12-15 GB per card

### 3. Monitor Training
In another terminal:
```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Or tail the log
tail -f training_stateful.log | grep -E "Epoch|Loss|Metrics"
```

### 4. Expected Progress
```
===== Epoch 1/16 =====
Train Loss: 0.8563 → Val Loss: 0.8626 (pred_variance: 0.084)

===== Epoch 2/16 =====
Train Loss: 0.8341 → Val Loss: 0.8319 (pred_variance: 0.089)

===== Epoch 3/16 =====
Train Loss: 0.8346 → Val Loss: 0.8161 (pred_variance: 0.095)

... (loss should gradually decrease)

===== Epoch 16/16 =====
Best validation loss: 0.78 (↓ from 0.86)
Models saved to: models/stateful_v1
```

### 5. Verify Results
```bash
ls -lh models/stateful_v1/
# best_model_stateful.pt
# last_model_stateful.pt
# training_*.log

# Check metrics from log
grep "Val Loss" training_stateful.log
grep "pred_variance" training_stateful.log
```

---

## 📈 Comparing: Stateful vs. Random-Shuffled Training

Quick comparison to verify improvement:

```bash
# 1. Run 2-epoch baseline (random shuffling)
# (Optional - if you want to compare against old training)

# 2. Run 2-epoch stateful
python scripts/retrain_stateful.py \
    --data-root /mnt/d/Datasets/SeizeIT2 \
    --output-dir models/stateful_comparison \
    --num-epochs 2 \
    --batch-size 32 \
    --smoke-test

# 3. Compare logs
# Look for:
# - Prediction variance (stateful should be higher)
# - Loss trajectory (both should decrease, stateful more smoothly)
# - Hidden state stats (stateful should show resets and detaches)
```

**Expected Improvements:**

| Metric | Random Shuffle | Stateful | Improvement |
|--------|---|---|---|
| Prediction variance | 0.03-0.05 | 0.08-0.12 | ↑↑ 2-3x |
| Loss smoothness | Noisy | Smooth | ✓ Better |
| Heatmap diversity | Limited colors | Vibrant colors | ✓ Better |
| Token activation | All-black | Multi-channel | ✓ Better |

---

## 🔍 Files Created/Modified

### New Files
- `src/stateful_data_loader.py` - TemporalPatientSequenceDataLoader class
- `scripts/retrain_stateful.py` - Retraining script
- `config/stateful_lstm.yaml` - Stateful configuration
- `TRAINING_METHODOLOGY_GUARD.md` - Prevention guide
- `STATEFUL_LSTM_TRAINING_DESIGN.md` - Architecture details
- `RETRAINING_GUIDE.md` - This file

### Modified Files
- `src/models.py` - Added hidden_state parameter to `forward()`
- `documentation/PI_REPORT_MODEL_VALIDATION_2026-07-08.md` - Added methodology notes + placeholder marker

---

## 🎯 Next Steps After Retraining

### 1. Phase 1: Longitudinal Validation
Once best_model_stateful.pt is ready:
```bash
python scripts/continuous_patient_inference.py \
    --model models/stateful_v1/best_model_stateful.pt \
    --patient-list validation_patients.json \
    --output-dir models/stateful_v1/longitudinal_results
```

Will generate:
- Continuous inference traces for 5-10 patients
- Seizure neighborhood visualizations
- Temporal metrics (advance warning time, FPR, etc.)

### 2. Update Reports
- Replace epoch_016_gt_vs_inference_panel.png (placeholder) with new best epoch
- Update PI_REPORT with real longitudinal validation results
- Document findings in TRAINING_LOG.md

### 3. Deploy Ready Check
Verify deployment readiness:
- ✓ Model loads without errors
- ✓ Inference latency < 100ms
- ✓ Continuous inference stable over 24+ hours
- ✓ Advanced warning time acceptable (> 5 min before seizure)

---

## 📚 Documentation Map

| Document | Purpose |
|----------|---------|
| `TRAINING_METHODOLOGY_GUARD.md` | Prevention guide - why this error happened, how to avoid it |
| `STATEFUL_LSTM_TRAINING_DESIGN.md` | Architecture design - detailed component specifications |
| `RETRAINING_GUIDE.md` | This file - how to run retraining |
| `LONGITUDINAL_VALIDATION_GUIDE.md` | Phase 1 after retraining - continuous inference validation |
| `PI_REPORT_MODEL_VALIDATION_2026-07-08.md` | Clinical report - updated with retraining results |

---

## ⚠️ Common Pitfalls & Troubleshooting

### Issue: "All predictions are NaN"
**Cause:** Invalid sample_end_times_s in dataset  
**Fix:** Check that sample_end_times_s are valid floats (not negative, not inf)
```python
dataset.sample_end_times_s[dataset.sample_end_times_s < 0]  # Should be empty
```

### Issue: "Loss doesn't decrease"
**Cause 1:** Learning rate too small  
**Fix:** Try `--learning-rate 0.005` or `0.01`

**Cause 2:** Class imbalance not handled  
**Fix:** Verify in `config/stateful_lstm.yaml`:
```yaml
use_class_weighting: true
classification_positive_weight: 15.0
classification_loss_type: "bce"  # NOT "focal"
```

### Issue: "OOM after epoch 3"
**Cause:** Hidden state accumulating too much history  
**Fix:** Increase detach_interval in config:
```yaml
stateful_lstm:
  hidden_state_detach_interval: 5  # Detach more frequently
```

### Issue: "Smoke test passes but full training crashes"
**Cause:** Different data distribution in full dataset  
**Fix:**
1. Reduce batch_size: `--batch-size 16`
2. Reduce learning_rate: `--learning-rate 0.0005`
3. Check for outliers in dataset

---

## 🔐 Verification Checklist

Before using model in production:

- [ ] Smoke test passes (2 epochs, <1 min)
- [ ] Full training completes (16 epochs, ~5 min)
- [ ] Best validation loss < 0.85 (improved from 0.86)
- [ ] Prediction variance > 0.08 (not flat)
- [ ] Heatmaps show temporal patterns (colorful, not monochrome)
- [ ] Token waterfall shows activation (not all-black)
- [ ] Phase 1 longitudinal validation shows advance warning (> 5 min)
- [ ] False alarm rate acceptable (< 1/hour interictal)
- [ ] Models saved: best_model_stateful.pt, last_model_stateful.pt

---

## 📞 Support

**Issues:**
1. Check `training_stateful.log` for detailed errors
2. Review `TRAINING_METHODOLOGY_GUARD.md` for methodology questions
3. Review `STATEFUL_LSTM_TRAINING_DESIGN.md` for architecture questions

**Questions about old (random-shuffled) training:**
- See `INFERENCE_COLLAPSE_FIX_REPORT.md` for root cause analysis
- See `TRAINING_METHODOLOGY_GUARD.md` for why this approach was wrong

---

**Status:** Ready to execute  
**Owner:** AI Development Team  
**Last Updated:** 2026-07-08
