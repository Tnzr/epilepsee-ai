# Oregon Watch Run Report (Confluence Ready)

Date: 2026-05-05

## Objective
Run a strict real-data training/evaluation pass on the Oregon Watch dataset and capture results for project tracking.

## Dataset Confirmation
- Dataset root used: $WEARABLE_DATASET_ROOT
- Strict mode enabled: `--strict-real-data`
- Source enabled: `--data-source wearable`
- Result: real wearable metadata loaded (13 recordings total, 4 with seizures)

## Final Completed Run (Reportable)
Command used:

```bash
python scripts/train.py \
  --model-type eegnet \
  --epochs 5 \
  --batch-size 64 \
  --learning-rate 0.0005 \
  --data-mode real \
  --data-source wearable \
  --dataset-root $WEARABLE_DATASET_ROOT \
  --strict-real-data \
  --max-recordings 3 \
  --max-samples-per-recording 120 \
  --real-stride-seconds 30.0 \
  --auto-threshold \
  --threshold-objective balanced_accuracy \
  --threshold-min-sensitivity 0.75 \
  --threshold-max-fpr 0.40
```

Notes:
- This run is strict real-data and completed end-to-end.
- Long-sweep attempts were also executed, but were too slow for immediate report completion in this session.

## Data Split Summary
- Total samples: 360
- Train: 230 (preictal 15, interictal 215)
- Val: 54 (preictal 1, interictal 53)
- Test: 90 (preictal 1, interictal 89)

## Training Summary
- Model: EEGNet
- Params: 4,018
- Epochs: 5
- Best validation MAE: 4.5439
- Training time: 11.45 seconds

Per-epoch (train_loss, val_loss, val_mae):
- E1: 0.8969, 0.2570, 4.5439
- E2: 0.5361, 0.2785, 4.8146
- E3: 0.5076, 0.2969, 4.9066
- E4: 0.4533, 0.3095, 4.9526
- E5: 0.4163, 0.2914, 5.1156

## Threshold Selection
- Selected threshold: 0.52
- Objective: balanced_accuracy = 0.6517
- Sensitivity: 1.0000
- Specificity: 0.3034
- FPR: 0.6966

## Test Metrics
- MAE: 5.7693
- RMSE: 5.7693
- MEDAE: 5.7693
- Accuracy: 0.3111
- AUROC: 0.6517
- F1: 0.0313
- Sensitivity: 1.0000
- Specificity: 0.3034
- Precision: 0.0159
- Sensitivity @3min: 0.0000
- Sensitivity @5min: 0.0000
- Sensitivity @10min: 0.0000
- Prediction stability: 1.7987

## Generated Artifacts
- models/results.json
- models/threshold_selection.json
- models/test_predictions.npz
- models/performance_overview.png
- models/gt_vs_inference_panel.png
- models/train_confusion_matrix.png
- models/val_confusion_matrix.png
- models/test_confusion_matrix.png

## Interpretation
- Pipeline and strict Oregon ingestion are functioning correctly.
- Current run has very limited preictal examples in val/test (1 each), so threshold and classification metrics are unstable and should not be treated as final scientific performance.
- Next reliable experiment should increase recording count and/or preictal coverage while keeping strict real-data mode.

## Recommended Next Run
For a stronger Oregon result (still strict real-data):
- Increase `--max-recordings` to 5 or 8.
- Keep `--max-samples-per-recording` bounded for runtime.
- If runtime allows, re-enable long-sweep mode and produce `test_predictions_long_sweep_bayes.npz`.
