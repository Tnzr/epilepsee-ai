# PI Update: AI Performance and Deployment Readiness

Date: 2026-07-04  
Project: Epilepsee-AI  
Run ID: seizeit2_tcn_longrun_e12_panelfix

## 1) Executive Summary

- Completed a full 12-epoch long-sweep training and evaluation cycle on SeizeIT2-derived data using the TCN model.
- Achieved best validation MAE of 3.1770 minutes and test MAE of 3.1294 minutes.
- Bayesian long-sweep memory simulation completed with bayes_auroc=0.7304 and bayes_ece=0.1577.
- Main blocker identified and now addressed in code: post-evaluation panel indexing crash during visualization generation (did not invalidate checkpoint or core evaluation metrics).
- For real-time watch demo preparation, development is now positioned to split into three services: AI Inference, Live Visualization, and Bluetooth Streaming.

## 2) Experimental Context

Configuration highlights from run logs:
- Model: tcn
- Device: CPU (no GPU detected)
- Data source: bids (fallback directory scanning; pybids unavailable in runtime)
- Train/Val/Test windows: 8640 / 2160 / 3600
- Unique recordings represented: 120
- Train class distribution: preictal=288, interictal=8352
- Val class distribution: preictal=73, interictal=2087
- Total trainable parameters: 238,338
- Training duration: 0.58 hours

Artifacts produced:
- Checkpoint: models/seizeit2_tcn_longrun_e12_panelfix/last_model.pt
- Training history: models/seizeit2_tcn_longrun_e12_panelfix/training_history.json
- Performance figure: models/seizeit2_tcn_longrun_e12_panelfix/performance_overview.png
- Epoch panels and anomaly reports through epoch 12 under models/seizeit2_tcn_longrun_e12_panelfix/

Generated figure outputs (direct links):
- [models/seizeit2_tcn_longrun_e12_panelfix/performance_overview.png](../models/seizeit2_tcn_longrun_e12_panelfix/performance_overview.png)
- [models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_010_gt_vs_inference_panel.png](../models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_010_gt_vs_inference_panel.png)
- [models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_011_gt_vs_inference_panel.png](../models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_011_gt_vs_inference_panel.png)
- [models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_012_gt_vs_inference_panel.png](../models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_012_gt_vs_inference_panel.png)

Preview (performance overview):

![Performance Overview](../models/seizeit2_tcn_longrun_e12_panelfix/performance_overview.png)

Preview (Epoch 12 GT vs Inference panel):

![Epoch 12 GT vs Inference](../models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_012_gt_vs_inference_panel.png)

## 3) Quantitative Results

## 3.1 Training and Validation

- Best validation MAE: 3.17697 minutes
- Final epoch (12/12):
  - Train loss: 0.3894
  - Val loss: 0.5643
  - Val MAE: 3.3844 minutes

Interpretation:
- Training converged stably overall, with expected epoch-to-epoch variance.
- Best MAE occurred before final epoch, indicating non-monotonic late-epoch validation behavior.

## 3.2 Test Evaluation

Regression metrics:
- MAE: 3.1294
- MedAE: 2.5685
- RMSE: 3.7859
- MAE std: 2.1306

Countdown bucket metrics:
- 0-2 min: n=19, MAE=1.7343
- 2-5 min: n=40, MAE=1.1796
- 5-10 min: n=61, MAE=4.8425

Classification metrics:
- Accuracy: 0.9667
- AUROC: 0.6398
- F1: 0.0000
- Sensitivity: 0.0000
- Specificity: 1.0000
- Precision: 0.0000

Clinical metrics:
- Sensitivity @3 min: 0.3793
- Sensitivity @5 min: 0.0000
- Sensitivity @10 min: 0.0000
- FPR per 8h: 0.0000

Stability:
- Prediction stability: 0.5367

Bayesian memory simulation:
- bayes_auroc: 0.7304
- bayes_ece: 0.1577

## 4) Key Technical Observations

1. Positive signal:
- Countdown error in 2-5 minute window is comparatively low (MAE ~1.18 min), indicating meaningful local anticipation behavior in that band.

2. Major performance gap:
- Classification head behavior is overly conservative at current thresholding (zero recall/sensitivity despite high apparent accuracy from class imbalance).

3. Horizon degradation:
- Error increases substantially in 5-10 minute bucket (MAE ~4.84 min), showing long-horizon timing remains the main weak point.

4. Data/modality anomaly trend:
- Epoch anomaly reports repeatedly flag missing/flat eeg/emg/mov and near-constant ecg proxy in selected panel windows.
- This suggests panel-source quality and modality availability are currently limiting interpretability and possibly affecting robustness.

5. Output semantics caveat:
- Under state-loss mode, countdown-like outputs can cluster in narrow ranges and should be interpreted carefully unless explicitly calibrated as minutes.

## 5) Reliability and Engineering Status

- Core training/evaluation path completed successfully and produced all primary artifacts.
- A post-evaluation visualization crash occurred due to panel index mismatch against partially materialized test windows.
- The index-guard fix has been applied to scripts/train.py to prevent recurrence.
- This issue affected final panel rendering stage only, not model checkpointing or metric computation.

## 6) Next Technical Actions (Priority-Ordered)

1. Classification operating point calibration
- Sweep thresholds and report recall/specificity trade-offs aligned to preictal detection goals.
- Target: non-zero sensitivity with controlled FPR per 8h.

2. Long-horizon countdown calibration
- Improve 5-10 minute bucket error via calibration and/or loss weighting adjustments.

3. Modality-quality gating
- Enforce signal-quality-aware panel and inference paths to avoid misleading flat/missing overlays.

4. Re-run validation after panel fix
- Produce a clean no-crash end-to-end report package for archival and PI review.

## 7) Containerized Service Roadmap (Planned)

When moving to live smartwatch integration, services will be separated into dedicated containers:

1. Bluetooth Streaming Service
- Responsibility: BLE device connectivity, packet decoding, timestamping, buffering.
- Inputs: Watch streams (PPG, EDA, optional ECG, IMU).
- Outputs: Normalized stream events on message bus.

2. AI Inference Service
- Responsibility: online preprocessing, feature windows, model inference, alert state logic.
- Inputs: Stream events from Bluetooth service.
- Outputs: predictions, confidence, alert state, quality flags.

3. Live Visualization Service
- Responsibility: operator dashboard for real-time traces, predictions, and event annotations.
- Inputs: predictions + stream telemetry.
- Outputs: web UI panels and session exports.

Cross-service requirements:
- Time-synchronized event schema
- Health endpoints and metrics
- Retry/degraded mode policies for stream loss
- Persistent run artifacts for reproducibility

## 8) PI-Facing Summary (Copy/Paste)

We completed a full long-sweep TCN run (12 epochs) and obtained a best validation MAE of 3.177 min with test MAE 3.129 min. Bayesian long-sweep simulation also completed (AUROC 0.7304, ECE 0.1577). Core model training/evaluation is stable and artifacts are produced consistently (checkpoint, history, performance overview). The main current gap is detection operating point: high class-imbalance accuracy (0.9667) but zero sensitivity at current thresholding, so threshold calibration is now top priority. We also resolved a post-eval panel rendering crash (index mismatch in visualization path), and the fix is in place. Next phase is deployment-oriented engineering with a three-container architecture: Bluetooth Streaming, AI Inference, and Live Visualization, to support real-time smartwatch integration (PPG/EDA with optional ECG).

## 9) Appendix: Exact Reported Metrics

From run logs:
- Best validation MAE: 3.1770
- Test MAE: 3.1294
- Test MedAE: 2.5685
- Test RMSE: 3.7859
- Accuracy: 0.9667
- AUROC: 0.6398
- F1: 0.0000
- Sensitivity: 0.0000
- Specificity: 1.0000
- Sensitivity@3m: 0.3793
- Sensitivity@5m: 0.0000
- Sensitivity@10m: 0.0000
- FPR/8h: 0.0000
- Stability: 0.5367
- bayes_auroc: 0.7304
- bayes_ece: 0.1577
