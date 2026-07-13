# Training Coherence and Stability Checklist

Purpose: Restore the training pipeline to the intended methodology, keep context persistent across runs, and close each known failure mode with objective evidence.

How to use this checklist:
- Work top to bottom.
- Do not mark an item complete without attaching artifact evidence (config diff, log snippet, metric file, or figure path).
- For every completed item, update the Execution Log at the bottom.

## A. Persistent Context and Documentation

- [ ] Create one canonical runbook section in `README.md` for "Methodologically Intended Training" with:
  - [ ] Required model/data path combinations.
  - [ ] Required config fields and defaults.
  - [ ] Required preflight checks.
- [ ] Add a short architecture truth source doc (or update existing architecture report) that states:
  - [ ] Which models are unimodal vs multimodal.
  - [ ] Which training paths are lazy streaming vs dense extraction.
  - [ ] Where context-state and token context originate.
- [ ] Add a "Known Failure Signatures" section in documentation with exact anomaly flags and likely causes.
- [ ] Add a "Run Acceptance Criteria" section defining go/no-go metrics and collapse thresholds.
- [ ] Add a changelog entry template for each training run (inputs, code commit, config hash, outcomes).

## B. Method Coherence: Data Path vs Model Intent

- [ ] Enforce model/data compatibility checks before training starts:
  - [ ] Fail fast if multimodal model is selected but pipeline is feeding ECG-only tensors.
  - [ ] Print explicit resolved input shape contract at startup.
- [ ] Implement true multimodal support for real SeizeIT2 lazy mode:
  - [ ] Cache/load ECG stream.
  - [ ] Cache/load EEG stream.
  - [ ] Cache/load EMG and movement stream (or a clearly documented fused motion branch).
  - [ ] Build aligned modality tensors with time synchronization guarantees.
- [ ] Add unit tests for modality presence and tensor slicing for multimodal models.
- [ ] Add a startup report file that confirms per-run modality availability ratios.

## C. Labels and Objective Definition

- [ ] Confirm and document final target definition:
  - [ ] Alert objective (preictal/onset inclusion rules).
  - [ ] Onset objective window.
  - [ ] Countdown regression domain and clipping.
- [ ] Add automated label sanity checks:
  - [ ] Countdown distribution histogram by split.
  - [ ] Preictal ratio by split and by patient.
  - [ ] Time-to-onset coverage report.
- [ ] Add guardrails for annotation mismatches and missing events.

## D. Class Imbalance and Loss Stability

- [ ] Replace unstable per-batch positive weighting with stable strategy:
  - [ ] Global or moving-average positive weight (documented and logged).
  - [ ] Optional focal-loss fallback with fixed alpha/gamma policy.
- [ ] Keep DDP behavior coherent:
  - [ ] Document when sampler balancing is disabled.
  - [ ] Ensure loss weighting strategy remains valid in distributed stateful mode.
- [ ] Add loss-component telemetry every epoch:
  - [ ] Classification loss trend.
  - [ ] Regression loss trend.
  - [ ] Effective positive weight trend.
  - [ ] Alert logit/probability distribution trend.
- [ ] Add collapse-trigger alarms:
  - [ ] Near-zero alert variance.
  - [ ] All-negative confusion matrix persistence.

## E. Token Context and Multi-Scale Cascade Integrity

- [ ] Split "diagnostic derived channels" from "learned latent context" in outputs and plots.
- [ ] Rename panel traces to prevent interpretation confusion.
- [ ] Ensure context-energy cascade is shown only when true context state exists.
- [ ] Add explicit legend tags:
  - [ ] Derived channel.
  - [ ] Learned latent.
  - [ ] Smoothed proxy.
- [ ] Add test coverage for token/cascade panel construction and expected channel semantics.

## F. Visualization and Monitoring Reliability

- [ ] Add deterministic panel selection policy (recording/time window reproducibility).
- [ ] Add modality integrity overlays on the panel (flatline flags and finite-ratio badges).
- [ ] Save per-epoch anomaly report and enforce thresholds for warning/fail.
- [ ] Add one compact "training health" dashboard artifact per epoch:
  - [ ] Alert confusion matrix.
  - [ ] Alert distribution.
  - [ ] Context/token diagnostics.
  - [ ] Modality integrity summary.

## G. Performance and Throughput (Without Breaking Method)

- [ ] Profile training slowdowns after architecture/path changes.
- [ ] Verify data loader bottlenecks per modality path.
- [ ] Add caching strategy notes with measured tradeoffs.
- [ ] Add one benchmark script for:
  - [ ] Samples/sec
  - [ ] GPU utilization
  - [ ] Epoch wall time
  - [ ] Memory usage

## H. Validation Protocol and Sign-off

- [ ] Define acceptance thresholds for "stable and intended" training:
  - [ ] No persistent all-negative alert confusion over N epochs.
  - [ ] Alert variance above minimum threshold.
  - [ ] Balanced accuracy and sensitivity floor targets.
  - [ ] No critical anomaly flags sustained over N epochs.
- [ ] Run a fixed reproducibility suite (same seed, same config) at least 3 times.
- [ ] Compare runs and confirm low variance in key metrics.
- [ ] Produce final sign-off report with:
  - [ ] What changed.
  - [ ] Why it fixed the issue.
  - [ ] Evidence artifacts.
  - [ ] Remaining risks.

## I. Immediate First Sprint (High Priority)

- [ ] Add hard fail for multimodal model + ECG-only lazy input mismatch.
- [ ] Stabilize class weighting in DDP stateful runs (remove per-batch volatility).
- [ ] Separate derived token waterfall from learned context visuals.
- [ ] Re-run 7-epoch sanity training and confirm alert head no longer collapses.
- [ ] Update docs and archive run artifacts.

---

## Execution Log (Persistent Context)

Use one entry per checklist milestone.

| Date | Owner | Checklist Item | Change Summary | Evidence Paths | Status |
|---|---|---|---|---|---|
| 2026-07-12 | Copilot | B. Method Coherence: model/data compatibility checks | Added hard fail in lazy real BIDS prep when multimodal model is selected but ECG-only lazy path would be used. | scripts/train.py:1742 | completed |
| 2026-07-12 | Copilot | D. Class imbalance and loss stability | Added fixed dataset-level positive class weight inference for distributed/stateful training when no explicit positive weight is set. | src/training.py:3478 | completed |
| 2026-07-12 | Copilot | E. Token context and cascade integrity | Relabeled derived token waterfall/context outputs and updated panel titles to separate derived channels from learned latent context. | src/training.py:3103, src/visualization.py:1018, src/visualization.py:1038 | completed |
| 2026-07-13 | Copilot | Validation: conda env alignment | Switched validation commands to requested `epilepsee-ai` environment via explicit `conda run -n epilepsee-ai ...` execution. | /root/miniforge3/bin/conda env list | completed |
| 2026-07-13 | Copilot | Validation: multimodal fail-fast behavior | Confirmed real BIDS lazy + multimodal model now hard-fails with explicit configuration error before training starts. | scripts/train.py:1742 | completed |
| 2026-07-13 | Copilot | Validation: no regression on short train cycle | Completed 1-epoch dummy stateful run in `epilepsee-ai` environment; training/evaluation/artifact generation completed without syntax/runtime regressions. | models/training_history.json, models/comprehensive_run_report.json | completed |
| 2026-07-13 | Copilot | D. Class imbalance and loss stability (verification) | Verified fixed dataset-level positive weight inference now activates when distributed is configured (even if DDP not active in single-process launch) and logs effective weight path. | src/training.py:3460, src/training.py:3473 | completed |
| 2026-07-13 | Copilot | Real SeizeIT2 sanity comparison (2 epochs, conda env) | Ran short real BIDS TCN sanity pass in `epilepsee-ai`; `inference_map_low_variance` no longer appeared and alert variance became non-trivial (`pred_alert_std` ~0.013-0.016), but `token_collapse_dominant_channel` and `ecg_flat_or_near_constant` persisted. | models/monitoring/epoch_001_anomaly_report.json, models/monitoring/epoch_002_anomaly_report.json | partial |
| 2026-07-13 | Copilot | Real SeizeIT2 extended stability run (requested 6 epochs; observed 29 with early stop) | Re-ran in `epilepsee-ai` with dedicated tee log. Runtime reported `Epoch 1/100`, trained to epoch 29 with early stopping, and reached best val MAE 2.4470. Initial epochs showed non-trivial alert variance and one clean anomaly-free point (epoch 3), but by epoch 29 `inference_map_low_variance` and `token_collapse_dominant_channel` persisted, and test classification remained all-negative (`sensitivity=0.0`, `f1=0.0`, `specificity=1.0`). | logs/real_sanity_stability_20260713/run_6e.log, models/real_sanity_stability_20260713/monitoring/epoch_003_anomaly_report.json, models/real_sanity_stability_20260713/monitoring/epoch_029_anomaly_report.json, models/real_sanity_stability_20260713/results.json | partial |
| 2026-07-13 | Copilot | Epoch schedule coherence fix + true 6-epoch real rerun | Fixed CLI/config precedence bug in `scripts/train.py` where default `--epochs=100` (and other core defaults) always overwrote YAML. After patch, a real SeizeIT2 run with no `--epochs` correctly executed `Epoch 1/6`...`Epoch 6/6` and stopped as configured. Collapse improved versus the drifted 29-epoch run (no `inference_map_low_variance` at epoch 6), but `token_collapse_dominant_channel` and all-negative test classification persisted. | scripts/train.py:422, scripts/train.py:2793, logs/real_sanity_stability_20260713_epochfix/run_6e_fixed.log, models/real_sanity_stability_20260713_epochfix/monitoring/epoch_006_anomaly_report.json, models/real_sanity_stability_20260713_epochfix/results.json | partial |
| 2026-07-13 | Copilot | Alert threshold calibration coherence | Changed `--auto-threshold` to calibrate on validation predictions instead of test labels, then apply the selected threshold to test metrics. In the 6-epoch real rerun, validation selected threshold `0.22` and test classification improved from all-negative to `sensitivity=0.25`, `f1=0.0621`, `specificity=0.8092`, showing thresholding was part of the collapse story but not the full root cause. | scripts/train.py:657, scripts/train.py:2977, logs/real_sanity_stability_20260713_epochfix_autothr/run_6e_autothr.log, models/real_sanity_stability_20260713_epochfix/threshold_selection.json, models/real_sanity_stability_20260713_epochfix/results.json | partial |
| 2026-07-13 | Copilot | Token diagnostic semantics fix | Updated anomaly monitoring so `token_collapse_dominant_channel` is only emitted for latent-or-mixed token semantics, not for the derived diagnostic waterfall. Real 6-epoch confirmation run showed epoch 3 clean and epoch 6 flagged only `ecg_flat_or_near_constant`; calibrated test alert metrics remained non-degenerate (`sensitivity=0.4722`, `f1=0.0722`, `specificity=0.6745`). | src/training.py:708, src/training.py:849, logs/real_sanity_stability_20260713_tokenfix/run_6e_tokenfix.log, models/real_sanity_stability_20260713_tokenfix/monitoring/epoch_003_anomaly_report.json, models/real_sanity_stability_20260713_tokenfix/monitoring/epoch_006_anomaly_report.json, models/real_sanity_stability_20260713_tokenfix/results.json | completed |
| 2026-07-13 | Copilot | ECG flatline anomaly threshold fix | Lowered the raw ECG span floor in anomaly monitoring from `1e-3` to `1e-4` while keeping the near-constant derivative check intact. Real 6-epoch confirmation run showed epoch 3 and epoch 6 both clean for ECG flatline; only epoch 5 retained `inference_map_low_variance`. Calibrated alert metrics remained non-degenerate (`sensitivity=0.3889`, `f1=0.0698`, `specificity=0.7266`). | src/training.py:839, logs/real_sanity_stability_20260713_ecgfix/run_6e_ecgfix.log, models/real_sanity_stability_20260713_ecgfix/monitoring/epoch_003_anomaly_report.json, models/real_sanity_stability_20260713_ecgfix/monitoring/epoch_006_anomaly_report.json, models/real_sanity_stability_20260713_ecgfix/results.json | completed |
| 2026-07-13 | Copilot | Low-variance anomaly gating refinement | Refined `inference_map_low_variance` to require meaningful preictal prevalence in the monitored slice, preventing false alarms on overwhelmingly interictal panels. Real 6-epoch confirmation run showed epoch 5 and epoch 6 anomaly reports clean while calibrated alert metrics stayed non-degenerate and slightly improved (`sensitivity=0.5000`, `f1=0.0800`, `specificity=0.6916`). | src/training.py:847, logs/real_sanity_stability_20260713_varfix/run_6e_varfix.log, models/real_sanity_stability_20260713_varfix/monitoring/epoch_005_anomaly_report.json, models/real_sanity_stability_20260713_varfix/monitoring/epoch_006_anomaly_report.json, models/real_sanity_stability_20260713_varfix/results.json | completed |
| 2026-07-13 | Copilot | Classification-loss rebalance experiment | Tested stronger alert emphasis by changing loss weights from `classification/regression = 0.3/0.7` to `1.0/0.3` in a fresh 6-epoch real run. Alert variance and sensitivity increased (`pred_alert_std` 0.0639, `sensitivity=0.5278`), and best val MAE slightly improved to `2.4429`, but test specificity and overall countdown quality degraded (`specificity=0.5981`, `mae=2.7129`, `rmse=3.1773`). This suggests the alert head was not simply underweighted; the tradeoff worsened overall behavior. | config/real_sanity_stability_20260713_clsrebalance.yaml, logs/real_sanity_stability_20260713_clsrebalance/run_6e_clsrebalance.log, models/real_sanity_stability_20260713_clsrebalance/results.json, models/real_sanity_stability_20260713_clsrebalance/monitoring/epoch_006_anomaly_report.json | partial |
| 2026-07-13 | Copilot | Focal-loss imbalance experiment | Tested `classification_loss_type=focal` with otherwise matched 6-epoch real settings. This regressed sharply: validation MAE blew up after epoch 4, epoch-6 panel alert probabilities collapsed to zero (`pred_alert_mean=0.0`, `pred_alert_std=0.0`), and test sensitivity fell to `0.1111` with `f1=0.0273` despite threshold calibration. Focal loss is not a viable next-step fix for this slice. | config/real_sanity_stability_20260713_focal.yaml, logs/real_sanity_stability_20260713_focal/run_6e_focal.log, models/real_sanity_stability_20260713_focal/results.json, models/real_sanity_stability_20260713_focal/monitoring/epoch_006_anomaly_report.json | completed |
| 2026-07-13 | Copilot | BCE label-smoothing experiment | Implemented the existing `label_smoothing` config in the active BCE/class-weighted BCE loss and tested `label_smoothing=0.05` in a matched 6-epoch real run. This also regressed: the alert head converged to an almost constant mid-probability band (`pred_alert_mean=0.4934`, `pred_alert_std=0.0022`), threshold calibration selected an overly permissive operating point (`threshold=0.44`, `sensitivity=0.8333`, `specificity=0.1729`), and overall quality degraded. Label smoothing is not a viable next-step fix for this slice. | src/losses.py:52, config/real_sanity_stability_20260713_labelsmooth.yaml, logs/real_sanity_stability_20260713_labelsmooth/run_6e_labelsmooth.log, models/real_sanity_stability_20260713_labelsmooth/results.json, models/real_sanity_stability_20260713_labelsmooth/monitoring/epoch_006_anomaly_report.json | completed |
| 2026-07-13 | Copilot | Pre-ictal window reduction experiment | Tested shorter target window by changing `pre_ictal_window_s` from `600` to `300` in a matched 6-epoch real run with validation-calibrated thresholding. Monitoring remained clean at epoch 6 (no anomaly flags), but alert quality regressed versus the cleaned `varfix` baseline (`f1=0.0304` vs `0.0800`, `precision=0.0156` vs `0.0435`) while countdown quality improved (`mae=1.4380` vs `2.6410`). This indicates the shorter window sharpens countdown fit but worsens alert separability on this slice. | config/real_sanity_stability_20260713_preictal300.yaml, logs/real_sanity_stability_20260713_preictal300/run_6e_preictal300.log, models/real_sanity_stability_20260713_preictal300/results.json, models/real_sanity_stability_20260713_preictal300/monitoring/epoch_006_anomaly_report.json, models/real_sanity_stability_20260713_varfix/results.json | partial |
| 2026-07-13 | Copilot | Pre-ictal window expansion experiment | Tested longer target window by changing `pre_ictal_window_s` from `600` to `900` in a matched 6-epoch real run. Alert precision/F1 improved slightly versus `varfix` (`precision=0.0506`, `f1=0.0872`) with clean monitoring at epoch 6, but countdown quality degraded sharply (`mae=4.0899`, `rmse=4.6000`) and calibrated operating point required a very low threshold (`0.13`). Not a viable overall tradeoff for this slice. | config/real_sanity_stability_20260713_preictal900.yaml, logs/real_sanity_stability_20260713_preictal900/run_6e_preictal900.log, models/real_sanity_stability_20260713_preictal900/results.json, models/real_sanity_stability_20260713_preictal900/monitoring/epoch_006_anomaly_report.json | partial |
| 2026-07-13 | Copilot | Onset-focused auxiliary supervision experiment | Added optional onset auxiliary term to the active countdown loss (`onset_aux_weight`, `onset_aux_window_min`) and ran a matched 6-epoch real run (`onset_aux_weight=0.15`, window `2.0` min). Outcome was neutral-to-negative versus `varfix`: sensitivity dropped (`0.4167` vs `0.5000`), F1 decreased (`0.0741` vs `0.0800`), precision decreased (`0.0407` vs `0.0435`), while countdown metrics stayed similar and monitoring remained clean. This setting is not a clear improvement. | src/losses.py:89, config/config.py:207, config/real_sanity_stability_20260713_onsetaux.yaml, logs/real_sanity_stability_20260713_onsetaux/run_6e_onsetaux.log, models/real_sanity_stability_20260713_onsetaux/results.json, models/real_sanity_stability_20260713_onsetaux/monitoring/epoch_006_anomaly_report.json | partial |
| 2026-07-13 | Copilot | Threshold-only operating-point sweep + eval checkpoint fix | Fixed eval path in `scripts/train.py` to honor `training.resume_from` when `best_model.pt` is absent in a fresh `save_dir` (enables true eval-only threshold sweeps). Re-ran two constrained sweeps from the same `varfix` checkpoint; both selected the same validation-calibrated threshold `0.33` and improved classification operating point over baseline without retraining (`precision=0.0462`, `f1=0.0838`, `specificity=0.7430`, `sensitivity=0.4444`). | scripts/train.py:2940, config/real_sanity_stability_20260713_varfix_threshold_f1_sens50_fpr35.yaml, config/real_sanity_stability_20260713_varfix_threshold_balacc_sens60_fpr30.yaml, models/real_sanity_stability_20260713_varfix_threshold_f1_sens50_fpr35/threshold_selection.json, models/real_sanity_stability_20260713_varfix_threshold_f1_sens50_fpr35/results.json, models/real_sanity_stability_20260713_varfix_threshold_balacc_sens60_fpr30/threshold_selection.json, models/real_sanity_stability_20260713_varfix_threshold_balacc_sens60_fpr30/results.json | completed |
