# Real-World Watch Deployment Readiness Checklist

## Purpose

This document defines the development principles, expected outcomes, and progress checklist for moving Epilepsee-AI from research training runs to a real-world smartwatch integration demo (Bluetooth streaming with PPG, optional ECG, and EDA).

Date baseline: 2026-07-04.

## Scope

This checklist is for a controlled technical demo of integration potential. It is not a clinical claim, medical device release, or patient-facing safety guarantee.

## Development Principles

1. Outcome-first engineering
- Every technical change must map to a visible outcome: better timeline coherence, safer alerts, lower false alarms, or higher deployment reliability.

2. Chronology integrity over convenience
- Keep strict time ordering for training/validation/test in long-sweep workflows. Avoid leakage from future windows into model fitting.

3. Modality fidelity
- Do not treat PPG as ECG. If ECG is absent on watch, preserve PPG-native semantics and signal quality handling.

4. Label semantics clarity
- In state-loss mode, the countdown head behaves as an onset-related score, not strictly calibrated minutes. Visual outputs must reflect this to avoid misinterpretation.

5. Reproducibility by default
- Paths and secrets via environment variables; deterministic seeds where possible; artifacted logs/checkpoints/metrics.

6. Deployment realism
- Build for intermittent Bluetooth, clock drift, dropped packets, variable sampling rates, and battery constraints.

7. Safety and fail-closed behavior
- If signal quality is poor or streams are missing, suppress hard alert decisions and surface degraded-confidence state.

## Expected Outcomes and Acceptance Criteria

### A. Data and Connectivity

Expected outcome: Stable watch-to-pipeline data ingestion with synchronized timeline.

Acceptance criteria:
- Bluetooth stream can run continuously for >= 30 minutes without fatal interruption.
- PPG and EDA packets are timestamped and parsed into a unified timeline.
- ECG is optional; pipeline remains valid when ECG is unavailable.
- Missing/choppy data is explicitly flagged in logs and metadata.

### B. Signal and Feature Pipeline

Expected outcome: Online feature extraction produces model-ready windows equivalent to offline contracts.

Acceptance criteria:
- Window and stride match configured training assumptions.
- Online features pass schema checks (shape, NaN handling, bounds).
- Signal quality gates (SQI/confidence) can mask low-quality intervals.

### C. Inference and Alerting

Expected outcome: Streaming inference produces stable outputs with interpretable confidence behavior.

Acceptance criteria:
- End-to-end inference latency per step within demo budget.
- Output values logged with timestamps and source quality context.
- Alert logic includes hysteresis/cooldown to avoid jitter.

### D. Evaluation and Explainability

Expected outcome: Demo outputs include enough context to explain decisions.

Acceptance criteria:
- Timeline panel includes true/pred traces where labels exist.
- Classification and countdown behavior is documented per loss mode.
- Post-run artifacts saved to reproducible paths.

### E. Operational Hardening

Expected outcome: Demo can be repeated by another developer with same outcomes.

Acceptance criteria:
- Runbook steps succeed on a clean shell with documented env vars.
- Failure states produce actionable logs.
- Crash in post-processing does not invalidate trained checkpoint artifacts.

## Current Progress Checklist (2026-07-04)

### Completed

- [x] Wearable data integration path for Oregon watch dataset (PPG/EDA/IMU-first).
- [x] Long-sweep training/evaluation flow with chronology-preserving setup.
- [x] Per-epoch monitoring artifacts (threshold, panel source, anomaly report).
- [x] Post-training evaluation output with regression/classification/clinical metrics.
- [x] Visualization crash guard for panel index mismatch after partial materialization.

### In Progress

- [ ] Clarify countdown visualization semantics under state-loss mode (onset score vs minutes).
- [ ] Add explicit calibration/translation policy if minute-level display is required in state mode.

### Pending for Real-World Demo

- [ ] Bluetooth ingestion adapter for watch streams (PPG, EDA, ECG optional).
- [ ] Real-time buffering/resampling module for asynchronous sensor rates.
- [ ] Online SQI gate and degraded-mode decision policy.
- [ ] Streaming inference service with latency and packet-loss telemetry.
- [ ] Demo dashboard/panel for live watch data + model outputs.
- [ ] Dry-run rehearsal with forced failure scenarios (disconnect, low battery, missing modality).

## Smartwatch Bluetooth Integration Blueprint

### Required stream contract

Each packet should include:
- device_id
- sensor_type in {ppg, eda, ecg, imu}
- timestamp_utc (ISO-8601)
- sampling_rate_hz
- samples (array)
- quality fields (confidence, SQI, battery if available)

### Runtime pipeline order

1. Bluetooth receive and decode
2. Time alignment and de-jitter buffering
3. Missing-data handling and quality gating
4. Feature window construction
5. Model inference
6. Alert/state output with confidence
7. Artifact logging (JSON + figures)

### Known Behavior to Communicate in Demo

- In current state-loss training, countdown scatter may cluster in a narrow band (for example 2-4 min) because the head is not optimized as direct minute regression.
- This does not invalidate classification utility, but it changes interpretation of the countdown axis.

## Demo Runbook (Execution Checklist)

Pre-demo:
- [ ] Confirm watch battery and Bluetooth stability.
- [ ] Verify environment variables and dataset/cache paths.
- [ ] Run short synthetic/recorded smoke test.

Live demo:
- [ ] Start stream and verify incoming packet counts per modality.
- [ ] Confirm timeline synchronization and quality flags.
- [ ] Start inference and observe output stability for >= 15 minutes.
- [ ] Capture logs, plots, and artifact bundle.

Post-demo:
- [ ] Archive run config + git commit hash + environment summary.
- [ ] Review anomalies, disconnects, and latency spikes.
- [ ] Record pass/fail against acceptance criteria.

## Exit Criteria for "Pilot-Ready Integration"

All items below must be true:
- Connectivity, feature pipeline, and inference acceptance criteria pass.
- At least one end-to-end live stream demo completes without fatal error.
- Output semantics are documented and understood by stakeholders.
- Failure handling and degraded-mode behavior are demonstrated.

## Related Documentation

- [Wearable Data Integration Guide](../WEARABLE_DATA_INTEGRATION.md)
- [Memory Management](MEMORY_MANAGEMENT.md)
- [Dataset Comparison: SeizeIT2 vs Oregon Wearable](DEV_Report_DATASET_COMPARISON_SEIZEIT2_VS_WEARABLE.md)
- [Root README](../README.md)
