# Turn-In Report: Dense Multimodal Panel + Token Activation Integration

Date: 2026-07-04
Project: Epilepsee-AI
Dataset: SeizeIT2 (BIDS)
Environment: WSL/Linux, CPU run, no Docker

## 1) Objective

Resolve visualization fidelity and interpretability issues by:
- Removing sparse/flat panel appearance.
- Restoring true multimodal overlays (ECG+BPM, EEG, EMG, MOV, HRV context).
- Integrating token activation output directly into the same GT-vs-inference figure.

## 2) Implemented Code Changes

### 2.1 src/training.py

Key updates:
- Added dense panel reconstruction helpers:
  - `_parse_recording_uid(...)`
  - `_interp_to_grid(...)`
  - `_extract_hr_hrv_from_dense_ecg(...)`
- Added imports required by new helpers (`re`, `List`).
- In epoch visualization path:
  - Rebuilds selected panel timeline on a dense grid (default 1.0 s) for BIDS runs.
  - Interpolates prediction/label series onto dense timeline.
  - Pulls dense ECG from cached signal binary when available.
  - Recomputes dense HR/HRV from dense ECG waveform.
  - Attempts dense EEG/EMG/MOV extraction for same recording and aligns to dense timeline.
  - Passes EMG and MOV arrays into the panel renderer.
- Token explainability:
  - Builds token activation occupancy matrix from the same arrays used in the main panel.
  - Passes `token_roll` and `token_window` directly into `plot_gt_vs_inference_panel(...)`.
  - Removes separate token-waterfall artifact generation path that was failing with shape mismatch.
- Stores validation dataset reference for metadata-backed dense panel reconstruction.

### 2.2 src/visualization.py

Key updates:
- `plot_gt_vs_inference_panel(...)` signature expanded to accept:
  - `emg_signal`
  - `mov_signal`
- Motion row now explicitly supports and labels EMG and MOV traces.
- Motion row title updated to `EMG/MOV + HRV (dual axis)`.
- Token activation row rendering order fixed:
  - Token axis now created after primary axes exist, preventing share-axis ordering errors.
  - Row title set to `Token Activation Waterfall`.

## 3) Validation Performed

### 3.1 Static validation
- Python compile checks passed:
  - `src/training.py`
  - `src/visualization.py`
- Module import smoke test passed.

### 3.2 Sanity run (Option 3)
Command executed:
- `OUTPUT_DIR=./models/seizeit2_tcn_sanity_panelfix_e1 ... --epochs 1 ...`

Observed results:
- Training and validation completed for epoch 1.
- New epoch panel artifact generated:
  - `models/seizeit2_tcn_sanity_panelfix_e1/epoch_visualizations/epoch_001_gt_vs_inference_panel.png`
- Checkpoints generated:
  - `models/seizeit2_tcn_sanity_panelfix_e1/best_model.pt`
  - `models/seizeit2_tcn_sanity_panelfix_e1/last_model.pt`

Important note:
- The sanity run completed epoch-level outputs correctly, then hit an existing downstream final-evaluation panel bug in `scripts/train.py`:
  - `IndexError: index 2313 is out of bounds for axis 0 with size 2000`
- This is in the final script-level panel path, not in the updated epoch panel path inside `src/training.py`.

## 4) Main Run Restarted (Option 2)

Previous run:
- Old long run terminal stopped.

New long run command started on patched code:
- `OUTPUT_DIR=./models/seizeit2_tcn_longrun_e12_panelfix ... --epochs 12 ...`

Current status at handoff:
- Run initialized and dataset loading started in new terminal session.

### Live Progress Update (Patched Long Run)

Run:
- `models/seizeit2_tcn_longrun_e12_panelfix/`

Confirmed completed:
- Epoch 1/12
  - Train Loss: 0.7570
  - Val Loss: 0.6911
  - Val MAE: 3.2320 min
  - Adaptive threshold: 0.22
  - Panel source (metadata-selected): `sub-022_ses-01_run-10`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_001_gt_vs_inference_panel.png`
  - Checkpoint artifact: `models/seizeit2_tcn_longrun_e12_panelfix/best_model.pt`

Current in-progress state:
- Epoch 2 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress:
- Epoch 2/12
  - Train Loss: 0.4474 (Δ -40.9%)
  - Val Loss: 0.6282 (Δ -9.1%)
  - Val MAE: 3.4732 min (Δ +7.5%)
  - Adaptive threshold: 0.15
  - Panel source (metadata-selected): `sub-028_ses-01_run-1`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_002_gt_vs_inference_panel.png`

Current in-progress state (updated):
- Epoch 3 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 3/12
  - Train Loss: 0.4238 (Δ -5.3%)
  - Val Loss: 0.7766 (Δ +23.6%)
  - Val MAE: 3.1770 min (Δ -8.5%)
  - Adaptive threshold: 0.15
  - Panel source (metadata-selected): `sub-034_ses-01_run-7`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_003_gt_vs_inference_panel.png`
  - Checkpoint artifact: `models/seizeit2_tcn_longrun_e12_panelfix/best_model.pt` (updated)

Current in-progress state (latest):
- Epoch 4 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 4/12
  - Train Loss: 0.4798 (Δ +13.2%)
  - Val Loss: 0.5639 (Δ -27.4%)
  - Val MAE: 3.4324 min (Δ +8.0%)
  - Adaptive threshold: 0.22
  - Panel source (metadata-selected): `sub-022_ses-01_run-10`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_004_gt_vs_inference_panel.png`

Current in-progress state (latest):
- Epoch 5 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 5/12
  - Train Loss: 0.4669 (Δ -2.7%)
  - Val Loss: 0.5345 (Δ -5.2%)
  - Val MAE: 3.9825 min (Δ +16.0%)
  - Adaptive threshold: 0.12
  - Panel source (metadata-selected): `sub-002_ses-01_run-10`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_005_gt_vs_inference_panel.png`

Current in-progress state (latest):
- Awaiting epoch 6 batch logs (no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 6/12
  - Train Loss: 0.4483 (Δ -4.0%)
  - Val Loss: 0.6548 (Δ +22.5%)
  - Val MAE: 3.5844 min (Δ -10.0%)
  - Adaptive threshold: 0.12
  - Panel source (metadata-selected): `sub-002_ses-01_run-5`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_006_gt_vs_inference_panel.png`

Current in-progress state (latest):
- Epoch 7 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 7/12
  - Train Loss: 0.4049 (Δ -9.7%)
  - Val Loss: 0.5211 (Δ -20.4%)
  - Val MAE: 3.5816 min (Δ -0.1%)
  - Adaptive threshold: 0.20
  - Panel source (metadata-selected): `sub-001_ses-01_run-8`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_007_gt_vs_inference_panel.png`

Current in-progress state (latest):
- Epoch 8 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 8/12
  - Train Loss: 0.4243 (Δ +4.8%)
  - Val Loss: 0.5014 (Δ -3.8%)
  - Val MAE: 3.5883 min (Δ +0.2%)
  - Adaptive threshold: 0.35
  - Panel source (metadata-selected): `sub-031_ses-01_run-3`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_008_gt_vs_inference_panel.png`

Current in-progress state (latest):
- Epoch 9 active (batch-level logs progressing; no interactive input prompt).

Additional confirmed progress (latest):
- Epoch 9/12
  - Train Loss: 0.3921 (Δ -7.6%)
  - Val Loss: 0.4768 (Δ -4.9%)
  - Val MAE: 3.8292 min (Δ +6.7%)
  - Adaptive threshold: 0.20
  - Panel source (metadata-selected): `sub-015_ses-01_run-3`
  - Panel artifact: `models/seizeit2_tcn_longrun_e12_panelfix/epoch_visualizations/epoch_009_gt_vs_inference_panel.png`

Current in-progress state (latest):
- Epoch 10 active (batch-level logs progressing; no interactive input prompt).

## 5) Delivered Artifacts

Code files changed:
- `src/training.py`
- `src/visualization.py`

Sanity output folder:
- `models/seizeit2_tcn_sanity_panelfix_e1/`

Main restarted output folder:
- `models/seizeit2_tcn_longrun_e12_panelfix/`

This report:
- `documentation/Research-Dev/TURN_IN_REPORT_PANELFIX_2026-07-04.md`

## 6) Remaining Item (Known)

One pre-existing script-level bug remains to patch next:
- Final evaluation panel indexing mismatch in `scripts/train.py` after partial test prediction materialization.
- Symptom observed in sanity run tail:
  - `panel_indices` computed against a larger timeline than available `signal_proxy` array.

## 7) Recommended Next Patch

Patch `scripts/train.py` final panel assembly to enforce consistent indexing domain:
- Clip or remap `panel_indices` to `len(signal_proxy)` before signal slicing.
- Use a common index source for all arrays in final panel generation.
- Add guard log when a truncation/remap occurs.
