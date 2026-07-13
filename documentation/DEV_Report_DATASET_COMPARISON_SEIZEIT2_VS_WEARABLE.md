# Dataset Dev Notes: SeizeIT2 (BIDS) vs. Oregon Wearable (VSM Watch)

This document summarizes the key differences between the two main datasets used in this project and highlights how those differences affect:

- Model inputs and modalities
- Sampling and windowing strategy
- Training behavior and visualizations
- Deployment assumptions for smartwatch‑class devices

Datasets covered:

- **SeizeIT2 BIDS dataset** (ds005873) – hospital vEEG + wearable EEG/ECG/EMG/motion
- **Oregon Wearable dataset** – VSM Study Watch wrist‑wearable (PPG + motion + other sensors)

---

## 1. High‑Level Summary

| Aspect                         | SeizeIT2 BIDS (ds005873)                                                | Oregon Wearable (VSM Watch)                                                     |
|--------------------------------|---------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| Recording context              | EMU (epilepsy monitoring unit), in‑hospital presurgical vEEG monitoring | Ambulatory wearable monitoring in the wild                                      |
| Primary reference              | Full vEEG system (10–20 / 25‑electrode arrays)                          | No scalp EEG reference; annotations via master Excel + clinical context         |
| Wearable modalities           | Back‑of‑ear EEG (bte‑EEG), ECG, EMG, accelerometer/gyroscope            | PPG (raw + HR), accelerometer, EDA, temperature, ECG (sometimes), HRV, others   |
| Typical sampling rates         | Wearable device at ~250 Hz (EEG/ECG/EMG), movement at ~25 Hz            | PPG raw ~100 Hz, accelerometer 50 Hz, processed HR/HRV at lower effective rates |
| Seizure coverage               | 125 patients, 886 focal seizures in total                               | Currently curated subset covers 7 patients with a small number of annotated seizures |
| Original modeling assumption   | Wrist dataset would also provide ECG/EEG‑like signals                   | Actual wrist modalities are PPG + motion (ADXL) first, plus optional extras     |
| Intended deployment analogue   | Clinical‑grade wearable with adhesive electrodes                         | Consumer‑style smartwatch / medical‑grade watch (PPG + IMU always, ECG optional)|

---

### 1.1 Dataset size and windowed sample distribution

- **SeizeIT2 (ds005873)**
  - Subjects: 125, with 886 focal seizures across EMU stays.
  - Each long‑duration recording yields tens of thousands of 60‑s windows when using a 1‑s stride; across all subjects this produces on the order of hundreds of thousands of windowed samples, heavily but not overwhelmingly skewed toward interictal windows.
  - Many subjects contribute multiple seizure‑positive windows, so countdown labels are reasonably well distributed across patients.

- **Oregon Wearable (VSM Watch, current subset)**
  - Subjects: **7** patients with seizure‑annotated sessions in the current integration (out of a larger raw corpus of mostly non‑seizure recordings).
  - Total seizures: only a **handful of events** spread across these 7 patients; the majority of wearable recordings are seizure‑free and are either down‑sampled or excluded to control class imbalance.
  - With 60‑s windows and a 1‑s stride, the wearable pipeline can still generate tens of thousands of windows, but **true preictal windows form a very small fraction** of the total; most windows are interictal/background and often come from subjects who never seize.

This mismatch means that, although the raw number of wearable windows can look similar to SeizeIT2 in logs, the **effective signal for learning preictal dynamics is much sparser** and concentrated in just a few seizure‑positive patients. All modeling and visualization results on the Oregon data should therefore be interpreted with this scarcity in mind.

## 2. SeizeIT2 BIDS Dataset (ds005873)

**Source**: [README](../../$BIDS_DATASET_ROOT/README) for the BIDS‑converted SeizeIT2 dataset.

### 2.1 Study context

- Multicenter prospective study (SeizeIT2, NCT04284072) in EMU settings.
- 125 patients with refractory focal epilepsy, monitored during presurgical vEEG.
- Includes adult and pediatric patients from several European EMUs.
- All participants have long‑term simultaneous:
  - full vEEG,
  - **Sensor Dot** wearable(s): bte‑EEG, ECG, EMG, movement.

### 2.2 Wearable device and modalities

- **Sensor Dot device** records multiple channels:
  - bte‑EEG (1–2 channels depending on suspected seizure focus),
  - ECG,
  - EMG,
  - movement (accelerometers, gyroscopes).
- Wearable sampling frequencies (per README):
  - EEG/ECG/EMG around **250 Hz**,
  - movement data at **25 Hz**.
- Every subject has **wearable EEG**; most also have ECG, EMG, and motion.

### 2.3 Data organization (BIDS)

- Reorganized from the original SeizeIT2 into BIDS:
  - Subject/session level folders with `sub-*/ses-*` structure.
  - `*.edf` files contain wearable EEG/ECG/EMG/movement.
  - Event annotations are in BIDS‑compatible `*_events.tsv` (seizure timings, labels, etc.).
- Metadata (age, sex, etc.) via `participants.tsv` and JSON sidecars.

### 2.4 Label geometry and modeling implications

- **Labels**:
  - Focal seizure events with start/end times.
  - Countdown labels typically constructed as minutes‑to‑seizure from EDF events.
- **Multimodal potential**:
  - True **ECG waveform** is available → can derive HR, HRV from R‑R intervals.
  - bte‑EEG channels → can model EEG activity jointly with ECG and movement.
- **Visualization expectations**:
  - Panels can show:
    - An actual ECG‑like trace (signal over time).
    - HR/HRV curves derived from that ECG.
    - EEG proxy traces.
  - Confusion matrices and inference maps operate on labels/contracts where ECG+EEG content is meaningful and relatively stable in sampling.

---

## 3. Oregon Wearable Dataset (VSM Study Watch)

**Source**: [Readme](../../$WEARABLE_DATASET_ROOT/Readme) for the VSM Study Watch dataset.

### 3.1 Study context

- Ambulatory monitoring sessions with the **VSM Study Watch** (Analog Devices).
- Goal: support development and validation of **wearable seizure detection** algorithms.
- Data is organized by **participant** and **watch**, with multiple sessions per watch.

### 3.2 Device and modalities

The watch exposes several sensor streams (JSON‑lines files), including but not limited to:

- **PPG (photoplethysmography)**:
  - `ADPDCLAppStream.json`: raw PPG (photodiode channels, 100 Hz effective sampling).
  - `PPGAppStream.json`: processed HR output (heart rate in bpm, confidence, state).
- **Motion**:
  - `ADXLAppStream.json`: raw accelerometer (x, y, z) at **50 Hz**.
- **Other sensors** (optional, depending on use‑case):
  - `EDAAppStream.json`: electrodermal activity.
  - `TemperatureStream.json`: skin temperature.
  - `ECGAppStream.json`: ECG waveform when configured.
  - `HRVStreamInfo.json`: HRV metrics and RR intervals.
  - `BIAAppStream.json`: bio‑impedance.
  - `SyncPPGAppStream.json`: synchronized PPG + motion samples.
  - `PedometerAppStream.json`, `SQIStream.json`, `BatteryStream.json`, etc.

**Key point for modeling**: for most smartwatch‑like deployments, **PPG + accelerometer (ADXL)** are the minimal, always‑on streams. ECG might be sporadic or absent, and there is **no scalp EEG**.

### 3.3 Ground‑truth labels

- Ground truth is in a master Excel: `Wearable Seizure Detection Master Database.xlsx`.
- Per session, provides:
  - Participant ID and watch ID.
  - Session start/stop times.
  - Whether seizures occurred.
  - For each seizure: date, time, duration.
- All timestamps are **local time**, with time‑zone offset available from session metadata (`DateTimeInfo.tz_sec`). Alignment to sensor data requires conversion to epoch time and matching timestamps.

### 3.4 Sampling characteristics

- **PPG raw (`ADPDCLAppStream`)**:
  - ~100 Hz, by grouping 6 samples per packet every ~60 ms.
- **Accelerometer (`ADXLAppStream`)**:
  - 50 Hz, by grouping up to 5 samples every 100 ms.
- **Processed HR / HRV streams**:
  - Much lower effective sampling rate (on the order of seconds) due to windowed estimation.

These sampling rates are **different** from SeizeIT2’s 250 Hz ECG/EEG and 25 Hz movement.

---

## 4. Original Assumption vs. Reality

### 4.1 Original assumption

- During early design, the working assumption was that the **wrist wearable dataset** would resemble SeizeIT2’s Sensor Dot setup:
  - Some form of **wearable EEG** or at least high‑fidelity ECG,
  - Similar per‑sample waveforms usable as input, with HR/HRV derivable as in SeizeIT2.
- That assumption informed:
  - Some visualization expectations (ECG‑like traces, biologically plausible HR/HRV derived from the same signal),
  - Early model design choices that treated the wearable stream as an “ECG‑like” channel.

### 4.2 Reality of the VSM Watch dataset

- The Oregon dataset is fundamentally **PPG + motion first**:
  - Raw PPG is optical (not ECG) and is more sensitive to motion artifacts.
  - Processed HR/HRV are already algorithm outputs – not raw R‑R intervals from ECG.
  - Accelerometer is a separate IMU stream at different sampling rate.
- There is **no scalp EEG**, and continuous ECG is not guaranteed or primary.
- For smartwatch‑style deployment, **PPG + ADXL** should be treated as the core multimodal input, with other modalities as optional enrichments.

**Implication**: visuals and models that assume a “biometric ECG waveform” will not transfer 1:1. For wearable runs, the panel’s top trace is currently a feature proxy, not a true ECG or PPG waveform, and HR/HRV proxies are synthetic rather than beat‑to‑beat estimates.

---

## 5. Impact on Modeling and Training

### 5.1 Modalities and feature engineering

- **SeizeIT2 (ECG + EEG + EMG + motion)**:
  - Model can ingest engineered features from ECG, EEG, motion, etc., or even raw segments.
  - HR/HRV can be computed directly from ECG R‑R intervals.
  - EEG features can complement ECG for seizure anticipation.

- **Oregon Wearable (PPG + ADXL + others)**:
  - Core multimodal stack is **PPG + accelerometer**.
  - HR/HRV are already derived by the device; additional feature engineering is needed if raw PPG is used.
  - Motion features (step count, RMS acceleration, etc.) should be integrated explicitly.

### 5.2 Windowing and sampling strategy

- Window length and stride in **training** depend on sampling rates:
  - SeizeIT2 ECG/EEG at 250 Hz allows short windows to capture many cardiac/EEG events.
  - Oregon raw PPG at 100 Hz and accelerometer at 50 Hz will change the number of samples per window for the same real‑time duration.
- In the current code, we often operate on **engineered per‑window features** rather than per‑sample waveforms for the wearable dataset. That:
  - Simplifies model input to a fixed feature dimension per time step,
  - But means the plotted “signal” is not a true waveform and may appear flat when standardized.

**Key risk**: If the training data is pre‑sampled at a given window cadence (e.g., 30–60 s windows with a certain stride) and deployment uses a different stride or sampling regime, the model’s performance and calibration can drift.

### 5.3 Labeling and countdown construction

- For both datasets, countdown labels are typically constructed as **minutes to seizure onset** from seizure annotations.
- In SeizeIT2, seizure events are aligned with continuous EDF recordings (hospital‑grade timing).
- In the VSM dataset, seizure times come from the Excel master; care must be taken about:
  - Local time vs. epoch,
  - Session boundaries,
  - Multiple seizures per session.

---

## 6. Impact on Visualization

### 6.1 Why SeizeIT2 panels look “biometric”

- Inputs include actual ECG waveforms; panels can:
  - Plot a **non‑flat ECG signal** over time.
  - Derive HR/HRV from real beat detection.
- Confusion matrices and inference heatmaps reflect behavior on a large, balanced dataset with rich multimodal signals.

### 6.2 Why wearable panels can look “flat”

- For the Oregon dataset, current panels for TCN runs typically use:
  - A **single feature channel** as a “signal proxy”, standardized → often centered near 0.
  - HR/HRV proxies mapped from feature means/stds, not from true R‑R intervals.
- If the underlying features are low‑variance across windows, all three traces (signal proxy, HR, HRV) can look nearly constant.
- The inference heatmap and confusion tiles for small, skewed windows may give the impression of “all‑preictal” behavior, even when full evaluation metrics show good specificity.

**Actionable takeaway**: For wearable runs, panels should ideally use:

- A **true time series** from PPG (e.g., downsampled raw PPG or processed HR) and from ADXL, in addition to feature proxies.
- Visualization time axes that reflect realistic durations (e.g., 10–60 min windows) and clearly mark GT seizure onset.

### 6.3 Example panels from the codebase

Below are two concrete examples from this repository that illustrate the contrast:

**(A) SeizeIT2 / ECG+EEG example (ideal behavior)**

![SeizeIT2 example panel – rich ECG signal, responsive inference map, and well-balanced confusion matrix](../models/epoch_visualizations/epoch_025_gt_vs_inference_panel.png)

- Source: SeizeIT2 BIDS run with mobilenet_1d.
- Top trace: non-flat ECG proxy with clear morphology and variability.
- HR (~30–100 bpm) and HRV traces follow naturally from that signal.
- Inference heatmap and countdown tracks respond visibly to preictal dynamics.
- Confusion matrix shows high TN and TP rates (e.g., TN~96.5%, TP~99.8%).

**(B) Oregon wearable / PPG+ADXL example (current behavior)**

![Wearable example panel – feature-based signal proxy with flatter traces and preictal-dominant confusion on a small window](../models/epoch_visualizations/epoch_001_gt_vs_inference_panel.png)

- Source: VSM Study Watch wearable run (PPG + ADXL feature stack).
- Top trace: feature-based “signal proxy” that is standardized and nearly flat around 0.
- HR and HRV proxies are synthetic mappings from window statistics, so they appear much flatter.
- Inference map shows less temporal structure, and, on this small 60-minute window, the right-hand confusion tile skews heavily to preictal.
- Full test-set confusion matrices and metrics, however, can still show good specificity and accuracy even when a single visualization window looks preictal-heavy.

In addition to the epoch panels above, the repository now also produces a **final GT-vs-inference panel** for wearable runs, saved as:

![Final wearable GT vs inference panel (multi-channel wearable signals + HR/HRV + countdown + confusion matrix)](../models/gt_vs_inference_panel.png)

This figure summarizes the current behavior of the best wearable model (for example, `mobilenet_1d` on the OHSU VSM dataset) over a longer 120-minute window, using multi-channel wearable signals (PPG+ADXL), HR/HRV traces, countdown references, inference heatmap, and a full confusion matrix. It serves as an alternative, more integrated view of the present progress on the wearable pipeline.

---

## 7. Deployment Considerations

### 7.1 Accessible modalities on smartwatch‑class hardware

- Real‑world deployment for a watch is constrained to:
  - Continuous **PPG** (for HR/HRV/SpO2‑like metrics).
  - Continuous **accelerometer** (and possibly gyroscope).
  - Optional or intermittent **ECG** (spot checks, not continuous stream).
- No scalp EEG; EMG may be implicit via motion, not direct electrodes.

### 7.2 Matching training to deployment

To make the model’s behavior and visualizations trustworthy at deployment:

- **Match modality set**:
  - Train primarily on PPG + ADXL features for smartwatch deployment, not on ECG + EEG when those will not be available.
- **Match sampling regime**:
  - Align training window length and stride to what the online system will use (e.g., 30–60 s windows with a fixed overlap).
  - Be explicit about how often predictions are evaluated (e.g., once per window vs. streaming per second).
- **Re‑interpret metrics and panels**:
  - Understand that “flat‑looking” traces on the wearable dataset do **not** necessarily mean the model is useless; they reflect the abstraction level (features vs. waveform).
  - Use confusion matrices from **full test sets** and threshold sweeps to assess sensitivity/specificity and FPR in clinically meaningful terms.

---

## 8. Practical Guidance for Future Development

1. **When building new models**:
   - For SeizeIT2, continue to exploit ECG + EEG + motion when possible.
   - For Oregon, design architectures and features around PPG + ADXL as first‑class modalities.

2. **When tuning sampling/windowing**:
   - Always consider the underlying physical sampling rates (250 Hz vs. 100/50 Hz).
   - Verify that the number of windows and their duration match intended real‑time behavior.

3. **When interpreting visualizations**:
   - Expect rich ECG‑like shapes and HR/HRV dynamics in SeizeIT2 panels.
   - Expect more abstract, possibly flatter traces for feature‑only wearable runs unless raw PPG/ADXL are explicitly plotted.

4. **When thinking about deployment**:
   - Treat the Oregon dataset as the realistic reference for smartwatch‑class deployment.
   - Use SeizeIT2 primarily as a high‑fidelity testbed for multimodal algorithms (ECG + EEG + motion), then adapt ideas to the PPG+ADXL regime.

---

## 9. Follow‑Up: Flat Wearable Segments and Variance‑Based Filtering

Subsequent inspection of raw Oregon wearable CSVs revealed that some sensor streams are effectively flat for long spans, particularly in the combined ADPD/ADXL files. For example:

- In A181C2C9_ADPD_ADXL_CombinedData_CSV.csv (Participant #7), the ADPD channels `Ch1` and `Ch2` are `0` for the entire file.
- In 90E16C12_ADPD_ADXL_CombinedData_CSV.csv (Participant #1), the accelerometer (`X`, `Y`, `Z`, `Magnitude`) shows real motion, but upstream ADPD channels can still be degenerate.

When such flat streams are included in the feature stack, entire windows can end up with (near) zero variance across all channels. These windows:

- Contribute nothing useful to training, and
- When used as visualization proxies, produce completely flat “Signal”, HR, and HRV traces that do not reflect real physiology.

### 9.1 Implemented Fix: Per‑Window Flatness Filter

To prevent corrupted or invalid segments from influencing the model or panels, a variance‑based filter was added in the wearable feature extraction pipeline (WearableDeviceDataLoader.extract_features_from_recording):

- After building all windows for a recording, compute the standard deviation across all time steps and channels for each window.
- Any window with near‑zero variance (std <= 1e‑6) is treated as **flat** and dropped.
- If **all** windows for a recording are flat, the entire recording is skipped with a warning in the logs.
- Otherwise, the loader logs how many windows were dropped vs. kept for that subject/watch.

This filtering happens before dataset splitting, so flat segments no longer appear in train/val/test and cannot dominate the GT‑vs‑inference panels.

### 9.2 Practical Impact

- **Training**: The model is no longer exposed to windows that are pure artifacts (all‑zero ADPD/ADXL/PPG), reducing noise and potential bias from invalid data.
- **Visualization**: Epoch and final panels are less likely to show constant 0 or constant 180/20 traces that do not exist in the real sensor dynamics; remaining flatness now reflects true low‑variance behavior, not corrupted exports.
- **Data hygiene**: This step effectively marks problematic recordings as unusable when they contain only flat windows, while preserving valid motion/PPG segments in mixed‑quality sessions.

This change should be viewed as a first‑line data‑quality safeguard for the Oregon wearable dataset, sitting alongside the existing alignment and sampling logic.

---

## 10. Current Situation: Visualization Hurdles and Model Behavior

This section tracks the present set of visualization quirks and modeling hurdles encountered while bringing the Oregon wearable pipeline up to parity with the SeizeIT2 panels.

### 10.1 Signal Going into the Model vs. Signal Shown in Panels

- **Model input (wearable)**:
  - The model always receives a full feature tensor `(T, F)` per window, built in `_prepare_wearable_datasets` from PPG + ADXL (and any additional wearable channels) via `_align_feature_matrix`.
  - During evaluation (`Evaluator.collect_predictions`), we pass this tensor to the model without additional filtering beyond float casting and basic NaN sanitization.
- **Visualization input (epoch / wandb)**:
  - Epoch panels (logged to wandb) currently use **only channel 0** as a wearable proxy (PPG‑derived HR series), z‑scored for the top “Signal” and kept in native units for the HR subplot.
  - This means wandb’s top trace is a single‑channel HR proxy, not the full multi‑channel wearable feature bank.
- **Visualization input (final offline GT‑vs‑inference panel)**:
  - The final test panel now overlays **all wearable feature channels** (PPG + ADXL, etc.) by taking a per‑window mean for each channel and standardizing the resulting `(N_panel, F)` matrix.
  - Matplotlib plots one line per feature column; the legend repeats the same label once per channel (e.g., "Wearable signals (PPG+ADXL, z‑score)" appears multiple times), which can make the top legend crowded while the traces visually overlap.

**Nuance**: The model is always trained and evaluated on the full wearable feature bank; the “flatness” users sometimes see in panels is a property of how we summarize that bank (single HR channel vs. multi‑channel overlay), not a change in what the ML model actually sees.

### 10.2 Flat Raw Probabilities and Countdown Predictions

- **Observation**: In some wearable runs, the GT countdown row shows a proper down‑ramp, but the `Raw Prob` and `Pred Countdown/10` rows look nearly constant across time.
- **Code path**:
  - `pred_preictal` and `pred_countdown` are taken directly from the model outputs in `Evaluator.collect_predictions` and only clipped to `[0, 1]` (or divided by 10 and clipped) in the visualization.
  - No extra smoothing is applied to the heatmap rows; the only smoothing is an optional causal filter used for the overlaid "Smoothed alarm" curve.
- **Interpretation**:
  - When `Raw Prob` or `Pred Countdown/10` are nearly constant, this reflects **model saturation** (e.g., almost‑always‑preictal or almost‑always‑far‑from‑seizure) rather than a plotting bug.
  - On highly imbalanced wearable segments or small test windows, the model can converge to a nearly constant operating point that still optimizes the loss locally but looks non‑responsive in the panel.

This is a **modeling hurdle**, not a visualization one: improving it will require adjusting loss weighting, sampling/augmentation, or architecture so that the learned pre‑ictal probabilities and countdowns vary meaningfully with the wearable input.

### 10.3 Offline vs. Wandb Panels: Different Projections, Different Intuition

- **Offline final panel**:
  - Uses test data and shows a multi‑channel wearable "Signal" (PPG + ADXL and others) with overlaid traces, plus HR/HRV derived from the first wearable feature channel.
  - Often looks more "alive" on top because motion channels contribute visible variability even when HR is relatively smooth.
- **Wandb epoch panels**:
  - Use validation data and a 60‑minute metadata‑driven window.
  - Top "Signal" is currently just the z‑scored HR proxy from channel 0; HR/HRV subplots are the corresponding per‑window mean/std.
  - Depending on which validation window is selected, these series can appear quite flat, especially when the chosen segment is physiologically quiet or dominated by low‑variance engineered features.

**Current hurdle**: The mismatch between HR‑only wandb panels and multi‑channel offline panels can be confusing when debugging. A future alignment step is to reuse the same multi‑channel wearable projection for epoch panels so that wandb and offline views share the same notion of "Signal" while still keeping HR/HRV subplots interpretable.

### 10.4 Legend and Readability Issues for Wearable Signals

- Because the final panel now overlays one line per wearable feature channel, Matplotlib creates one legend entry per channel using the same label.
- This leads to legends with repeated text (e.g., the same "Wearable signals (PPG+ADXL, z‑score)" label 5–7 times) and can obscure more important legend entries (GT preictal, predicted preictal, seizure onset).

**Planned refinement**:

- Collapse the wearable channels into one or two logically named groups in the legend (e.g., a single "PPG features" entry and a single "Motion features" entry), while still plotting multiple lines, or selectively plot only the most informative subset of channels.

### 10.5 Summary of Open Visualization Tasks

- Unify signal projections between wandb epoch panels and offline final panels for wearable runs.
- Improve legends and labeling when multiple wearable channels are overlaid, avoiding repeated labels.
- Add lightweight diagnostics (min/max/std logs) for `pred_preictal` and `pred_countdown` on wearable runs to distinguish true model saturation from visualization scaling issues.
- Consider exposing a separate "waveform" view that plots a raw or downsampled PPG/ADXL segment alongside the feature‑based GT‑vs‑inference panel for richer qualitative analysis.

These items currently represent the main interpretability and debugging hurdles for the Oregon wearable pipeline, even after the flat‑window variance filtering has been applied.
