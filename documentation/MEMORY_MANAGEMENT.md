# Memory Management in Epilepsee-AI Training

> Draft report on current issues and proposed solutions for efficient, robust training on limited host RAM (e.g., 32 GB) with multi-GPU DDP.

## 1. Problem Overview

Training seizure countdown models on long-duration ECG and wearable recordings uses a *windowed* representation: each fixed-length window (e.g., 60 s) becomes one training sample. When feature windows are extracted with a small step (e.g., 1 s), a single 24 h recording can yield tens of thousands of windows.

In the current design, all windowed samples are:
- Fully materialized into NumPy arrays with shape roughly `(N, 600, F)` for each split (train/val/test).
- Held in host RAM simultaneously.
- Duplicated across all DDP ranks, because each process loads its own copy of the full dataset into a `SeizureDataset`.

On the Oregon wearable dataset this leads to:
- Very large `N` because there are hundreds of long recordings.
- Severe class imbalance: only a handful of seizures (< 10) from hundreds of subjects/recordings.
- Host RAM exhaustion (e.g., 32 GB fully consumed) when using `DATA_SOURCE=wearable`, `DATA_MODE=real`, `MAX_SAMPLES_PER_RECORDING=0`, and multi-GPU DDP, sometimes causing the OS to log off or reboot instead of raising a Python exception.

## 2. Root Causes

1. **Full in-memory materialization of all windows**
   - `scripts/train.py::_prepare_real_datasets` and `_prepare_wearable_datasets` both build full feature arrays via `np.stack`/`np.concatenate` before splitting.
   - Each feature array is stored as `float32` with time axis length 600 and feature dimension `F` (e.g., 14 for ECG), so memory scales as `O(N * 600 * F * 4 bytes)`.

2. **DDP process-level duplication**
   - Each DDP rank constructs its own `SeizureDataset` instances from NumPy arrays, so memory usage is multiplied by the number of ranks (e.g., ×2 on a 2-GPU system).
   - Until very recently, cached datasets were loaded without memory mapping, forcing each rank to allocate its own copy of the NPZ arrays.

3. **Many non-seizure wearable recordings**
   - The wearable dataset contains ~800 recordings but only a handful include any seizures.
   - By default, real-mode selection may include many non-seizure recordings, greatly increasing `N` without adding useful preictal labels.

4. **Aggressive window density**
   - A small `feature_step_s` (e.g., 1 s) across long recordings yields very dense sampling (up to ~86,400 windows per 24 h per stream).
   - When combined with hundreds of recordings, this quickly explodes total sample count.

5. **Preictal augmentation (train set only)**
   - `_augment_train_dataset` can multiplicatively expand the number of preictal samples to rebalance classes, further increasing memory usage for the training split.

## 3. Existing Safeguards

Several safeguards are already present in the codebase:

- **Per-recording window cap** (`--max-samples-per-recording`):
  - After feature extraction for each recording, the code can sub-sample to a fixed maximum number of windows per recording.
- **Recording count cap** (`--max-recordings`):
  - Limits the total number of recordings used in real mode.
- **Preictal-aware train/val/test split** (`_split_indices_with_preictal_coverage`):
  - Attempts to ensure each split receives some preictal samples.
- **Wearable global sample cap** (`data.max_wearable_global_samples`):
  - Recently added; caps the *total* number of wearable windows across all recordings with stratified subsampling to preserve preictal coverage.
- **Memory-mapped NPZ caching** (`np.load(..., mmap_mode="r")`):
  - For cached real-mode datasets, allows multiple DDP ranks to share the same underlying arrays rather than duplicating them.

These measures help, but with dense windows, hundreds of recordings, and DDP, host RAM can still be exhausted before capping kicks in or even with conservative caps when `N` is very large.

## 4. Impact on Training

- **System instability**: host OOM can terminate the user session or trigger a full reboot/logoff, rather than failing cleanly at the Python level.
- **Limited scalability**: training configurations that should be feasible on 2×12 GB GPUs with 32 GB RAM (e.g., batch sizes of 4–8) become impractical due to CPU RAM limits.
- **Slow iteration**: repeated full reconstruction of large windowed datasets (when cache keys change) further slows experimentation.
- **Underutilized data**: many non-seizure recordings consume memory without improving preictal coverage or model generalization.

## 5. Planned Improvements

To address these issues and enable robust training on realistic hardware, we will implement the following best-practice improvements:

1. **Seizure-first recording selection for wearable data**
   - Explicitly prioritize wearable recordings that contain seizures, and strongly limit or exclude purely non-seizure recordings.
   - Introduce configuration options to cap the number of non-seizure recordings used (e.g., `max_nonseizure_recordings`).

2. **Stronger global sample control and monitoring**
   - Tighten `max_wearable_global_samples` defaults for 32 GB environments and add detailed logging of sample counts and approximate memory usage before/after capping.
   - Optionally make the cap adaptive to `feature_step_s`, recording duration, and class balance.

3. **Streaming-friendly dataset representation**
   - Move toward `Dataset` implementations that can:
     - Load windowed data lazily from disk or memory-mapped files.
     - Avoid holding all `(N, 600, F)` windows in RAM at once.
   - This will trade some I/O overhead for bounded memory, which is critical for DDP.

4. **Wearable-specific sampling strategies**
   - Prefer denser sampling around seizure onsets and sparser sampling far from seizures (e.g., random sub-sampling of long interictal stretches).
   - This reduces `N` while preserving preictal information.

5. **Safer augmentation and DDP behavior**
   - Make preictal augmentation optional and more conservative for large real-mode runs.
   - Ensure that augmented train sets do not blow past memory caps.

## 6. Next Steps

- Implement seizure-driven recording selection and non-seizure caps in the wearable preparation path.
- Refine global sample capping and logging for the wearable dataset.
- Prototype a streaming / memory-mapped `SeizureDataset` variant suitable for DDP.
- Update training documentation (including `Makefile` recipes) to recommend safe defaults for 32 GB + 2×GPU setups.

This document will be updated as these changes are rolled out and validated on the target hardware configuration.
