"""
Training engine with multi-GPU support (Distributed Data Parallel).

Features:
- Distributed training across 2 GPUs
- Early stopping with best model checkpointing
- Learning rate scheduling
- Gradient accumulation
- Weights & Biases (wandb) logging
"""

import os
import sys
import json
import re
import math
import logging
import signal
import importlib
from contextlib import nullcontext
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import numpy as np
from collections import defaultdict
import time
from pathlib import Path

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

def _import_wandb_safely():
    """Import real wandb package even when local ./wandb folder shadows it."""
    try:
        import wandb as _wandb
        if hasattr(_wandb, 'init') and hasattr(_wandb, 'Image'):
            return _wandb, True
    except Exception:
        pass

    try:
        import sys
        import site

        repo_root = str(Path(__file__).resolve().parents[1])
        blocked = {
            '',
            os.getcwd(),
            repo_root,
        }

        orig_path = list(sys.path)
        try:
            sys.modules.pop('wandb', None)
            clean_path = [
                p for p in sys.path
                if str(Path(p).resolve()) not in {str(Path(x).resolve()) for x in blocked if x}
            ]
            for sp in site.getsitepackages():
                if sp not in clean_path:
                    clean_path.insert(0, sp)
            user_site = site.getusersitepackages()
            if user_site and user_site not in clean_path:
                clean_path.insert(0, user_site)

            sys.path = clean_path
            _wandb = importlib.import_module('wandb')
            if hasattr(_wandb, 'init') and hasattr(_wandb, 'Image'):
                return _wandb, True
        finally:
            sys.path = orig_path
    except Exception:
        pass

    return None, False


wandb, HAS_WANDB = _import_wandb_safely()

from config.config import Config, TrainingConfig
from src.models import ModelFactory
from src.losses import LossFactory
from src.data_loader import SeizureDataset
from src.visualization import SignalVisualizer
from src.stateful_data_loader import (
    TemporalPatientSequenceDataLoader,
    HiddenStateManager,
    IndexedTemporalDatasetView,
    partition_patients_by_batch_count,
)


logger = logging.getLogger(__name__)


def _causal_smooth(probs: np.ndarray,
                   rise_alpha: float = 0.50,
                   fall_alpha: float = 0.12,
                   streak_threshold: float = 0.30,
                   streak_window: int = 8,
                   streak_max_bonus: float = 0.15) -> np.ndarray:
    """Causal asymmetric EMA with alarm-streak bonus (see evaluation.causal_smooth_predictions)."""
    n = len(probs)
    if n == 0:
        return np.array([], dtype=np.float32)
    out = np.empty(n, dtype=np.float32)
    state = float(probs[0])
    consecutive = 0
    for i in range(n):
        p = float(probs[i])
        alpha = rise_alpha if p >= state else fall_alpha
        state = alpha * p + (1.0 - alpha) * state
        consecutive = consecutive + 1 if p >= streak_threshold else 0
        bonus = min(consecutive / max(streak_window, 1), 1.0) * streak_max_bonus
        out[i] = float(np.clip(state + bonus, 0.0, 1.0))
    return out


def _uniform_pick_indices(indices: np.ndarray, count: int) -> np.ndarray:
    """Pick roughly uniformly spaced indices from an index array."""
    if count <= 0 or len(indices) == 0:
        return np.array([], dtype=np.int64)
    if len(indices) <= count:
        return indices.astype(np.int64)

    positions = np.linspace(0, len(indices) - 1, num=count, dtype=np.int64)
    return indices[positions].astype(np.int64)


def _select_detection_threshold(
    true_countdown: np.ndarray,
    pred_preictal_prob: np.ndarray,
    default_threshold: float,
) -> float:
    """Choose threshold that maximizes balanced accuracy on available labels."""
    true_preictal = (np.asarray(true_countdown) >= 0).astype(np.int32)
    probs = np.asarray(pred_preictal_prob, dtype=np.float32)

    if len(true_preictal) == 0 or len(np.unique(true_preictal)) < 2:
        return float(default_threshold)

    best_threshold = float(default_threshold)
    best_score = -1.0

    for threshold in np.linspace(0.10, 0.90, num=33, dtype=np.float32):
        pred_binary = (probs >= threshold).astype(np.int32)

        tp = np.sum((true_preictal == 1) & (pred_binary == 1))
        tn = np.sum((true_preictal == 0) & (pred_binary == 0))
        fp = np.sum((true_preictal == 0) & (pred_binary == 1))
        fn = np.sum((true_preictal == 1) & (pred_binary == 0))

        tpr = tp / max(1, tp + fn)
        tnr = tn / max(1, tn + fp)
        balanced_accuracy = 0.5 * (tpr + tnr)

        if balanced_accuracy > best_score:
            best_score = float(balanced_accuracy)
            best_threshold = float(threshold)

    return best_threshold


def _compute_binary_metrics(true_labels: np.ndarray,
                            pred_probs: np.ndarray,
                            threshold: float) -> Dict[str, float]:
    """Compute thresholded binary classification metrics."""
    true_arr = np.asarray(true_labels, dtype=np.int32)
    prob_arr = np.asarray(pred_probs, dtype=np.float32)

    if true_arr.size == 0 or prob_arr.size == 0:
        return {
            'balanced_accuracy': 0.0,
            'sensitivity': 0.0,
            'specificity': 0.0,
            'precision': 0.0,
            'f1': 0.0,
            'tp': 0.0,
            'tn': 0.0,
            'fp': 0.0,
            'fn': 0.0,
        }

    pred_arr = (prob_arr >= float(threshold)).astype(np.int32)
    tp = int(np.sum((true_arr == 1) & (pred_arr == 1)))
    tn = int(np.sum((true_arr == 0) & (pred_arr == 0)))
    fp = int(np.sum((true_arr == 0) & (pred_arr == 1)))
    fn = int(np.sum((true_arr == 1) & (pred_arr == 0)))

    sensitivity = tp / max(1, tp + fn)
    specificity = tn / max(1, tn + fp)
    precision = tp / max(1, tp + fp)
    f1 = 2.0 * precision * sensitivity / max(precision + sensitivity, 1e-8)

    return {
        'balanced_accuracy': 0.5 * (sensitivity + specificity),
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1': f1,
        'tp': float(tp),
        'tn': float(tn),
        'fp': float(fp),
        'fn': float(fn),
    }


def _derive_state_targets(true_countdown: np.ndarray,
                          onset_threshold_min: float) -> np.ndarray:
    """Map countdown labels to 3 states: interictal / preictal / onset."""
    true_cd = np.asarray(true_countdown, dtype=np.float32)
    state = np.zeros_like(true_cd, dtype=np.int32)
    active_mask = true_cd >= 0.0
    onset_mask = active_mask & (true_cd < float(onset_threshold_min))
    preictal_mask = active_mask & ~onset_mask
    state[preictal_mask] = 1
    state[onset_mask] = 2
    return state


def _derive_state_probabilities(alert_prob: np.ndarray,
                                onset_logits: np.ndarray) -> Dict[str, np.ndarray]:
    """Derive joint state probabilities from alert and onset heads."""
    alert_arr = np.clip(np.asarray(alert_prob, dtype=np.float32), 0.0, 1.0)
    onset_prob = 1.0 / (1.0 + np.exp(-np.asarray(onset_logits, dtype=np.float32)))
    onset_prob = np.clip(onset_prob, 0.0, 1.0)

    onset_joint = np.clip(alert_arr * onset_prob, 0.0, 1.0)
    preictal_only = np.clip(alert_arr * (1.0 - onset_prob), 0.0, 1.0)
    interictal = np.clip(1.0 - alert_arr, 0.0, 1.0)

    stacked = np.stack([interictal, preictal_only, onset_joint], axis=1)
    pred_state = np.argmax(stacked, axis=1).astype(np.int32)
    return {
        'alert_prob': alert_arr,
        'onset_prob': onset_prob,
        'preictal_only_prob': preictal_only,
        'onset_joint_prob': onset_joint,
        'interictal_prob': interictal,
        'pred_state': pred_state,
    }


def _compute_multiclass_balanced_accuracy(true_labels: np.ndarray,
                                          pred_labels: np.ndarray,
                                          n_classes: int) -> Tuple[float, np.ndarray, np.ndarray]:
    """Return macro recall, confusion matrix, and per-class recall."""
    true_arr = np.asarray(true_labels, dtype=np.int32)
    pred_arr = np.asarray(pred_labels, dtype=np.int32)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_value, pred_value in zip(true_arr, pred_arr):
        if 0 <= true_value < n_classes and 0 <= pred_value < n_classes:
            cm[true_value, pred_value] += 1

    row_sums = cm.sum(axis=1)
    recalls = np.divide(
        np.diag(cm).astype(np.float64),
        np.maximum(row_sums, 1),
        out=np.zeros(n_classes, dtype=np.float64),
        where=np.maximum(row_sums, 1) > 0,
    )
    return float(np.mean(recalls)), cm, recalls


def _build_confusion_matrix_figure(cm: np.ndarray,
                                   class_labels: Tuple[str, ...],
                                   title: str):
    """Create a compact confusion matrix figure for wandb logging."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap='Blues')

    for row_idx in range(cm.shape[0]):
        row_total = max(int(cm[row_idx].sum()), 1)
        for col_idx in range(cm.shape[1]):
            count = int(cm[row_idx, col_idx])
            pct = 100.0 * count / row_total
            ax.text(col_idx, row_idx, f'{count}\n({pct:.1f}%)', ha='center', va='center', fontsize=9)

    ax.set_xticks(np.arange(len(class_labels)))
    ax.set_yticks(np.arange(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=20, ha='right')
    ax.set_yticklabels(class_labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Ground Truth')
    ax.set_title(title, fontweight='bold', fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.040, pad=0.02)
    fig.tight_layout()
    return fig


def _standardize_for_plot(series: np.ndarray) -> np.ndarray:
    """Return a z-scored copy for visualization only.

    This keeps even low-variance signals from appearing numerically flat in
    plots while leaving all training/evaluation computations unchanged.
    """
    arr = np.asarray(series, dtype=np.float32)
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    if not np.isfinite(std) or std < 1e-6:
        std = 1.0
    return (arr - mean) / std


def _select_panel_indices_with_metadata(
    true_countdown: np.ndarray,
    feature_step_s: float,
    window_minutes: float,
    seed: int,
    sample_end_times_s: Optional[np.ndarray],
    recording_ids: Optional[np.ndarray],
    preferred_recording_substring: Optional[str] = None,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Select contiguous panel indices from a single recording when metadata exists.

    This prefers recordings that contain both preictal and interictal samples so the
    panel shows a realistic context around seizure onset. Falls back to dataset-order
    contiguous indices when metadata is unavailable or inconsistent.
    """
    true_countdown = np.asarray(true_countdown, dtype=np.float32)
    n_total = len(true_countdown)
    if n_total == 0:
        return np.array([], dtype=np.int64), None

    step_s = max(float(feature_step_s), 1e-6)
    window_seconds = float(window_minutes) * 60.0
    window_samples = max(1, int(np.ceil(window_seconds / step_s)))

    if (
        sample_end_times_s is None
        or recording_ids is None
        or len(sample_end_times_s) != n_total
        or len(recording_ids) != n_total
    ):
        # Fallback: contiguous by dataset index anchored near a preictal sample.
        rng = np.random.default_rng(seed)
        if n_total <= window_samples:
            selected = np.arange(n_total, dtype=np.int64)
        else:
            pre_idx = np.where(true_countdown >= 0)[0]
            if len(pre_idx) > 0:
                anchor_idx = int(rng.choice(pre_idx))
            else:
                anchor_idx = int(rng.integers(0, n_total))
            half = window_samples // 2
            start = max(0, anchor_idx - half)
            end = min(n_total, start + window_samples)
            start = max(0, end - window_samples)
            selected = np.arange(start, end, dtype=np.int64)

        time_minutes = selected.astype(np.float64) * step_s / 60.0
        return selected, time_minutes

    # Metadata path: choose a single recording and time-sorted contiguous window.
    rng = np.random.default_rng(seed)
    sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float64)
    recording_ids = np.asarray(recording_ids)

    unique_recs = np.unique(recording_ids)
    tier_full = []
    tier_preictal = []
    tier_any = []

    for rec_id in unique_recs:
        rec_mask = recording_ids == rec_id
        rec_count = int(np.sum(rec_mask))
        if rec_count < 2:
            continue

        rec_true = true_countdown[rec_mask]
        has_preictal = bool(np.any(rec_true >= 0))
        has_interictal = bool(np.any(rec_true < 0))

        if has_preictal and has_interictal:
            tier_full.append(rec_id)
        elif has_preictal:
            tier_preictal.append(rec_id)
        else:
            tier_any.append(rec_id)

    pref = (preferred_recording_substring or "").strip().lower()
    preferred_pool = []
    if pref:
        for rec_id in np.concatenate([np.asarray(tier_full), np.asarray(tier_preictal), np.asarray(tier_any)]):
            if pref in str(rec_id).lower():
                preferred_pool.append(rec_id)

    if len(preferred_pool) > 0:
        chosen_rec = rng.choice(preferred_pool)
    elif len(tier_full) > 0:
        chosen_rec = rng.choice(tier_full)
    elif len(tier_preictal) > 0:
        chosen_rec = rng.choice(tier_preictal)
    elif len(tier_any) > 0:
        chosen_rec = rng.choice(tier_any)
    else:
        chosen_rec = None

    if chosen_rec is None:
        # Fallback to contiguous-by-index behavior.
        return _select_panel_indices_with_metadata(
            true_countdown,
            feature_step_s,
            window_minutes,
            seed,
            None,
            None,
        )

    rec_indices = np.where(recording_ids == chosen_rec)[0]
    if rec_indices.size < 2:
        return _select_panel_indices_with_metadata(
            true_countdown,
            feature_step_s,
            window_minutes,
            seed,
            None,
            None,
        )

    rec_order = np.argsort(sample_end_times_s[rec_indices])
    rec_indices = rec_indices[rec_order]
    rec_times_s = sample_end_times_s[rec_indices]
    rec_true = true_countdown[rec_indices]

    pre_mask = rec_true >= 0
    if np.any(pre_mask):
        # Anchor near seizure onset: smallest countdown (closest to 0 minutes).
        pre_indices = np.where(pre_mask)[0]
        anchor_local = int(pre_indices[np.argmin(rec_true[pre_indices])])
    else:
        # No preictal in this recording: center of recording.
        anchor_local = int(len(rec_indices) // 2)

    rec_len = len(rec_indices)
    half = window_samples // 2
    start_local = max(0, anchor_local - half)
    end_local = min(rec_len, start_local + window_samples)
    start_local = max(0, end_local - window_samples)

    selected = rec_indices[start_local:end_local]

    # Rebase time axis so that the panel covers a local window from 0
    # to ~window_minutes instead of large absolute minutes since
    # recording start. This makes plots easier to read (e.g., "0–10 min"
    # around seizure) while keeping relative timing intact.
    base_time_s = float(rec_times_s[start_local]) if rec_times_s.size > 0 else 0.0
    time_minutes = (rec_times_s[start_local:end_local] - base_time_s) / 60.0
    return selected.astype(np.int64), time_minutes.astype(np.float64)


def _extract_hr_hrv_from_sequences(ecg_sequences: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate per-sample HR (bpm) and HRV (ms) from ECG proxy sequences.

    This routine assumes a waveform-like signal with identifiable peaks
    (e.g., ECG). For wearable PPG runs where channel 0 is already an
    HR time series, a simpler summary is computed instead (see
    Trainer.validate).
    """
    if ecg_sequences.ndim != 2 or ecg_sequences.shape[1] < 5 or sampling_rate_hz <= 0:
        zeros = np.zeros(ecg_sequences.shape[0], dtype=np.float32)
        return zeros, zeros

    min_peak_distance = max(1, int(0.3 * sampling_rate_hz))
    hr_values = np.zeros(ecg_sequences.shape[0], dtype=np.float32)
    hrv_values = np.zeros(ecg_sequences.shape[0], dtype=np.float32)

    for idx, sequence in enumerate(ecg_sequences):
        centered = sequence - np.mean(sequence)
        std = np.std(centered)
        if std < 1e-6:
            continue

        threshold = 0.5 * std
        candidates = np.where(
            (centered[1:-1] > centered[:-2])
            & (centered[1:-1] >= centered[2:])
            & (centered[1:-1] > threshold)
        )[0] + 1

        if len(candidates) == 0:
            continue

        selected_peaks = [int(candidates[0])]
        for peak in candidates[1:]:
            if int(peak) - selected_peaks[-1] >= min_peak_distance:
                selected_peaks.append(int(peak))

        if len(selected_peaks) < 2:
            continue

        rr_intervals_s = np.diff(selected_peaks).astype(np.float32) / float(sampling_rate_hz)
        mean_rr = float(np.mean(rr_intervals_s))
        if mean_rr <= 1e-6:
            continue

        hr_bpm = float(np.clip(60.0 / mean_rr, 20.0, 220.0))
        hrv_ms = float(np.std(rr_intervals_s) * 1000.0) if len(rr_intervals_s) > 1 else 0.0

        hr_values[idx] = hr_bpm
        hrv_values[idx] = hrv_ms

    return hr_values, hrv_values


def _parse_recording_uid(recording_uid: str) -> Optional[Tuple[str, str, int]]:
    """Parse canonical recording IDs like sub-001_ses-01_run-3."""
    if not recording_uid:
        return None
    m = re.match(r"sub-([^_]+)_ses-([^_]+)_run-([^_]+)", str(recording_uid))
    if m is None:
        return None
    try:
        return m.group(1), m.group(2), int(m.group(3))
    except ValueError:
        return None


def _interp_to_grid(source_t: np.ndarray, source_v: np.ndarray, target_t: np.ndarray) -> np.ndarray:
    """1D interpolation with edge fill for dense panel alignment."""
    st = np.asarray(source_t, dtype=np.float64)
    sv = np.asarray(source_v, dtype=np.float32)
    tt = np.asarray(target_t, dtype=np.float64)
    if st.size == 0 or sv.size == 0 or tt.size == 0:
        return np.zeros(tt.shape[0], dtype=np.float32)

    order = np.argsort(st)
    st = st[order]
    sv = sv[order]

    uniq_t, uniq_idx = np.unique(st, return_index=True)
    uniq_v = sv[uniq_idx]
    if uniq_t.size == 1:
        return np.full(tt.shape[0], float(uniq_v[0]), dtype=np.float32)

    out = np.interp(tt, uniq_t, uniq_v, left=float(uniq_v[0]), right=float(uniq_v[-1]))
    return out.astype(np.float32)


def _extract_hr_hrv_from_dense_ecg(ecg_signal: np.ndarray, sampling_rate_hz: float) -> Tuple[np.ndarray, np.ndarray]:
    """Estimate continuous HR/HRV arrays from a dense ECG signal."""
    ecg = np.asarray(ecg_signal, dtype=np.float32)
    n = ecg.size
    if n < 5 or sampling_rate_hz <= 0:
        z = np.zeros(n, dtype=np.float32)
        return z, z

    centered = ecg - float(np.mean(ecg))
    std = float(np.std(centered))
    if std < 1e-6:
        z = np.zeros(n, dtype=np.float32)
        return z, z

    peaks: List[int]
    try:
        from scipy.signal import find_peaks  # type: ignore

        prominence = max(std * 0.35, 1e-7)
        min_dist = max(1, int(0.25 * sampling_rate_hz))
        peak_idx, _ = find_peaks(centered, distance=min_dist, prominence=prominence)
        peaks = peak_idx.astype(np.int64).tolist()
    except Exception:
        threshold = 0.5 * std
        candidates = np.where(
            (centered[1:-1] > centered[:-2])
            & (centered[1:-1] >= centered[2:])
            & (centered[1:-1] > threshold)
        )[0] + 1

        min_dist = max(1, int(0.3 * sampling_rate_hz))
        peaks = []
        for pk in candidates.tolist():
            if not peaks or (pk - peaks[-1]) >= min_dist:
                peaks.append(int(pk))

    if len(peaks) < 2:
        z = np.zeros(n, dtype=np.float32)
        return z, z

    rr_s = np.diff(peaks).astype(np.float32) / float(sampling_rate_hz)
    inst_hr = np.clip(60.0 / np.maximum(rr_s, 1e-6), 20.0, 220.0).astype(np.float32)

    inst_hrv = np.zeros_like(inst_hr, dtype=np.float32)
    for i in range(inst_hrv.size):
        left = max(0, i - 5)
        window = rr_s[left:i + 1]
        inst_hrv[i] = float(np.std(window) * 1000.0) if window.size > 1 else 0.0

    peak_times = (np.asarray(peaks[1:], dtype=np.float32) / float(sampling_rate_hz)).astype(np.float64)
    full_times = (np.arange(n, dtype=np.float64) / float(sampling_rate_hz)).astype(np.float64)
    hr = _interp_to_grid(peak_times, inst_hr, full_times)
    hrv = _interp_to_grid(peak_times, inst_hrv, full_times)
    return hr.astype(np.float32), hrv.astype(np.float32)


def _choose_panel_recording_uid(
    panel_recording_ids: np.ndarray,
    all_recording_ids: Optional[np.ndarray],
    panel_loader=None,
) -> Optional[str]:
    """Prefer recordings with richer modalities for dense visualization rebuilds."""
    rec_ids = np.asarray(panel_recording_ids)
    if rec_ids.size == 0:
        return None

    unique_recs, rec_counts = np.unique(rec_ids, return_counts=True)
    chosen_rec = str(unique_recs[int(np.argmax(rec_counts))])
    if panel_loader is None:
        return chosen_rec

    modality_rank: List[Tuple[int, int, str]] = []

    def _modality_score_for_recording(sub_id: str, ses_id: str, run_id: int) -> int:
        # Prefer EEG availability first, then EMG/MOV as secondary richness.
        # This keeps the dedicated EEG subplot alive whenever possible.
        score = 0
        try:
            panel_loader.resolve_subject_edf_path(sub_id, session_id=ses_id, datatype='eeg', run_id=run_id)
            score += 3
        except Exception:
            pass
        for datatype in ('emg', 'mov'):
            try:
                panel_loader.resolve_subject_edf_path(sub_id, session_id=ses_id, datatype=datatype, run_id=run_id)
                score += 1
            except Exception:
                continue
        return score

    for rec_id, freq in zip(unique_recs, rec_counts):
        rec_name = str(rec_id)
        parsed = _parse_recording_uid(rec_name)
        if parsed is None:
            modality_rank.append((-1, int(freq), rec_name))
            continue

        sub_id, ses_id, run_id = parsed
        score = _modality_score_for_recording(sub_id, ses_id, run_id)
        modality_rank.append((score, int(freq), rec_name))

    if modality_rank:
        modality_rank.sort(key=lambda item: (item[0], item[1]), reverse=True)
        chosen_score, _, chosen_name = modality_rank[0]
        chosen_rec = str(chosen_name)

        if chosen_score <= 0 and all_recording_ids is not None:
            global_recs, global_counts = np.unique(np.asarray(all_recording_ids), return_counts=True)
            global_rank: List[Tuple[int, int, str]] = []
            for rec_id, rec_n in zip(global_recs, global_counts):
                rec_name = str(rec_id)
                parsed = _parse_recording_uid(rec_name)
                if parsed is None or int(rec_n) < 2:
                    continue

                sub_id, ses_id, run_id = parsed
                score = _modality_score_for_recording(sub_id, ses_id, run_id)
                global_rank.append((score, int(rec_n), rec_name))

            if global_rank:
                global_rank.sort(key=lambda item: (item[0], item[1]), reverse=True)
                best_score, _, best_rec = global_rank[0]
                if best_score > chosen_score:
                    chosen_rec = str(best_rec)

    return chosen_rec


def _compute_local_balanced_accuracy(
    true_binary: np.ndarray,
    pred_binary: np.ndarray,
    window: int,
) -> np.ndarray:
    """Compute a rolling local balanced accuracy trace for panel diagnostics."""
    truth = np.asarray(true_binary, dtype=np.int32)
    pred = np.asarray(pred_binary, dtype=np.int32)
    out = np.zeros_like(truth, dtype=np.float32)
    for idx in range(truth.size):
        left = max(0, idx - window + 1)
        win_truth = truth[left:idx + 1]
        win_pred = pred[left:idx + 1]
        pos_mask = win_truth == 1
        neg_mask = win_truth == 0

        sensitivity = float(np.mean((win_pred[pos_mask] == 1).astype(np.float32))) if np.any(pos_mask) else 1.0
        specificity = float(np.mean((win_pred[neg_mask] == 0).astype(np.float32))) if np.any(neg_mask) else 1.0
        out[idx] = 0.5 * (sensitivity + specificity)
    return out


def _safe_float(x: float, default: float = 0.0) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else float(default)
    except Exception:
        return float(default)


def _compute_panel_anomaly_report(
    epoch: int,
    data_source: str,
    ecg_signal: np.ndarray,
    hr_series: np.ndarray,
    hrv_series: np.ndarray,
    pred_preictal_prob: np.ndarray,
    true_countdown: np.ndarray,
    eeg_signal: Optional[np.ndarray] = None,
    emg_signal: Optional[np.ndarray] = None,
    mov_signal: Optional[np.ndarray] = None,
    token_roll: Optional[np.ndarray] = None,
    context_energy: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Compute automated panel anomaly checks from epoch visualization inputs."""
    ecg = np.asarray(ecg_signal, dtype=np.float32)
    hr = np.asarray(hr_series, dtype=np.float32)
    hrv = np.asarray(hrv_series, dtype=np.float32)
    p_alert = np.asarray(pred_preictal_prob, dtype=np.float32)
    cd = np.asarray(true_countdown, dtype=np.float32)

    finite_ecg = np.isfinite(ecg)
    if np.any(finite_ecg):
        ecg_f = ecg[finite_ecg]
        ecg_std = _safe_float(np.nanstd(ecg_f))
        ecg_span95 = _safe_float(np.nanpercentile(ecg_f, 95.0) - np.nanpercentile(ecg_f, 5.0))
        ecg_centered = ecg_f - _safe_float(np.nanmedian(ecg_f))
        ecg_absmax = _safe_float(np.nanmax(np.abs(ecg_centered)))
        if ecg.size > 1:
            diff = np.diff(np.asarray(ecg, dtype=np.float32))
            dthr = max(1e-6, 0.01 * max(ecg_span95, 1e-6))
            ecg_near_const_ratio = _safe_float(np.mean(np.abs(diff[np.isfinite(diff)]) < dthr)) if np.any(np.isfinite(diff)) else 1.0
        else:
            ecg_near_const_ratio = 1.0
    else:
        ecg_std = 0.0
        ecg_span95 = 0.0
        ecg_absmax = 0.0
        ecg_near_const_ratio = 1.0

    finite_hr = np.isfinite(hr)
    plausible_hr = finite_hr & (hr >= 20.0) & (hr <= 240.0)
    hr_finite_ratio = _safe_float(np.mean(finite_hr)) if hr.size else 0.0
    hr_plausible_ratio = _safe_float(np.mean(plausible_hr)) if hr.size else 0.0
    hr_min = _safe_float(np.nanmin(hr[finite_hr])) if np.any(finite_hr) else 0.0
    hr_max = _safe_float(np.nanmax(hr[finite_hr])) if np.any(finite_hr) else 0.0
    hr_std = _safe_float(np.nanstd(hr[finite_hr])) if np.any(finite_hr) else 0.0

    def _mod_stats(arr: Optional[np.ndarray]) -> Dict[str, object]:
        if arr is None:
            return {"present": False, "finite_ratio": 0.0, "std": 0.0, "span95": 0.0}
        v = np.asarray(arr, dtype=np.float32)
        f = np.isfinite(v)
        if not np.any(f):
            return {"present": True, "finite_ratio": 0.0, "std": 0.0, "span95": 0.0}
        vf = v[f]
        return {
            "present": True,
            "finite_ratio": _safe_float(np.mean(f)),
            "std": _safe_float(np.nanstd(vf)),
            "span95": _safe_float(np.nanpercentile(vf, 95.0) - np.nanpercentile(vf, 5.0)),
        }

    eeg_stats = _mod_stats(eeg_signal)
    emg_stats = _mod_stats(emg_signal)
    mov_stats = _mod_stats(mov_signal)

    alert_finite = np.isfinite(p_alert)
    alert_std = _safe_float(np.nanstd(p_alert[alert_finite])) if np.any(alert_finite) else 0.0
    alert_mean = _safe_float(np.nanmean(p_alert[alert_finite])) if np.any(alert_finite) else 0.0
    preictal_ratio = _safe_float(np.mean(cd >= 0.0)) if cd.size else 0.0

    token_stats: Dict[str, object] = {
        "present": token_roll is not None,
        "n_channels": 0,
        "dominant_channel_ratio": 0.0,
        "normalized_entropy": 0.0,
        "mean_channel_std": 0.0,
        "mean_token_strength": 0.0,
        "mean_token_certainty": 0.0,
        "mean_token_transition_density": 0.0,
        "context_energy_mean": 0.0,
        "context_energy_std": 0.0,
    }
    if token_roll is not None:
        tok = np.asarray(token_roll, dtype=np.float32)
        if tok.ndim == 2 and tok.shape[0] > 0 and tok.shape[1] > 0:
            n_ch = int(tok.shape[1])
            token_stats["n_channels"] = n_ch
            ch_std = np.nanstd(tok, axis=0)
            token_stats["mean_channel_std"] = _safe_float(np.nanmean(ch_std))
            winners = np.argmax(np.nan_to_num(tok, nan=-1e9), axis=1)
            counts = np.bincount(winners, minlength=n_ch).astype(np.float64)
            probs = counts / max(1.0, float(np.sum(counts)))
            dom = float(np.max(probs)) if probs.size else 0.0
            token_stats["dominant_channel_ratio"] = _safe_float(dom)
            entropy = -np.sum([p * np.log(p + 1e-12) for p in probs if p > 0.0])
            token_stats["normalized_entropy"] = _safe_float(entropy / np.log(max(2, n_ch)))
            clipped = np.clip(tok, 0.0, None)
            row_sum = np.maximum(np.sum(clipped, axis=1, keepdims=True), 1e-6)
            norm = clipped / row_sum
            entropy_ts = -np.sum(norm * np.log(norm + 1e-12), axis=1)
            entropy_ts = entropy_ts / np.log(max(2, n_ch))
            certainty_ts = np.clip(1.0 - entropy_ts, 0.0, 1.0)
            token_stats["mean_token_strength"] = _safe_float(np.mean(np.max(clipped, axis=1)))
            token_stats["mean_token_certainty"] = _safe_float(np.mean(certainty_ts))
            dominant_index = np.argmax(clipped, axis=1)
            switch_events = np.zeros(dominant_index.shape[0], dtype=np.float32)
            if dominant_index.shape[0] > 1:
                switch_events[1:] = (dominant_index[1:] != dominant_index[:-1]).astype(np.float32)
            window = max(3, min(25, dominant_index.shape[0] // 20 if dominant_index.shape[0] >= 20 else 3))
            kernel = np.ones(window, dtype=np.float32) / float(window)
            token_stats["mean_token_transition_density"] = _safe_float(np.mean(np.convolve(switch_events, kernel, mode='same')))

    if context_energy is not None:
        ctx = np.asarray(context_energy, dtype=np.float32)
        ctx_f = np.isfinite(ctx)
        if np.any(ctx_f):
            token_stats["context_energy_mean"] = _safe_float(np.mean(ctx[ctx_f]))
            token_stats["context_energy_std"] = _safe_float(np.std(ctx[ctx_f]))

    ds = str(data_source).lower()
    expected_eeg = ds == 'bids'
    expected_emg_mov = ds == 'bids'

    anomalies: List[str] = []
    if ecg_span95 < 1e-3 or ecg_near_const_ratio > 0.985:
        anomalies.append("ecg_flat_or_near_constant")
    if hr_finite_ratio > 0.0 and hr_plausible_ratio < 0.50:
        anomalies.append("bpm_implausible_range")
    if hr_finite_ratio > 0.0 and hr_std < 1.0:
        anomalies.append("bpm_low_variability")
    if expected_eeg and (not eeg_stats["present"] or float(eeg_stats["finite_ratio"]) < 0.50 or float(eeg_stats["span95"]) < 1e-4):
        anomalies.append("eeg_missing_or_flat")
    if expected_emg_mov and (not emg_stats["present"] or float(emg_stats["finite_ratio"]) < 0.50 or float(emg_stats["span95"]) < 1e-4):
        anomalies.append("emg_missing_or_flat")
    if expected_emg_mov and (not mov_stats["present"] or float(mov_stats["finite_ratio"]) < 0.50 or float(mov_stats["span95"]) < 1e-4):
        anomalies.append("mov_missing_or_flat")
    if alert_std < 0.01:
        anomalies.append("inference_map_low_variance")
    if token_roll is not None:
        if float(token_stats["dominant_channel_ratio"]) > 0.95 and float(token_stats["normalized_entropy"]) < 0.20:
            anomalies.append("token_collapse_dominant_channel")

    return {
        "epoch": int(epoch),
        "data_source": str(data_source),
        "n_samples": int(ecg.size),
        "summary": {
            "anomaly_count": int(len(anomalies)),
            "anomalies": anomalies,
            "has_anomaly": bool(len(anomalies) > 0),
        },
        "expectations": {
            "eeg_expected": bool(expected_eeg),
            "emg_expected": bool(expected_emg_mov),
            "mov_expected": bool(expected_emg_mov),
        },
        "metrics": {
            "ecg_std": ecg_std,
            "ecg_span95": ecg_span95,
            "ecg_absmax_centered": ecg_absmax,
            "ecg_near_const_ratio": ecg_near_const_ratio,
            "hr_finite_ratio": hr_finite_ratio,
            "hr_plausible_ratio": hr_plausible_ratio,
            "hr_min": hr_min,
            "hr_max": hr_max,
            "hr_std": hr_std,
            "hrv_std": _safe_float(np.nanstd(hrv[np.isfinite(hrv)])) if np.any(np.isfinite(hrv)) else 0.0,
            "pred_alert_mean": alert_mean,
            "pred_alert_std": alert_std,
            "gt_preictal_ratio": preictal_ratio,
        },
        "modalities": {
            "eeg": eeg_stats,
            "emg": emg_stats,
            "mov": mov_stats,
        },
        "token": token_stats,
    }


def _compute_patient_difficulty_summary(
    pred_preictal: np.ndarray,
    true_preictal: np.ndarray,
    pred_countdown: np.ndarray,
    true_countdown: np.ndarray,
    patient_ids: np.ndarray,
    threshold: float,
    countdown_max_minutes: float,
    top_k: int = 5,
) -> Dict[str, object]:
    """Compute per-patient difficulty ranking from prediction errors."""
    pred_preictal = np.asarray(pred_preictal, dtype=np.float32)
    true_preictal = np.asarray(true_preictal, dtype=np.float32)
    pred_countdown = np.asarray(pred_countdown, dtype=np.float32)
    true_countdown = np.asarray(true_countdown, dtype=np.float32)
    patient_ids = np.asarray(patient_ids).astype(str)

    n = len(pred_preictal)
    if not (n == len(true_preictal) == len(pred_countdown) == len(true_countdown) == len(patient_ids)):
        return {
            "n_samples": int(n),
            "n_patients": 0,
            "threshold": float(threshold),
            "patient_rows": [],
            "hardest_patients": [],
            "easiest_patients": [],
        }

    if n == 0:
        return {
            "n_samples": 0,
            "n_patients": 0,
            "threshold": float(threshold),
            "patient_rows": [],
            "hardest_patients": [],
            "easiest_patients": [],
        }

    countdown_scale = max(float(countdown_max_minutes), 1e-6)
    prob_abs_err = np.abs(pred_preictal - true_preictal)
    preictal_mask = true_countdown >= 0
    countdown_abs_err = np.abs(pred_countdown - true_countdown)
    countdown_abs_err_norm = np.zeros_like(countdown_abs_err, dtype=np.float32)
    countdown_abs_err_norm[preictal_mask] = np.clip(
        countdown_abs_err[preictal_mask] / countdown_scale,
        0.0,
        3.0,
    )
    difficulty = 0.65 * prob_abs_err + 0.35 * countdown_abs_err_norm

    pred_binary = (pred_preictal >= float(threshold)).astype(np.int32)
    true_binary = (true_preictal >= 0.5).astype(np.int32)

    rows: List[Dict[str, object]] = []
    for patient_id in np.unique(patient_ids):
        mask = patient_ids == patient_id
        if not np.any(mask):
            continue
        p_true = true_binary[mask]
        p_pred = pred_binary[mask]
        tp = int(np.sum((p_true == 1) & (p_pred == 1)))
        tn = int(np.sum((p_true == 0) & (p_pred == 0)))
        fp = int(np.sum((p_true == 0) & (p_pred == 1)))
        fn = int(np.sum((p_true == 1) & (p_pred == 0)))
        sens = float(tp / max(tp + fn, 1))
        spec = float(tn / max(tn + fp, 1))
        bal_acc = 0.5 * (sens + spec)

        p_preictal_mask = preictal_mask[mask]
        p_countdown_mae = float(np.mean(countdown_abs_err[mask][p_preictal_mask])) if np.any(p_preictal_mask) else None

        rows.append({
            "patient_id": str(patient_id),
            "n_samples": int(np.sum(mask)),
            "n_preictal": int(np.sum(p_true)),
            "n_interictal": int(np.sum(mask) - np.sum(p_true)),
            "alert_balanced_accuracy": float(bal_acc),
            "alert_sensitivity": float(sens),
            "alert_specificity": float(spec),
            "mean_prob_abs_error": float(np.mean(prob_abs_err[mask])),
            "preictal_countdown_mae": p_countdown_mae,
            "difficulty_score": float(np.mean(difficulty[mask])),
        })

    rows.sort(key=lambda item: float(item.get("difficulty_score", 0.0)), reverse=True)
    k = max(1, int(top_k))

    return {
        "n_samples": int(n),
        "n_patients": int(len(rows)),
        "threshold": float(threshold),
        "patient_rows": rows,
        "hardest_patients": rows[:k],
        "easiest_patients": rows[-k:][::-1] if rows else [],
    }


class Trainer:
    """
    Training engine with distributed data parallel support for multi-GPU training.
    """
    
    def __init__(
        self,
        config: Config,
        device: torch.device = None,
        use_wandb: bool = True,
        wandb_project: str = "epilepsee-ai",
        wandb_run_name: str = None,
    ):
        """Initialize trainer.
        
        Args:
            config: Master Config object
            device: torch device (if None, auto-select based on availability)
            use_wandb: Whether to use Weights & Biases logging
            wandb_project: Weights & Biases project name
            wandb_run_name: Optional explicit Weights & Biases run name
        """
        self.config = config
        self.training_config = config.training
        self.use_wandb = use_wandb and HAS_WANDB
        self.wandb_project = str(wandb_project)
        self.wandb_run_name = wandb_run_name
        
        # Device setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
        
        logger.info(f"Using device: {self.device}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Distributed training setup
        requested_world_size = int(os.environ.get('WORLD_SIZE', '1'))
        self.distributed = (
            self.training_config.distributed
            and torch.cuda.device_count() > 1
            and requested_world_size > 1
        )
        self.rank = 0
        self.local_rank = int(os.environ.get('LOCAL_RANK', 0))
        self.world_size = 1
        
        if self.distributed:
            if self.device.type == 'cuda':
                self.device = torch.device(f'cuda:{self.local_rank}')
                torch.cuda.set_device(self.local_rank)
            self._setup_distributed()
        
        # Create save directory
        self.save_dir = Path(self.training_config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Early stopping
        self.early_stopping_counter = 0
        self.best_val_metric = float('inf')
        self.best_model_path = None
        
        # Initialize wandb (only on main process)
        self.wandb_run = None
        self._shutdown_requested = False
        if self.use_wandb and self._is_main_process():
            self._init_wandb()

        self._register_signal_handlers()
        self.epoch_visualizer = None
        if self._is_main_process():
            self.epoch_visualizer = SignalVisualizer(
                save_dir=str(self.save_dir / "epoch_visualizations"),
                upload_to_wandb=self.use_wandb,
            )
        
        logger.info(f"Trainer initialized. Rank: {self.rank}/{self.world_size}")
    
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        os.environ.setdefault('NCCL_TIMEOUT', '1800')
        os.environ.setdefault('TORCH_NCCL_TIMEOUT', '1800')
        os.environ.setdefault('NCCL_DEBUG', 'WARN')

        if not dist.is_initialized():
            from datetime import timedelta
            dist.init_process_group(
                backend=self.training_config.backend,
                init_method='env://',
                world_size=int(os.environ.get('WORLD_SIZE', 1)),
                rank=int(os.environ.get('RANK', 0)),
                # Stateful validation on rank 0 can take >10 minutes while
                # other ranks wait at synchronization barriers.
                timeout=timedelta(minutes=90),
            )
        
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        
        # Synchronize
        if self.device.type == 'cuda':
            dist.barrier(device_ids=[self.local_rank])
        else:
            dist.barrier()
        
        logger.info(f"Distributed training initialized. Rank: {self.rank}/{self.world_size}")
    
    def _is_main_process(self) -> bool:
        """Check if current process is main process."""
        return self.rank == 0

    def _register_signal_handlers(self) -> None:
        """Register termination handlers to flush wandb cleanly."""
        if not hasattr(signal, 'SIGTERM'):
            return

        def _handle_signal(signum, frame):
            logger.warning("Received termination signal %s. Attempting clean shutdown.", signum)
            if self.wandb_run is not None and HAS_WANDB:
                try:
                    wandb.finish()
                except Exception as finish_exc:
                    logger.warning("Failed to finish wandb on signal: %s", finish_exc)
            sys.exit(128 + signum)

        try:
            signal.signal(signal.SIGTERM, _handle_signal)
            signal.signal(signal.SIGINT, _handle_signal)
        except Exception as exc:
            logger.warning("Could not register signal handlers: %s", exc)

    def _distributed_train_loader_lengths_match(self, train_loader) -> bool:
        """Return True if all distributed ranks have the same train loader length."""
        if not self.distributed:
            return True
        local_len = torch.tensor([len(train_loader)], dtype=torch.int64, device=self.device)
        max_len = local_len.clone()
        min_len = local_len.clone()
        dist.all_reduce(max_len, op=dist.ReduceOp.MAX)
        dist.all_reduce(min_len, op=dist.ReduceOp.MIN)
        return int(max_len.item()) == int(min_len.item())

    def _get_visualization_indices(self, total_batches: int) -> List[Tuple[int, int]]:
        """Return quarter-epoch batch indices and quarter numbers for visualization."""
        quarters = int(getattr(self.training_config, 'visualization_quarters', 0))
        if quarters <= 1 or total_batches <= 0:
            return []

        indices: List[Tuple[int, int]] = []
        for q in range(1, quarters):
            idx = int(math.ceil(float(total_batches) * float(q) / float(quarters))) - 1
            idx = min(max(0, idx), total_batches - 1)
            if not any(existing_idx == idx for existing_idx, _ in indices):
                indices.append((idx, q))
        return indices

    def _run_visualization_checkpoint(
        self,
        model: nn.Module,
        val_loader,
        criterion: nn.Module,
        epoch: int,
        train_viz_payload: Optional[Dict[str, np.ndarray]] = None,
        suffix: Optional[str] = None,
        max_val_batches: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Run validation and save the epoch visualization, optionally with a suffix."""
        # Validation sets model.eval(); preserve original mode so mid-epoch
        # checkpoints do not leave the training loop in eval mode.
        was_training = bool(model.training)
        module_was_training = bool(model.module.training) if isinstance(model, DDP) else was_training

        if self._use_stateful_training():
            val_metrics, viz_payload = self.validate_stateful(
                model,
                val_loader,
                criterion,
                max_batches=max_val_batches,
            )
        else:
            val_metrics, viz_payload = self.validate(model, val_loader, criterion)

        if was_training:
            model.train()
        if isinstance(model, DDP) and module_was_training:
            model.module.train()

        self._save_epoch_visualizations(epoch, viz_payload, train_viz_payload=train_viz_payload, suffix=suffix)
        return val_metrics, viz_payload

    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not HAS_WANDB:
            logger.warning("wandb not available, skipping initialization")
            return
        
        try:
            # Initialize wandb run
            run_name = self.wandb_run_name or f"{self.config.model.model_type}_{int(time.time())}"
            wandb_mode = os.environ.get('WANDB_MODE', 'online')
            if wandb_mode not in ('online', 'offline', 'disabled'):
                wandb_mode = 'online'

            logger.info("Initializing wandb (project=%s, mode=%s)", self.wandb_project, wandb_mode)

            self.wandb_run = wandb.init(
                project=self.wandb_project,
                name=run_name,
                mode=wandb_mode,
                config={
                    "model_type": self.config.model.model_type,
                    "hidden_dim": self.config.model.hidden_dim,
                    "dropout": self.config.model.dropout,
                    "learning_rate": self.training_config.learning_rate,
                    "batch_size": self.training_config.batch_size,
                    "optimizer": self.training_config.optimizer,
                    "epochs": self.training_config.num_epochs,
                    "dataset_size": "883 seizures",
                    "num_gpus": self.world_size,
                },
                tags=["seizure-prediction", "countdown", self.config.model.model_type],
            )
            logger.info(f"wandb initialized: {self.wandb_run.url}")
        except Exception as e:
            logger.warning(f"wandb initialization failed: {str(e)}")
            self.wandb_run = None

    def _use_stateful_training(self) -> bool:
        return (
            str(getattr(self.training_config, 'training_mode', 'stateless')).lower() == 'stateful'
            and str(getattr(self.training_config, 'batching_strategy', 'random')).lower() == 'patient_sequential'
            and str(self.config.model.model_type).lower() == 'ecg_lstm'
        )

    def _forward_model(
        self,
        model: nn.Module,
        features: torch.Tensor,
        hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_hidden_state: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass wrapper handling single-modal and multimodal inputs."""
        from src.models import ModelFactory as _MF

        if _MF.is_multimodal(self.config.model.model_type):
            ecg_dim = int(self.config.model.ecg_feature_dim)
            eeg_dim = int(self.config.model.eeg_feature_dim)
            motion_dim = int(self.config.model.motion_feature_dim)
            required_dim = ecg_dim + eeg_dim + motion_dim

            if features.shape[-1] < required_dim:
                raise ValueError(
                    f"Multimodal model requires >= {required_dim} features, got {features.shape[-1]}"
                )

            ecg_x = features[:, :, :ecg_dim]
            eeg_x = features[:, :, ecg_dim:ecg_dim + eeg_dim]
            motion_x = features[:, :, ecg_dim + eeg_dim:ecg_dim + eeg_dim + motion_dim]
            outputs = model(ecg_x, eeg_x, motion_x)
        else:
            if hidden_state is not None:
                outputs = model(features, hidden_state)
            else:
                outputs = model(features)

        next_hidden_state = None
        context_state = None

        # Support models that may emit auxiliary outputs in addition to
        # (pre_ictal_pred, countdown_pred).
        if isinstance(outputs, dict):
            pre_ictal_pred = (
                outputs.get('pre_ictal_pred')
                or outputs.get('classification')
                or outputs.get('pre_ictal_logits')
            )
            countdown_pred = (
                outputs.get('countdown_pred')
                or outputs.get('regression')
                or outputs.get('countdown')
            )
            if pre_ictal_pred is None or countdown_pred is None:
                raise ValueError(
                    "Model forward dict output must include pre-ictal and countdown tensors. "
                    f"Available keys: {sorted(outputs.keys())}"
                )
            if return_hidden_state:
                context_state = outputs.get('context_state')
                if context_state is None:
                    context_state = getattr(model, 'last_context_state', None)
                    if context_state is None and hasattr(model, 'module'):
                        context_state = getattr(model.module, 'last_context_state', None)
                return pre_ictal_pred, countdown_pred, outputs.get('hidden_state'), context_state
            return pre_ictal_pred, countdown_pred

        if isinstance(outputs, (tuple, list)):
            if len(outputs) < 2:
                raise ValueError(
                    "Model forward must return at least two outputs: "
                    "(pre_ictal_pred, countdown_pred)"
                )
            if len(outputs) >= 3:
                next_hidden_state = outputs[2]
            if return_hidden_state:
                context_state = getattr(model, 'last_context_state', None)
                if context_state is None and hasattr(model, 'module'):
                    context_state = getattr(model.module, 'last_context_state', None)
                return outputs[0], outputs[1], next_hidden_state, context_state
            return outputs[0], outputs[1]

        raise ValueError(
            "Model forward returned unsupported output type. Expected tuple/list/dict with "
            "pre-ictal and countdown predictions."
        )

    def _create_stateful_loaders(self, train_dataset, val_dataset):
        """Create patient-partitioned stateful loaders for training/validation."""
        if getattr(train_dataset, 'subject_ids', None) is None:
            raise ValueError('Stateful training requires subject_ids on train dataset')

        rank_train_dataset = train_dataset
        if self.distributed:
            assignments = partition_patients_by_batch_count(
                train_dataset,
                batch_size=self.training_config.batch_size,
                world_size=self.world_size,
            )
            my_patients = assignments[self.rank]
            subject_ids = np.asarray(train_dataset.subject_ids).astype(str)
            keep_idx = np.where(np.isin(subject_ids, np.asarray(my_patients, dtype=str)))[0]
            rank_train_dataset = IndexedTemporalDatasetView(train_dataset, keep_idx)
            if self._is_main_process():
                logger.info('Stateful DDP patient partitioning enabled across %d ranks', self.world_size)

        train_loader = TemporalPatientSequenceDataLoader(
            rank_train_dataset,
            batch_size=self.training_config.batch_size,
            reset_hidden_between_epochs=bool(getattr(self.training_config, 'reset_hidden_between_epochs', False)),
            allow_partial_batches=True,
            shuffle_patients=True,
        )

        val_loader = None
        if (not self.distributed) or self._is_main_process():
            val_loader = TemporalPatientSequenceDataLoader(
                val_dataset,
                batch_size=self.training_config.batch_size,
                reset_hidden_between_epochs=True,
                allow_partial_batches=True,
                shuffle_patients=False,
            )

        if self._is_main_process():
            logger.info('Stateful train loader: %d patient-sequential batches', len(train_loader))
            if val_loader is not None:
                logger.info('Stateful val loader: %d patient-sequential batches', len(val_loader))

        return train_loader, val_loader

    def _progress(self, iterable, desc: str, total: Optional[int] = None):
        """Return tqdm iterator on main process, plain iterable otherwise."""
        if HAS_TQDM and self._is_main_process():
            return tqdm(iterable, desc=desc, total=total, leave=False)
        return iterable
    
    def create_dataloaders(self, train_dataset: SeizureDataset, 
                          val_dataset: SeizureDataset) -> Tuple[DataLoader, DataLoader]:
        """Create distributed dataloaders.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
        
        Returns:
            Tuple of (train_loader, val_loader)
        """
        if self._use_stateful_training():
            return self._create_stateful_loaders(train_dataset, val_dataset)

        long_sweep_mode = bool(getattr(self.training_config, 'long_sweep_training', False))

        # Distributed sampler for training
        if self.distributed:
            # In distributed mode, use stratified sampling to ensure each GPU sees balanced batches
            train_sampler = DistributedSampler(
                train_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=(not long_sweep_mode),
                seed=self.config.data.random_seed
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
                seed=self.config.data.random_seed
            )
            
            if self._is_main_process():
                logger.info("Using DistributedSampler for multi-GPU training")
                logger.info("Note: Class weighting handled by loss function in distributed mode")
        else:
            val_sampler = None

            # Weighted random sampling for class imbalance (single-process only)
            if (not long_sweep_mode) and getattr(self.training_config, 'use_weighted_sampling', True):
                preictal_mask = train_dataset.preictal_labels > 0.5
                n_preictal = int(np.sum(preictal_mask))
                n_interictal = len(train_dataset) - n_preictal
                
                # Create weights: preictal samples get higher weight
                imbalance_ratio = n_interictal / max(1, n_preictal)
                sample_weights = np.where(preictal_mask, imbalance_ratio, 1.0)
                
                # Oversample to see more preictal samples per epoch
                num_samples = len(train_dataset) * 2

                train_sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=num_samples,
                    replacement=True,
                )
                
                logger.info(f"WeightedRandomSampler: {n_preictal} preictal, {n_interictal} interictal")
                logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1, oversampling {num_samples} samples/epoch")
            else:
                train_sampler = None
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.training_config.batch_size,
            sampler=train_sampler,
            shuffle=(False if long_sweep_mode else (train_sampler is None)),
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            drop_last=(False if long_sweep_mode else True)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.training_config.batch_size * 2,  # Can use larger batch for eval
            sampler=val_sampler,
            shuffle=False,
            num_workers=self.training_config.num_workers,
            pin_memory=self.training_config.pin_memory,
            drop_last=False
        )
        
        if self._is_main_process():
            logger.info(f"Train loader: {len(train_loader)} batches")
            logger.info(f"Val loader: {len(val_loader)} batches")
            if long_sweep_mode:
                logger.info("Long-sweep dataloader mode enabled: preserving temporal order, no random sampling")
                online_aug_enabled = bool(getattr(train_dataset, '_online_aug_enabled', False))
                online_aug_prob = float(getattr(train_dataset, '_online_aug_probability', 0.0))
                logger.info(
                    "Long-sweep online augmentation: enabled=%s, probability=%.2f",
                    online_aug_enabled,
                    online_aug_prob,
                )
            if hasattr(train_dataset, 'class_distribution'):
                logger.info(f"Train class distribution: {train_dataset.class_distribution}")
            if hasattr(val_dataset, 'class_distribution'):
                logger.info(f"Val class distribution: {val_dataset.class_distribution}")
            if self.distributed and getattr(self.training_config, 'use_weighted_sampling', False):
                logger.warning("WeightedRandomSampler is disabled in distributed mode; using loss class weighting instead.")
        
        return train_loader, val_loader

    def train_epoch_stateful(
        self,
        model: nn.Module,
        train_loader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_loader=None,
        visualization_indices: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Train one epoch with hidden-state carry across sequential patient batches."""
        model.train()
        if isinstance(model, DDP):
            model.module.train()
        if not model.training or (isinstance(model, DDP) and not model.module.training):
            logger.warning("Model training mode was not enabled; forcing train() on DDP wrapper and module.")
            model.train()
            if isinstance(model, DDP):
                model.module.train()
        hidden_mgr = HiddenStateManager(
            device=str(self.device),
            detach_interval=max(1, int(getattr(self.training_config, 'hidden_state_detach_interval', 10))),
        )
        if bool(getattr(self.training_config, 'reset_hidden_between_epochs', False)):
            hidden_mgr.reset()

        metrics_sum = defaultdict(float)
        num_batches = 0
        all_pred_preictal = []
        all_true_preictal = []
        all_pred_countdown = []
        all_true_countdown = []

        train_iter = self._progress(train_loader, desc=f"Train Epoch {epoch}", total=len(train_loader))
        join_ctx = model.join() if self.distributed and hasattr(model, 'join') else nullcontext()
        visualization_index_map = {idx: q for idx, q in (visualization_indices or [])}
        distributed_viz_sync = bool(self.distributed)
        if self.distributed and visualization_index_map and not self._distributed_train_loader_lengths_match(train_loader):
            if self._is_main_process():
                logger.warning(
                    "Unequal distributed stateful train loader lengths detected; "
                    "running intermediate quarter-epoch visualizations on rank 0 without cross-rank barriers."
                )
            distributed_viz_sync = False
            if not self._is_main_process():
                visualization_index_map = {}
        last_grad_norm: Optional[float] = None
        last_batch_size: Optional[int] = None
        optimizer_steps = 0
        batch_loss_ema: Optional[float] = None

        with join_ctx:
            for batch_idx, (patient_id, batch_features, batch_labels, batch_weights) in enumerate(train_iter):
                if (not model.training) or (isinstance(model, DDP) and not model.module.training):
                    logger.warning(
                        "Model switched to eval mode during stateful training loop; restoring train mode "
                        "(epoch=%d, batch=%d).",
                        int(epoch),
                        int(batch_idx),
                    )
                    model.train()
                    if isinstance(model, DDP):
                        model.module.train()

                features = torch.from_numpy(batch_features).to(self.device)
                labels = torch.from_numpy(batch_labels).to(self.device)
                weights = torch.from_numpy(batch_weights).to(self.device)

                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

                features = torch.clamp(features, -200.0, 200.0)
                labels = torch.clamp(labels, -1.0, float(self.config.model.output_countdown_max))
                weights = torch.clamp(weights, 0.0, 10.0)

                pre_ictal_labels = (labels >= 0).float()
                hidden_state = hidden_mgr.get_hidden_state()
                current_batch_size = int(features.shape[0])
                if last_batch_size is not None and last_batch_size != current_batch_size:
                    hidden_state = None
                last_batch_size = current_batch_size
                pre_ictal_pred, countdown_pred, hidden_state_new, _context_state = self._forward_model(
                    model,
                    features,
                    hidden_state=hidden_state,
                    return_hidden_state=True,
                )

                loss_dict = criterion(pre_ictal_pred, countdown_pred, pre_ictal_labels, labels, weights)
                optimizer.zero_grad()
                loss_dict['total'].backward()

                if self.training_config.gradient_clip_norm > 0:
                    last_grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), self.training_config.gradient_clip_norm))

                optimizer.step()
                optimizer_steps += 1

                if hidden_state_new is not None:
                    if isinstance(hidden_state_new, tuple):
                        hidden_state_new = tuple(h.detach() for h in hidden_state_new)
                    else:
                        hidden_state_new = hidden_state_new.detach()
                hidden_mgr.update_for_batch(str(patient_id), hidden_state_new)

                num_batches += 1
                for key, value in loss_dict.items():
                    metrics_sum[f'train_{key}'] += float(value.detach().cpu())

                all_pred_preictal.extend(pre_ictal_pred.detach().cpu().numpy())
                all_true_preictal.extend(pre_ictal_labels.detach().cpu().numpy())
                all_pred_countdown.extend(countdown_pred.detach().cpu().numpy())
                all_true_countdown.extend(labels.detach().cpu().numpy())

                if batch_idx % self.training_config.log_interval == 0 and self._is_main_process():
                    avg_loss = metrics_sum['train_total'] / max(num_batches, 1)
                    batch_loss_ema = avg_loss if batch_loss_ema is None else (0.98 * batch_loss_ema + 0.02 * avg_loss)
                    if last_grad_norm is not None:
                        logger.info(
                            f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {avg_loss:.4f} "
                            f"| grad_norm: {last_grad_norm:.4e}"
                        )
                    if HAS_TQDM and hasattr(train_iter, 'set_postfix'):
                        train_iter.set_postfix({
                            'loss': f'{avg_loss:.4f}',
                            'lr': f"{optimizer.param_groups[0]['lr']:.2e}",
                            'steps': optimizer_steps,
                        })

                    if self.wandb_run:
                        log_payload = {
                            'epoch': epoch,
                            'batch': batch_idx,
                            'train/batch_loss': avg_loss,
                            'train/batch_loss_ema': batch_loss_ema,
                            'train/learning_rate': optimizer.param_groups[0]['lr'],
                            'train/optimizer_steps': optimizer_steps,
                        }
                        if last_grad_norm is not None:
                            log_payload['train/grad_norm'] = last_grad_norm
                        wandb.log(log_payload)

                if batch_idx in visualization_index_map:
                    if distributed_viz_sync:
                        dist.barrier()

                    if self._is_main_process():
                        quarter = visualization_index_map[batch_idx]
                        suffix = f"q{quarter}"
                        logger.info(
                            "Starting intermediate visualization checkpoint %s at epoch %d batch %d/%d",
                            suffix,
                            int(epoch),
                            int(batch_idx),
                            int(len(train_loader)),
                        )
                        self._run_visualization_checkpoint(
                            model,
                            val_loader,
                            criterion,
                            epoch,
                            train_viz_payload={
                                'pred_preictal': np.array(all_pred_preictal, dtype=np.float32),
                                'true_preictal': np.array(all_true_preictal, dtype=np.float32),
                            },
                            suffix=suffix,
                        )
                        logger.info(
                            "Completed intermediate visualization checkpoint %s at epoch %d",
                            suffix,
                            int(epoch),
                        )

                    if distributed_viz_sync:
                        try:
                            from datetime import timedelta
                            dist.barrier(timeout=timedelta(minutes=45))
                        except TypeError:
                            dist.barrier()

        stat_tensor = torch.tensor([
            metrics_sum.get('train_total', 0.0),
            metrics_sum.get('train_classification', 0.0),
            metrics_sum.get('train_regression', 0.0),
            metrics_sum.get('train_ranking', 0.0),
            float(num_batches),
        ], device=self.device, dtype=torch.float64)
        if self.distributed:
            dist.all_reduce(stat_tensor, op=dist.ReduceOp.SUM)

        global_batches = max(int(stat_tensor[4].item()), 1)
        metrics = {
            'train_total': float(stat_tensor[0].item() / global_batches),
            'train_classification': float(stat_tensor[1].item() / global_batches),
            'train_regression': float(stat_tensor[2].item() / global_batches),
            'train_ranking': float(stat_tensor[3].item() / global_batches),
        }

        train_viz_payload = {
            'pred_preictal': np.array(all_pred_preictal, dtype=np.float32),
            'true_preictal': np.array(all_true_preictal, dtype=np.float32),
        }
        return metrics, train_viz_payload

    def validate_stateful(
        self,
        model: nn.Module,
        val_loader,
        criterion: nn.Module,
        max_batches: Optional[int] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Validate with continuous hidden-state carry per patient on rank 0."""
        if val_loader is None:
            return ({'val_total': 0.0, 'val_classification': 0.0, 'val_regression': 0.0, 'val_ranking': 0.0, 'val_mae': 0.0, 'val_medae': 0.0, 'val_rmse': 0.0}, {})

        if self.distributed and not self._is_main_process():
            return ({'val_total': 0.0, 'val_classification': 0.0, 'val_regression': 0.0, 'val_ranking': 0.0, 'val_mae': 0.0, 'val_medae': 0.0, 'val_rmse': 0.0}, {})

        model.eval()
        # In distributed stateful mode, only rank 0 runs validation. Bypass the
        # DDP wrapper so this rank does not participate in DDP collectives while
        # non-main ranks wait at synchronization barriers.
        eval_model = model.module if isinstance(model, DDP) else model
        hidden_mgr = HiddenStateManager(device=str(self.device), detach_interval=0)
        metrics = defaultdict(float)
        num_batches = 0
        all_pred_countdown = []
        all_true_countdown = []
        all_pred_preictal = []
        all_true_preictal = []
        all_ecg_proxy = []
        all_eeg_proxy = []
        all_ppg_proxy = []
        all_eda_proxy = []
        all_adxl_proxy = []
        all_hr_proxy = []
        all_hrv_proxy = []
        all_context_energy = []
        last_batch_size: Optional[int] = None

        with torch.no_grad():
            val_iter = self._progress(val_loader, desc='Validation', total=len(val_loader))
            for patient_id, batch_features, batch_labels, batch_weights in val_iter:
                features = torch.from_numpy(batch_features).to(self.device)
                labels = torch.from_numpy(batch_labels).to(self.device)
                weights = torch.from_numpy(batch_weights).to(self.device)

                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)
                features_raw_cpu = features.cpu()

                features = torch.clamp(features, -200.0, 200.0)
                labels = torch.clamp(labels, -1.0, float(self.config.model.output_countdown_max))
                weights = torch.clamp(weights, 0.0, 10.0)
                pre_ictal_labels = (labels >= 0).float()

                hidden_state = hidden_mgr.get_hidden_state()
                current_batch_size = int(features.shape[0])
                if last_batch_size is not None and last_batch_size != current_batch_size:
                    hidden_state = None
                last_batch_size = current_batch_size
                pre_ictal_pred, countdown_pred, hidden_state_new, context_state = self._forward_model(
                    eval_model,
                    features,
                    hidden_state=hidden_state,
                    return_hidden_state=True,
                )
                if hidden_state_new is not None:
                    if isinstance(hidden_state_new, tuple):
                        hidden_state_new = tuple(h.detach() for h in hidden_state_new)
                    else:
                        hidden_state_new = hidden_state_new.detach()
                hidden_mgr.update_for_batch(str(patient_id), hidden_state_new)

                loss_dict = criterion(pre_ictal_pred, countdown_pred, pre_ictal_labels, labels, weights)
                num_batches += 1
                for key, value in loss_dict.items():
                    metrics[f'val_{key}'] += float(value.detach().cpu())
                if max_batches is not None and num_batches >= max_batches:
                    break

                all_pred_preictal.extend(pre_ictal_pred.detach().cpu().numpy())
                all_true_preictal.extend(pre_ictal_labels.detach().cpu().numpy())
                all_pred_countdown.extend(countdown_pred.detach().cpu().numpy())
                all_true_countdown.extend(labels.cpu().numpy())

                full_features = features_raw_cpu.numpy()
                n_feat = full_features.shape[-1]
                source = getattr(self.config.data, "data_source", "bids")
                is_multimodal = ModelFactory.is_multimodal(self.config.model.model_type)
                ecg_dim = int(self.config.model.ecg_feature_dim)
                eeg_dim = int(self.config.model.eeg_feature_dim) if is_multimodal else 0

                if is_multimodal and n_feat >= ecg_dim + eeg_dim and ecg_dim > 0 and eeg_dim > 0:
                    ecg_block = full_features[:, :, :ecg_dim]
                    eeg_block = full_features[:, :, ecg_dim:ecg_dim + eeg_dim]

                    ecg_std = ecg_block.reshape(-1, ecg_block.shape[-1]).std(axis=0)
                    eeg_std = eeg_block.reshape(-1, eeg_block.shape[-1]).std(axis=0)

                    ecg_best_ch = int(np.argmax(ecg_std)) if ecg_block.shape[-1] > 0 else 0
                    eeg_best_ch = int(np.argmax(eeg_std)) if eeg_block.shape[-1] > 0 else 0

                    ecg_sequences = ecg_block[:, :, ecg_best_ch]
                    eeg_sequences = eeg_block[:, :, eeg_best_ch]
                    mid_idx = ecg_sequences.shape[1] // 2

                    all_ecg_proxy.extend(ecg_sequences[:, mid_idx])
                    all_eeg_proxy.extend(eeg_sequences[:, mid_idx])

                    if source == "wearable":
                        ppg_sequences = ecg_block[:, :, 0] if ecg_block.shape[-1] > 0 else ecg_sequences
                        adxl_sequences = ecg_block[:, :, 1] if ecg_block.shape[-1] > 1 else np.zeros_like(ppg_sequences)
                        eda_sequences = eeg_block[:, :, 0] if eeg_block.shape[-1] > 0 else np.zeros_like(ppg_sequences)
                        all_adxl_proxy.extend(adxl_sequences[:, mid_idx])
                    else:
                        ppg_sequences = ecg_sequences
                        eda_sequences = np.zeros_like(ecg_sequences)

                    all_ppg_proxy.extend(ppg_sequences[:, mid_idx])
                    all_eda_proxy.extend(eda_sequences[:, mid_idx])
                    signal_sequences = ecg_sequences
                else:
                    signal_sequences = full_features[:, :, 0]
                    mid_idx = signal_sequences.shape[1] // 2
                    all_ecg_proxy.extend(signal_sequences[:, mid_idx])
                    all_ppg_proxy.extend(signal_sequences[:, mid_idx])
                    all_eda_proxy.extend(np.zeros(signal_sequences.shape[0], dtype=np.float32))
                    all_adxl_proxy.extend(np.zeros(signal_sequences.shape[0], dtype=np.float32))

                hr_batch = signal_sequences.mean(axis=1).astype(np.float32)
                hrv_batch = signal_sequences.std(axis=1).astype(np.float32)
                all_hr_proxy.extend(hr_batch)
                all_hrv_proxy.extend(hrv_batch)
                if context_state is not None:
                    ctx = context_state.detach().cpu().numpy()
                    if ctx.ndim == 2:
                        all_context_energy.extend(np.linalg.norm(ctx, axis=1).astype(np.float32))
                    else:
                        all_context_energy.extend(np.asarray(ctx, dtype=np.float32).reshape(-1).tolist())

        all_pred_countdown = np.array(all_pred_countdown)
        all_true_countdown = np.array(all_true_countdown)
        preictal_mask = all_true_countdown >= 0
        if np.sum(preictal_mask) > 0:
            mae = np.mean(np.abs(all_pred_countdown[preictal_mask] - all_true_countdown[preictal_mask]))
            medae = np.median(np.abs(all_pred_countdown[preictal_mask] - all_true_countdown[preictal_mask]))
            rmse = np.sqrt(np.mean((all_pred_countdown[preictal_mask] - all_true_countdown[preictal_mask]) ** 2))
        else:
            mae = medae = rmse = 0.0

        for key in list(metrics.keys()):
            metrics[key] /= max(num_batches, 1)
        metrics['val_mae'] = float(mae)
        metrics['val_medae'] = float(medae)
        metrics['val_rmse'] = float(rmse)

        alert_metrics = _compute_binary_metrics(
            (all_true_countdown >= 0).astype(np.int32),
            np.array(all_pred_preictal, dtype=np.float32),
            float(self.config.loss.detection_threshold),
        )
        metrics['val_alert_balanced_accuracy'] = alert_metrics['balanced_accuracy']
        metrics['val_alert_sensitivity'] = alert_metrics['sensitivity']
        metrics['val_alert_specificity'] = alert_metrics['specificity']
        metrics['val_alert_precision'] = alert_metrics['precision']
        metrics['val_alert_f1'] = alert_metrics['f1']

        viz_payload = {
            'pred_preictal': np.array(all_pred_preictal, dtype=np.float32),
            'pred_preictal_smooth': _causal_smooth(np.array(all_pred_preictal, dtype=np.float32)),
            'pred_countdown': np.array(all_pred_countdown, dtype=np.float32),
            'true_countdown': np.array(all_true_countdown, dtype=np.float32),
            'true_preictal': np.array(all_true_preictal, dtype=np.float32),
            'ecg_proxy': np.array(all_ecg_proxy, dtype=np.float32),
            'ppg_proxy': np.array(all_ppg_proxy, dtype=np.float32),
            'eda_proxy': np.array(all_eda_proxy, dtype=np.float32),
            'adxl_proxy': np.array(all_adxl_proxy, dtype=np.float32),
            'hr_proxy': np.array(all_hr_proxy, dtype=np.float32),
            'hrv_proxy': np.array(all_hrv_proxy, dtype=np.float32),
        }
        if len(all_context_energy) == len(all_true_countdown):
            viz_payload['context_energy'] = np.array(all_context_energy, dtype=np.float32)
        if len(all_eeg_proxy) == len(all_ecg_proxy) and len(all_eeg_proxy) > 0:
            viz_payload['eeg_proxy'] = np.array(all_eeg_proxy, dtype=np.float32)

        val_dataset = getattr(val_loader, 'seizure_dataset', getattr(val_loader, 'dataset', None))
        self._cached_val_dataset_ref = val_dataset
        sample_end_times_s = getattr(val_dataset, 'sample_end_times_s', None) if val_dataset is not None else None
        recording_ids = getattr(val_dataset, 'recording_ids', None) if val_dataset is not None else None
        n_points = int(len(all_true_countdown))
        if (
            sample_end_times_s is not None
            and recording_ids is not None
            and len(sample_end_times_s) >= n_points
            and len(recording_ids) >= n_points
            and n_points > 0
        ):
            viz_payload['sample_end_times_s'] = np.asarray(sample_end_times_s[:n_points], dtype=np.float32)
            viz_payload['recording_ids'] = np.asarray(recording_ids[:n_points])

        recordings_meta = getattr(val_dataset, 'recordings_meta', None) if val_dataset is not None else None
        if recordings_meta is not None:
            viz_payload['recordings_meta'] = recordings_meta

        return dict(metrics), viz_payload
    
    def create_model_optimizer_scheduler(self) -> Tuple[nn.Module, optim.Optimizer, object]:
        """Create model, optimizer, and learning rate scheduler.
        
        Returns:
            Tuple of (model, optimizer, scheduler)
        """
        # Create model
        model = ModelFactory.create_model(self.config.model)
        model = model.to(self.device)
        
        # Wrap with DDP if distributed
        if self.distributed:
            model = DDP(
                model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False
            )
        
        num_params = sum(p.numel() for p in model.parameters())
        if self._is_main_process():
            logger.info(f"Model created: {self.config.model.model_type}")
            logger.info(f"Total parameters: {num_params:,}")
        
        # Create optimizer
        if self.training_config.optimizer == "adam":
            optimizer = optim.Adam(
                model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "adamw":
            optimizer = optim.AdamW(
                model.parameters(),
                lr=self.training_config.learning_rate,
                weight_decay=self.training_config.weight_decay
            )
        elif self.training_config.optimizer == "sgd":
            optimizer = optim.SGD(
                model.parameters(),
                lr=self.training_config.learning_rate,
                momentum=self.training_config.momentum,
                weight_decay=self.training_config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.training_config.optimizer}")
        
        # Create learning rate scheduler
        if self.training_config.lr_scheduler == "reduce_on_plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=self.training_config.lr_factor,
                patience=self.training_config.lr_patience,
                min_lr=self.training_config.lr_min
            )
        elif self.training_config.lr_scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.training_config.num_epochs
            )
        elif self.training_config.lr_scheduler == "step":
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=10,
                gamma=0.5
            )
        else:
            scheduler = None
        
        if self._is_main_process():
            logger.info(f"Optimizer: {self.training_config.optimizer}")
            logger.info(f"LR scheduler: {self.training_config.lr_scheduler}")
        
        return model, optimizer, scheduler
    
    def train_epoch(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: optim.Optimizer,
        epoch: int,
        val_loader=None,
        visualization_indices: Optional[List[Tuple[int, int]]] = None,
    ) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Train for one epoch.
        
        Args:
            model: Neural network model
            train_loader: Training dataloader
            criterion: Loss function
            optimizer: Optimizer
            epoch: Epoch number
        
        Returns:
            Dictionary with epoch metrics
        """
        model.train()
        
        metrics = defaultdict(float)
        batch_loss_ema = None
        num_batches = 0
        all_pred_preictal = []
        all_true_preictal = []
        all_pred_countdown = []
        all_true_countdown = []
        all_patient_ids = []
        
        # Update sampler for distributed training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_iter = self._progress(
            enumerate(train_loader),
            desc=f"Train Epoch {epoch}",
            total=len(train_loader)
        )
        visualization_index_map = {idx: q for idx, q in (visualization_indices or [])}
        distributed_viz_sync = bool(self.distributed)
        if self.distributed and visualization_index_map and not self._distributed_train_loader_lengths_match(train_loader):
            if self._is_main_process():
                logger.warning(
                    "Unequal distributed train loader lengths detected; "
                    "running intermediate quarter-epoch visualizations on rank 0 without cross-rank barriers."
                )
            distributed_viz_sync = False
            if not self._is_main_process():
                visualization_index_map = {}

        # Track last gradient norm for debugging numerical issues
        last_grad_norm: Optional[float] = None

        # Optimizer step scheduling:
        # - batch: step every batch (legacy behavior)
        # - patient: step when patient ID boundary is crossed
        # - patients_group: step every N patient boundaries
        step_scope_raw = str(getattr(self.training_config, 'optimizer_step_scope', 'batch')).lower()
        if step_scope_raw == 'n_patients':
            # Backward-compatible alias if old configs use this spelling.
            step_scope_raw = 'patients_group'
        valid_step_scopes = {'batch', 'patient', 'patients_group'}
        step_scope = step_scope_raw if step_scope_raw in valid_step_scopes else 'batch'
        patients_per_step = max(1, int(getattr(self.training_config, 'patients_per_step', 1)))
        grouped_patient_mode = (step_scope == 'patients_group') and (patients_per_step > 1)

        # Resolve per-sample patient IDs if available (required for boundary stepping).
        patient_ids: Optional[np.ndarray] = None
        if step_scope != 'batch':
            dataset_obj = train_loader.dataset
            base_dataset = getattr(dataset_obj, '_dataset', dataset_obj)
            if hasattr(base_dataset, 'subject_ids'):
                try:
                    if hasattr(dataset_obj, '_dataset') and hasattr(dataset_obj, '_offset') and hasattr(dataset_obj, '_buf_size'):
                        start = int(dataset_obj._offset)
                        stop = start + int(dataset_obj._buf_size)
                        patient_ids = np.atleast_1d(np.asarray(base_dataset.subject_ids[start:stop]))
                    else:
                        patient_ids = np.atleast_1d(np.asarray(base_dataset.subject_ids))
                    if patient_ids.ndim == 0 or len(patient_ids) == 0:
                        patient_ids = None
                        step_scope = 'batch'
                except Exception as exc:
                    if self._is_main_process():
                        logger.warning(
                            "Failed to read subject_ids for patient-level stepping; falling back to batch stepping: %s",
                            exc,
                        )
                    step_scope = 'batch'
            else:
                if self._is_main_process():
                    logger.warning(
                        "optimizer_step_scope=%s requested but dataset has no subject_ids; falling back to batch stepping.",
                        step_scope,
                    )
                step_scope = 'batch'

        # Track boundary state for patient/group stepping.
        current_patient_id: Optional[object] = None
        patients_accumulated_for_step = 0
        batches_since_step = 0
        optimizer_steps = 0
        self._last_train_patient_summary = None

        optimizer.zero_grad()

        def _sanitize_clip_step(step_batch_idx: int) -> Optional[float]:
            """Apply gradient safety checks, optional clipping, and optimizer step."""
            nonlocal optimizer_steps
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param.grad is None or param.grad.data.numel() == 0:
                        continue
                    if not torch.isfinite(param.grad).all():
                        clean_grad = torch.nan_to_num(
                            param.grad,
                            nan=0.0,
                            posinf=0.0,
                            neginf=0.0,
                        )
                        clean_grad = torch.clamp(clean_grad, -1000.0, 1000.0)
                        logger.error(
                            "Non-finite gradient detected; sanitizing before optimizer step "
                            f"(epoch={epoch}, batch={step_batch_idx}, param={name}, "
                            f"max_abs={float(clean_grad.abs().max()):.4e})"
                        )
                        param.grad.copy_(clean_grad)

            step_grad_norm: Optional[float] = None
            if self.training_config.gradient_clip_norm > 0:
                step_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                )

            optimizer.step()
            optimizer_steps += 1

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param is None or param.data.numel() == 0:
                        continue
                    if not torch.isfinite(param.data).all():
                        clean = torch.nan_to_num(param.data)
                        logger.error(
                            "Non-finite parameter after optimizer step "
                            f"(epoch={epoch}, batch={step_batch_idx}, param={name}, "
                            f"min={float(clean.min()):.4e}, max={float(clean.max()):.4e})"
                        )
                        raise RuntimeError(
                            "Non-finite model parameters encountered after optimizer step; "
                            "aborting training. See logs for diagnostics."
                        )

            return step_grad_norm

        for batch_idx, batch in train_iter:
            if (not model.training) or (isinstance(model, DDP) and not model.module.training):
                logger.warning(
                    "Model switched to eval mode during training loop; restoring train mode "
                    "(epoch=%d, batch=%d).",
                    int(epoch),
                    int(batch_idx),
                )
                model.train()
                if isinstance(model, DDP):
                    model.module.train()

            # Move data to device
            features = batch[0].to(self.device)
            labels = batch[1].to(self.device)
            weights = batch[2].to(self.device)

            # Robustness: sanitize inputs to avoid propagating NaNs/Infs
            features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
            labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
            weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

            # Clamp to reasonable ranges (features in a modest band, labels in
            # [-1, output_max], weights positive and not extreme).
            features = torch.clamp(features, -200.0, 200.0)
            labels = torch.clamp(labels, -1.0, float(self.config.model.output_countdown_max))
            weights = torch.clamp(weights, 0.0, 10.0)
            
            # Forward pass
            pre_ictal_pred, countdown_pred = self._forward_model(model, features)
            # Detect non-finite model outputs early
            if not torch.isfinite(pre_ictal_pred).all() or not torch.isfinite(countdown_pred).all():
                with torch.no_grad():
                    msg_parts = [
                        f"Non-finite model output at epoch={epoch} batch={batch_idx}",
                        f"pre_ictal_pred[min,max]=[{float(torch.nan_to_num(pre_ictal_pred).min()):.4f}, {float(torch.nan_to_num(pre_ictal_pred).max()):.4f}]",
                        f"countdown_pred[min,max]=[{float(torch.nan_to_num(countdown_pred).min()):.4f}, {float(torch.nan_to_num(countdown_pred).max()):.4f}]",
                    ]
                    logger.error(" | ".join(msg_parts))
                raise RuntimeError("Non-finite model output encountered; aborting training. See logs for diagnostics.")
            
            # Compute loss
            pre_ictal_labels = (labels >= 0).float()
            loss_dict = criterion(pre_ictal_pred, countdown_pred,
                                 pre_ictal_labels, labels, weights)
            
            # Detect non-finite losses early for debugging
            if not torch.isfinite(loss_dict['total']):
                with torch.no_grad():
                    msg_parts = [
                        f"Non-finite loss at epoch={epoch} batch={batch_idx}",
                        f"pre_ictal_pred[min,max]=[{float(torch.nan_to_num(pre_ictal_pred).min()):.4f}, {float(torch.nan_to_num(pre_ictal_pred).max()):.4f}]",
                        f"countdown_pred[min,max]=[{float(torch.nan_to_num(countdown_pred).min()):.4f}, {float(torch.nan_to_num(countdown_pred).max()):.4f}]",
                        f"labels[min,max]=[{float(torch.nan_to_num(labels).min()):.4f}, {float(torch.nan_to_num(labels).max()):.4f}]",
                        f"weights[min,max]=[{float(torch.nan_to_num(weights).min()):.4f}, {float(torch.nan_to_num(weights).max()):.4f}]",
                    ]
                    logger.error(" | ".join(msg_parts))
                raise RuntimeError("Non-finite loss encountered; aborting training. See logs for diagnostics.")
            
            # Backward pass (step timing depends on configured scope)
            loss_dict['total'].backward()

            batch_size = int(labels.shape[0])
            sample_start = int(batch_idx) * int(train_loader.batch_size)
            sample_end = min(sample_start + batch_size, len(train_loader.dataset))
            batch_patient_ids = None
            if patient_ids is not None and sample_start < len(patient_ids):
                batch_patient_ids = patient_ids[sample_start:sample_end]

            do_step_now = False
            if step_scope == 'batch':
                do_step_now = True
            else:
                if patient_ids is None or sample_start >= len(patient_ids):
                    do_step_now = True
                else:
                    if batch_patient_ids.size == 0:
                        do_step_now = True
                    else:
                        first_patient = batch_patient_ids[0]
                        if current_patient_id is None:
                            current_patient_id = first_patient
                            patients_accumulated_for_step = 1
                        elif first_patient != current_patient_id:
                            current_patient_id = first_patient
                            patients_accumulated_for_step += 1

                        if step_scope == 'patient':
                            next_sample_start = sample_end
                            is_last_batch = (batch_idx + 1) >= len(train_loader)
                            if is_last_batch:
                                do_step_now = True
                            elif next_sample_start < len(patient_ids):
                                next_patient = patient_ids[next_sample_start]
                                if next_patient != current_patient_id:
                                    do_step_now = True
                        elif step_scope == 'patients_group':
                            next_sample_start = sample_end
                            is_last_batch = (batch_idx + 1) >= len(train_loader)
                            boundary_after_batch = False
                            if is_last_batch:
                                boundary_after_batch = True
                            elif next_sample_start < len(patient_ids):
                                next_patient = patient_ids[next_sample_start]
                                boundary_after_batch = (next_patient != current_patient_id)

                            if boundary_after_batch and patients_accumulated_for_step >= patients_per_step:
                                do_step_now = True
                            elif is_last_batch:
                                do_step_now = True

            batches_since_step += 1
            if do_step_now:
                last_grad_norm = _sanitize_clip_step(batch_idx)
                optimizer.zero_grad()
                batches_since_step = 0
                if grouped_patient_mode and step_scope == 'patients_group':
                    patients_accumulated_for_step = 0
            
            # Accumulate metrics
            num_batches += 1
            for key, value in loss_dict.items():
                metrics[f'train_{key}'] += float(value.detach().cpu())
            all_pred_preictal.extend(pre_ictal_pred.detach().cpu().numpy())
            all_true_preictal.extend(pre_ictal_labels.detach().cpu().numpy())
            all_pred_countdown.extend(countdown_pred.detach().cpu().numpy())
            all_true_countdown.extend(labels.detach().cpu().numpy())
            if batch_patient_ids is not None and len(batch_patient_ids) == batch_size:
                all_patient_ids.extend(batch_patient_ids.tolist())
            
            # Log batch
            if batch_idx % self.training_config.log_interval == 0 and self._is_main_process():
                avg_loss = metrics['train_total'] / num_batches
                batch_loss_ema = avg_loss if batch_loss_ema is None else (0.98 * batch_loss_ema + 0.02 * avg_loss)
                if last_grad_norm is not None:
                    logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {avg_loss:.4f} "
                        f"| grad_norm: {last_grad_norm:.4e}"
                    )
                else:
                    logger.info(
                        f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {avg_loss:.4f}"
                    )

                if HAS_TQDM and hasattr(train_iter, "set_postfix"):
                    train_iter.set_postfix({
                        "loss": f"{avg_loss:.4f}",
                        "lr": f"{optimizer.param_groups[0]['lr']:.2e}",
                        "steps": optimizer_steps,
                    })

                    if self.wandb_run:
                        log_payload = {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "train/batch_loss": avg_loss,
                            "train/batch_loss_ema": batch_loss_ema,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                            "train/optimizer_steps": optimizer_steps,
                        }
                        if last_grad_norm is not None:
                            log_payload["train/grad_norm"] = last_grad_norm
                        wandb.log(log_payload)

                if batch_idx in visualization_index_map:
                    if distributed_viz_sync:
                        dist.barrier()

                    if self._is_main_process():
                        quarter = visualization_index_map[batch_idx]
                        suffix = f"q{quarter}"
                        logger.info(
                            "Starting intermediate visualization checkpoint %s at epoch %d batch %d/%d",
                            suffix,
                            int(epoch),
                            int(batch_idx),
                            int(len(train_loader)),
                        )
                        self._run_visualization_checkpoint(
                            model,
                            val_loader,
                            criterion,
                            epoch,
                            train_viz_payload={
                                'pred_preictal': np.array(all_pred_preictal, dtype=np.float32),
                                'true_preictal': np.array(all_true_preictal, dtype=np.float32),
                            },
                            suffix=suffix,
                        )
                        logger.info(
                            "Completed intermediate visualization checkpoint %s at epoch %d",
                            suffix,
                            int(epoch),
                        )

                    if distributed_viz_sync:
                        try:
                            from datetime import timedelta
                            dist.barrier(timeout=timedelta(minutes=45))
                        except TypeError:
                            dist.barrier()

            if len(all_patient_ids) > 0:
                patient_summary = _compute_patient_difficulty_summary(
                    pred_preictal=np.asarray(all_pred_preictal, dtype=np.float32),
                    true_preictal=np.asarray(all_true_preictal, dtype=np.float32),
                    pred_countdown=np.asarray(all_pred_countdown, dtype=np.float32),
                    true_countdown=np.asarray(all_true_countdown, dtype=np.float32),
                    patient_ids=np.asarray(all_patient_ids, dtype=str),
                    threshold=float(getattr(self.config.loss, 'detection_threshold', 0.5)),
                    countdown_max_minutes=float(getattr(self.config.model, 'output_countdown_max', 10.0)),
                    top_k=5,
                )
            else:
                patient_summary = {
                    'n_samples': len(all_pred_preictal),
                    'n_patients': 0,
                    'threshold': float(getattr(self.config.loss, 'detection_threshold', 0.5)),
                    'patient_rows': [],
                    'hardest_patients': [],
                    'easiest_patients': [],
                }

            self._last_train_patient_summary = patient_summary

            hardest = patient_summary.get('hardest_patients', [])
            easiest = patient_summary.get('easiest_patients', [])
            if hardest:
                metrics['train_hardest_patient_difficulty'] = float(hardest[0].get('difficulty_score', 0.0))
                metrics['train_hardest_patient_bal_acc'] = float(hardest[0].get('alert_balanced_accuracy', 0.0))
            if easiest:
                metrics['train_easiest_patient_difficulty'] = float(easiest[0].get('difficulty_score', 0.0))
                metrics['train_easiest_patient_bal_acc'] = float(easiest[0].get('alert_balanced_accuracy', 0.0))
            metrics['train_patient_count'] = float(patient_summary.get('n_patients', 0))

            train_viz_payload = {
                'pred_preictal': np.array(all_pred_preictal, dtype=np.float32),
                'true_preictal': np.array(all_true_preictal, dtype=np.float32),
            }
        
        return dict(metrics), train_viz_payload
    
    def validate(self, model: nn.Module, val_loader: DataLoader,
                criterion: nn.Module) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
        """Validation loop.
        
        Args:
            model: Neural network model
            val_loader: Validation dataloader
            criterion: Loss function
        
        Returns:
            Dictionary with validation metrics
        """
        model.eval()
        
        metrics = defaultdict(float)
        num_batches = 0
        
        all_pred_countdown = []
        all_true_countdown = []
        all_pred_preictal = []
        all_true_preictal = []
        all_ecg_proxy = []
        all_eeg_proxy = []
        all_ppg_proxy = []
        all_eda_proxy = []
        all_adxl_proxy = []
        all_hr_proxy = []
        all_hrv_proxy = []
        
        signal_label: Optional[str] = None
        loss_type = str(getattr(self.config.loss, 'loss_type', 'countdown')).lower()
        state_mode = loss_type == 'state'
        onset_threshold_min = float(getattr(self.config.loss, 'onset_threshold_min', 2.0))

        with torch.no_grad():
            val_iter = self._progress(
                val_loader,
                desc="Validation",
                total=len(val_loader)
            )

            for batch in val_iter:
                features = batch[0].to(self.device)
                labels = batch[1].to(self.device)
                weights = batch[2].to(self.device)

                # Apply same sanitization/clamping as in training for
                # consistent behavior and to avoid NaNs in metrics.
                features = torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
                labels = torch.nan_to_num(labels, nan=0.0, posinf=0.0, neginf=0.0)
                weights = torch.nan_to_num(weights, nan=1.0, posinf=1.0, neginf=1.0)

                # Save raw (pre-clamp) features for proxy/visualization extraction
                # so that physiological values (e.g., HR in BPM ~60-120) are not
                # destroyed by the [-20, 20] clamp used for model stability.
                features_raw_cpu = features.cpu()

                features = torch.clamp(features, -200.0, 200.0)
                labels = torch.clamp(labels, -1.0, float(self.config.model.output_countdown_max))
                weights = torch.clamp(weights, 0.0, 10.0)
                
                # Forward pass
                pre_ictal_pred, countdown_pred = self._forward_model(model, features)
                
                # Compute loss
                pre_ictal_labels = (labels >= 0).float()
                loss_dict = criterion(pre_ictal_pred, countdown_pred,
                                     pre_ictal_labels, labels, weights)
                
                # Accumulate metrics
                num_batches += 1
                for key, value in loss_dict.items():
                    metrics[f'val_{key}'] += float(value.detach().cpu())
                
                # Store predictions for additional metrics
                all_pred_preictal.extend(pre_ictal_pred.detach().cpu().numpy())
                all_true_preictal.extend(pre_ictal_labels.detach().cpu().numpy())
                all_pred_countdown.extend(countdown_pred.cpu().numpy())
                all_true_countdown.extend(labels.cpu().numpy())

                # Signal proxies for panels. For multimodal models, keep
                # separate ECG and EEG proxy scalars per sample so that the
                # visualization can show paired traces. For single-modal
                # models, fall back to a single best-variance channel.
                full_features = features_raw_cpu.numpy()
                n_feat = full_features.shape[-1]
                source = getattr(self.config.data, "data_source", "bids")

                from src.models import ModelFactory as _MF
                is_multimodal = _MF.is_multimodal(self.config.model.model_type)
                ecg_dim = int(self.config.model.ecg_feature_dim)
                eeg_dim = int(self.config.model.eeg_feature_dim) if is_multimodal else 0

                if is_multimodal and n_feat >= ecg_dim + eeg_dim and ecg_dim > 0 and eeg_dim > 0:
                    ecg_block = full_features[:, :, :ecg_dim]
                    eeg_block = full_features[:, :, ecg_dim:ecg_dim + eeg_dim]

                    ecg_std = ecg_block.reshape(-1, ecg_block.shape[-1]).std(axis=0)
                    eeg_std = eeg_block.reshape(-1, eeg_block.shape[-1]).std(axis=0)

                    ecg_best_ch = int(np.argmax(ecg_std)) if ecg_block.shape[-1] > 0 else 0
                    eeg_best_ch = int(np.argmax(eeg_std)) if eeg_block.shape[-1] > 0 else 0

                    ecg_sequences = ecg_block[:, :, ecg_best_ch]
                    eeg_sequences = eeg_block[:, :, eeg_best_ch]
                    mid_idx = ecg_sequences.shape[1] // 2

                    all_ecg_proxy.extend(ecg_sequences[:, mid_idx])
                    all_eeg_proxy.extend(eeg_sequences[:, mid_idx])

                    if source == "wearable":
                        ppg_sequences = ecg_block[:, :, 0] if ecg_block.shape[-1] > 0 else ecg_sequences
                        # Channel 1 of ecg_block = ADXL accelerometer stream
                        adxl_sequences = ecg_block[:, :, 1] if ecg_block.shape[-1] > 1 else np.zeros_like(ppg_sequences)
                        eda_sequences = eeg_block[:, :, 0] if eeg_block.shape[-1] > 0 else np.zeros_like(ppg_sequences)
                        all_adxl_proxy.extend(adxl_sequences[:, mid_idx])
                    else:
                        ppg_sequences = ecg_sequences
                        eda_sequences = np.zeros_like(ecg_sequences)

                    all_ppg_proxy.extend(ppg_sequences[:, mid_idx])
                    all_eda_proxy.extend(eda_sequences[:, mid_idx])

                    # Use ECG sequences for downstream HR/HRV estimation.
                    signal_sequences = ecg_sequences
                else:
                    # Single-branch models: choose proxy channel.
                    # For wearable runs, always treat feature channel 0 as
                    # the heart-rate series derived from PPGAppStream.
                    if n_feat > 0:
                        if source == "wearable":
                            best_ch = 0
                        else:
                            feat_std = full_features.reshape(-1, n_feat).std(axis=0)
                            best_ch = int(np.argmax(feat_std))
                    else:
                        best_ch = 0

                    signal_sequences = full_features[:, :, best_ch]
                    mid_idx = signal_sequences.shape[1] // 2
                    all_ecg_proxy.extend(signal_sequences[:, mid_idx])

                    if source == "wearable":
                        ppg_sequences = full_features[:, :, 0] if n_feat > 0 else signal_sequences
                        # Channel 1 = ADXL accelerometer stream
                        adxl_sequences = full_features[:, :, 1] if n_feat > 1 else np.zeros_like(ppg_sequences)
                        eda_sequences = full_features[:, :, 2] if n_feat > 2 else np.zeros_like(ppg_sequences)
                        all_adxl_proxy.extend(adxl_sequences[:, mid_idx])
                    else:
                        ppg_sequences = signal_sequences
                        eda_sequences = np.zeros_like(signal_sequences)

                    all_ppg_proxy.extend(ppg_sequences[:, mid_idx])
                    all_eda_proxy.extend(eda_sequences[:, mid_idx])
                    # Track which feature channel is used for the proxy so
                    # downstream visualizations can label the signal source
                    # (e.g., PPG for wearable runs), but keep this separate
                    # from numeric metrics.
                    if signal_label is None:
                        if source == "wearable":
                            signal_label = f"HR signal (feat_ch={best_ch})"
                        else:
                            signal_label = f"ECG proxy (feat_ch={best_ch})"

                # HR/HRV estimation: modality-aware.
                if getattr(self.config.data, "data_source", "bids") == "wearable":
                    # Wearable runs: interpret the selected feature channel
                    # as a heart-rate time series derived from the PPG
                    # stream. Use the raw per-window mean as HR so that
                    # traces reflect actual dataset dynamics instead of a
                    # synthetic mapping. Do not force it into an arbitrary
                    # physiological range; keep native units.
                    hr_batch = signal_sequences.mean(axis=1).astype(np.float32)

                    # HRV proxy: simple within-window variability of the same
                    # sequence. Units are "HRV proxy" rather than true ms but
                    # remain monotonically tied to underlying HR dynamics.
                    hrv_batch = signal_sequences.std(axis=1).astype(np.float32)
                else:
                    # BIDS ECG runs: approximate HR/HRV from waveform peaks.
                    sequence_rate = float(features.shape[1]) / float(self.config.data.feature_window_s)
                    hr_batch, hrv_batch = _extract_hr_hrv_from_sequences(signal_sequences, sequence_rate)

                all_hr_proxy.extend(hr_batch)
                all_hrv_proxy.extend(hrv_batch)
        
        # Compute additional metrics
        all_pred_countdown = np.array(all_pred_countdown)
        all_true_countdown = np.array(all_true_countdown)
        
        # Filter to pre-ictal epochs only
        preictal_mask = all_true_countdown >= 0
        if np.sum(preictal_mask) > 0:
            mae = np.mean(np.abs(all_pred_countdown[preictal_mask] - all_true_countdown[preictal_mask]))
            medae = np.median(np.abs(all_pred_countdown[preictal_mask] - all_true_countdown[preictal_mask]))
            rmse = np.sqrt(np.mean((all_pred_countdown[preictal_mask] - all_true_countdown[preictal_mask]) ** 2))
        else:
            mae = medae = rmse = 0.0
        
        # Average metrics
        for key in metrics:
            metrics[key] /= num_batches
        
        # Add additional metrics
        metrics['val_mae'] = mae
        metrics['val_medae'] = medae
        metrics['val_rmse'] = rmse

        alert_metrics = _compute_binary_metrics(
            (all_true_countdown >= 0).astype(np.int32),
            np.array(all_pred_preictal, dtype=np.float32),
            float(self.config.loss.detection_threshold),
        )
        metrics['val_alert_balanced_accuracy'] = alert_metrics['balanced_accuracy']
        metrics['val_alert_sensitivity'] = alert_metrics['sensitivity']
        metrics['val_alert_specificity'] = alert_metrics['specificity']
        metrics['val_alert_precision'] = alert_metrics['precision']
        metrics['val_alert_f1'] = alert_metrics['f1']
        
        ecg_proxy_arr = np.array(all_ecg_proxy, dtype=np.float32)
        eeg_proxy_arr = (
            np.array(all_eeg_proxy, dtype=np.float32)
            if len(all_eeg_proxy) == len(all_ecg_proxy) and len(all_eeg_proxy) > 0
            else None
        )
        ppg_proxy_arr = (
            np.array(all_ppg_proxy, dtype=np.float32)
            if len(all_ppg_proxy) == len(all_ecg_proxy) and len(all_ppg_proxy) > 0
            else None
        )
        eda_proxy_arr = (
            np.array(all_eda_proxy, dtype=np.float32)
            if len(all_eda_proxy) == len(all_ecg_proxy) and len(all_eda_proxy) > 0
            else None
        )
        adxl_proxy_arr = (
            np.array(all_adxl_proxy, dtype=np.float32)
            if len(all_adxl_proxy) == len(all_ecg_proxy) and len(all_adxl_proxy) > 0
            else None
        )

        viz_payload = {
            'pred_preictal': np.array(all_pred_preictal, dtype=np.float32),
            'pred_preictal_smooth': _causal_smooth(np.array(all_pred_preictal, dtype=np.float32)),
            'true_preictal': np.array(all_true_preictal, dtype=np.float32),
            'pred_countdown': all_pred_countdown.astype(np.float32),
            'true_countdown': all_true_countdown.astype(np.float32),
            # For backward compatibility, keep the original key while also
            # exposing explicit ECG/EEG proxies.
            'signal_proxy': ecg_proxy_arr,
            'ecg_proxy': ecg_proxy_arr,
            'hr_proxy': np.array(all_hr_proxy, dtype=np.float32),
            'hrv_proxy': np.array(all_hrv_proxy, dtype=np.float32),
        }
        if state_mode:
            state_probs = _derive_state_probabilities(
                viz_payload['pred_preictal'],
                viz_payload['pred_countdown'],
            )
            true_state = _derive_state_targets(all_true_countdown, onset_threshold_min)
            state_balanced_accuracy, state_cm, state_recalls = _compute_multiclass_balanced_accuracy(
                true_state,
                state_probs['pred_state'],
                3,
            )
            onset_metrics = _compute_binary_metrics(
                (true_state == 2).astype(np.int32),
                state_probs['onset_joint_prob'],
                0.5,
            )
            metrics['val_state_balanced_accuracy'] = state_balanced_accuracy
            metrics['val_state_accuracy'] = float(np.mean(state_probs['pred_state'] == true_state))
            metrics['val_interictal_recall'] = float(state_recalls[0])
            metrics['val_preictal_recall'] = float(state_recalls[1])
            metrics['val_onset_recall'] = float(state_recalls[2])
            metrics['val_onset_balanced_accuracy'] = onset_metrics['balanced_accuracy']
            metrics['val_onset_sensitivity'] = onset_metrics['sensitivity']
            metrics['val_onset_specificity'] = onset_metrics['specificity']
            metrics['val_onset_precision'] = onset_metrics['precision']
            metrics['val_onset_f1'] = onset_metrics['f1']
            viz_payload['state_mode'] = True
            viz_payload['true_state'] = true_state.astype(np.int32)
            viz_payload['pred_state'] = state_probs['pred_state'].astype(np.int32)
            viz_payload['pred_onset_prob'] = state_probs['onset_joint_prob'].astype(np.float32)
            viz_payload['pred_preictal_only_prob'] = state_probs['preictal_only_prob'].astype(np.float32)
            viz_payload['pred_interictal_prob'] = state_probs['interictal_prob'].astype(np.float32)
            viz_payload['state_confusion_matrix'] = state_cm.astype(np.int32)
        else:
            viz_payload['state_mode'] = False
        if eeg_proxy_arr is not None:
            viz_payload['eeg_proxy'] = eeg_proxy_arr
        if ppg_proxy_arr is not None:
            viz_payload['ppg_proxy'] = ppg_proxy_arr
        if eda_proxy_arr is not None:
            viz_payload['eda_proxy'] = eda_proxy_arr
        if adxl_proxy_arr is not None:
            viz_payload['adxl_proxy'] = adxl_proxy_arr

        # Propagate signal source label if we tracked it.
        if signal_label is not None:
            viz_payload['signal_label'] = str(signal_label)

        # Attach validation metadata for visualization when available and
        # aligned. This lets epoch panels show contiguous segments from a
        # single recording rather than shuffled samples across recordings.
        val_dataset = getattr(val_loader, 'dataset', None)
        self._cached_val_dataset_ref = val_dataset
        sample_end_times_s = getattr(val_dataset, 'sample_end_times_s', None) if val_dataset is not None else None
        recording_ids = getattr(val_dataset, 'recording_ids', None) if val_dataset is not None else None
        n_points = int(len(all_true_countdown))
        if (
            sample_end_times_s is not None
            and recording_ids is not None
            and len(sample_end_times_s) >= n_points
            and len(recording_ids) >= n_points
            and n_points > 0
        ):
            viz_payload['sample_end_times_s'] = np.asarray(sample_end_times_s[:n_points], dtype=np.float32)
            viz_payload['recording_ids'] = np.asarray(recording_ids[:n_points])

        recordings_meta = getattr(val_dataset, 'recordings_meta', None) if val_dataset is not None else None
        if recordings_meta is not None:
            viz_payload['recordings_meta'] = recordings_meta

        return dict(metrics), viz_payload

    def _save_epoch_visualizations(
        self,
        epoch: int,
        viz_payload: Dict[str, np.ndarray],
        train_viz_payload: Optional[Dict[str, np.ndarray]] = None,
        suffix: Optional[str] = None,
    ) -> None:
        """Save per-epoch validation visualizations on main process."""
        if not self._is_main_process() or self.epoch_visualizer is None:
            return

        pred_preictal = viz_payload['pred_preictal']
        true_preictal = viz_payload['true_preictal']
        pred_countdown = viz_payload['pred_countdown']
        true_countdown = viz_payload['true_countdown']
        state_mode = bool(viz_payload.get('state_mode', False))
        pred_onset_prob_full = viz_payload.get('pred_onset_prob')
        pred_preictal_only_prob_full = viz_payload.get('pred_preictal_only_prob')
        true_state_full = viz_payload.get('true_state')
        pred_state_full = viz_payload.get('pred_state')
        context_energy_full = viz_payload.get('context_energy')

        # Prefer explicit ECG/EEG proxies when available, but fall back to
        # the original single-channel signal_proxy for backward compatibility.
        ecg_proxy = viz_payload.get('ecg_proxy', viz_payload.get('signal_proxy'))
        ppg_proxy = viz_payload.get('ppg_proxy', ecg_proxy)
        eda_proxy = viz_payload.get('eda_proxy', None)
        eeg_proxy = viz_payload.get('eeg_proxy', None)
        adxl_proxy = viz_payload.get('adxl_proxy', None)
        hr_proxy = viz_payload.get('hr_proxy', np.zeros_like(ecg_proxy, dtype=np.float32))
        hrv_proxy = viz_payload.get('hrv_proxy', np.zeros_like(ecg_proxy, dtype=np.float32))
        eeg_signal_label = str(viz_payload.get('eeg_signal_label', 'EEG proxy'))

        # Prefer a contiguous segment from a single recording that includes a
        # seizure (when metadata is available and we are not in distributed
        # mode). This matches clinical expectations: 1–5–10 minutes of
        # continuous context around a seizure, not a shuffled concatenation
        # of many independent windows.
        time_axis_minutes: Optional[np.ndarray]
        if 'sample_end_times_s' in viz_payload and 'recording_ids' in viz_payload:
            panel_window_minutes = float(getattr(self.training_config, 'epoch_panel_window_minutes', 60.0))
            preferred_recording = getattr(self.training_config, 'epoch_panel_preferred_recording', None)
            selected, time_axis_minutes = _select_panel_indices_with_metadata(
                true_countdown,
                feature_step_s=float(self.config.data.feature_step_s),
                window_minutes=panel_window_minutes,
                seed=int(self.config.data.random_seed) + int(epoch),
                sample_end_times_s=viz_payload['sample_end_times_s'],
                recording_ids=viz_payload['recording_ids'],
                preferred_recording_substring=preferred_recording,
            )
        else:
            # Fallback: keep native sample order for this rank.
            selected = np.arange(len(true_countdown), dtype=np.int64)
            time_axis_minutes = None
        # For metadata-backed contiguous panels (time_axis_minutes is not None),
        # keep the full continuous segment so the x-axis and y-series remain
        # aligned. For fallback (no metadata), optionally subsample to a
        # maximum number of points while preserving a mix of preictal and
        # interictal samples.
        if time_axis_minutes is None:
            max_points = 5000
            if len(selected) > max_points:
                rng = np.random.default_rng(int(self.config.data.random_seed) + int(epoch))
                pre_idx = np.where(true_countdown >= 0)[0]
                inter_idx = np.where(true_countdown < 0)[0]

                target_pre = min(len(pre_idx), max_points // 2)
                target_inter = min(len(inter_idx), max_points - target_pre)

                chosen = []
                if target_pre > 0:
                    chosen.append(rng.choice(pre_idx, size=target_pre, replace=False))
                if target_inter > 0:
                    chosen.append(rng.choice(inter_idx, size=target_inter, replace=False))

                if len(chosen) == 0:
                    selected = selected[:max_points]
                else:
                    selected = np.sort(np.concatenate(chosen)).astype(np.int64)

        if len(selected) == 0:
            return

        panel_sampling_rate_hz = 1.0 / max(float(self.config.data.feature_step_s), 1e-6)

        pred_preictal = pred_preictal[selected]
        true_preictal = true_preictal[selected]
        pred_countdown = pred_countdown[selected]
        true_countdown = true_countdown[selected]
        pred_onset_prob = (
            pred_onset_prob_full[selected]
            if pred_onset_prob_full is not None and len(pred_onset_prob_full) == len(viz_payload['pred_preictal'])
            else None
        )
        pred_preictal_only_prob = (
            pred_preictal_only_prob_full[selected]
            if pred_preictal_only_prob_full is not None and len(pred_preictal_only_prob_full) == len(viz_payload['pred_preictal'])
            else None
        )
        true_state = (
            true_state_full[selected]
            if true_state_full is not None and len(true_state_full) == len(viz_payload['pred_preictal'])
            else None
        )
        pred_state = (
            pred_state_full[selected]
            if pred_state_full is not None and len(pred_state_full) == len(viz_payload['pred_preictal'])
            else None
        )
        context_energy = (
            np.asarray(context_energy_full, dtype=np.float32)[selected]
            if context_energy_full is not None and len(context_energy_full) == len(viz_payload['pred_preictal'])
            else None
        )
        pred_preictal_smooth_sel = viz_payload.get('pred_preictal_smooth')
        if pred_preictal_smooth_sel is not None:
            pred_preictal_smooth_sel = pred_preictal_smooth_sel[selected]

        # Keep raw proxies for any downstream numeric use.
        ecg_proxy_raw = ecg_proxy[selected]
        ppg_proxy_raw = ppg_proxy[selected] if ppg_proxy is not None and len(ppg_proxy) == len(ecg_proxy) else ecg_proxy_raw
        eda_proxy_raw = eda_proxy[selected] if eda_proxy is not None and len(eda_proxy) == len(ecg_proxy) else None
        eeg_proxy_raw = eeg_proxy[selected] if eeg_proxy is not None and len(eeg_proxy) == len(ecg_proxy) else None
        adxl_proxy_raw = adxl_proxy[selected] if adxl_proxy is not None and len(adxl_proxy) == len(ecg_proxy) else None
        hr_proxy_raw = hr_proxy[selected]
        hrv_proxy_raw = hrv_proxy[selected]
        emg_proxy_raw: Optional[np.ndarray] = None
        mov_proxy_raw: Optional[np.ndarray] = None

        if getattr(self.config.data, 'data_source', 'bids') != 'wearable':
            hr_proxy_raw = np.clip(60.0 + 30.0 * np.asarray(hr_proxy_raw, dtype=np.float32), 30.0, 180.0).astype(np.float32)
            hrv_proxy_raw = np.clip(20.0 + 40.0 * np.asarray(hrv_proxy_raw, dtype=np.float32), 0.0, 250.0).astype(np.float32)

        # Rebuild a dense timeline (default 1 Hz) for metadata-backed BIDS
        # panels so overlays do not look sparsely sampled when validation
        # windows were uniformly subsampled per recording.
        rec_ids_meta = viz_payload.get('recording_ids')
        times_meta = viz_payload.get('sample_end_times_s')
        if (
            getattr(self.config.data, 'data_source', 'bids') != 'wearable'
            and rec_ids_meta is not None
            and times_meta is not None
            and len(rec_ids_meta) == len(viz_payload['true_countdown'])
            and len(times_meta) == len(viz_payload['true_countdown'])
            and len(selected) >= 2
        ):
            rec_ids_all = np.asarray(rec_ids_meta)
            times_all = np.asarray(times_meta, dtype=np.float64)
            rec_ids_sel = rec_ids_all[selected]
            times_sel = times_all[selected]

            uniq_rec, counts = np.unique(rec_ids_sel, return_counts=True)
            panel_loader = None
            if getattr(self.config.data, 'data_source', 'bids') != 'wearable':
                try:
                    from src.data_loader import BIDSDataLoader as _BIDSLoader
                    panel_loader = _BIDSLoader(self.config.data)
                except Exception:
                    panel_loader = None

            chosen_rec = _choose_panel_recording_uid(rec_ids_sel, rec_ids_all, panel_loader)
            if chosen_rec is not None:
                keep = rec_ids_sel == chosen_rec
                if not np.any(keep):
                    rec_all_times = np.sort(times_all[rec_ids_all == chosen_rec])
                    if rec_all_times.size >= 2:
                        center_s = float(rec_all_times[rec_all_times.size // 2])
                        half_window_s = max(60.0, panel_window_minutes * 30.0)
                        start_s = max(float(rec_all_times[0]), center_s - half_window_s)
                        end_s = min(float(rec_all_times[-1]), center_s + half_window_s)
                        if end_s <= start_s:
                            start_s = float(rec_all_times[0])
                            end_s = float(rec_all_times[-1])
                        global_mask = (rec_ids_all == chosen_rec) & (times_all >= start_s) & (times_all <= end_s)
                        sel_local = np.where(global_mask)[0]
                        if sel_local.size >= 2:
                            t_local = np.asarray(times_all[sel_local], dtype=np.float64)
                            order = np.argsort(t_local)
                            sel_local = sel_local[order]
                            t_local = t_local[order]
                            dense_step_s = float(max(0.5, getattr(self.training_config, 'epoch_panel_dense_step_s', 1.0)))
                            dense_t = np.arange(float(t_local[0]), float(t_local[-1]) + dense_step_s * 0.5, dense_step_s, dtype=np.float64)
                            keep = np.ones(sel_local.shape[0], dtype=bool)
                if np.any(keep):
                    if 'sel_local' not in locals() or sel_local.size == 0:
                        sel_local = selected[keep]
                        t_local = np.asarray(times_all[sel_local], dtype=np.float64)
                        order = np.argsort(t_local)
                        sel_local = sel_local[order]
                        t_local = t_local[order]

                    dense_step_s = float(max(0.5, getattr(self.training_config, 'epoch_panel_dense_step_s', 1.0)))
                    if 'dense_t' not in locals() or dense_t.size == 0:
                        dense_t = np.arange(float(t_local[0]), float(t_local[-1]) + dense_step_s * 0.5, dense_step_s, dtype=np.float64)

                    if dense_t.size >= 8:
                        # Interpolate predictions/targets to the dense timeline.
                        pred_preictal = _interp_to_grid(t_local, viz_payload['pred_preictal'][sel_local], dense_t)
                        true_countdown = _interp_to_grid(t_local, viz_payload['true_countdown'][sel_local], dense_t)
                        pred_countdown = _interp_to_grid(t_local, viz_payload['pred_countdown'][sel_local], dense_t)
                        true_preictal = (true_countdown >= 0.0).astype(np.float32)

                        if pred_preictal_smooth_sel is not None:
                            pred_preictal_smooth_sel = _interp_to_grid(
                                t_local,
                                np.asarray(pred_preictal_smooth_sel, dtype=np.float32)[keep][order],
                                dense_t,
                            )

                        if pred_onset_prob is not None:
                            pred_onset_prob = _interp_to_grid(t_local, np.asarray(pred_onset_prob, dtype=np.float32)[keep][order], dense_t)
                        if pred_preictal_only_prob is not None:
                            pred_preictal_only_prob = _interp_to_grid(
                                t_local,
                                np.asarray(pred_preictal_only_prob, dtype=np.float32)[keep][order],
                                dense_t,
                            )
                        if true_state is not None:
                            true_state = np.rint(_interp_to_grid(t_local, np.asarray(true_state, dtype=np.float32)[keep][order], dense_t)).astype(np.int32)
                        if pred_state is not None:
                            pred_state = np.rint(_interp_to_grid(t_local, np.asarray(pred_state, dtype=np.float32)[keep][order], dense_t)).astype(np.int32)
                        if context_energy is not None:
                            context_energy = _interp_to_grid(t_local, np.asarray(context_energy, dtype=np.float32)[keep][order], dense_t)

                        # Start from interpolated proxies, then override with
                        # dense raw streams when available.
                        ecg_proxy_local = np.asarray(viz_payload.get('ecg_proxy', viz_payload.get('signal_proxy')), dtype=np.float32)[sel_local]
                        ppg_source = viz_payload.get('ppg_proxy', viz_payload.get('ecg_proxy', viz_payload.get('signal_proxy')))
                        ppg_proxy_local = np.asarray(ppg_source, dtype=np.float32)[sel_local]
                        ecg_proxy_raw = _interp_to_grid(t_local, ecg_proxy_local, dense_t)
                        ppg_proxy_raw = _interp_to_grid(t_local, ppg_proxy_local, dense_t)
                        if eda_proxy_raw is not None:
                            eda_proxy_raw = _interp_to_grid(t_local, np.asarray(viz_payload['eda_proxy'], dtype=np.float32)[sel_local], dense_t)
                        if eeg_proxy_raw is not None:
                            eeg_proxy_raw = _interp_to_grid(t_local, np.asarray(viz_payload['eeg_proxy'], dtype=np.float32)[sel_local], dense_t)
                        if adxl_proxy_raw is not None:
                            adxl_proxy_raw = _interp_to_grid(t_local, np.asarray(viz_payload['adxl_proxy'], dtype=np.float32)[sel_local], dense_t)
                        hr_proxy_raw = _interp_to_grid(t_local, np.asarray(hr_proxy, dtype=np.float32)[sel_local], dense_t)
                        hrv_proxy_raw = _interp_to_grid(t_local, np.asarray(hrv_proxy, dtype=np.float32)[sel_local], dense_t)

                        # Use cached ECG binary when available in LazyRealDataset metadata.
                        try:
                            val_dataset_obj = getattr(self, '_cached_val_dataset_ref', None)
                            recordings_meta = getattr(val_dataset_obj, 'recordings_meta', None)
                            if recordings_meta is None:
                                recordings_meta = viz_payload.get('recordings_meta')
                            if recordings_meta is not None:
                                for meta in recordings_meta:
                                    if str(meta.get('recording_uid', '')) == chosen_rec:
                                        fs = float(meta.get('fs', 0.0))
                                        n_samples = int(meta.get('n_samples', 0))
                                        signal_path = str(meta.get('signal_path', ''))
                                        if fs > 0 and n_samples > 0 and signal_path:
                                            dense_idx = np.clip((dense_t * fs).astype(np.int64), 0, n_samples - 1)
                                            sig = np.memmap(signal_path, dtype=np.float32, mode='r', shape=(n_samples,))
                                            ecg_proxy_raw = np.asarray(sig[dense_idx], dtype=np.float32)

                                            # Compute HR/HRV from a raw-rate ECG segment (not from
                                            # the 1 Hz display-downsampled signal), then sample those
                                            # trajectories to the dense display grid.
                                            seg_margin = max(1, int(10.0 * fs))
                                            seg_start = max(0, int(dense_idx.min()) - seg_margin)
                                            seg_end = min(n_samples, int(dense_idx.max()) + seg_margin + 1)
                                            ecg_segment = np.asarray(sig[seg_start:seg_end], dtype=np.float32)
                                            if ecg_segment.size >= 8:
                                                hr_seg, hrv_seg = _extract_hr_hrv_from_dense_ecg(ecg_segment, fs)
                                                local_idx = np.clip(dense_idx - seg_start, 0, hr_seg.shape[0] - 1)
                                                hr_proxy_raw = np.asarray(hr_seg[local_idx], dtype=np.float32)
                                                hrv_proxy_raw = np.asarray(hrv_seg[local_idx], dtype=np.float32)
                                        break
                        except Exception as dense_ecg_exc:
                            logger.debug('Dense ECG panel rebuild skipped for %s: %s', chosen_rec, str(dense_ecg_exc))

                        # Try loading EEG/EMG/MOV from BIDS streams for the
                        # same recording to keep modality overlays visible.
                        try:
                            parsed = _parse_recording_uid(chosen_rec)
                            if parsed is not None and panel_loader is not None:
                                sub_id, ses_id, run_id = parsed

                                def _load_dense_modality(datatype: str) -> Optional[np.ndarray]:
                                    nonlocal eeg_signal_label
                                    try:
                                        data, fs_local = panel_loader.load_subject_edf(sub_id, ses_id, datatype, run_id)
                                        if data is None:
                                            return None
                                        arr = np.asarray(data, dtype=np.float32)
                                        if arr.ndim > 1:
                                            if datatype in ('eeg', 'emg'):
                                                ch_std = np.nanstd(arr, axis=1)
                                                if np.any(np.isfinite(ch_std)):
                                                    best_ch = int(np.nanargmax(ch_std))
                                                else:
                                                    best_ch = 0
                                                arr = arr[best_ch]
                                                if datatype == 'eeg':
                                                    eeg_signal_label = f'EEG (ch{best_ch}, dense raw)'
                                            elif datatype == 'mov':
                                                arr = np.sqrt(np.mean(arr ** 2, axis=0))
                                            else:
                                                ch_std = np.nanstd(arr, axis=1)
                                                if np.any(np.isfinite(ch_std)):
                                                    best_ch = int(np.nanargmax(ch_std))
                                                else:
                                                    best_ch = 0
                                                arr = arr[best_ch]
                                        elif datatype == 'eeg':
                                            eeg_signal_label = 'EEG (dense raw)'
                                        arr = np.asarray(arr, dtype=np.float32).reshape(-1)
                                        if arr.size == 0 or fs_local <= 0:
                                            return None
                                        idx = np.clip((dense_t * float(fs_local)).astype(np.int64), 0, arr.shape[0] - 1)
                                        return np.asarray(arr[idx], dtype=np.float32)
                                    except Exception:
                                        return None

                                ecg_raw = _load_dense_modality('ecg')
                                eeg_raw = _load_dense_modality('eeg')
                                emg_raw = _load_dense_modality('emg')
                                mov_raw = _load_dense_modality('mov')

                                if ecg_raw is not None:
                                    ecg_proxy_raw = ecg_raw
                                if eeg_raw is not None:
                                    eeg_proxy_raw = eeg_raw
                                if emg_raw is not None:
                                    emg_proxy_raw = emg_raw
                                if mov_raw is not None:
                                    mov_proxy_raw = mov_raw

                                if emg_proxy_raw is not None or mov_proxy_raw is not None:
                                    parts = []
                                    if emg_proxy_raw is not None:
                                        parts.append(_standardize_for_plot(np.abs(emg_proxy_raw - float(np.mean(emg_proxy_raw)))))
                                    if mov_proxy_raw is not None:
                                        parts.append(_standardize_for_plot(mov_proxy_raw))
                                    if len(parts) == 1:
                                        adxl_proxy_raw = parts[0]
                                    elif len(parts) >= 2:
                                        adxl_proxy_raw = (0.65 * parts[0] + 0.35 * parts[1]).astype(np.float32)
                        except Exception as dense_modal_exc:
                            logger.debug('Dense EEG/EMG/MOV panel rebuild skipped for %s: %s', chosen_rec, str(dense_modal_exc))

                        time_axis_minutes = ((dense_t - float(dense_t[0])) / 60.0).astype(np.float64)
                        panel_sampling_rate_hz = 1.0 / max(dense_step_s, 1e-6)

        # For wearable runs, show a z-scored heart-rate proxy as the top
        # "Signal" trace (so that the HR subplot can display the raw bpm
        # values separately). For BIDS ECG runs, keep the primary trace raw.
        if getattr(self.config.data, "data_source", "bids") == "wearable":
            ecg_proxy_plot = _standardize_for_plot(ppg_proxy_raw)
        else:
            ecg_proxy_plot = ecg_proxy_raw.astype(np.float32)
        ppg_proxy_plot = _standardize_for_plot(ppg_proxy_raw)
        eda_proxy_plot = _standardize_for_plot(eda_proxy_raw) if eda_proxy_raw is not None else None
        eeg_proxy_plot = _standardize_for_plot(eeg_proxy_raw) if eeg_proxy_raw is not None else None
        adxl_proxy_plot = adxl_proxy_raw if adxl_proxy_raw is not None else None
        # For wearable runs, keep HR/HRV proxies in their synthetic
        # physiological units so that plots show meaningful ranges instead of
        # z-scored lines around zero; the top "Signal" uses the z-scored
        # version. For BIDS ECG runs, standardize for better visual contrast.
        if getattr(self.config.data, "data_source", "bids") == "wearable":
            hr_proxy_plot = hr_proxy_raw.astype(np.float32)
            hrv_proxy_plot = hrv_proxy_raw.astype(np.float32)
        else:
            hr_proxy_plot = hr_proxy_raw.astype(np.float32)
            hrv_proxy_plot = hrv_proxy_raw.astype(np.float32)

        # Runtime/local inference accuracy using local history only.
        gt_binary = (true_countdown >= 0).astype(np.float32)
        panel_alarm_prob = (
            np.asarray(pred_preictal_smooth_sel, dtype=np.float32)
            if pred_preictal_smooth_sel is not None
            else np.asarray(pred_preictal, dtype=np.float32)
        )
        pred_binary = (panel_alarm_prob >= float(self.config.loss.detection_threshold)).astype(np.float32)
        acc_window = max(3, int(getattr(self.training_config, 'epoch_panel_accuracy_window', 31)))
        local_accuracy = _compute_local_balanced_accuracy(gt_binary, pred_binary, acc_window)

        gt_event_mask = np.zeros_like(gt_binary, dtype=np.int32)
        if len(gt_binary) > 0:
            gt_event_mask[0] = int(gt_binary[0] > 0)
        if len(gt_binary) > 1:
            gt_event_mask[1:] = ((gt_binary[1:] > 0) & (gt_binary[:-1] <= 0)).astype(np.int32)
        # Build token activation waterfall from longitudinal causal channels
        # so token rows reflect temporal digestion, not static bucket IDs.
        token_roll = None
        token_labels: Optional[List[str]] = None
        token_window = max(1, int(getattr(self.training_config, 'token_waterfall_window', 25)))
        try:
            alert = np.clip(pred_preictal.astype(np.float32), 0.0, 1.0)
            alert_smooth = np.clip(
                pred_preictal_smooth_sel.astype(np.float32) if pred_preictal_smooth_sel is not None else alert,
                0.0,
                1.0,
            )
            d_alert = np.zeros_like(alert, dtype=np.float32)
            if alert.size > 1:
                d_alert[1:] = alert[1:] - alert[:-1]

            # Causal streak of above-threshold alertness.
            thr = float(self.config.loss.detection_threshold)
            streak = np.zeros_like(alert, dtype=np.float32)
            run = 0
            for i, p in enumerate(alert):
                run = run + 1 if p >= thr else 0
                streak[i] = float(run)
            streak_norm = np.clip(streak / max(1.0, float(token_window)), 0.0, 1.0)

            # Countdown-head imminent risk in [0, 1] for non-state mode.
            pred_cd = np.asarray(pred_countdown, dtype=np.float32)
            tau_min = max(0.5, float(np.nanpercentile(np.clip(true_countdown[true_countdown >= 0], 0.0, None), 50)) if np.any(true_countdown >= 0) else 2.0)
            imminent = np.clip(np.exp(-np.clip(pred_cd, 0.0, None) / tau_min), 0.0, 1.0)

            channels: List[np.ndarray] = [
                alert,
                alert_smooth,
                np.clip((alert - 0.2) / 0.8, 0.0, 1.0),
                np.clip((alert - 0.5) / 0.5, 0.0, 1.0),
                np.clip((alert - 0.8) / 0.2, 0.0, 1.0),
                np.clip(np.maximum(d_alert, 0.0) / 0.10, 0.0, 1.0),
                np.clip(np.maximum(-d_alert, 0.0) / 0.10, 0.0, 1.0),
                streak_norm,
                np.asarray(local_accuracy, dtype=np.float32) if 'local_accuracy' in locals() else np.zeros_like(alert),
                imminent,
            ]
            labels: List[str] = [
                'Alert Raw',
                'Alert Smoothed',
                'Alert Low+',
                'Alert Med+',
                'Alert High+',
                'Alert Rising',
                'Alert Falling',
                'Alert Persistence',
                'Local Accuracy',
                'Imminent Risk',
            ]

            if state_mode and pred_onset_prob is not None and pred_preictal_only_prob is not None:
                onset = np.clip(np.asarray(pred_onset_prob, dtype=np.float32), 0.0, 1.0)
                pre_only = np.clip(np.asarray(pred_preictal_only_prob, dtype=np.float32), 0.0, 1.0)
                d_onset = np.zeros_like(onset, dtype=np.float32)
                if onset.size > 1:
                    d_onset[1:] = onset[1:] - onset[:-1]
                channels.extend([
                    pre_only,
                    onset,
                    np.clip(np.maximum(d_onset, 0.0) / 0.10, 0.0, 1.0),
                ])
                labels.extend([
                    'Preictal Head',
                    'Onset Head',
                    'Onset Rising',
                ])

            channel_mat = np.stack(channels, axis=1).astype(np.float32)

            # Causal rolling average per channel to make token flow visibly longitudinal.
            token_roll = np.zeros_like(channel_mat)
            for col in range(channel_mat.shape[1]):
                c = channel_mat[:, col]
                for i in range(c.shape[0]):
                    left = max(0, i - token_window + 1)
                    token_roll[i, col] = float(np.mean(c[left:i + 1]))

            token_labels = labels
        except Exception as token_exc:
            logger.warning('Token activation build skipped for epoch %d: %s', int(epoch), str(token_exc))
            token_roll = None
            token_labels = None

        panel_threshold = _select_detection_threshold(
            viz_payload['true_countdown'],
            viz_payload['pred_preictal'],
            self.config.loss.detection_threshold,
        )
        logger.info("Epoch %d adaptive detection threshold: %.2f", epoch, panel_threshold)

        # Use the configured detection threshold for panels to avoid
        # degenerate "all-preictal" confusion matrices caused by extremely
        # low adaptive thresholds on highly imbalanced small datasets.
        panel_threshold = float(self.config.loss.detection_threshold)

        data_source = getattr(self.config.data, "data_source", "bids")
        signal_label = viz_payload.get(
            'signal_label',
            'ECG raw amplitude + BPM' if data_source != "wearable" else "PPG proxy",
        )

        # Human-readable dataset tag for figure titles/footers.
        if data_source == "wearable":
            dataset_tag = "Wearable (OHSU VSM)"
        else:
            dataset_tag = "SeizeIT2 (BIDS)"

        # Transparent metadata: if validation metadata is available, report
        # which recording(s) and original time ranges contributed to this
        # panel so flat or unexpected behavior can be traced back to raw
        # wearable/BIDS data.
        panel_footer = (
            f"Dataset: {dataset_tag} | Model: {self.config.model.model_type} | Epoch: {epoch} | "
            f"Threshold: {panel_threshold:.2f} | Signal: {signal_label}"
        )
        rec_ids_meta = viz_payload.get('recording_ids')
        times_meta = viz_payload.get('sample_end_times_s')
        if rec_ids_meta is not None and times_meta is not None and len(rec_ids_meta) == len(viz_payload['true_countdown']):
            rec_ids_meta = np.asarray(rec_ids_meta)
            times_meta = np.asarray(times_meta, dtype=np.float64)
            rec_ids_panel = rec_ids_meta[selected]
            times_panel = times_meta[selected]
            unique_recs = np.unique(rec_ids_panel)
            pieces = []
            for rec_id in unique_recs:
                mask = rec_ids_panel == rec_id
                if not np.any(mask):
                    continue
                n_local = int(np.sum(mask))
                t_min = float(times_panel[mask].min() / 60.0)
                t_max = float(times_panel[mask].max() / 60.0)
                pieces.append(f"{rec_id}: n={n_local}, t={t_min:.1f}-{t_max:.1f} min")
            if pieces:
                meta_str = " | Panel samples: " + "; ".join(pieces)
                panel_footer = panel_footer + meta_str
                logger.info("Epoch %d panel sources: %s", epoch, "; ".join(pieces))

        fig_panel = self.epoch_visualizer.plot_gt_vs_inference_panel(
            ecg_signal=ecg_proxy_plot,
            ppg_signal=ppg_proxy_plot if data_source == 'wearable' else None,
            eda_signal=eda_proxy_plot if data_source == 'wearable' else None,
            eeg_signal=eeg_proxy_plot,
            emg_signal=emg_proxy_raw,
            mov_signal=mov_proxy_raw,
            adxl_series=adxl_proxy_plot,
            hr_series=hr_proxy_plot,
            hrv_series=hrv_proxy_plot,
            true_countdown=true_countdown,
            pred_countdown=pred_countdown,
            pred_preictal_prob=pred_preictal,
            pred_preictal_smooth=pred_preictal_smooth_sel,
            train_cm_true_preictal=None if train_viz_payload is None else train_viz_payload.get('true_preictal'),
            train_cm_pred_preictal_prob=None if train_viz_payload is None else train_viz_payload.get('pred_preictal'),
            cm_true_countdown=viz_payload['true_countdown'],
            cm_pred_preictal_prob=viz_payload['pred_preictal'],
            sampling_rate=panel_sampling_rate_hz,
            detection_threshold=panel_threshold,
            time_axis_minutes=time_axis_minutes,
            title_prefix=f"{self.config.model.model_type} — {dataset_tag} — epoch {epoch}",
            footer_text=panel_footer,
            signal_label=signal_label,
            eeg_signal_label=eeg_signal_label,
            pred_onset_prob=pred_onset_prob,
            pred_preictal_only_prob=pred_preictal_only_prob,
            gt_onset=None if true_state is None else (true_state == 2).astype(np.int32),
            gt_preictal_only=None if true_state is None else (true_state == 1).astype(np.int32),
            visualization_mode='state' if state_mode else 'countdown',
            token_roll=token_roll,
            token_window=token_window,
            token_labels=token_labels,
            local_accuracy=local_accuracy,
            gt_event_mask=gt_event_mask,
            data_source=data_source,
            context_energy=context_energy,
        )
        panel_filename = f"epoch_{epoch:03d}{('_' + suffix) if suffix else ''}_gt_vs_inference_panel"
        self.epoch_visualizer.save_figure(fig_panel, panel_filename, step=epoch)

        # Automated anomaly checks for panel inputs; persist machine-readable
        # reports so run health can be monitored without manual figure review.
        try:
            anomaly_report = _compute_panel_anomaly_report(
                epoch=int(epoch),
                data_source=str(data_source),
                ecg_signal=np.asarray(ecg_proxy_plot, dtype=np.float32),
                hr_series=np.asarray(hr_proxy_plot, dtype=np.float32),
                hrv_series=np.asarray(hrv_proxy_plot, dtype=np.float32),
                pred_preictal_prob=np.asarray(pred_preictal, dtype=np.float32),
                true_countdown=np.asarray(true_countdown, dtype=np.float32),
                eeg_signal=None if eeg_proxy_plot is None else np.asarray(eeg_proxy_plot, dtype=np.float32),
                emg_signal=None if emg_proxy_raw is None else np.asarray(emg_proxy_raw, dtype=np.float32),
                mov_signal=None if mov_proxy_raw is None else np.asarray(mov_proxy_raw, dtype=np.float32),
                token_roll=None if token_roll is None else np.asarray(token_roll, dtype=np.float32),
                context_energy=None if context_energy is None else np.asarray(context_energy, dtype=np.float32),
            )

            monitor_dir = self.save_dir / "monitoring"
            monitor_dir.mkdir(parents=True, exist_ok=True)

            epoch_report_path = monitor_dir / f"epoch_{int(epoch):03d}_anomaly_report.json"
            with open(epoch_report_path, "w", encoding="utf-8") as fh:
                json.dump(anomaly_report, fh, indent=2, sort_keys=True)

            jsonl_path = monitor_dir / "anomaly_reports.jsonl"
            with open(jsonl_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(anomaly_report, sort_keys=True) + "\n")

            anomalies = anomaly_report.get("summary", {}).get("anomalies", [])
            if anomalies:
                logger.warning(
                    "Epoch %d anomaly flags: %s | report=%s",
                    int(epoch),
                    ",".join([str(a) for a in anomalies]),
                    str(epoch_report_path),
                )
            else:
                logger.info("Epoch %d anomaly checks: clean | report=%s", int(epoch), str(epoch_report_path))
        except Exception as anomaly_exc:
            logger.warning("Epoch %d anomaly checks failed: %s", int(epoch), str(anomaly_exc))


        # Log interactive per-epoch timeline table and main panel to W&B
        if self.wandb_run is not None and HAS_WANDB:
            try:
                if time_axis_minutes is None:
                    x_minutes = np.arange(len(pred_preictal), dtype=np.float32) * float(self.config.data.feature_step_s) / 60.0
                else:
                    x_minutes = np.asarray(time_axis_minutes, dtype=np.float32)

                alarm_smooth = pred_preictal_smooth_sel if pred_preictal_smooth_sel is not None else pred_preictal
                if state_mode and pred_onset_prob is not None and pred_preictal_only_prob is not None and true_state is not None and pred_state is not None:
                    table = wandb.Table(columns=[
                        'time_min', 'ppg_proxy', 'eda_proxy', 'eeg_proxy', 'hr', 'hrv',
                        'alert_prob', 'alert_smooth', 'preictal_prob', 'onset_prob',
                        'gt_active', 'gt_preictal', 'gt_onset', 'pred_state', 'gt_state'
                    ])
                    for i in range(len(x_minutes)):
                        table.add_data(
                            float(x_minutes[i]),
                            float(ppg_proxy_plot[i]),
                            float(eda_proxy_plot[i]) if eda_proxy_plot is not None else 0.0,
                            float(eeg_proxy_plot[i]) if eeg_proxy_plot is not None else 0.0,
                            float(hr_proxy_plot[i]),
                            float(hrv_proxy_plot[i]),
                            float(np.clip(pred_preictal[i], 0.0, 1.0)),
                            float(np.clip(alarm_smooth[i], 0.0, 1.0)),
                            float(np.clip(pred_preictal_only_prob[i], 0.0, 1.0)),
                            float(np.clip(pred_onset_prob[i], 0.0, 1.0)),
                            int(true_state[i] > 0),
                            int(true_state[i] == 1),
                            int(true_state[i] == 2),
                            int(pred_state[i]),
                            int(true_state[i]),
                        )
                else:
                    gt_mask = true_countdown >= 0
                    positive_gt = true_countdown[gt_mask]
                    observed_gt_max = float(np.max(positive_gt)) if positive_gt.size > 0 else 0.0
                    if observed_gt_max <= 1.5:
                        countdown_ref_max = 1.0
                    else:
                        countdown_ref_max = max(30.0, observed_gt_max)

                    gt_countdown_ref = np.full_like(true_countdown, countdown_ref_max, dtype=np.float32)
                    if np.any(gt_mask):
                        gt_countdown_ref[gt_mask] = true_countdown[gt_mask]
                    gt_countdown_ref = np.clip(gt_countdown_ref / countdown_ref_max, 0.0, 1.0)

                    pred_scale = max(countdown_ref_max, float(np.max(np.abs(pred_countdown))) + 1e-6)
                    pred_countdown_ref = np.clip(pred_countdown.astype(np.float32) / pred_scale, 0.0, 1.0)

                    table = wandb.Table(columns=[
                        'time_min', 'ppg_proxy', 'eda_proxy', 'eeg_proxy', 'hr', 'hrv',
                        'alarm_raw', 'alarm_smooth', 'gt_countdown_ref', 'pred_countdown_ref', 'gt_preictal'
                    ])
                    for i in range(len(x_minutes)):
                        table.add_data(
                            float(x_minutes[i]),
                            float(ppg_proxy_plot[i]),
                            float(eda_proxy_plot[i]) if eda_proxy_plot is not None else 0.0,
                            float(eeg_proxy_plot[i]) if eeg_proxy_plot is not None else 0.0,
                            float(hr_proxy_plot[i]),
                            float(hrv_proxy_plot[i]),
                            float(np.clip(pred_preictal[i], 0.0, 1.0)),
                            float(np.clip(alarm_smooth[i], 0.0, 1.0)),
                            float(gt_countdown_ref[i]),
                            float(pred_countdown_ref[i]),
                            int(true_countdown[i] >= 0),
                        )

                # Log both the main panel and the table
                log_payload = {
                    'visualizations/epoch_timeseries_table': table,
                    'visualizations/epoch': int(epoch),
                    'visualizations/gt_vs_inference_panel': wandb.Image(str(self.save_dir / 'epoch_visualizations' / f'{panel_filename}.png')),
                }

                if token_roll is not None:
                    log_payload['visualizations/token_activation_enabled'] = 1

                try:
                    alert_metrics = _compute_binary_metrics(
                        (viz_payload['true_countdown'] >= 0).astype(np.int32),
                        viz_payload['pred_preictal'],
                        float(self.config.loss.detection_threshold),
                    )
                    alert_cm = np.array([
                        [int(alert_metrics['tn']), int(alert_metrics['fp'])],
                        [int(alert_metrics['fn']), int(alert_metrics['tp'])],
                    ], dtype=np.int32)
                    fig_alert_cm = _build_confusion_matrix_figure(
                        alert_cm,
                        ('Interictal', 'Alert'),
                        f'Alert confusion matrix (raw) | epoch {int(epoch):03d}',
                    )
                    log_payload['visualizations/alert_confusion_matrix_raw'] = wandb.Image(fig_alert_cm)

                    # Use a conservative smoothed signal for confusion analysis.
                    # The panel can still display aggressive alert smoothing for
                    # trend visibility, but confusion matrices should avoid
                    # streak-bonus saturation that can collapse to all-alert.
                    smooth_probs = _causal_smooth(
                        np.asarray(viz_payload['pred_preictal'], dtype=np.float32),
                        rise_alpha=0.25,
                        fall_alpha=0.25,
                        streak_threshold=1.0,
                        streak_window=1,
                        streak_max_bonus=0.0,
                    )
                    smooth_metrics = _compute_binary_metrics(
                        (viz_payload['true_countdown'] >= 0).astype(np.int32),
                        smooth_probs,
                        float(self.config.loss.detection_threshold),
                    )
                    smooth_cm = np.array([
                        [int(smooth_metrics['tn']), int(smooth_metrics['fp'])],
                        [int(smooth_metrics['fn']), int(smooth_metrics['tp'])],
                    ], dtype=np.int32)
                    logger.info(
                        "Epoch %d alert CM raw tn/fp/fn/tp=%d/%d/%d/%d | smooth=%d/%d/%d/%d",
                        int(epoch),
                        int(alert_metrics['tn']), int(alert_metrics['fp']), int(alert_metrics['fn']), int(alert_metrics['tp']),
                        int(smooth_metrics['tn']), int(smooth_metrics['fp']), int(smooth_metrics['fn']), int(smooth_metrics['tp']),
                    )
                    fig_smooth_cm = _build_confusion_matrix_figure(
                        smooth_cm,
                        ('Interictal', 'Alert'),
                        f'Alert confusion matrix (smoothed) | epoch {int(epoch):03d}',
                    )
                    log_payload['visualizations/alert_confusion_matrix_smoothed'] = wandb.Image(fig_smooth_cm)

                    log_payload['visualizations/alert_balanced_accuracy_raw'] = float(alert_metrics['balanced_accuracy'])
                    log_payload['visualizations/alert_balanced_accuracy_smoothed'] = float(smooth_metrics['balanced_accuracy'])

                    if state_mode and 'state_confusion_matrix' in viz_payload:
                        fig_state_cm = _build_confusion_matrix_figure(
                            np.asarray(viz_payload['state_confusion_matrix']),
                            ('Interictal', 'Preictal', 'Onset'),
                            f'State confusion matrix | epoch {int(epoch):03d}',
                        )
                        log_payload['visualizations/state_confusion_matrix'] = wandb.Image(fig_state_cm)
                except Exception as cm_exc:
                    logger.warning('Failed to generate/log confusion matrices for epoch %d: %s', int(epoch), str(cm_exc))

                wandb.log(log_payload)
            except Exception as e:
                logger.warning("Failed to log interactive epoch timeline table to wandb: %s", str(e))
    
    def train(self, train_dataset: SeizureDataset, val_dataset: SeizureDataset,
             test_dataset: Optional[SeizureDataset] = None) -> Dict:
        """Full training loop.
        
        Args:
            train_dataset: Training dataset
            val_dataset: Validation dataset
            test_dataset: Optional test dataset
        
        Returns:
            Dictionary with training results
        """
        if self._is_main_process():
            logger.info("="*60)
            logger.info("Starting training...")
            logger.info("="*60)

        stateful_mode = self._use_stateful_training()
        if stateful_mode and self._is_main_process():
            logger.info("Stateful patient-sequential training mode enabled")
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Create model and optimizer
        model, optimizer, scheduler = self.create_model_optimizer_scheduler()
        
        # Create loss function — supports 'state' (safe 3-class, no countdown regression)
        # or 'countdown' (legacy regression).  Read from config or auto-detect.
        _loss_type = str(getattr(self.config.loss, 'loss_type', 'countdown')).lower()
        criterion = LossFactory.create_loss(self.config.loss, loss_type=_loss_type)
        criterion = criterion.to(self.device)

        preflight_enabled = bool(getattr(self.training_config, 'preflight_visualization', True))
        preflight_max_batches_cfg = int(getattr(self.training_config, 'preflight_max_val_batches', 0))
        preflight_max_batches = preflight_max_batches_cfg if preflight_max_batches_cfg > 0 else None
        if preflight_enabled:
            if self._is_main_process():
                logger.info(
                    "Running pre-training visualization preflight (max_val_batches=%s)",
                    str(preflight_max_batches) if preflight_max_batches is not None else 'full',
                )

            if self.distributed:
                dist.barrier()

            if self._is_main_process():
                self._run_visualization_checkpoint(
                    model,
                    val_loader,
                    criterion,
                    epoch=0,
                    train_viz_payload=None,
                    suffix='preflight',
                    max_val_batches=preflight_max_batches,
                )
                logger.info("Pre-training visualization preflight completed")

            if self.distributed:
                try:
                    from datetime import timedelta
                    dist.barrier(timeout=timedelta(minutes=45))
                except TypeError:
                    dist.barrier()
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_mae': [],
            'learning_rates': [],
            # Per-epoch percentage changes (relative to previous epoch)
            'train_loss_pct_change': [],
            'val_loss_pct_change': [],
            'val_mae_pct_change': [],
        }
        
        # Training loop
        start_time = time.time()
        prev_train_loss: Optional[float] = None
        prev_val_loss: Optional[float] = None
        prev_val_mae: Optional[float] = None

        def _compute_delta_pct(current: float, previous: Optional[float]):
            """Return (delta, pct_change) vs previous, or (None, None) for baseline.

            pct_change is expressed as percentage relative to |previous|.
            """
            if previous is None or not np.isfinite(previous) or abs(previous) < 1e-8:
                return None, None
            delta = float(current) - float(previous)
            pct = 100.0 * delta / max(abs(previous), 1e-8)
            return delta, pct
        
        for epoch in range(1, self.training_config.num_epochs + 1):
            # Advance ring-buffer window if training dataset supports it.
            # Step size defaults to 10 % of the buffer so the window overlaps
            # and the model sees each sample several times in different contexts.
            from src.data_loader import TemporalRingBufferDataset as _RingDS
            if isinstance(train_dataset, _RingDS):
                _ring_step = max(1, int(len(train_dataset) * 0.10))
                train_dataset.advance_to_epoch(epoch - 1, step=_ring_step)
                if self._is_main_process():
                    logger.info(
                        "Ring buffer epoch %d: offset=%d / %d  [window_size=%d]",
                        epoch,
                        train_dataset._offset,
                        len(train_dataset._dataset),
                        len(train_dataset),
                    )

            # Train
            visualization_indices = self._get_visualization_indices(len(train_loader))
            if stateful_mode:
                train_metrics, train_viz_payload = self.train_epoch_stateful(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    epoch,
                    val_loader=val_loader,
                    visualization_indices=visualization_indices,
                )
            else:
                train_metrics, train_viz_payload = self.train_epoch(
                    model,
                    train_loader,
                    criterion,
                    optimizer,
                    epoch,
                    val_loader=val_loader,
                    visualization_indices=visualization_indices,
                )
            
            # Validate
            if stateful_mode:
                if self.distributed:
                    dist.barrier()
                val_metrics, viz_payload = self.validate_stateful(model, val_loader, criterion)
                if self.distributed:
                    # Some stateful validation runs can exceed the default NCCL barrier
                    # timeout (10 minutes). Use a longer wait for rank synchronization.
                    try:
                        from datetime import timedelta
                        dist.barrier(timeout=timedelta(minutes=45))
                    except TypeError:
                        dist.barrier()
            else:
                val_metrics, viz_payload = self.validate(model, val_loader, criterion)
            
            # Gather metrics across ranks (distributed training)
            if self.distributed and not stateful_mode:
                self._sync_metrics(train_metrics)
                self._sync_metrics(val_metrics)
            
            if self._is_main_process():
                # Current epoch metrics
                train_loss = float(train_metrics['train_total'])
                val_loss = float(val_metrics['val_total'])
                val_mae = float(val_metrics['val_mae'])

                # Compute deltas vs previous epoch
                d_train, pct_train = _compute_delta_pct(train_loss, prev_train_loss)
                d_val, pct_val = _compute_delta_pct(val_loss, prev_val_loss)
                d_mae, pct_mae = _compute_delta_pct(val_mae, prev_val_mae)

                # Log metrics with per-epoch percentage change for quick significance check
                logger.info(f"Epoch {epoch}/{self.training_config.num_epochs}")
                if d_train is None:
                    logger.info(f"  Train Loss: {train_loss:.4f} (baseline)")
                else:
                    logger.info(
                        f"  Train Loss: {train_loss:.4f} (Δ {d_train:+.4f}, {pct_train:+.1f}%)"
                    )

                if d_val is None:
                    logger.info(f"  Val Loss: {val_loss:.4f} (baseline)")
                else:
                    logger.info(
                        f"  Val Loss: {val_loss:.4f} (Δ {d_val:+.4f}, {pct_val:+.1f}%)"
                    )

                if d_mae is None:
                    logger.info(f"  Val MAE: {val_mae:.4f} min (baseline)")
                else:
                    logger.info(
                        f"  Val MAE: {val_mae:.4f} min (Δ {d_mae:+.4f}, {pct_mae:+.1f}%)"
                    )

                self._save_epoch_visualizations(epoch, viz_payload, train_viz_payload)
                
                # Update history
                history['train_loss'].append(train_loss)
                history['val_loss'].append(val_loss)
                history['val_mae'].append(val_mae)
                history['train_loss_pct_change'].append(pct_train)
                history['val_loss_pct_change'].append(pct_val)
                history['val_mae_pct_change'].append(pct_mae)
                history['learning_rates'].append(optimizer.param_groups[0]['lr'])

                patient_summary = getattr(self, '_last_train_patient_summary', None)
                if isinstance(patient_summary, dict) and patient_summary.get('n_patients', 0) > 0:
                    monitor_dir = self.save_dir / "monitoring"
                    monitor_dir.mkdir(parents=True, exist_ok=True)

                    patient_epoch_payload = {
                        'epoch': int(epoch),
                        'summary': patient_summary,
                    }
                    patient_epoch_path = monitor_dir / f"epoch_{int(epoch):03d}_patient_difficulty.json"
                    with open(patient_epoch_path, "w", encoding="utf-8") as fh:
                        json.dump(patient_epoch_payload, fh, indent=2, sort_keys=True)

                    patient_jsonl_path = monitor_dir / "patient_difficulty_reports.jsonl"
                    with open(patient_jsonl_path, "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(patient_epoch_payload, sort_keys=True) + "\n")

                    hardest = patient_summary.get('hardest_patients', [])
                    easiest = patient_summary.get('easiest_patients', [])
                    if hardest:
                        logger.info(
                            "Epoch %d hardest patient: %s | difficulty=%.4f | bal_acc=%.4f | n=%d",
                            int(epoch),
                            str(hardest[0].get('patient_id', 'unknown')),
                            float(hardest[0].get('difficulty_score', 0.0)),
                            float(hardest[0].get('alert_balanced_accuracy', 0.0)),
                            int(hardest[0].get('n_samples', 0)),
                        )
                    if easiest:
                        logger.info(
                            "Epoch %d easiest patient: %s | difficulty=%.4f | bal_acc=%.4f | n=%d",
                            int(epoch),
                            str(easiest[0].get('patient_id', 'unknown')),
                            float(easiest[0].get('difficulty_score', 0.0)),
                            float(easiest[0].get('alert_balanced_accuracy', 0.0)),
                            int(easiest[0].get('n_samples', 0)),
                        )

                # Update previous metrics for next epoch comparison
                prev_train_loss = train_loss
                prev_val_loss = val_loss
                prev_val_mae = val_mae
                
                # Log to wandb
                if self.wandb_run:
                    log_payload = {
                        "epoch": epoch,
                        "train/loss": train_loss,
                        "train/classification": train_metrics.get('train_classification', 0),
                        "train/regression": train_metrics.get('train_regression', 0),
                        "val/loss": val_loss,
                        "val/classification": val_metrics.get('val_classification', 0),
                        "val/regression": val_metrics.get('val_regression', 0),
                        "val/mae": val_mae,
                        "val/medae": val_metrics['val_medae'],
                        "val/rmse": val_metrics['val_rmse'],
                        "val/alert_balanced_accuracy": val_metrics.get('val_alert_balanced_accuracy', 0.0),
                        "val/alert_sensitivity": val_metrics.get('val_alert_sensitivity', 0.0),
                        "val/alert_specificity": val_metrics.get('val_alert_specificity', 0.0),
                        "val/alert_precision": val_metrics.get('val_alert_precision', 0.0),
                        "val/alert_f1": val_metrics.get('val_alert_f1', 0.0),
                        "learning_rate": optimizer.param_groups[0]['lr'],
                    }

                    if 'val_state_balanced_accuracy' in val_metrics:
                        log_payload["val/state_balanced_accuracy"] = val_metrics['val_state_balanced_accuracy']
                        log_payload["val/state_accuracy"] = val_metrics.get('val_state_accuracy', 0.0)
                        log_payload["val/interictal_recall"] = val_metrics.get('val_interictal_recall', 0.0)
                        log_payload["val/preictal_recall"] = val_metrics.get('val_preictal_recall', 0.0)
                        log_payload["val/onset_recall"] = val_metrics.get('val_onset_recall', 0.0)
                        log_payload["val/onset_balanced_accuracy"] = val_metrics.get('val_onset_balanced_accuracy', 0.0)
                        log_payload["val/onset_sensitivity"] = val_metrics.get('val_onset_sensitivity', 0.0)
                        log_payload["val/onset_specificity"] = val_metrics.get('val_onset_specificity', 0.0)
                        log_payload["val/onset_precision"] = val_metrics.get('val_onset_precision', 0.0)
                        log_payload["val/onset_f1"] = val_metrics.get('val_onset_f1', 0.0)

                    # Also expose per-epoch percentage changes for charts (0.0 on baseline)
                    if pct_train is not None:
                        log_payload["train/loss_delta_pct"] = pct_train
                    if pct_val is not None:
                        log_payload["val/loss_delta_pct"] = pct_val
                    if pct_mae is not None:
                        log_payload["val/mae_delta_pct"] = pct_mae

                    if isinstance(patient_summary, dict) and patient_summary.get('n_patients', 0) > 0:
                        hardest = patient_summary.get('hardest_patients', [])
                        easiest = patient_summary.get('easiest_patients', [])
                        log_payload["train/patient_count"] = float(patient_summary.get('n_patients', 0))
                        if hardest:
                            log_payload["train/hardest_patient_difficulty"] = float(hardest[0].get('difficulty_score', 0.0))
                            log_payload["train/hardest_patient_balanced_accuracy"] = float(hardest[0].get('alert_balanced_accuracy', 0.0))
                            log_payload["train/hardest_patient_preictal_countdown_mae"] = float(hardest[0].get('preictal_countdown_mae', 0.0) or 0.0)
                        if easiest:
                            log_payload["train/easiest_patient_difficulty"] = float(easiest[0].get('difficulty_score', 0.0))
                            log_payload["train/easiest_patient_balanced_accuracy"] = float(easiest[0].get('alert_balanced_accuracy', 0.0))
                            log_payload["train/easiest_patient_preictal_countdown_mae"] = float(easiest[0].get('preictal_countdown_mae', 0.0) or 0.0)

                    wandb.log(log_payload)
                
                # Learning rate scheduling
                if scheduler is not None:
                    if self.training_config.lr_scheduler == "reduce_on_plateau":
                        scheduler.step(val_metrics['val_mae'])
                    else:
                        scheduler.step()
                
                # Early stopping check
                if self.training_config.early_stopping:
                    metric_key = self.training_config.early_stopping_metric
                    if not metric_key.startswith("val_"):
                        metric_key = f"val_{metric_key}"

                    metric_value = val_metrics.get(metric_key, val_metrics['val_total'])
                    
                    if metric_value < self.best_val_metric:
                        self.best_val_metric = metric_value
                        self.early_stopping_counter = 0
                        
                        # Save best model
                        self._save_checkpoint(model, optimizer, epoch, checkpoint_name="best_model.pt")
                    else:
                        self.early_stopping_counter += 1
                        if self.early_stopping_counter >= self.training_config.early_stopping_patience:
                            logger.info(f"Early stopping triggered at epoch {epoch}")
                            break

                # Optional periodic checkpointing
                if self.training_config.save_interval > 0 and epoch % self.training_config.save_interval == 0:
                    self._save_checkpoint(model, optimizer, epoch, checkpoint_name=f"epoch_{epoch:03d}.pt")
        
        elapsed_time = time.time() - start_time
        
        if self._is_main_process():
            logger.info(f"Training completed in {elapsed_time/3600:.2f} hours")
            logger.info(f"Best validation MAE: {self.best_val_metric:.4f}")

            # Always save a final checkpoint for resumability
            self._save_checkpoint(
                model,
                optimizer,
                epoch=epoch if 'epoch' in locals() else 0,
                checkpoint_name="last_model.pt"
            )

            # Save training history artifact
            self._save_training_artifacts(history, elapsed_time)
            
            # Log final summary to wandb
            if self.wandb_run:
                wandb.log({
                    "final/training_time_hours": elapsed_time/3600,
                    "final/best_val_mae": self.best_val_metric,
                })
                wandb.finish()
            
            # Load best model for final evaluation
            if self.best_model_path:
                checkpoint = torch.load(self.best_model_path, map_location=self.device, weights_only=False)
                if self.distributed:
                    model.module.load_state_dict(checkpoint['model_state'])
                else:
                    model.load_state_dict(checkpoint['model_state'])
        
        # Synchronize after training
        if self.distributed:
            if self.device.type == 'cuda':
                dist.barrier(device_ids=[self.local_rank])
            else:
                dist.barrier()
        
        return {
            'history': history,
            'best_val_metric': self.best_val_metric,
            'training_time': elapsed_time,
        }
    
    def _save_checkpoint(self, model: nn.Module, optimizer: optim.Optimizer,
                        epoch: int, checkpoint_name: str = "best_model.pt") -> None:
        """Save model checkpoint.
        
        Args:
            model: Neural network model
            optimizer: Optimizer
            epoch: Epoch number
        """
        checkpoint = {
            'epoch': epoch,
            'model_state': model.module.state_dict() if self.distributed else model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'best_metric': self.best_val_metric,
        }
        
        checkpoint_path = self.save_dir / checkpoint_name
        torch.save(checkpoint, checkpoint_path)

        if checkpoint_name == "best_model.pt":
            self.best_model_path = checkpoint_path
        
        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def _save_training_artifacts(self, history: Dict[str, list], elapsed_time: float) -> None:
        """Save training history/summary artifacts for reproducibility."""
        def _to_jsonable(value):
            if isinstance(value, np.generic):
                return value.item()
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, dict):
                return {k: _to_jsonable(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [_to_jsonable(v) for v in value]
            return value

        artifact = {
            "history": _to_jsonable(history),
            "best_val_metric": float(self.best_val_metric),
            "training_time_seconds": float(elapsed_time),
            "training_time_hours": float(elapsed_time / 3600),
            "num_epochs_completed": len(history.get('train_loss', [])),
            "world_size": self.world_size,
            "distributed": self.distributed,
        }

        history_path = self.save_dir / "training_history.json"
        with open(history_path, "w", encoding="utf-8") as file_handle:
            json.dump(artifact, file_handle, indent=2)

        logger.info(f"Saved training history to {history_path}")
    
    def _sync_metrics(self, metrics: Dict) -> None:
        """Synchronize metrics across distributed processes."""
        if not self.distributed:
            return
        
        for key in list(metrics.keys()):
            tensor = torch.tensor(metrics[key], device=self.device)
            dist.all_reduce(tensor)
            metrics[key] = (tensor / self.world_size).item()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config.config import DEFAULT_CONFIG
    
    # Create trainer
    trainer = Trainer(DEFAULT_CONFIG)
    
    # Create dummy datasets
    X_train = np.random.randn(100, 600, 14).astype(np.float32)
    y_train = np.random.rand(100) * 10
    train_dataset = SeizureDataset(X_train, y_train)
    
    X_val = np.random.randn(30, 600, 14).astype(np.float32)
    y_val = np.random.rand(30) * 10
    val_dataset = SeizureDataset(X_val, y_val)
    
    # Train
    results = trainer.train(train_dataset, val_dataset)
    print(f"Training completed. Best MAE: {results['best_val_metric']:.4f}")
