#!/usr/bin/env python
"""
Main training script for seizure anticipation model with multi-GPU support.

Usage:
    # Single GPU
    python scripts/train.py

    # Multi-GPU (2 GPUs) with Distributed Data Parallel
    python -m torch.distributed.launch --nproc_per_node=2 scripts/train.py

    # With custom config
    python scripts/train.py --config config/custom_config.yaml --model-type cnn_lstm
"""

import os
import sys
import json
import logging
import argparse
import hashlib
import time
from pathlib import Path
from typing import List, Tuple, Optional

# Ensure headless plotting backend (must be set before importing project modules)
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import torch.distributed as dist
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, DEFAULT_CONFIG
from src import (
    BIDSDataLoader,
    WearableDeviceDataLoader,
    Trainer,
    Evaluator,
    FeatureExtractor,
    SignalProcessor,
)
from src.visualization import SignalVisualizer


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _to_jsonable(value):
    """Convert numpy/torch scalar containers into JSON-serializable primitives."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train seizure anticipation model'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file'
    )
    
    parser.add_argument(
        '--model-type',
        type=str,
        choices=[
            # Original
            'ecg_lstm', 'cnn_lstm', 'multimodal',
            # TinyML / Edge
            'eegnet', 'mobilenet_1d', 'tcn', 'inception_1d',
            # Server / Cloud
            'temporal_transformer', 'multimodal_transformer',
        ],
        default='ecg_lstm',
        help=(
            'Model architecture. '
            'TinyML/edge: eegnet (ESP32), mobilenet_1d (smartwatch), tcn, inception_1d. '
            'Server: temporal_transformer, multimodal_transformer.'
        )
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of epochs'
    )
    
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    
    parser.add_argument(
        '--no-cuda',
        action='store_true',
        help='Disable GPU'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )

    parser.add_argument(
        '--dummy-train-size',
        type=int,
        default=2000,
        help='Dummy dataset train sample count'
    )

    parser.add_argument(
        '--dummy-val-size',
        type=int,
        default=600,
        help='Dummy dataset validation sample count'
    )

    parser.add_argument(
        '--dummy-test-size',
        type=int,
        default=600,
        help='Dummy dataset test sample count'
    )

    parser.add_argument(
        '--dummy-preictal-ratio',
        type=float,
        default=0.35,
        help='Dummy dataset preictal ratio in [0,1]'
    )

    parser.add_argument(
        '--data-mode',
        type=str,
        choices=['real', 'dummy'],
        default='real',
        help='Dataset mode: real uses SeizeIT2 recordings, dummy uses synthetic data'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        choices=['bids', 'wearable'],
        default='bids',
        help='Real data source: bids (ds005873 EDF) or wearable (WearablwDevice-Oregon CSV)'
    )

    parser.add_argument(
        '--strict-real-data',
        action='store_true',
        help='Fail immediately if real dataset is unavailable instead of falling back to dummy data'
    )

    parser.add_argument(
        '--dataset-root',
        type=str,
        default=None,
        help='Override dataset root path (useful when external drive mount path changed)'
    )

    parser.add_argument(
        '--max-recordings',
        type=int,
        default=120,
        help='Maximum number of recordings to use in real mode (for faster iteration)'
    )

    parser.add_argument(
        '--max-samples-per-recording',
        type=int,
        default=120,
        help='Maximum extracted windows per recording in real mode'
    )

    parser.add_argument(
        '--real-stride-seconds',
        type=float,
        default=None,
        help='Stride between extracted real-data windows in seconds; lower means more points per minute'
    )

    parser.add_argument(
        '--long-sweep-training',
        action='store_true',
        help=(
            'Enable day-like continuous sweep training: preserve chronological '
            'order within recordings, split by recording (no cross-split leakage), '
            'and disable random/context subsampling shortcuts.'
        )
    )

    parser.add_argument(
        '--panel-window-minutes',
        type=float,
        default=180.0,
        help='Window length in minutes for final GT-vs-inference panel (default: 180)'
    )

    parser.add_argument(
        '--threshold-objective',
        type=str,
        choices=['balanced_accuracy', 'f1'],
        default='balanced_accuracy',
        help='Objective for post-training threshold sweep'
    )

    parser.add_argument(
        '--threshold-min-sensitivity',
        type=float,
        default=0.0,
        help='Minimum sensitivity constraint for threshold sweep (0-1)'
    )

    parser.add_argument(
        '--threshold-max-fpr',
        type=float,
        default=1.0,
        help='Maximum false-positive-rate constraint for threshold sweep (0-1)'
    )

    parser.add_argument(
        '--auto-threshold',
        action='store_true',
        help='Run threshold sweep on test predictions and save selected threshold'
    )

    parser.add_argument(
        '--write-threshold-to-config',
        action='store_true',
        help='Persist selected threshold into saved models/config.yaml'
    )

    # DDP launch compatibility (torch.distributed.launch / torchrun)
    parser.add_argument(
        '--local-rank',
        '--local_rank',
        type=int,
        default=None,
        help='Local rank passed by distributed launchers'
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _extract_seizure_onsets(events_df) -> np.ndarray:
    """Extract seizure onset times in seconds from events dataframe."""
    if events_df is None or events_df.empty or 'onset' not in events_df.columns:
        return np.array([], dtype=np.float32)

    if 'eventType' in events_df.columns:
        seizure_mask = events_df['eventType'].astype(str).str.startswith('sz', na=False)
    elif 'trial_type' in events_df.columns:
        seizure_mask = events_df['trial_type'].astype(str).str.contains('seizure|sz', case=False, na=False)
    else:
        return np.array([], dtype=np.float32)

    onsets = events_df.loc[seizure_mask, 'onset'].astype(float).values
    return np.sort(onsets.astype(np.float32))


def _segment_to_feature_matrix(segment: np.ndarray, feature_dim: int, target_steps: int = 600) -> np.ndarray:
    """Convert 1D segment into (target_steps, feature_dim) feature matrix.

    This helper is modality-agnostic: it projects any 1D waveform into a
    small bank of engineered channels (base, gradients, powers, rolls, etc.)
    and then keeps the first *feature_dim* channels. It is used for ECG, EEG
    and motion proxies in real-mode feature construction.
    """
    if len(segment) < 4:
        segment = np.pad(segment, (0, max(0, 4 - len(segment))), mode='edge')

    x_old = np.linspace(0.0, 1.0, num=len(segment), endpoint=True)
    x_new = np.linspace(0.0, 1.0, num=target_steps, endpoint=True)
    base = np.interp(x_new, x_old, segment)

    base = (base - np.mean(base)) / (np.std(base) + 1e-6)
    grad1 = np.gradient(base)
    grad2 = np.gradient(grad1)
    abs_base = np.abs(base)
    sq_base = base ** 2
    signed_sqrt = np.sign(base) * np.sqrt(np.abs(base) + 1e-6)
    cumsum = np.cumsum(base) / np.arange(1, target_steps + 1)
    dev_from_mean = base - cumsum

    candidates = [
        base,
        grad1,
        grad2,
        abs_base,
        sq_base,
        signed_sqrt,
        cumsum,
        dev_from_mean,
        np.tanh(base),
        np.tanh(grad1),
        np.sin(base),
        np.cos(base),
        np.clip(base, -2, 2),
        np.clip(grad1, -2, 2),
        np.clip(abs_base, 0, 3),
        np.clip(sq_base, 0, 4),
        np.roll(base, 1),
        np.roll(base, 2),
        np.roll(grad1, 1),
        np.roll(grad1, 2),
        np.roll(abs_base, 1),
        np.roll(abs_base, 2),
    ]

    feature_list = candidates[:feature_dim]
    feature_mat = np.stack(feature_list, axis=-1).astype(np.float32)
    return feature_mat


def _align_feature_matrix(feature_mat: np.ndarray, feature_dim: int, target_steps: int = 600) -> np.ndarray:
    """Align an existing (T, F) feature matrix to model-required (target_steps, feature_dim).

    Resamples time dimension if needed and pads/truncates feature dimension.
    """
    feature_mat = np.asarray(feature_mat, dtype=np.float32)

    if feature_mat.ndim == 1:
        feature_mat = feature_mat[:, None]

    time_steps, input_dim = feature_mat.shape

    if time_steps != target_steps:
        x_old = np.linspace(0.0, 1.0, num=time_steps, endpoint=True)
        x_new = np.linspace(0.0, 1.0, num=target_steps, endpoint=True)
        resized = np.zeros((target_steps, input_dim), dtype=np.float32)
        for col in range(input_dim):
            resized[:, col] = np.interp(x_new, x_old, feature_mat[:, col]).astype(np.float32)
        feature_mat = resized

    input_dim = feature_mat.shape[1]
    if input_dim == feature_dim:
        return feature_mat.astype(np.float32)

    if input_dim > feature_dim:
        return feature_mat[:, :feature_dim].astype(np.float32)

    padded = np.zeros((feature_mat.shape[0], feature_dim), dtype=np.float32)
    padded[:, :input_dim] = feature_mat
    return padded


def _select_recordings_for_real_mode(recordings: List[dict], max_recordings: int) -> List[dict]:
    """Select recordings with seizure-first prioritization for better preictal coverage."""
    if max_recordings and max_recordings > 0:
        candidates = recordings[:]
    else:
        candidates = recordings

    seizure_recs = [recording for recording in candidates if int(recording.get('num_seizures', 0)) > 0]
    nonseizure_recs = [recording for recording in candidates if int(recording.get('num_seizures', 0)) <= 0]

    seizure_recs.sort(key=lambda recording: int(recording.get('num_seizures', 0)), reverse=True)

    selected: List[dict] = []
    if max_recordings and max_recordings > 0:
        n_target = max_recordings
    else:
        n_target = len(candidates)

    selected.extend(seizure_recs[:n_target])
    if len(selected) < n_target:
        selected.extend(nonseizure_recs[: n_target - len(selected)])

    return selected


def _split_indices_with_preictal_coverage(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """Build split indices while trying to keep preictal samples in each split."""
    rng = np.random.default_rng(seed)

    n_total = len(y)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val

    preictal_idx = np.where(y >= 0)[0]
    interictal_idx = np.where(y < 0)[0]
    rng.shuffle(preictal_idx)
    rng.shuffle(interictal_idx)

    split_sizes = [n_train, n_val, n_test]
    split_names = ['train', 'val', 'test']

    split_preictal_alloc = {name: 0 for name in split_names}
    if len(preictal_idx) > 0:
        for split_name in split_names:
            if len(preictal_idx) > sum(split_preictal_alloc.values()):
                split_preictal_alloc[split_name] += 1

        remaining_pos = len(preictal_idx) - sum(split_preictal_alloc.values())
        if remaining_pos > 0:
            size_weights = np.array(split_sizes, dtype=np.float32)
            if size_weights.sum() <= 0:
                size_weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
            size_weights /= size_weights.sum()
            extra_alloc = np.floor(size_weights * remaining_pos).astype(int)
            for split_name, alloc in zip(split_names, extra_alloc):
                split_preictal_alloc[split_name] += int(alloc)

            assigned = int(extra_alloc.sum())
            leftover = remaining_pos - assigned
            if leftover > 0:
                for split_name in split_names:
                    if leftover <= 0:
                        break
                    split_preictal_alloc[split_name] += 1
                    leftover -= 1

    split_indices = {}
    pos_cursor = 0
    neg_cursor = 0
    for split_name, split_size in zip(split_names, split_sizes):
        n_pos = min(split_preictal_alloc[split_name], split_size)
        n_neg = max(0, split_size - n_pos)

        pos_slice = preictal_idx[pos_cursor: pos_cursor + n_pos]
        neg_slice = interictal_idx[neg_cursor: neg_cursor + n_neg]
        pos_cursor += len(pos_slice)
        neg_cursor += len(neg_slice)

        split_idx = np.concatenate([pos_slice, neg_slice])
        rng.shuffle(split_idx)
        split_indices[split_name] = split_idx

    remaining_pos = preictal_idx[pos_cursor:]
    remaining_neg = interictal_idx[neg_cursor:]
    remaining_all = np.concatenate([remaining_pos, remaining_neg])
    if len(remaining_all) > 0:
        rng.shuffle(remaining_all)
        split_indices['test'] = np.concatenate([split_indices['test'], remaining_all])

    return split_indices['train'], split_indices['val'], split_indices['test']


def _sort_indices_by_timeline(indices: np.ndarray,
                              recording_ids: np.ndarray,
                              sample_end_times_s: np.ndarray) -> np.ndarray:
    """Sort sample indices by (recording_id, sample_time)."""
    if len(indices) == 0:
        return indices.astype(np.int64)

    rec_local = np.asarray(recording_ids)[indices].astype(str)
    t_local = np.asarray(sample_end_times_s, dtype=np.float64)[indices]
    order = np.lexsort((t_local, rec_local))
    return indices[order].astype(np.int64)


def _split_indices_by_recording_timeline(
    y: np.ndarray,
    recording_ids: np.ndarray,
    sample_end_times_s: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """Split by recording to avoid leakage, then keep timeline order within each split."""
    rng = np.random.default_rng(seed)

    y = np.asarray(y, dtype=np.float32)
    recording_ids = np.asarray(recording_ids)
    sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float64)

    unique_recs = np.unique(recording_ids)
    if unique_recs.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    rec_meta = []
    for rec_id in unique_recs:
        rec_idx = np.where(recording_ids == rec_id)[0]
        if rec_idx.size == 0:
            continue
        rec_meta.append(
            {
                'rec_id': rec_id,
                'n': int(rec_idx.size),
                'has_preictal': bool(np.any(y[rec_idx] >= 0)),
                't0': float(np.min(sample_end_times_s[rec_idx])),
            }
        )

    if len(rec_meta) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    pre_recs = [row for row in rec_meta if row['has_preictal']]
    non_recs = [row for row in rec_meta if not row['has_preictal']]
    rng.shuffle(pre_recs)
    rng.shuffle(non_recs)

    # Place preictal-containing recordings first to increase chance each split
    # receives seizure context, then fill with non-seizure recordings.
    ordered_recs = pre_recs + non_recs

    n_total = int(len(y))
    target = {
        'train': float(train_ratio) * n_total,
        'val': float(val_ratio) * n_total,
        'test': max(0.0, (1.0 - float(train_ratio) - float(val_ratio)) * n_total),
    }
    assigned = {'train': 0, 'val': 0, 'test': 0}
    split_recs = {'train': [], 'val': [], 'test': []}
    rec_size = {row['rec_id']: int(row['n']) for row in rec_meta}

    for row in ordered_recs:
        # Greedy assignment by smallest target-fill ratio.
        best_split = min(
            ['train', 'val', 'test'],
            key=lambda name: assigned[name] / max(target[name], 1.0)
        )
        split_recs[best_split].append(row['rec_id'])
        assigned[best_split] += int(row['n'])

    # For small-N recording scenarios, avoid empty splits when feasible.
    # If we have >=3 recordings total, try to keep at least one recording in
    # each split by moving one recording from the currently largest donor split.
    if len(unique_recs) >= 3:
        for need_split in ['train', 'val', 'test']:
            if len(split_recs[need_split]) > 0:
                continue

            donor = None
            donor_size = -1
            for candidate in ['train', 'val', 'test']:
                if candidate == need_split:
                    continue
                if len(split_recs[candidate]) <= 1:
                    continue
                if assigned[candidate] > donor_size:
                    donor = candidate
                    donor_size = assigned[candidate]

            if donor is None:
                continue

            # Move the smallest recording from donor to minimize disturbance.
            donor_rec = min(split_recs[donor], key=lambda rec_id: rec_size.get(rec_id, 0))
            split_recs[donor].remove(donor_rec)
            split_recs[need_split].append(donor_rec)
            moved_n = rec_size.get(donor_rec, 0)
            assigned[donor] -= moved_n
            assigned[need_split] += moved_n

    def _collect(split_name: str) -> np.ndarray:
        recs = split_recs[split_name]
        if len(recs) == 0:
            return np.array([], dtype=np.int64)
        mask = np.isin(recording_ids, np.asarray(recs))
        idx = np.where(mask)[0]
        return _sort_indices_by_timeline(idx, recording_ids, sample_end_times_s)

    return _collect('train'), _collect('val'), _collect('test')


def _select_panel_indices(true_countdown: np.ndarray,
                          feature_step_s: float,
                          window_minutes: float = 60.0,
                          seed: int = 42,
                          sample_end_times_s: Optional[np.ndarray] = None,
                          recording_ids: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Select panel indices and time axis from a true contiguous timeline.

    If recording/time metadata is available, selects one recording then extracts
    a contiguous one-hour window around seizure-onset context. Otherwise falls
    back to dataset-order contiguous selection.
    """
    n_total = len(true_countdown)
    if n_total == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float64)

    rng = np.random.default_rng(seed)
    step_s = max(float(feature_step_s), 1e-6)
    window_seconds = float(window_minutes) * 60.0
    window_samples = max(1, int(np.ceil(window_seconds / step_s)))

    has_meta = (
        sample_end_times_s is not None and
        recording_ids is not None and
        len(sample_end_times_s) == n_total and
        len(recording_ids) == n_total
    )

    if has_meta:
        sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float64)
        recording_ids = np.asarray(recording_ids)

        unique_recordings = np.unique(recording_ids)
        tier_full = []
        tier_preictal = []
        tier_any = []

        for rec_id in unique_recordings:
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

        if len(tier_full) > 0:
            chosen_rec = rng.choice(tier_full)
        elif len(tier_preictal) > 0:
            chosen_rec = rng.choice(tier_preictal)
        elif len(tier_any) > 0:
            chosen_rec = rng.choice(tier_any)
        else:
            chosen_rec = None

        if chosen_rec is not None:
            rec_indices = np.where(recording_ids == chosen_rec)[0]
            rec_order = np.argsort(sample_end_times_s[rec_indices])
            rec_indices = rec_indices[rec_order]
            rec_times_s = sample_end_times_s[rec_indices]
            rec_true = true_countdown[rec_indices]

            pre_mask = rec_true >= 0
            if np.any(pre_mask):
                onset_estimates_s = rec_times_s[pre_mask] + rec_true[pre_mask] * 60.0
                onset_time_s = float(np.median(onset_estimates_s))
                desired_start_s = onset_time_s - max(window_seconds - 10.0 * 60.0, 0.0)
            else:
                desired_start_s = float(rec_times_s[0])

            min_start_s = float(rec_times_s[0])
            max_start_s = max(min_start_s, float(rec_times_s[-1]) - window_seconds)
            start_s = float(np.clip(desired_start_s, min_start_s, max_start_s))
            end_s = start_s + window_seconds

            in_window = (rec_times_s >= start_s) & (rec_times_s <= end_s)
            if np.sum(in_window) >= 2:
                selected = rec_indices[in_window]
                # Rebase to a local timeline so the panel x-axis runs from
                # ~0 to window_minutes instead of large absolute minutes
                # since recording start. This matches the epoch panels and
                # is easier to interpret.
                base_time_s = float(rec_times_s[in_window][0])
                time_minutes = (rec_times_s[in_window] - base_time_s) / 60.0
                return selected.astype(np.int64), time_minutes.astype(np.float64)

            # Fallback within selected recording: contiguous sample chunk around onset anchor.
            if np.any(pre_mask):
                onset_idx_local = int(np.where(pre_mask)[0][np.argmin(np.abs(rec_true[pre_mask]))])
            else:
                onset_idx_local = len(rec_indices) // 2

            half = window_samples // 2
            start = max(0, onset_idx_local - half)
            end = min(len(rec_indices), start + window_samples)
            start = max(0, end - window_samples)

            selected = rec_indices[start:end]
            base_time_s = float(rec_times_s[start]) if rec_times_s.size > 0 else 0.0
            time_minutes = (rec_times_s[start:end] - base_time_s) / 60.0
            return selected.astype(np.int64), time_minutes.astype(np.float64)

    # Fallback: contiguous by dataset index when metadata is unavailable.
    if n_total <= window_samples:
        selected = np.arange(n_total, dtype=np.int64)
    else:
        preictal_idx = np.where(true_countdown >= 0)[0]
        if len(preictal_idx) > 0:
            anchor_idx = int(rng.choice(preictal_idx))
        else:
            anchor_idx = int(rng.integers(0, n_total))

        half = window_samples // 2
        start = max(0, anchor_idx - half)
        end = min(n_total, start + window_samples)
        start = max(0, end - window_samples)
        selected = np.arange(start, end, dtype=np.int64)

    time_minutes = selected.astype(np.float64) * step_s / 60.0
    return selected, time_minutes


def _augment_train_dataset(X_train: np.ndarray, y_train: np.ndarray, 
                          config: Config) -> Tuple[np.ndarray, np.ndarray]:
    """Augment preictal samples in training set to balance classes.
    
    Args:
        X_train: Training features (N, T, F)
        y_train: Training labels (N,)
        config: Config with augmentation settings
    
    Returns:
        Augmented (X_train, y_train) with more preictal samples
    """
    if not config.data.augment_preictal:
        logger.info("Preictal augmentation disabled")
        return X_train, y_train
    
    from src.data_loader import augment_preictal_sample
    
    preictal_mask = y_train >= 0
    preictal_count_orig = int(np.sum(preictal_mask))
    
    if preictal_count_orig == 0:
        logger.warning("No preictal samples to augment")
        return X_train, y_train
    
    rng = np.random.default_rng(config.data.random_seed)
    
    augmented_features = []
    augmented_labels = []
    
    # Keep all interictal samples
    interictal_mask = ~preictal_mask
    augmented_features.extend(X_train[interictal_mask])
    augmented_labels.extend(y_train[interictal_mask])
    
    # Augment each preictal sample
    preictal_indices = np.where(preictal_mask)[0]
    for idx in preictal_indices:
        sample_features = X_train[idx]
        sample_label = y_train[idx]
        
        aug_samples = augment_preictal_sample(sample_features, sample_label, config.data, rng)
        
        for aug_feat, aug_label in aug_samples:
            augmented_features.append(aug_feat)
            augmented_labels.append(aug_label)
    
    X_augmented = np.stack(augmented_features).astype(np.float32)
    y_augmented = np.array(augmented_labels, dtype=np.float32)
    
    preictal_count_aug = int(np.sum(y_augmented >= 0))
    interictal_count = int(np.sum(y_augmented < 0))
    
    logger.info(f"Augmentation: {preictal_count_orig} → {preictal_count_aug} preictal samples")
    logger.info(f"New balance: {preictal_count_aug} preictal vs {interictal_count} interictal")
    logger.info(f"Imbalance ratio: {interictal_count / max(1, preictal_count_aug):.1f}:1")
    
    return X_augmented, y_augmented


def _prepare_real_datasets(config: Config, args, loader: BIDSDataLoader, recordings: List[dict]):
    """Build datasets from real ECG(+EEG/motion) recordings and seizure onsets.

    When a multimodal model is requested (see ModelFactory.is_multimodal), this
    routine will attempt to construct per-window feature matrices that contain
    concatenated ECG, EEG, and motion proxies in the following layout along the
    feature dimension:

        [0 : ecg_feature_dim)        → ECG-derived channels
        [ecg : ecg+eeg)              → EEG-derived channels (if available)
        [ecg+eeg : ecg+eeg+motion)   → Motion-derived channels (currently zero
                                       placeholders unless motion data is
                                       wired in later).

    For single-modal models, only ECG-derived channels are constructed.
    """
    logger.info("Preparing REAL dataset from SeizeIT2 ECG(+EEG) + events...")
    prep_start = time.time()

    recordings = _select_recordings_for_real_mode(recordings, args.max_recordings)
    seizure_recordings = sum(1 for recording in recordings if int(recording.get('num_seizures', 0)) > 0)
    logger.info(
        "Selected %d recordings for real mode (%d with seizures)",
        len(recordings),
        seizure_recordings,
    )

    from src.models import ModelFactory as _MF
    is_multimodal = _MF.is_multimodal(config.model.model_type)

    ecg_dim = int(config.model.ecg_feature_dim)
    eeg_dim = int(config.model.eeg_feature_dim) if is_multimodal else 0
    motion_dim = int(config.model.motion_feature_dim) if is_multimodal else 0

    if is_multimodal:
        logger.info(
            "REAL mode: constructing multimodal features with dims (ECG=%d, EEG=%d, motion=%d)",
            ecg_dim,
            eeg_dim,
            motion_dim,
        )
    else:
        logger.info("REAL mode: constructing ECG-only features with dim=%d", ecg_dim)

    sequence_seconds = 120.0
    stride_seconds = float(config.data.feature_step_s)
    preictal_window_s = float(config.data.pre_ictal_window_s)

    all_features = []
    all_labels = []
    all_sample_end_times_s = []
    all_recording_ids = []
    low_density_warning_emitted = False

    for rec_idx, recording in enumerate(recordings):
        try:
            rec_start = time.time()
            if rec_idx < 5 or (rec_idx + 1) % 5 == 0:
                logger.info(
                    "[REAL PREP] Processing recording %d/%d: sub-%s ses-%s run-%s",
                    rec_idx + 1,
                    len(recordings),
                    recording.get('subject_id', '?'),
                    recording.get('session_id', '?'),
                    recording.get('run_id', '?'),
                )

            ecg_data, fs = loader.load_subject_edf(
                recording['subject_id'],
                recording['session_id'],
                'ecg',
                recording['run_id']
            )
            signal_1d = ecg_data[0] if ecg_data.ndim > 1 else ecg_data

            # Optionally load EEG and motion streams when training multimodal
            # models. If alignment or loading fails, fall back to zero-filled
            # channels so that the architecture remains consistent while still
            # benefiting from ECG.
            eeg_1d = None
            fs_eeg = None
            if is_multimodal and eeg_dim > 0:
                try:
                    eeg_data, fs_eeg = loader.load_subject_edf(
                        recording['subject_id'],
                        recording['session_id'],
                        'eeg',
                        recording['run_id'],
                    )
                    eeg_1d = eeg_data[0] if eeg_data.ndim > 1 else eeg_data

                    if not np.isfinite(fs_eeg) or abs(float(fs_eeg) - float(fs)) > 1e-3:
                        logger.warning(
                            "Skipping EEG for sub-%s ses-%s run-%s due to sampling-rate mismatch (ECG=%.3f Hz, EEG=%.3f Hz)",
                            recording.get('subject_id', '?'),
                            recording.get('session_id', '?'),
                            recording.get('run_id', '?'),
                            float(fs),
                            float(fs_eeg),
                        )
                        eeg_1d = None
                except Exception as eeg_error:
                    logger.warning(
                        "Could not load EEG for sub-%s ses-%s run-%s: %s",
                        recording.get('subject_id', '?'),
                        recording.get('session_id', '?'),
                        recording.get('run_id', '?'),
                        str(eeg_error),
                    )

            events_df = loader.load_events_tsv(
                recording['subject_id'],
                recording['session_id'],
                recording['run_id']
            )
            seizure_onsets = _extract_seizure_onsets(events_df)

            seq_samples = int(sequence_seconds * fs)
            stride_samples = max(1, int(stride_seconds * fs))

            candidate_end_indices = np.arange(seq_samples, len(signal_1d), stride_samples, dtype=np.int64)
            if len(candidate_end_indices) == 0:
                continue

            # Long-sweep mode keeps all chronological windows to mimic true
            # day-long streaming operation.
            if (not args.long_sweep_training) and args.max_samples_per_recording > 0 and len(candidate_end_indices) > args.max_samples_per_recording:
                selected_positions = np.linspace(
                    0,
                    len(candidate_end_indices) - 1,
                    num=args.max_samples_per_recording,
                    dtype=np.int64,
                )
                selected_end_indices = candidate_end_indices[selected_positions]
            else:
                selected_end_indices = candidate_end_indices

            if (not low_density_warning_emitted) and len(selected_end_indices) > 1:
                covered_seconds = float(selected_end_indices[-1] - selected_end_indices[0]) / float(fs)
                effective_hz = float(len(selected_end_indices) - 1) / max(covered_seconds, 1e-6)
                if effective_hz < 0.2:
                    logger.warning(
                        "Low temporal sampling detected for visualization (effective %.3f Hz over %.1f min). "
                        "This can make 60-minute panels appear sparse. "
                        "Current --max-samples-per-recording=%d; consider MAX_SAMPLES_PER_RECORDING=0 "
                        "or a much larger value.",
                        effective_hz,
                        covered_seconds / 60.0,
                        int(args.max_samples_per_recording),
                    )
                    low_density_warning_emitted = True

            sample_count_for_recording = 0
            recording_uid = f"sub-{recording['subject_id']}_ses-{recording['session_id']}_run-{recording['run_id']}"
            for end_idx in selected_end_indices:
                # ECG window
                if end_idx <= seq_samples:
                    continue
                segment = signal_1d[end_idx - seq_samples:end_idx]

                # Always construct ECG-derived channels.
                ecg_features = _segment_to_feature_matrix(segment, ecg_dim, target_steps=600)

                # EEG window (if available); otherwise, zeros.
                if is_multimodal and eeg_dim > 0 and eeg_1d is not None and len(eeg_1d) >= end_idx:
                    eeg_segment = eeg_1d[end_idx - seq_samples:end_idx]
                    eeg_features = _segment_to_feature_matrix(eeg_segment, eeg_dim, target_steps=600)
                elif is_multimodal and eeg_dim > 0:
                    eeg_features = np.zeros((600, eeg_dim), dtype=np.float32)
                else:
                    eeg_features = None

                # Motion window: currently a zero placeholder that preserves
                # architecture compatibility. This can be replaced with true
                # motion-derived features once motion streams are wired in.
                if is_multimodal and motion_dim > 0:
                    motion_segment = np.zeros_like(segment, dtype=np.float32)
                    motion_features = _segment_to_feature_matrix(motion_segment, motion_dim, target_steps=600)
                else:
                    motion_features = None

                if is_multimodal:
                    feature_blocks = [ecg_features]
                    if eeg_features is not None:
                        feature_blocks.append(eeg_features)
                    if motion_features is not None:
                        feature_blocks.append(motion_features)
                    feature_mat = np.concatenate(feature_blocks, axis=-1)
                else:
                    feature_mat = ecg_features

                t_end = end_idx / fs
                future_onsets = seizure_onsets[seizure_onsets >= t_end]
                if len(future_onsets) > 0:
                    dt = float(future_onsets[0] - t_end)
                    countdown_label = dt / 60.0 if dt <= preictal_window_s else -1.0
                else:
                    countdown_label = -1.0

                all_features.append(feature_mat)
                all_labels.append(countdown_label)
                all_sample_end_times_s.append(float(t_end))
                all_recording_ids.append(recording_uid)
                sample_count_for_recording += 1

            elapsed = time.time() - prep_start
            avg_per_recording = elapsed / max(1, rec_idx + 1)
            remaining = max(0, len(recordings) - (rec_idx + 1))
            eta_seconds = avg_per_recording * remaining
            if rec_idx < 5 or (rec_idx + 1) % 5 == 0:
                logger.info(
                    "[REAL PREP] Done %d/%d | +%d samples (total=%d) | rec_time=%.1fs | ETA=%.1f min",
                    rec_idx + 1,
                    len(recordings),
                    sample_count_for_recording,
                    len(all_labels),
                    time.time() - rec_start,
                    eta_seconds / 60.0,
                )

        except Exception as error:
            logger.debug(f"Skipping recording due to load/process issue: {error}")
            continue

    if len(all_labels) < 100:
        logger.warning("Real dataset extraction produced too few samples; falling back to dummy mode.")
        return None

    logger.info(
        "[REAL PREP] Completed in %.1f min with %d samples",
        (time.time() - prep_start) / 60.0,
        len(all_labels),
    )

    X = np.stack(all_features).astype(np.float32)
    y = np.array(all_labels, dtype=np.float32)
    sample_end_times_s = np.array(all_sample_end_times_s, dtype=np.float32)
    recording_ids = np.array(all_recording_ids)

    if args.long_sweep_training:
        train_idx, val_idx, test_idx = _split_indices_by_recording_timeline(
            y,
            recording_ids=recording_ids,
            sample_end_times_s=sample_end_times_s,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices_with_preictal_coverage(
            y,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]
    val_sample_end_times_s = sample_end_times_s[val_idx]
    val_recording_ids = recording_ids[val_idx]
    test_sample_end_times_s = sample_end_times_s[test_idx]
    test_recording_ids = recording_ids[test_idx]
    n_total = len(y)

    # Long-sweep mode avoids augmentation to preserve real temporal dynamics.
    if args.long_sweep_training:
        logger.info("Long-sweep mode: disabling augmentation to preserve true timeline")
        train_sample_end_times_s = sample_end_times_s[train_idx]
        train_recording_ids = recording_ids[train_idx]
    else:
        # Apply data augmentation to training set (preictal samples only)
        X_train, y_train = _augment_train_dataset(X_train, y_train, config)
        train_sample_end_times_s = None
        train_recording_ids = None

    from src import SeizureDataset
    train_dataset = SeizureDataset(
        X_train,
        y_train,
        sample_end_times_s=train_sample_end_times_s,
        recording_ids=train_recording_ids,
    )
    val_dataset = SeizureDataset(
        X_val,
        y_val,
        sample_end_times_s=val_sample_end_times_s,
        recording_ids=val_recording_ids,
    )
    test_dataset = SeizureDataset(
        X_test,
        y_test,
        sample_end_times_s=test_sample_end_times_s,
        recording_ids=test_recording_ids,
    )

    logger.info(f"Real dataset prepared: total={n_total}, train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    logger.info(f"Train class distribution: {train_dataset.class_distribution}")
    logger.info(f"Val class distribution: {val_dataset.class_distribution}")
    logger.info(f"Test class distribution: {test_dataset.class_distribution}")

    return train_dataset, val_dataset, test_dataset


def _prepare_wearable_datasets(config: Config, args, loader: WearableDeviceDataLoader, recordings: List[dict]):
    """Build datasets from wearable recordings and absolute seizure-time annotations."""
    logger.info("Preparing REAL dataset from wearable CSV streams + master seizure annotations...")
    prep_start = time.time()
    # Keep a stable wearable channel order for downstream multimodal mapping:
    #   0: PPG, 1: ADXL, 2: EDA, 3: Temperature, 4: SQI.
    signal_types = ['ppg', 'adxl', 'eda', 'temperature', 'sqi']
    
    # Wearable-specific seizure-first filtering: cap purely non-seizure
    # recordings before real-mode selection to avoid loading hundreds of
    # long interictal recordings that add little preictal coverage.
    max_nonseiz = int(getattr(config.data, "max_wearable_nonseizure_recordings", 0) or 0)
    if args.long_sweep_training:
        max_nonseiz = 0
    if max_nonseiz > 0:
        seizure_recs = [r for r in recordings if int(r.get('num_seizures', 0)) > 0]
        nonseizure_recs = [r for r in recordings if int(r.get('num_seizures', 0)) <= 0]

        if len(nonseizure_recs) > max_nonseiz:
            rng = np.random.default_rng(int(config.data.random_seed))
            keep_non = list(rng.choice(nonseizure_recs, size=max_nonseiz, replace=False))
        else:
            keep_non = nonseizure_recs

        recordings = seizure_recs + keep_non
        logger.info(
            "Wearable selection: %d seizure recordings + %d/%d non-seizure recordings (max_nonseizure_recordings=%d)",
            len(seizure_recs),
            len(keep_non),
            len(nonseizure_recs),
            max_nonseiz,
        )

    recordings = _select_recordings_for_real_mode(recordings, args.max_recordings)
    seizure_recordings = sum(1 for recording in recordings if int(recording.get('num_seizures', 0)) > 0)
    logger.info(
        "Selected %d wearable recordings for real mode (%d with seizures)",
        len(recordings),
        seizure_recordings,
    )

    from src.models import ModelFactory as _MF
    is_multimodal = _MF.is_multimodal(config.model.model_type)

    ecg_dim = int(config.model.ecg_feature_dim)
    eeg_dim = int(config.model.eeg_feature_dim) if is_multimodal else 0
    motion_dim = int(config.model.motion_feature_dim) if is_multimodal else 0

    if is_multimodal:
        logger.info(
            "Wearable mode: constructing multimodal features from wearable channels (PPG+EDA+Temp+SQI+ADXL) with dims (ECG=%d, EEG=%d, motion=%d)",
            ecg_dim,
            eeg_dim,
            motion_dim,
        )
        feature_dim = ecg_dim + eeg_dim + motion_dim
    else:
        logger.info(
            "Wearable mode: constructing single-branch features from PPG with dim=%d",
            ecg_dim,
        )
        feature_dim = ecg_dim

    all_features = []
    all_labels = []
    all_sample_end_times_s = []
    all_recording_ids = []

    def _context_sample_indices(rec: dict, labels_arr: np.ndarray, sample_times_arr: np.ndarray, seed: int) -> np.ndarray:
        """Select memory-efficient samples around seizure context (5/10/20 min).

        Keeps all preictal windows when feasible and stratifies interictal
        windows by time-to-seizure bands to preserve clinically relevant
        context while reducing total memory footprint.
        """
        n = len(labels_arr)
        if n == 0:
            return np.array([], dtype=np.int64)

        rng = np.random.default_rng(seed)
        idx_all = np.arange(n, dtype=np.int64)

        # Non-seizure recordings: cap aggressively for memory.
        if int(rec.get('num_seizures', 0)) <= 0 or not rec.get('seizure_times'):
            cap_nonseiz = int(getattr(config.data, 'max_samples_per_recording_nonseizure', 400) or 400)
            if n <= cap_nonseiz:
                return idx_all
            return np.sort(rng.choice(idx_all, size=cap_nonseiz, replace=False)).astype(np.int64)

        start_dt = rec.get('start_datetime')
        if start_dt is None:
            return idx_all

        seizure_offsets = np.array(
            [(sz - start_dt).total_seconds() for sz in rec['seizure_times']],
            dtype=np.float64,
        )
        seizure_offsets = seizure_offsets[np.isfinite(seizure_offsets)]
        if seizure_offsets.size == 0:
            return idx_all

        sample_t = np.asarray(sample_times_arr, dtype=np.float64)

        # Min future time-to-seizure for each sample (seconds); inf means no future seizure.
        dt_future = np.full(sample_t.shape, np.inf, dtype=np.float64)
        for sz in seizure_offsets:
            dt = sz - sample_t
            valid = dt >= 0
            dt_future[valid] = np.minimum(dt_future[valid], dt[valid])

        # Bands requested by user: around seizure in 5 / 10 / 20 minute windows.
        idx_0_5 = np.where((dt_future >= 0) & (dt_future <= 5 * 60))[0]
        idx_5_10 = np.where((dt_future > 5 * 60) & (dt_future <= 10 * 60))[0]
        idx_10_20 = np.where((dt_future > 10 * 60) & (dt_future <= 20 * 60))[0]

        # Always include preictal labels if present.
        idx_preictal = np.where(labels_arr >= 0)[0]

        # Per-band caps to keep memory stable.
        cap_0_5 = int(getattr(config.data, 'max_samples_per_band_0_5min', 1200) or 1200)
        cap_5_10 = int(getattr(config.data, 'max_samples_per_band_5_10min', 1200) or 1200)
        cap_10_20 = int(getattr(config.data, 'max_samples_per_band_10_20min', 1200) or 1200)
        cap_far = int(getattr(config.data, 'max_samples_far_interictal', 800) or 800)

        def _pick(indices: np.ndarray, cap: int) -> np.ndarray:
            if indices.size <= cap:
                return indices.astype(np.int64)
            return np.sort(rng.choice(indices, size=cap, replace=False)).astype(np.int64)

        keep_parts = [
            idx_preictal.astype(np.int64),
            _pick(idx_0_5, cap_0_5),
            _pick(idx_5_10, cap_5_10),
            _pick(idx_10_20, cap_10_20),
        ]

        # Add a bounded sample of far interictal windows for background context.
        near_mask = np.zeros(n, dtype=bool)
        for arr in (idx_0_5, idx_5_10, idx_10_20):
            near_mask[arr] = True
        far_inter = np.where((labels_arr < 0) & (~near_mask))[0]
        keep_parts.append(_pick(far_inter, cap_far))

        keep = np.unique(np.concatenate([p for p in keep_parts if p.size > 0]))
        if keep.size == 0:
            return idx_all
        return keep.astype(np.int64)

    # Global safety cap for wearable datasets to prevent host OOM when
    # recordings are long and feature_step_s is small. Enforce this cap
    # *during* accumulation so that we never hold more than the budgeted
    # number of windows in memory at once.
    max_global = int(getattr(config.data, "max_wearable_global_samples", 0) or 0)
    total_kept = 0

    for rec_idx, recording in enumerate(recordings):
        recording_dir = recording.get('recording_dir', None)
        if recording_dir is None:
            continue

        try:
            features, labels, sample_times_s = loader.extract_features_from_recording(
                recording,
                signal_types=signal_types,
                window_s=float(config.data.feature_window_s),
            )
            if features is None or len(features) == 0:
                continue

            # For multimodal wearable runs, map stable wearable channels into
            # model branches:
            #   ECG branch   <- PPG
            #   EEG branch   <- fused (EDA, Temperature, SQI)
            #   Motion branch<- ADXL
            if is_multimodal:
                engineered = []
                for window in features:
                    base = np.asarray(window, dtype=np.float32)
                    if base.ndim == 1:
                        base = base[:, None]
                    elif base.ndim != 2:
                        base = base.reshape(base.shape[0], -1)

                    n_t = base.shape[0]

                    # Stable fallback helpers for recordings missing channels.
                    ppg_segment = base[:, 0] if base.shape[1] > 0 else np.zeros(n_t, dtype=np.float32)
                    adxl_segment = base[:, 1] if base.shape[1] > 1 else np.zeros(n_t, dtype=np.float32)

                    aux_parts = []
                    if base.shape[1] > 2:
                        aux_parts.append(base[:, 2])  # EDA
                    if base.shape[1] > 3:
                        aux_parts.append(base[:, 3])  # Temperature
                    if base.shape[1] > 4:
                        aux_parts.append(base[:, 4])  # SQI

                    if aux_parts:
                        aux_stack = np.stack(aux_parts, axis=0)
                        eeg_source = np.nanmean(aux_stack, axis=0).astype(np.float32)
                    else:
                        eeg_source = np.zeros(n_t, dtype=np.float32)

                    if not np.isfinite(eeg_source).all() or float(np.nanstd(eeg_source)) <= 1e-8:
                        eeg_source = ppg_segment.astype(np.float32)

                    ecg_features = _segment_to_feature_matrix(ppg_segment, ecg_dim, target_steps=600)
                    eeg_features = _segment_to_feature_matrix(eeg_source, eeg_dim, target_steps=600) if eeg_dim > 0 else None
                    motion_features = _segment_to_feature_matrix(adxl_segment, motion_dim, target_steps=600) if motion_dim > 0 else None

                    blocks = [ecg_features]
                    if eeg_features is not None:
                        blocks.append(eeg_features)
                    if motion_features is not None:
                        blocks.append(motion_features)

                    feature_mat = np.concatenate(blocks, axis=-1)
                    engineered.append(feature_mat)

                features = np.stack(engineered).astype(np.float32)
            else:
                # Single-branch wearable models use the aligned PPG feature
                # matrix directly.
                features = np.stack([
                    _align_feature_matrix(window, feature_dim=feature_dim, target_steps=600)
                    for window in features
                ]).astype(np.float32)

            # Optional cap per recording for faster iteration. Long-sweep mode
            # keeps full chronological coverage.
            if (not args.long_sweep_training) and args.max_samples_per_recording > 0 and len(features) > args.max_samples_per_recording:
                keep_positions = np.linspace(
                    0,
                    len(features) - 1,
                    num=args.max_samples_per_recording,
                    dtype=np.int64,
                )
                features = features[keep_positions]
                labels = labels[keep_positions]
                sample_times_s = sample_times_s[keep_positions]

            if not args.long_sweep_training:
                # Memory-optimized seizure-context sampling: 5/10/20 minute bands.
                keep_idx = _context_sample_indices(
                    recording,
                    np.asarray(labels, dtype=np.float32),
                    np.asarray(sample_times_s, dtype=np.float32),
                    seed=int(config.data.random_seed) + rec_idx,
                )
                if keep_idx.size > 0 and keep_idx.size < len(labels):
                    features = features[keep_idx]
                    labels = labels[keep_idx]
                    sample_times_s = sample_times_s[keep_idx]

            # Enforce global sample budget early: if we are at or beyond
            # the cap, stop processing further recordings. If this
            # recording would exceed the budget, subsample within this
            # recording while trying to preserve some preictal coverage.
            if (not args.long_sweep_training) and max_global > 0:
                remaining = max_global - total_kept
                if remaining <= 0:
                    logger.info(
                        "Reached wearable global sample budget (%d); skipping remaining recordings",
                        max_global,
                    )
                    break

                if len(labels) > remaining:
                    rng_local = np.random.default_rng(int(config.data.random_seed) + rec_idx)
                    idx_all = np.arange(len(labels), dtype=np.int64)
                    pre_idx = idx_all[labels >= 0]
                    inter_idx = idx_all[labels < 0]

                    if len(pre_idx) == 0 or remaining <= len(pre_idx):
                        chosen_local = rng_local.choice(idx_all, size=remaining, replace=False)
                    else:
                        target_pre = min(len(pre_idx), max(1, int(0.4 * remaining)))
                        target_inter = remaining - target_pre
                        chosen_pre = rng_local.choice(pre_idx, size=target_pre, replace=False)
                        if len(inter_idx) >= target_inter:
                            chosen_inter = rng_local.choice(inter_idx, size=target_inter, replace=False)
                        else:
                            chosen_inter = inter_idx
                        chosen_local = np.concatenate([chosen_pre, chosen_inter])
                        rng_local.shuffle(chosen_local)

                    features = features[chosen_local]
                    labels = labels[chosen_local]
                    sample_times_s = sample_times_s[chosen_local]

            rec_uid = f"{recording.get('subject_id', 'S?')}_{recording.get('watch_id', 'W?')}"

            all_features.append(features)
            all_labels.append(labels)
            all_sample_end_times_s.append(sample_times_s.astype(np.float32))
            all_recording_ids.extend([rec_uid] * len(labels))
            total_kept += len(labels)

            if rec_idx < 5 or (rec_idx + 1) % 5 == 0:
                logger.info(
                    "[WEARABLE PREP] Done %d/%d | +%d samples (total=%d)",
                    rec_idx + 1,
                    len(recordings),
                    len(labels),
                    sum(len(x) for x in all_labels),
                )
        except Exception as error:
            logger.debug(f"Skipping wearable recording due to load/process issue: {error}")
            continue

    if len(all_labels) == 0:
        logger.warning("Wearable dataset extraction produced no usable samples; falling back to dummy mode.")
        return None

    y_concat = np.concatenate(all_labels).astype(np.float32)
    if len(y_concat) < 100:
        logger.warning("Wearable dataset extraction produced too few samples; falling back to dummy mode.")
        return None

    X = np.concatenate(all_features, axis=0).astype(np.float32)
    y = y_concat
    sample_end_times_s = np.concatenate(all_sample_end_times_s, axis=0).astype(np.float32)
    recording_ids = np.array(all_recording_ids)

    # Global safety cap for wearable datasets to prevent host OOM when
    # recordings are long and feature_step_s is small. This trims the
    # total number of samples across all recordings while preserving
    # class balance as much as possible via random subsampling.
    max_global = int(getattr(config.data, "max_wearable_global_samples", 0) or 0)
    n_before_cap = len(y)
    if (not args.long_sweep_training) and max_global > 0 and n_before_cap > max_global:
        logger.warning(
            "Wearable dataset has %d samples; applying global cap max_wearable_global_samples=%d",
            n_before_cap,
            max_global,
        )
        rng = np.random.default_rng(int(config.data.random_seed))

        # Stratified subsampling to keep some preictal coverage.
        pre_idx = np.where(y >= 0)[0]
        inter_idx = np.where(y < 0)[0]

        if len(pre_idx) == 0 or max_global <= len(pre_idx):
            # No preictal or very tiny cap: plain random subset.
            chosen = rng.choice(n_before_cap, size=max_global, replace=False)
        else:
            # Reserve up to 20% of slots for preictal (but not exceeding
            # available preictal samples), remainder from interictal.
            target_pre = min(len(pre_idx), max(1, int(0.2 * max_global)))
            target_inter = max_global - target_pre

            chosen_pre = rng.choice(pre_idx, size=target_pre, replace=False)
            chosen_inter = (
                rng.choice(inter_idx, size=target_inter, replace=False)
                if len(inter_idx) >= target_inter
                else inter_idx
            )
            chosen = np.concatenate([chosen_pre, chosen_inter])
            rng.shuffle(chosen)

        X = X[chosen]
        y = y[chosen]
        sample_end_times_s = sample_end_times_s[chosen]
        recording_ids = recording_ids[chosen]

        logger.info(
            "Wearable global cap applied: %d → %d samples (preictal=%d, interictal=%d)",
            n_before_cap,
            len(y),
            int(np.sum(y >= 0)),
            int(np.sum(y < 0)),
        )

    preictal_count = int(np.sum(y >= 0))
    if preictal_count == 0:
        logger.error("Wearable dataset preparation produced zero preictal samples after alignment.")
        return None

    if args.long_sweep_training:
        train_idx, val_idx, test_idx = _split_indices_by_recording_timeline(
            y,
            recording_ids=recording_ids,
            sample_end_times_s=sample_end_times_s,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices_with_preictal_coverage(
            y,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_sample_end_times_s = sample_end_times_s[val_idx]
    val_recording_ids = recording_ids[val_idx]
    test_sample_end_times_s = sample_end_times_s[test_idx]
    test_recording_ids = recording_ids[test_idx]
    n_total = len(y)

    if args.long_sweep_training:
        logger.info("Long-sweep mode: disabling augmentation to preserve true timeline")
        train_sample_end_times_s = sample_end_times_s[train_idx]
        train_recording_ids = recording_ids[train_idx]
    else:
        X_train, y_train = _augment_train_dataset(X_train, y_train, config)
        train_sample_end_times_s = None
        train_recording_ids = None

    from src import SeizureDataset
    train_dataset = SeizureDataset(
        X_train,
        y_train,
        sample_end_times_s=train_sample_end_times_s,
        recording_ids=train_recording_ids,
    )
    val_dataset = SeizureDataset(
        X_val,
        y_val,
        sample_end_times_s=val_sample_end_times_s,
        recording_ids=val_recording_ids,
    )
    test_dataset = SeizureDataset(
        X_test,
        y_test,
        sample_end_times_s=test_sample_end_times_s,
        recording_ids=test_recording_ids,
    )

    logger.info(
        "Wearable dataset prepared in %.1f min: total=%d, train=%d, val=%d, test=%d",
        (time.time() - prep_start) / 60.0,
        n_total,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    logger.info(f"Train class distribution: {train_dataset.class_distribution}")
    logger.info(f"Val class distribution: {val_dataset.class_distribution}")
    logger.info(f"Test class distribution: {test_dataset.class_distribution}")

    return train_dataset, val_dataset, test_dataset


def _real_cache_path(config: Config, args) -> Path:
    """Create deterministic cache path for real-mode dataset extraction."""
    cache_root = Path('/tmp/epilepsee_ai_cache')
    cache_root.mkdir(parents=True, exist_ok=True)

    key_payload = {
        'data_source': str(args.data_source),
        'dataset_root': str(config.data.dataset_root),
        'model_type': config.model.model_type,
        'feature_dim': int(config.model.ecg_feature_dim),
        # Bump cache key when wearable signal composition changes (e.g.,
        # adding ADXL alongside PPG) so that a fresh real dataset is
        # prepared instead of reusing an incompatible NPZ.
        'wearable_signal_version': 3,
        'max_recordings': int(args.max_recordings),
        'max_samples_per_recording': int(args.max_samples_per_recording),
        'window_sampling_strategy_version': 5,
        'feature_step_s': float(config.data.feature_step_s),
        'pre_ictal_window_s': float(config.data.pre_ictal_window_s),
        'seed': int(config.data.random_seed),
        'train_ratio': float(config.data.train_ratio),
        'val_ratio': float(config.data.val_ratio),
        'augment_preictal': bool(config.data.augment_preictal),
        'augmentation_factor': int(config.data.augmentation_factor) if config.data.augment_preictal else 0,
        'long_sweep_training': bool(getattr(args, 'long_sweep_training', False)),
    }
    key = hashlib.md5(json.dumps(key_payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    return cache_root / f"real_dataset_{key}.npz"


def _save_real_dataset_cache(cache_path: Path, train_dataset, val_dataset, test_dataset) -> None:
    """Persist prepared datasets to NPZ cache."""
    tmp_path = cache_path.with_suffix('.tmp.npz')
    np.savez_compressed(
        tmp_path,
        train_features=train_dataset.features,
        train_labels=train_dataset.labels,
        val_features=val_dataset.features,
        val_labels=val_dataset.labels,
        val_sample_end_times_s=(np.array([], dtype=np.float32) if getattr(val_dataset, 'sample_end_times_s', None) is None else val_dataset.sample_end_times_s),
        val_recording_ids=(np.array([], dtype='<U1') if getattr(val_dataset, 'recording_ids', None) is None else np.asarray(val_dataset.recording_ids)),
        test_features=test_dataset.features,
        test_labels=test_dataset.labels,
        test_sample_end_times_s=(np.array([], dtype=np.float32) if getattr(test_dataset, 'sample_end_times_s', None) is None else test_dataset.sample_end_times_s),
        test_recording_ids=(np.array([], dtype='<U1') if getattr(test_dataset, 'recording_ids', None) is None else np.asarray(test_dataset.recording_ids)),
    )
    os.replace(tmp_path, cache_path)


def _load_real_dataset_cache(cache_path: Path):
    """Load cached real datasets."""
    from src import SeizureDataset

    # Use memory-mapped loading so multiple DDP ranks can share the
    # same underlying arrays without duplicating them in host RAM.
    data = np.load(cache_path, mmap_mode="r")
    train_dataset = SeizureDataset(data['train_features'], data['train_labels'])

    val_sample_end_times_s = data['val_sample_end_times_s'] if 'val_sample_end_times_s' in data else None
    val_recording_ids = data['val_recording_ids'] if 'val_recording_ids' in data else None
    if val_sample_end_times_s is not None and len(val_sample_end_times_s) == 0:
        val_sample_end_times_s = None
    if val_recording_ids is not None and len(val_recording_ids) == 0:
        val_recording_ids = None

    test_sample_end_times_s = data['test_sample_end_times_s'] if 'test_sample_end_times_s' in data else None
    test_recording_ids = data['test_recording_ids'] if 'test_recording_ids' in data else None
    if test_sample_end_times_s is not None and len(test_sample_end_times_s) == 0:
        test_sample_end_times_s = None
    if test_recording_ids is not None and len(test_recording_ids) == 0:
        test_recording_ids = None

    val_dataset = SeizureDataset(
        data['val_features'],
        data['val_labels'],
        sample_end_times_s=val_sample_end_times_s,
        recording_ids=val_recording_ids,
    )
    test_dataset = SeizureDataset(
        data['test_features'],
        data['test_labels'],
        sample_end_times_s=test_sample_end_times_s,
        recording_ids=test_recording_ids,
    )
    return train_dataset, val_dataset, test_dataset


def prepare_datasets(config: Config, args):
    """Prepare train/val/test datasets.
    
    Args:
        config: Config object
    
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    logger.info("Loading dataset...")
    
    if args.data_mode == 'real':
        try:
            if args.data_source == 'wearable':
                wearable_loader = WearableDeviceDataLoader(
                    config.data,
                    dataset_root=config.data.dataset_root,
                )
                recordings = wearable_loader.get_all_recordings()
                logger.info("Data source: wearable")
            else:
                loader = BIDSDataLoader(config.data)
                recordings = loader.list_all_recordings()
                logger.info("Data source: bids")

            logger.info(f"Total recordings: {len(recordings)}")
            logger.info(f"Total seizures: {sum(int(r.get('num_seizures', 0)) for r in recordings)}")
        except FileNotFoundError as exc:
            if args.strict_real_data:
                raise FileNotFoundError(
                    "Real dataset is required but unavailable. "
                    f"dataset_root='{config.data.dataset_root}'. "
                    "Fix mount/path or remove --strict-real-data."
                ) from exc
            logger.warning(
                "Real dataset unavailable at '%s' (%s). Falling back to dummy mode. "
                "Use --dataset-root to point to the correct location.",
                config.data.dataset_root,
                exc,
            )
            args.data_mode = 'dummy'

    if args.data_mode == 'real':
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        cache_path = _real_cache_path(config, args)

        if cache_path.exists():
            logger.info(f"Loading real dataset from cache: {cache_path}")
            return _load_real_dataset_cache(cache_path)

        if world_size == 1 or rank == 0:
            if args.data_source == 'wearable':
                prepared = _prepare_wearable_datasets(config, args, wearable_loader, recordings)
            else:
                prepared = _prepare_real_datasets(config, args, loader, recordings)
            if prepared is not None:
                _save_real_dataset_cache(cache_path, *prepared)
                logger.info(f"Saved real dataset cache: {cache_path}")
                return prepared
            if args.strict_real_data:
                raise RuntimeError(
                    f"Real {args.data_source} dataset preparation failed to produce a usable train/val/test split."
                )
        else:
            logger.info(f"Rank {rank} waiting for rank 0 dataset cache: {cache_path}")
            wait_seconds = 0
            timeout_seconds = 7200
            while wait_seconds < timeout_seconds and not cache_path.exists():
                time.sleep(5)
                wait_seconds += 5
                if wait_seconds % 60 == 0:
                    logger.info(
                        "Rank %d still waiting for dataset cache (%ds elapsed)...",
                        rank,
                        wait_seconds,
                    )
            if cache_path.exists():
                logger.info(f"Rank {rank} loading real dataset cache after wait: {cache_path}")
                return _load_real_dataset_cache(cache_path)
            logger.warning(f"Rank {rank} timed out waiting for dataset cache, using dummy fallback.")

    logger.warning("Using dummy datasets for testing.")
    
    # Create dummy data (for pipeline verification only)
    n_train = args.dummy_train_size
    n_val = args.dummy_val_size
    n_test = args.dummy_test_size
    preictal_ratio = float(np.clip(args.dummy_preictal_ratio, 0.0, 1.0))

    from src.models import ModelFactory as _MF
    if _MF.is_multimodal(config.model.model_type):
        feature_dim = (
            config.model.ecg_feature_dim
            + config.model.eeg_feature_dim
            + config.model.motion_feature_dim
        )
    else:
        feature_dim = config.model.ecg_feature_dim
    
    def make_labels(num_samples: int, ratio: float) -> np.ndarray:
        num_preictal = int(num_samples * ratio)
        num_interictal = num_samples - num_preictal
        preictal = np.random.uniform(0.0, 10.0, size=num_preictal)
        interictal = -np.ones(num_interictal, dtype=np.float32)
        labels = np.concatenate([preictal, interictal]).astype(np.float32)
        np.random.shuffle(labels)
        return labels

    X_train = np.random.randn(n_train, 600, feature_dim).astype(np.float32)
    y_train = make_labels(n_train, preictal_ratio)
    
    X_val = np.random.randn(n_val, 600, feature_dim).astype(np.float32)
    y_val = make_labels(n_val, preictal_ratio)
    
    X_test = np.random.randn(n_test, 600, feature_dim).astype(np.float32)
    y_test = make_labels(n_test, preictal_ratio)
    
    # Import SeizureDataset
    from src import SeizureDataset
    
    train_dataset = SeizureDataset(X_train, y_train)
    val_dataset = SeizureDataset(X_val, y_val)
    test_dataset = SeizureDataset(X_test, y_test)
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    logger.info(f"Dummy preictal ratio target: {preictal_ratio:.2f}")
    logger.info(f"Train class distribution: {train_dataset.class_distribution}")
    logger.info(f"Val class distribution: {val_dataset.class_distribution}")
    logger.info(f"Test class distribution: {test_dataset.class_distribution}")
    
    return train_dataset, val_dataset, test_dataset


def main():
    """Main training function."""
    args = parse_args()

    # Normalize local rank from CLI/env for torchrun/launch compatibility
    local_rank = args.local_rank
    if local_rank is None:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    if args.config:
        config = Config.from_yaml(args.config)
    else:
        config = DEFAULT_CONFIG
    
    # Override with command line args
    config.model.model_type = args.model_type
    config.training.batch_size = args.batch_size
    config.training.num_epochs = args.epochs
    config.training.learning_rate = args.learning_rate
    if args.real_stride_seconds is not None:
        config.data.feature_step_s = float(args.real_stride_seconds)
    if args.long_sweep_training:
        config.training.long_sweep_training = True
        config.training.use_weighted_sampling = False
        config.data.augment_preictal = False
        if args.max_samples_per_recording > 0:
            logger.info(
                "Long-sweep mode enabled: ignoring --max-samples-per-recording=%d to preserve full timeline",
                int(args.max_samples_per_recording),
            )
    if args.dataset_root:
        config.data.dataset_root = args.dataset_root
    elif args.data_source == 'wearable' and 'ds005873' in str(config.data.dataset_root):
        # Prefer env var; warn if not set so users know they must provide the path
        wearable_root = os.environ.get('WEARABLE_DATASET_ROOT', '')
        if not wearable_root:
            logger.error(
                "Wearable dataset path not set. "
                "Pass --dataset-root or set the WEARABLE_DATASET_ROOT environment variable."
            )
            raise SystemExit(1)
        config.data.dataset_root = wearable_root
        logger.info("Auto-set wearable dataset_root to %s", config.data.dataset_root)
    # Propagate data_source into config for downstream components
    config.data.data_source = args.data_source
    
    # Device
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device('cpu')
        config.training.distributed = False
    else:
        device = torch.device(f'cuda:{local_rank}')
        torch.cuda.set_device(local_rank)
    
    logger.info("="*60)
    logger.info("SEIZURE ANTICIPATION TRAINING")
    logger.info("="*60)
    logger.info(f"Model type: {config.model.model_type}")
    logger.info(f"Device: {device}")
    logger.info(f"Local rank: {local_rank}")
    logger.info(f"Global rank: {os.environ.get('RANK', 0)}")
    logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
    logger.info(f"Distributed training: {config.training.distributed and torch.cuda.device_count() > 1}")
    
    # Prepare datasets
    train_dataset, val_dataset, test_dataset = prepare_datasets(config, args)
    
    # Create trainer
    trainer = Trainer(config, device=device)
    
    # Train
    logger.info("Starting training...")
    results = trainer.train(train_dataset, val_dataset, test_dataset)

    is_main_process = int(os.environ.get('RANK', 0)) == 0
    if config.training.distributed and not is_main_process:
        if dist.is_initialized():
            dist.destroy_process_group()
        return 0
    
    logger.info(f"Training completed!")
    logger.info(f"Best validation MAE: {results['best_val_metric']:.4f}")
    logger.info(f"Total training time: {results['training_time']/3600:.2f} hours")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    evaluator = Evaluator(config)
    
    # Load best model
    checkpoint = torch.load(trainer.save_dir / "best_model.pt", map_location=device, weights_only=False)
    from src import ModelFactory
    model = ModelFactory.create_model(config.model)
    model.load_state_dict(checkpoint['model_state'])
    model = model.to(device)
    
    # Evaluate
    test_metrics = evaluator.evaluate(model, test_dataset, device)

    # Collect full predictions for artifact export
    prediction_payload = evaluator.collect_predictions(model, test_dataset, device)

    bayes_sweep_payload = None
    test_times = getattr(test_dataset, 'sample_end_times_s', None)
    test_rec_ids = getattr(test_dataset, 'recording_ids', None)
    if args.long_sweep_training and test_times is not None and test_rec_ids is not None:
        try:
            bayes_sweep_payload = evaluator.simulate_bayesian_long_sweep(
                pred_preictal=prediction_payload['pred_preictal'],
                pred_countdown=prediction_payload['pred_countdown'],
                true_countdown=prediction_payload['true_countdown'],
                sample_end_times_s=np.asarray(test_times),
                recording_ids=np.asarray(test_rec_ids),
            )
            if 'metrics' in bayes_sweep_payload:
                test_metrics.update(bayes_sweep_payload['metrics'])
                logger.info(
                    "Long-sweep Bayesian memory simulation complete | bayes_auroc=%.4f bayes_ece=%.4f",
                    float(test_metrics.get('bayes_auroc', 0.0)),
                    float(test_metrics.get('bayes_ece', 0.0)),
                )
        except Exception as bayes_error:
            logger.warning("Bayesian long-sweep simulation skipped due to error: %s", str(bayes_error))

    threshold_sweep = None
    if args.auto_threshold:
        threshold_sweep = evaluator.select_optimal_threshold(
            prediction_payload['pred_preictal'],
            prediction_payload['true_preictal'],
            objective=args.threshold_objective,
            min_sensitivity=float(np.clip(args.threshold_min_sensitivity, 0.0, 1.0)),
            max_fpr=float(np.clip(args.threshold_max_fpr, 0.0, 1.0)),
        )
        selected_threshold = float(threshold_sweep['threshold'])
        config.loss.detection_threshold = selected_threshold
        logger.info(
            "Auto-threshold selected: %.3f (objective=%s, sens=%.3f, spec=%.3f, fpr=%.3f)",
            selected_threshold,
            args.threshold_objective,
            threshold_sweep['sensitivity'],
            threshold_sweep['specificity'],
            threshold_sweep['fpr'],
        )

        updated_cls = evaluator._compute_classification_metrics(
            prediction_payload['pred_preictal'],
            prediction_payload['true_preictal'],
        )
        for key in ['accuracy', 'auroc', 'f1_score', 'sensitivity', 'specificity', 'precision']:
            if key in updated_cls:
                test_metrics[key] = updated_cls[key]
    
    # Print results
    evaluator.print_results(test_metrics)
    
    # Save results
    results['test_metrics'] = test_metrics
    if threshold_sweep is not None:
        results['threshold_sweep'] = threshold_sweep

    # Save machine-readable artifacts
    results_path = trainer.save_dir / "results.json"
    with open(results_path, "w", encoding="utf-8") as file_handle:
        json.dump(_to_jsonable(results), file_handle, indent=2)

    predictions_path = trainer.save_dir / "test_predictions.npz"
    np.savez_compressed(
        predictions_path,
        pred_preictal=prediction_payload['pred_preictal'],
        pred_countdown=prediction_payload['pred_countdown'],
        true_preictal=prediction_payload['true_preictal'],
        true_countdown=prediction_payload['true_countdown'],
    )

    if bayes_sweep_payload is not None:
        bayes_predictions_path = trainer.save_dir / "test_predictions_long_sweep_bayes.npz"
        np.savez_compressed(
            bayes_predictions_path,
            fused_preictal=bayes_sweep_payload['fused_preictal'],
            fused_preictal_smooth=bayes_sweep_payload['fused_preictal_smooth'],
            memory_risk=bayes_sweep_payload['memory_risk'],
            uncertainty=bayes_sweep_payload['uncertainty'],
            token_id=bayes_sweep_payload['token_id'],
            timeline_order_idx=bayes_sweep_payload['timeline_order_idx'],
            recording_ids=np.asarray(test_rec_ids),
            sample_end_times_s=np.asarray(test_times),
        )

    # Save summary performance figure (classification + countdown)
    perf_plot_path = trainer.save_dir / "performance_overview.png"
    plot_evaluator = Evaluator(config)
    plot_evaluator.plot_classification_and_countdown(
        prediction_payload['pred_preictal'],
        prediction_payload['pred_countdown'],
        prediction_payload['true_preictal'],
        prediction_payload['true_countdown'],
        save_path=perf_plot_path,
    )

    # Save GT-vs-inference panel and confusion matrix-focused figure
    visualizer = SignalVisualizer(save_dir=str(trainer.save_dir), upload_to_wandb=False)

    # Use waveform-like signal proxy aligned with per-window predictions.
    # For wearable runs, treat feature channel 0 as the PPG-derived
    # heart-rate series; for BIDS runs, choose the most dynamic channel.
    features_arr = np.asarray(test_dataset.features, dtype=np.float32)
    mid_idx = features_arr.shape[1] // 2

    data_source = getattr(config.data, "data_source", "bids")
    if data_source == "wearable":
        best_ch = 0 if features_arr.shape[-1] > 0 else 0
    else:
        flat = features_arr.reshape(-1, features_arr.shape[-1])
        var_per_channel = flat.std(axis=0)
        best_ch = int(np.argmax(var_per_channel)) if features_arr.shape[-1] > 0 else 0

    signal_proxy = features_arr[:, mid_idx, best_ch]

    panel_ppg_full = None
    panel_eda_full = None
    panel_eeg_full = None
    if features_arr.shape[-1] > 0:
        panel_ppg_full = features_arr[:, mid_idx, 0].astype(np.float32)
    if data_source == "wearable" and features_arr.shape[-1] > 2:
        panel_eda_full = features_arr[:, mid_idx, 2].astype(np.float32)

    if config.model.model_type in ('multimodal', 'multimodal_transformer'):
        ecg_dim = int(config.model.ecg_feature_dim)
        eeg_dim = int(config.model.eeg_feature_dim)
        if features_arr.shape[-1] >= ecg_dim + max(1, eeg_dim):
            panel_eeg_full = features_arr[:, mid_idx, ecg_dim].astype(np.float32)

    # Window-wise statistics for HR/HRV-style traces.
    seq_first_channel = features_arr[:, :, best_ch]

    if getattr(config.data, "data_source", "bids") == "wearable":
        # Wearable runs: interpret the selected feature channel as a
        # heart-rate time series derived from the PPG stream. Use raw
        # per-window mean/std so traces reflect actual dataset dynamics
        # without forcing them into a fixed bpm range.
        panel_hr_full = seq_first_channel.mean(axis=1).astype(np.float32)
        panel_hrv_full = seq_first_channel.std(axis=1).astype(np.float32)
    else:
        # BIDS ECG runs: map normalized statistics into synthetic
        # physiological ranges for visually informative HR/HRV traces.
        norm_mean = seq_first_channel.mean(axis=1).astype(np.float32)
        norm_std = seq_first_channel.std(axis=1).astype(np.float32)

        panel_hr_full = 60.0 + 30.0 * norm_mean
        panel_hr_full = np.clip(panel_hr_full, 30.0, 180.0).astype(np.float32)

        panel_hrv_full = 20.0 + 40.0 * norm_std
        panel_hrv_full = np.clip(panel_hrv_full, 0.0, 250.0).astype(np.float32)

    panel_indices, panel_time_minutes = _select_panel_indices(
        prediction_payload['true_countdown'],
        feature_step_s=float(config.data.feature_step_s),
        window_minutes=float(args.panel_window_minutes),
        seed=args.seed,
        sample_end_times_s=getattr(test_dataset, 'sample_end_times_s', None),
        recording_ids=getattr(test_dataset, 'recording_ids', None),
    )

    if len(panel_time_minutes) > 1:
        panel_span_minutes = float(panel_time_minutes[-1] - panel_time_minutes[0])
        panel_effective_hz = float(len(panel_time_minutes) - 1) / max(panel_span_minutes * 60.0, 1e-6)
        if panel_effective_hz < 0.2:
            logger.warning(
                "Panel timeline is sparse (effective %.3f Hz across %.1f min, %d points). "
                "For denser plots, increase sampling density in dataset prep "
                "(e.g., MAX_SAMPLES_PER_RECORDING=0).",
                panel_effective_hz,
                panel_span_minutes,
                len(panel_time_minutes),
            )

    # Extract panel-aligned ECG proxy and HR/HRV proxies. For BIDS ECG runs
    # we standardize for contrast; for wearable runs keep synthetic
    # physiological units so traces are not centered at zero.
    panel_ecg_raw = signal_proxy[panel_indices]
    panel_ppg_raw = panel_ppg_full[panel_indices] if panel_ppg_full is not None else panel_ecg_raw
    panel_eda_raw = panel_eda_full[panel_indices] if panel_eda_full is not None else None
    panel_eeg_raw = panel_eeg_full[panel_indices] if panel_eeg_full is not None else None
    panel_hr_raw = panel_hr_full[panel_indices]
    panel_hrv_raw = panel_hrv_full[panel_indices]

    def _standardize_for_plot(series: np.ndarray) -> np.ndarray:
        arr = np.asarray(series, dtype=np.float32)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        if not np.isfinite(std) or std < 1e-6:
            std = 1.0
        return (arr - mean) / std

    if data_source == "wearable":
        # For wearable runs, build the top "Signal" trace from *all*
        # available wearable feature channels (PPG HR + motion, etc.) so
        # that the panel reflects the full set of wearable dynamics rather
        # than a single averaged channel. We collapse the time axis within
        # each window via a mean per feature and then standardize across
        # the resulting feature matrix so multiple channels can be
        # overlaid in the top subplot.
        if features_arr.ndim == 3 and features_arr.shape[-1] > 0:
            # (N_panel, F) = mean over time within each window for all
            # feature channels, including PPG HR and ADXL-derived motion.
            panel_feats = features_arr[panel_indices].mean(axis=1).astype(np.float32)
            panel_ecg = _standardize_for_plot(panel_feats)
        else:
            # Fallback: use HR proxy only.
            panel_ecg = _standardize_for_plot(panel_hr_raw)
        panel_hr = panel_hr_raw.astype(np.float32)
        panel_hrv = panel_hrv_raw.astype(np.float32)
    else:
        panel_ecg = _standardize_for_plot(panel_ecg_raw)
        panel_hr = _standardize_for_plot(panel_hr_raw)
        panel_hrv = _standardize_for_plot(panel_hrv_raw)

    panel_ppg = _standardize_for_plot(panel_ppg_raw)
    panel_eda = _standardize_for_plot(panel_eda_raw) if panel_eda_raw is not None else None
    panel_eeg = _standardize_for_plot(panel_eeg_raw) if panel_eeg_raw is not None else None

    panel_true_countdown = prediction_payload['true_countdown'][panel_indices]
    panel_pred_countdown = prediction_payload['pred_countdown'][panel_indices]
    panel_pred_preictal = prediction_payload['pred_preictal'][panel_indices]
    if bayes_sweep_payload is not None:
        panel_pred_smooth = bayes_sweep_payload['fused_preictal_smooth'][panel_indices]
    else:
        panel_pred_smooth = (
            prediction_payload['pred_preictal_smooth'][panel_indices]
            if 'pred_preictal_smooth' in prediction_payload else None
        )

    display_sampling_hz = 1.0 / max(float(config.data.feature_step_s), 1e-6)

    # Human-readable dataset tag for figure title/footer.
    if data_source == "wearable":
        dataset_tag = "Wearable (OHSU VSM)"
    else:
        dataset_tag = "SeizeIT2 (BIDS)"

    # Transparent metadata: if the test dataset carries recording IDs and
    # sample end-times, summarize which recording(s) and time ranges this
    # panel spans. This makes it easy to trace flat or surprising behavior
    # back to specific wearable/BIDS segments.
    panel_footer = (
        f"Dataset: {dataset_tag} | Model: {config.model.model_type} | "
        f"Threshold: {config.loss.detection_threshold:.2f} | "
        f"n_test={len(prediction_payload['true_preictal'])} | "
        f"window={float(args.panel_window_minutes):.0f}min | "
        f"Signal: {'PPG proxy' if data_source == 'wearable' else 'ECG proxy'}"
    )
    rec_ids_meta = getattr(test_dataset, 'recording_ids', None)
    times_meta = getattr(test_dataset, 'sample_end_times_s', None)
    if rec_ids_meta is not None and times_meta is not None and len(rec_ids_meta) == len(prediction_payload['true_countdown']):
        rec_ids_meta = np.asarray(rec_ids_meta)
        times_meta = np.asarray(times_meta, dtype=np.float64)
        rec_ids_panel = rec_ids_meta[panel_indices]
        times_panel = times_meta[panel_indices]
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
            logger.info("Final panel sources: %s", "; ".join(pieces))

    # Log basic stats of the panel signals so we can see whether they are
    # truly flat or just low-dynamic-range for this selection.
    logger.info(
        "Final panel signal stats | ppg: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "eda: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "eeg: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "ecg: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "HR: min=%.3f max=%.3f mean=%.3f std=%.3f | HRV: min=%.3f max=%.3f mean=%.3f std=%.3f",
        float(panel_ppg.min()), float(panel_ppg.max()), float(panel_ppg.mean()), float(panel_ppg.std()),
        float(panel_eda.min()) if panel_eda is not None else 0.0,
        float(panel_eda.max()) if panel_eda is not None else 0.0,
        float(panel_eda.mean()) if panel_eda is not None else 0.0,
        float(panel_eda.std()) if panel_eda is not None else 0.0,
        float(panel_eeg.min()) if panel_eeg is not None else 0.0,
        float(panel_eeg.max()) if panel_eeg is not None else 0.0,
        float(panel_eeg.mean()) if panel_eeg is not None else 0.0,
        float(panel_eeg.std()) if panel_eeg is not None else 0.0,
        float(panel_ecg.min()), float(panel_ecg.max()), float(panel_ecg.mean()), float(panel_ecg.std()),
        float(panel_hr.min()), float(panel_hr.max()), float(panel_hr.mean()), float(panel_hr.std()),
        float(panel_hrv.min()), float(panel_hrv.max()), float(panel_hrv.mean()), float(panel_hrv.std()),
    )

    panel_fig = visualizer.plot_gt_vs_inference_panel(
        ecg_signal=panel_ecg,
        ppg_signal=panel_ppg,
        eda_signal=panel_eda,
        eeg_signal=panel_eeg,
        hr_series=panel_hr,
        hrv_series=panel_hrv,
        true_countdown=panel_true_countdown,
        pred_countdown=panel_pred_countdown,
        pred_preictal_prob=panel_pred_preictal,
        pred_preictal_smooth=panel_pred_smooth,
        cm_true_countdown=prediction_payload['true_countdown'],
        cm_pred_preictal_prob=prediction_payload['pred_preictal'],
        sampling_rate=display_sampling_hz,
        detection_threshold=config.loss.detection_threshold,
        time_axis_minutes=panel_time_minutes,
        title_prefix=f"{config.model.model_type} — {dataset_tag}",
        footer_text=panel_footer,
        signal_label='Wearable signals (PPG+ADXL, z-score)' if data_source == 'wearable' else 'ECG proxy',
    )
    visualizer.save_figure(panel_fig, "gt_vs_inference_panel")

    # ── Train / Val / Test confusion matrices ──────────────────────────────────
    import matplotlib.pyplot as _plt
    logger.info("Generating train/val/test confusion matrices...")
    threshold = float(config.loss.detection_threshold)

    for split_name, split_dataset in [
        ('train', train_dataset),
        ('val',   val_dataset),
        ('test',  test_dataset),
    ]:
        split_preds = evaluator.collect_predictions(model, split_dataset, device)
        split_true_preictal = split_preds['true_preictal']  # already int32 0/1
        split_cm_fig = visualizer.plot_confusion_matrix(
            true_preictal=split_true_preictal,
            pred_preictal_prob=split_preds['pred_preictal'],
            threshold=threshold,
            split_label=split_name.capitalize(),
        )
        visualizer.save_figure(split_cm_fig, f"{split_name}_confusion_matrix")
        _plt.close(split_cm_fig)
        logger.info(
            "Saved %s confusion matrix  (n=%d  preictal=%d)",
            split_name,
            len(split_true_preictal),
            int(split_true_preictal.sum()),
        )

    # Save config
    config.save_yaml(trainer.save_dir / "config.yaml")

    if args.auto_threshold:
        threshold_path = trainer.save_dir / "threshold_selection.json"
        with open(threshold_path, "w", encoding="utf-8") as file_handle:
            json.dump(_to_jsonable(threshold_sweep), file_handle, indent=2)
        logger.info(f"Threshold sweep saved to {threshold_path}")

        if args.write_threshold_to_config:
            config.save_yaml(trainer.save_dir / "config.yaml")
            logger.info(
                "Persisted detection_threshold=%.3f to %s",
                config.loss.detection_threshold,
                trainer.save_dir / "config.yaml",
            )

    # Generate and upload token embedding explainability visualization to W&B
    try:
        import subprocess
        logger.info("Generating token embedding explainability visualization...")
        
        predictions_npz = str(trainer.save_dir / "test_predictions.npz")
        bayes_npz = str(trainer.save_dir / "test_predictions_long_sweep_bayes.npz")
        explainability_dir = str(trainer.save_dir / "visualizations" / "explainability")
        
        cmd = [
            "python", "scripts/visualize_token_explainability.py",
            "--predictions-npz", predictions_npz,
            "--bayes-npz", bayes_npz,
            "--output-dir", explainability_dir,
            "--n-countdown-bins", "8",
            "--n-prob-bins", "4",
            "--max-countdown-min", "10.0",
            "--rolling-window", "25",
            "--tokenization-scheme", "auto",
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0:
            logger.info("Token embedding visualization generated successfully")
            
            # Upload PNG to W&B if available
            try:
                import wandb
                if wandb.run is not None:
                    token_fig_path = Path(explainability_dir) / "token_embedding_explainability.png"
                    if token_fig_path.exists():
                        try:
                            wandb.log({
                                "visualizations/token_embedding_explainability": wandb.Image(str(token_fig_path))
                            })
                            logger.info(f"Uploaded token embedding visualization to W&B: {token_fig_path}")
                        except Exception as e:
                            logger.warning(f"Failed to log to W&B: {e}")
            except ImportError:
                logger.debug("wandb not available, skipping W&B upload")
        else:
            logger.warning(f"Token visualization generation failed: {result.stderr}")
    except Exception as e:
        logger.warning(f"Token embedding visualization skipped: {e}")
    
    logger.info(f"Results saved to {trainer.save_dir}")

    if dist.is_initialized():
        dist.destroy_process_group()

    return 0


if __name__ == '__main__':
    sys.exit(main())
