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
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from pathlib import Path
from typing import Tuple, Optional, Dict
import numpy as np
from collections import defaultdict
import time

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from config.config import Config, TrainingConfig
from src.models import ModelFactory
from src.losses import LossFactory
from src.data_loader import SeizureDataset
from src.visualization import SignalVisualizer


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


class Trainer:
    """
    Training engine with distributed data parallel support for multi-GPU training.
    """
    
    def __init__(self, config: Config, device: torch.device = None, use_wandb: bool = True):
        """Initialize trainer.
        
        Args:
            config: Master Config object
            device: torch device (if None, auto-select based on availability)
            use_wandb: Whether to use Weights & Biases logging
        """
        self.config = config
        self.training_config = config.training
        self.use_wandb = use_wandb and HAS_WANDB
        
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
        if self.use_wandb and self._is_main_process():
            self._init_wandb()

        self.epoch_visualizer = None
        if self._is_main_process():
            self.epoch_visualizer = SignalVisualizer(
                save_dir=str(self.save_dir / "epoch_visualizations"),
                upload_to_wandb=self.use_wandb,
            )
        
        logger.info(f"Trainer initialized. Rank: {self.rank}/{self.world_size}")
    
    def _setup_distributed(self):
        """Initialize distributed training environment."""
        if not dist.is_initialized():
            dist.init_process_group(
                backend=self.training_config.backend,
                init_method='env://',
                world_size=int(os.environ.get('WORLD_SIZE', 1)),
                rank=int(os.environ.get('RANK', 0))
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
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        if not HAS_WANDB:
            logger.warning("wandb not available, skipping initialization")
            return
        
        try:
            # Initialize wandb run
            self.wandb_run = wandb.init(
                project="epilepsee-ai",
                name=f"{self.config.model.model_type}_{int(time.time())}",
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

    def _forward_model(self, model: nn.Module, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
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
            return model(ecg_x, eeg_x, motion_x)

        return model(features)

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
    
    def train_epoch(self, model: nn.Module, train_loader: DataLoader,
                   criterion: nn.Module, optimizer: optim.Optimizer,
                   epoch: int) -> Tuple[Dict[str, float], Dict[str, np.ndarray]]:
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
        num_batches = 0
        all_pred_preictal = []
        all_true_preictal = []
        
        # Update sampler for distributed training
        if hasattr(train_loader.sampler, 'set_epoch'):
            train_loader.sampler.set_epoch(epoch)
        
        train_iter = self._progress(
            enumerate(train_loader),
            desc=f"Train Epoch {epoch}",
            total=len(train_loader)
        )

        # Track last gradient norm for debugging numerical issues
        last_grad_norm: Optional[float] = None

        for batch_idx, batch in train_iter:
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
            
            # Backward pass
            optimizer.zero_grad()
            loss_dict['total'].backward()

            # Sanitize gradients to avoid propagating NaNs/Infs into
            # optimizer state. This is a last line of defence in case
            # some rare batch produces unstable gradients while the
            # forward pass and loss remain finite.
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
                        # Optionally clamp extremely large cleaned gradients
                        clean_grad = torch.clamp(clean_grad, -1000.0, 1000.0)
                        logger.error(
                            "Non-finite gradient detected; sanitizing before optimizer step "
                            f"(epoch={epoch}, batch={batch_idx}, param={name}, "
                            f"max_abs={float(clean_grad.abs().max()):.4e})"
                        )
                        param.grad.copy_(clean_grad)

            # Gradient clipping (also capture norm for optional logging)
            if self.training_config.gradient_clip_norm > 0:
                last_grad_norm = float(
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        self.training_config.gradient_clip_norm
                    )
                )
            
            optimizer.step()

            # Sanity-check model parameters after optimizer step to catch
            # the earliest sign of numerical instability (NaNs/Infs).
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if param is None or param.data.numel() == 0:
                        continue
                    if not torch.isfinite(param.data).all():
                        clean = torch.nan_to_num(param.data)
                        logger.error(
                            "Non-finite parameter after optimizer step "
                            f"(epoch={epoch}, batch={batch_idx}, param={name}, "
                            f"min={float(clean.min()):.4e}, max={float(clean.max()):.4e})"
                        )
                        raise RuntimeError(
                            "Non-finite model parameters encountered after optimizer step; "
                            "aborting training. See logs for diagnostics."
                        )
            
            # Accumulate metrics
            num_batches += 1
            for key, value in loss_dict.items():
                metrics[f'train_{key}'] += float(value.detach().cpu())
            all_pred_preictal.extend(pre_ictal_pred.detach().cpu().numpy())
            all_true_preictal.extend(pre_ictal_labels.detach().cpu().numpy())
            
            # Log batch
            if batch_idx % self.training_config.log_interval == 0 and self._is_main_process():
                avg_loss = metrics['train_total'] / num_batches
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
                    })
                
                # Log to wandb
                    if self.wandb_run:
                        log_payload = {
                            "epoch": epoch,
                            "batch": batch_idx,
                            "train/batch_loss": avg_loss,
                            "train/learning_rate": optimizer.param_groups[0]['lr'],
                        }
                        if last_grad_norm is not None:
                            log_payload["train/grad_norm"] = last_grad_norm
                        wandb.log(log_payload)
        
        # Average metrics over epoch
        for key in metrics:
            metrics[key] /= num_batches

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
        sample_end_times_s = getattr(val_dataset, 'sample_end_times_s', None) if val_dataset is not None else None
        recording_ids = getattr(val_dataset, 'recording_ids', None) if val_dataset is not None else None
        if (
            sample_end_times_s is not None
            and recording_ids is not None
            and len(sample_end_times_s) == len(all_true_countdown)
            and len(recording_ids) == len(all_true_countdown)
        ):
            viz_payload['sample_end_times_s'] = np.asarray(sample_end_times_s, dtype=np.float32)
            viz_payload['recording_ids'] = np.asarray(recording_ids)

        return dict(metrics), viz_payload

    def _save_epoch_visualizations(self, epoch: int, viz_payload: Dict[str, np.ndarray],
                                   train_viz_payload: Optional[Dict[str, np.ndarray]] = None) -> None:
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

        # Prefer explicit ECG/EEG proxies when available, but fall back to
        # the original single-channel signal_proxy for backward compatibility.
        ecg_proxy = viz_payload.get('ecg_proxy', viz_payload.get('signal_proxy'))
        ppg_proxy = viz_payload.get('ppg_proxy', ecg_proxy)
        eda_proxy = viz_payload.get('eda_proxy', None)
        eeg_proxy = viz_payload.get('eeg_proxy', None)
        adxl_proxy = viz_payload.get('adxl_proxy', None)
        hr_proxy = viz_payload.get('hr_proxy', np.zeros_like(ecg_proxy, dtype=np.float32))
        hrv_proxy = viz_payload.get('hrv_proxy', np.zeros_like(ecg_proxy, dtype=np.float32))

        # Prefer a contiguous segment from a single recording that includes a
        # seizure (when metadata is available and we are not in distributed
        # mode). This matches clinical expectations: 1–5–10 minutes of
        # continuous context around a seizure, not a shuffled concatenation
        # of many independent windows.
        time_axis_minutes: Optional[np.ndarray]
        if 'sample_end_times_s' in viz_payload and 'recording_ids' in viz_payload:
            panel_window_minutes = float(getattr(self.training_config, 'epoch_panel_window_minutes', 180.0))
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

        # Keep raw proxies for any downstream numeric use.
        ecg_proxy_raw = ecg_proxy[selected]
        ppg_proxy_raw = ppg_proxy[selected] if ppg_proxy is not None and len(ppg_proxy) == len(ecg_proxy) else ecg_proxy_raw
        eda_proxy_raw = eda_proxy[selected] if eda_proxy is not None and len(eda_proxy) == len(ecg_proxy) else None
        eeg_proxy_raw = eeg_proxy[selected] if eeg_proxy is not None and len(eeg_proxy) == len(ecg_proxy) else None
        adxl_proxy_raw = adxl_proxy[selected] if adxl_proxy is not None and len(adxl_proxy) == len(ecg_proxy) else None
        hr_proxy_raw = hr_proxy[selected]
        hrv_proxy_raw = hrv_proxy[selected]

        # For wearable runs, show a z-scored heart-rate proxy as the top
        # "Signal" trace (so that the HR subplot can display the raw bpm
        # values separately). For BIDS ECG runs we standardize the ECG proxy
        # for contrast.
        if getattr(self.config.data, "data_source", "bids") == "wearable":
            ecg_proxy_plot = _standardize_for_plot(ppg_proxy_raw)
        else:
            ecg_proxy_plot = _standardize_for_plot(ecg_proxy_raw)
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
            if np.allclose(hrv_proxy_raw, 0.0):
                hrv_proxy_plot = hr_proxy_raw.astype(np.float32)
            else:
                hrv_proxy_plot = hrv_proxy_raw.astype(np.float32)
        else:
            hr_proxy_plot = _standardize_for_plot(hr_proxy_raw)
            hrv_proxy_plot = _standardize_for_plot(hr_proxy_raw if np.allclose(hrv_proxy_raw, 0.0) else hrv_proxy_raw)
        pred_preictal_smooth_sel = viz_payload.get('pred_preictal_smooth')
        if pred_preictal_smooth_sel is not None:
            pred_preictal_smooth_sel = pred_preictal_smooth_sel[selected]

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
            'ECG proxy' if data_source != "wearable" else "PPG proxy",
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
        if rec_ids_meta is not None and times_meta is not None and len(rec_ids_meta) == len(true_countdown):
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
            ppg_signal=ppg_proxy_plot,
            eda_signal=eda_proxy_plot,
            eeg_signal=eeg_proxy_plot,
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
            pred_onset_prob=pred_onset_prob,
            pred_preictal_only_prob=pred_preictal_only_prob,
            gt_onset=None if true_state is None else (true_state == 2).astype(np.int32),
            gt_preictal_only=None if true_state is None else (true_state == 1).astype(np.int32),
            visualization_mode='state' if state_mode else 'countdown',
        )
        self.epoch_visualizer.save_figure(fig_panel, f"epoch_{epoch:03d}_gt_vs_inference_panel", step=epoch)

        # Log interactive per-epoch timeline table so W&B can zoom into
        # long-duration traces (hours) beyond static PNG panels.
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

                log_payload = {
                    'visualizations/epoch_timeseries_table': table,
                    'visualizations/epoch': int(epoch),
                }

                # Token waterfall explainability artifact (epoch-level).
                try:
                    n_countdown_bins = int(getattr(self.training_config, 'token_countdown_bins', 8))
                    n_prob_bins = int(getattr(self.training_config, 'token_prob_bins', 4))
                    token_window = max(1, int(getattr(self.training_config, 'token_waterfall_window', 25)))

                    if state_mode and pred_onset_prob is not None:
                        alert_prob = np.clip(pred_preictal.astype(np.float32), 0.0, 0.999999)
                        onset_prob_sel = np.clip(pred_onset_prob.astype(np.float32), 0.0, 0.999999)
                        alert_bin = np.minimum((alert_prob * n_prob_bins).astype(np.int32), n_prob_bins - 1)
                        onset_bin = np.minimum((onset_prob_sel * n_prob_bins).astype(np.int32), n_prob_bins - 1)
                        token_id = alert_bin * n_prob_bins + onset_bin
                        token_total = n_prob_bins * n_prob_bins
                    else:
                        max_countdown = max(10.0, float(np.max(np.clip(true_countdown, 0.0, None))))
                        pred_prob = np.clip(pred_preictal.astype(np.float32), 0.0, 0.999999)
                        pred_cd = pred_countdown.astype(np.float32)

                        countdown_bin = np.zeros_like(pred_cd, dtype=np.int32)
                        pre_mask = pred_cd >= 0.0
                        if np.any(pre_mask):
                            frac = np.clip(pred_cd[pre_mask] / max_countdown, 0.0, 0.999999)
                            countdown_bin[pre_mask] = 1 + np.minimum((frac * n_countdown_bins).astype(np.int32), n_countdown_bins - 1)

                        prob_bin = np.minimum((pred_prob * n_prob_bins).astype(np.int32), n_prob_bins - 1)
                        token_id = countdown_bin * n_prob_bins + prob_bin
                        token_total = (n_countdown_bins + 1) * n_prob_bins

                    one_hot = np.zeros((len(token_id), token_total), dtype=np.float32)
                    one_hot[np.arange(len(token_id)), np.clip(token_id, 0, token_total - 1)] = 1.0

                    kernel = np.ones(token_window, dtype=np.float32) / float(token_window)
                    token_roll = np.zeros_like(one_hot)
                    for col in range(one_hot.shape[1]):
                        token_roll[:, col] = np.convolve(one_hot[:, col], kernel, mode='same')

                    import matplotlib.pyplot as _plt
                    fig_tw, ax_tw = _plt.subplots(figsize=(12, 4))
                    im = ax_tw.imshow(
                        token_roll.T,
                        aspect='auto',
                        origin='lower',
                        interpolation='nearest',
                        cmap='magma',
                    )
                    waterfall_title = 'State Token Waterfall' if state_mode else 'Token Waterfall'
                    ax_tw.set_title(f'{waterfall_title} (epoch {int(epoch):03d}, window={token_window})')
                    ax_tw.set_xlabel('Sample index')
                    ax_tw.set_ylabel('Token ID')
                    fig_tw.colorbar(im, ax=ax_tw, fraction=0.03, pad=0.01, label='Rolling occupancy')

                    token_path = self.save_dir / 'epoch_visualizations' / f'epoch_{int(epoch):03d}_token_waterfall.png'
                    fig_tw.savefig(token_path, dpi=160, bbox_inches='tight')
                    _plt.close(fig_tw)

                    log_payload['visualizations/token_waterfall'] = wandb.Image(str(token_path))
                except Exception as token_exc:
                    logger.warning('Failed to generate/log token waterfall for epoch %d: %s', int(epoch), str(token_exc))

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
                        f'Alert confusion matrix | epoch {int(epoch):03d}',
                    )
                    log_payload['visualizations/alert_confusion_matrix'] = wandb.Image(fig_alert_cm)

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
        
        # Create dataloaders
        train_loader, val_loader = self.create_dataloaders(train_dataset, val_dataset)
        
        # Create model and optimizer
        model, optimizer, scheduler = self.create_model_optimizer_scheduler()
        
        # Create loss function — supports 'state' (safe 3-class, no countdown regression)
        # or 'countdown' (legacy regression).  Read from config or auto-detect.
        _loss_type = str(getattr(self.config.loss, 'loss_type', 'countdown')).lower()
        criterion = LossFactory.create_loss(self.config.loss, loss_type=_loss_type)
        criterion = criterion.to(self.device)
        
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
            train_metrics, train_viz_payload = self.train_epoch(model, train_loader, criterion, optimizer, epoch)
            
            # Validate
            val_metrics, viz_payload = self.validate(model, val_loader, criterion)
            
            # Gather metrics across ranks (distributed training)
            if self.distributed:
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
                if (
                    not self.training_config.save_best_only
                    and self.training_config.save_interval > 0
                    and epoch % self.training_config.save_interval == 0
                ):
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
