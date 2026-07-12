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
import csv
import logging
import argparse
import hashlib
import time
import shutil
import tempfile
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any

# Ensure headless plotting backend (must be set before importing project modules)
os.environ.setdefault("MPLBACKEND", "Agg")

import torch
import torch.distributed as dist
import numpy as np
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, DEFAULT_CONFIG

_OPTIONAL_IMPORT_ERROR = None
try:
    from src import (
        BIDSDataLoader,
        WearableDeviceDataLoader,
        Trainer,
        Evaluator,
        FeatureExtractor,
        SignalProcessor,
    )
    from src.training import _causal_smooth
    from src.visualization import SignalVisualizer
except Exception as import_exc:  # pragma: no cover - environment dependent
    _OPTIONAL_IMPORT_ERROR = import_exc
    BIDSDataLoader = None
    WearableDeviceDataLoader = None
    Trainer = None
    Evaluator = None
    FeatureExtractor = None
    SignalProcessor = None
    SignalVisualizer = None
    _causal_smooth = None


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


def _compute_binary_metrics_from_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    """Compute thresholded binary metrics from probability-like scores."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_score = np.asarray(y_score, dtype=np.float32)
    if y_true.size == 0 or y_score.size == 0 or y_true.size != y_score.size:
        return {
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0,
            "sensitivity": 0.0,
            "specificity": 0.0,
            "precision": 0.0,
            "f1": 0.0,
            "balanced_accuracy": 0.0,
        }

    y_pred = (y_score >= float(threshold)).astype(np.int32)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    sensitivity = float(tp / max(tp + fn, 1))
    specificity = float(tn / max(tn + fp, 1))
    precision = float(tp / max(tp + fp, 1))
    f1 = float(2.0 * precision * sensitivity / max(precision + sensitivity, 1e-8))
    balanced_accuracy = float(0.5 * (sensitivity + specificity))

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
    }


def _infer_subject_ids_from_recordings(recording_ids: Optional[np.ndarray], n: int) -> Optional[np.ndarray]:
    """Infer subject IDs from BIDS-like recording IDs when explicit IDs are absent."""
    if recording_ids is None:
        return None
    rec = np.asarray(recording_ids).astype(str)
    if len(rec) != int(n):
        return None
    out = []
    for rec_id in rec:
        token = str(rec_id).split('_')[0]
        out.append(token.replace('sub-', ''))
    return np.asarray(out).astype(str)


def _compute_patient_report(
    prediction_payload: Dict[str, np.ndarray],
    subject_ids: Optional[np.ndarray],
    threshold: float,
    countdown_max: float,
    top_k: int = 10,
    top_examples: int = 25,
) -> Dict[str, Any]:
    """Build per-patient difficulty summary plus hardest individual examples."""
    pred_preictal = np.asarray(prediction_payload.get('pred_preictal', []), dtype=np.float32)
    true_preictal = np.asarray(prediction_payload.get('true_preictal', []), dtype=np.int32)
    pred_countdown = np.asarray(prediction_payload.get('pred_countdown', []), dtype=np.float32)
    true_countdown = np.asarray(prediction_payload.get('true_countdown', []), dtype=np.float32)
    recording_ids = prediction_payload.get('recording_ids', None)
    sample_end_times_s = prediction_payload.get('sample_end_times_s', None)

    n = len(pred_preictal)
    if n == 0:
        return {
            "n_samples": 0,
            "n_patients": 0,
            "threshold": float(threshold),
            "patients": [],
            "hardest_patients": [],
            "easiest_patients": [],
            "hardest_examples": [],
            "best_examples": [],
        }

    if subject_ids is None:
        subject_ids = _infer_subject_ids_from_recordings(recording_ids, n)
    if subject_ids is None or len(subject_ids) != n:
        subject_ids = np.array(["unknown"] * n, dtype=object)

    subject_ids = np.asarray(subject_ids).astype(str)
    score_err = np.abs(pred_preictal - true_preictal.astype(np.float32))

    countdown_abs_err = np.abs(pred_countdown - true_countdown)
    preictal_mask = true_countdown >= 0
    countdown_norm = np.zeros_like(countdown_abs_err, dtype=np.float32)
    countdown_norm[preictal_mask] = np.clip(
        countdown_abs_err[preictal_mask] / max(float(countdown_max), 1e-6),
        0.0,
        3.0,
    )
    difficulty = 0.65 * score_err + 0.35 * countdown_norm

    patient_rows: List[Dict[str, Any]] = []
    for patient_id in np.unique(subject_ids):
        mask = subject_ids == patient_id
        if not np.any(mask):
            continue
        metrics = _compute_binary_metrics_from_scores(
            y_true=true_preictal[mask],
            y_score=pred_preictal[mask],
            threshold=float(threshold),
        )
        local_preictal = preictal_mask[mask]
        patient_rows.append({
            "patient_id": str(patient_id),
            "n_samples": int(np.sum(mask)),
            "n_preictal": int(np.sum(true_preictal[mask])),
            "n_interictal": int(np.sum(mask) - np.sum(true_preictal[mask])),
            "alert_balanced_accuracy": float(metrics["balanced_accuracy"]),
            "alert_sensitivity": float(metrics["sensitivity"]),
            "alert_specificity": float(metrics["specificity"]),
            "alert_precision": float(metrics["precision"]),
            "alert_f1": float(metrics["f1"]),
            "mean_prob_abs_error": float(np.mean(score_err[mask])),
            "preictal_countdown_mae": (
                float(np.mean(countdown_abs_err[mask][local_preictal])) if np.any(local_preictal) else None
            ),
            "difficulty_score": float(np.mean(difficulty[mask])),
        })

    patient_rows.sort(key=lambda item: float(item.get("difficulty_score", 0.0)), reverse=True)
    k = max(1, int(top_k))

    global_order = np.argsort(-difficulty)
    hard_rows: List[Dict[str, Any]] = []
    best_rows: List[Dict[str, Any]] = []

    def _example_row(sample_idx: int) -> Dict[str, Any]:
        row = {
            "sample_index": int(sample_idx),
            "patient_id": str(subject_ids[sample_idx]),
            "difficulty_score": float(difficulty[sample_idx]),
            "pred_preictal": float(pred_preictal[sample_idx]),
            "true_preictal": int(true_preictal[sample_idx]),
            "pred_countdown": float(pred_countdown[sample_idx]),
            "true_countdown": float(true_countdown[sample_idx]),
            "countdown_abs_error": float(countdown_abs_err[sample_idx]),
            "prob_abs_error": float(score_err[sample_idx]),
        }
        if recording_ids is not None and len(recording_ids) == n:
            row["recording_id"] = str(np.asarray(recording_ids).astype(str)[sample_idx])
        if sample_end_times_s is not None and len(sample_end_times_s) == n:
            row["sample_end_time_s"] = float(np.asarray(sample_end_times_s, dtype=np.float64)[sample_idx])
        return row

    for idx in global_order[: max(1, int(top_examples))]:
        hard_rows.append(_example_row(int(idx)))
    for idx in np.argsort(difficulty)[: max(1, int(top_examples))]:
        best_rows.append(_example_row(int(idx)))

    return {
        "n_samples": int(n),
        "n_patients": int(len(patient_rows)),
        "threshold": float(threshold),
        "patients": patient_rows,
        "hardest_patients": patient_rows[:k],
        "easiest_patients": patient_rows[-k:][::-1] if patient_rows else [],
        "hardest_examples": hard_rows,
        "best_examples": best_rows,
    }


def _write_comprehensive_run_report(
    save_dir: Path,
    config: Config,
    args,
    results: Dict[str, Any],
    test_metrics: Dict[str, Any],
    patient_report: Dict[str, Any],
    threshold_sweep: Optional[Dict[str, Any]] = None,
) -> Tuple[Path, Path]:
    """Write a comprehensive JSON + Markdown report for the run."""
    history = results.get('history', {})
    train_loss_hist = history.get('train_loss', [])
    val_loss_hist = history.get('val_loss', [])
    val_mae_hist = history.get('val_mae', [])

    report_payload: Dict[str, Any] = {
        "generated_at_utc": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        "run_context": {
            "model_type": str(config.model.model_type),
            "data_source": str(getattr(config.data, 'data_source', 'unknown')),
            "dataset_root": str(getattr(config.data, 'dataset_root', '')),
            "seed": int(getattr(args, 'seed', 42)),
            "optimizer_step_scope": str(getattr(config.training, 'optimizer_step_scope', 'batch')),
            "patients_per_step": int(getattr(config.training, 'patients_per_step', 1)),
            "patient_sequential": bool(getattr(args, 'patient_sequential', False)),
            "wandb_mode": str(getattr(args, 'wandb_mode', 'unknown')),
        },
        "training": {
            "num_epochs_completed": int(len(train_loss_hist)),
            "best_val_metric": float(results.get('best_val_metric', 0.0)),
            "training_time_hours": float(results.get('training_time', 0.0) / 3600.0),
            "last_train_loss": float(train_loss_hist[-1]) if train_loss_hist else None,
            "last_val_loss": float(val_loss_hist[-1]) if val_loss_hist else None,
            "last_val_mae": float(val_mae_hist[-1]) if val_mae_hist else None,
        },
        "test_metrics": _to_jsonable(test_metrics),
        "threshold_sweep": _to_jsonable(threshold_sweep) if threshold_sweep is not None else None,
        "patient_difficulty": _to_jsonable(patient_report),
        "artifacts": {
            "results_json": str(save_dir / 'results.json'),
            "training_history_json": str(save_dir / 'training_history.json'),
            "patient_difficulty_json": str(save_dir / 'patient_difficulty_report.json'),
            "predictions_npz": str(save_dir / 'test_predictions.npz'),
            "performance_overview_png": str(save_dir / 'performance_overview.png'),
            "gt_vs_inference_panel_png": str(save_dir / 'gt_vs_inference_panel.png'),
            "config_yaml": str(save_dir / 'config.yaml'),
        },
    }

    report_json_path = save_dir / "comprehensive_run_report.json"
    with open(report_json_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(report_payload), fh, indent=2)

    hardest = patient_report.get('hardest_patients', []) if isinstance(patient_report, dict) else []
    easiest = patient_report.get('easiest_patients', []) if isinstance(patient_report, dict) else []

    lines: List[str] = []
    lines.append("# Comprehensive Run Report")
    lines.append("")
    lines.append(f"Generated (UTC): {report_payload['generated_at_utc']}")
    lines.append("")
    lines.append("## Run Context")
    lines.append(f"- Model: {report_payload['run_context']['model_type']}")
    lines.append(f"- Data source: {report_payload['run_context']['data_source']}")
    lines.append(f"- Optimizer step scope: {report_payload['run_context']['optimizer_step_scope']}")
    lines.append(f"- Patients per step: {report_payload['run_context']['patients_per_step']}")
    lines.append(f"- Patient sequential: {report_payload['run_context']['patient_sequential']}")
    lines.append(f"- W&B mode: {report_payload['run_context']['wandb_mode']}")
    lines.append("")
    lines.append("## Training Summary")
    lines.append(f"- Epochs completed: {report_payload['training']['num_epochs_completed']}")
    lines.append(f"- Best validation metric: {report_payload['training']['best_val_metric']:.4f}")
    lines.append(f"- Training time (hours): {report_payload['training']['training_time_hours']:.3f}")
    if report_payload['training']['last_val_mae'] is not None:
        lines.append(f"- Last validation MAE (min): {report_payload['training']['last_val_mae']:.4f}")
    lines.append("")
    lines.append("## Test Metrics")
    for key in sorted(test_metrics.keys()):
        val = test_metrics[key]
        if isinstance(val, (int, float, np.integer, np.floating)):
            lines.append(f"- {key}: {float(val):.6f}")
    lines.append("")

    lines.append("## Patient Difficulty (Top)")
    lines.append(f"- Patients analyzed: {int(patient_report.get('n_patients', 0)) if isinstance(patient_report, dict) else 0}")
    lines.append("")
    lines.append("### Hardest Patients")
    lines.append("| patient_id | n_samples | difficulty | bal_acc | preictal_countdown_mae |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in hardest[:10]:
        mae = row.get('preictal_countdown_mae', None)
        mae_text = "-" if mae is None else f"{float(mae):.4f}"
        lines.append(
            f"| {row.get('patient_id', 'na')} | {int(row.get('n_samples', 0))} | "
            f"{float(row.get('difficulty_score', 0.0)):.4f} | {float(row.get('alert_balanced_accuracy', 0.0)):.4f} | {mae_text} |"
        )
    lines.append("")
    lines.append("### Easiest Patients")
    lines.append("| patient_id | n_samples | difficulty | bal_acc | preictal_countdown_mae |")
    lines.append("|---|---:|---:|---:|---:|")
    for row in easiest[:10]:
        mae = row.get('preictal_countdown_mae', None)
        mae_text = "-" if mae is None else f"{float(mae):.4f}"
        lines.append(
            f"| {row.get('patient_id', 'na')} | {int(row.get('n_samples', 0))} | "
            f"{float(row.get('difficulty_score', 0.0)):.4f} | {float(row.get('alert_balanced_accuracy', 0.0)):.4f} | {mae_text} |"
        )
    lines.append("")
    lines.append("## Artifacts")
    for name, path in report_payload['artifacts'].items():
        lines.append(f"- {name}: {path}")

    report_md_path = save_dir / "COMPREHENSIVE_RUN_REPORT.md"
    report_md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return report_json_path, report_md_path


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
        help='Real data source: bids (SeizeIT2 BIDS EDF) or wearable (WearableDevice-Oregon CSV)'
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
            'Enable long-sweep training mode: keep all chronological windows per recording '
            'to mimic true day-long streaming operation. Disables per-recording sample capping.'
        )
    )

    parser.add_argument(
        '--patient-sequential',
        action='store_true',
        help=(
            'Enable patient-by-patient sequential training: split by patient to avoid '
            'cross-patient data mixing, then process each patient\'s recordings sequentially. '
            'Prevents memory issues and improves generalization by avoiding unrelated sample mixing.'
        )
    )

    parser.add_argument(
        '--training-mode',
        type=str,
        choices=['stateful', 'stateless'],
        default=None,
        help='Training mode override: stateful or stateless. If omitted, config default is used.'
    )

    parser.add_argument(
        '--batching-strategy',
        type=str,
        choices=['patient_sequential', 'random'],
        default=None,
        help='Batching strategy override for stateful/stateless training.'
    )

    parser.add_argument(
        '--augment-preictal',
        dest='augment_preictal',
        action='store_true',
        default=None,
        help='Enable preictal augmentation (offline for non-long-sweep, online for long-sweep).'
    )

    parser.add_argument(
        '--no-augment-preictal',
        dest='augment_preictal',
        action='store_false',
        help='Disable preictal augmentation.'
    )

    parser.add_argument(
        '--online-aug-prob',
        type=float,
        default=None,
        help='Override long-sweep online preictal augmentation probability in [0,1].'
    )

    parser.add_argument(
        '--panel-window-minutes',
        type=float,
        default=60.0,
        help='Window length in minutes for final GT-vs-inference panel (default: 60, approximately -30/+30 around onset)'
    )

    parser.add_argument(
        '--preflight-visualization',
        dest='preflight_visualization',
        action='store_true',
        default=True,
        help='Run an early visualization checkpoint before epoch 1 to fail fast on plotting/panel issues.'
    )

    parser.add_argument(
        '--no-preflight-visualization',
        dest='preflight_visualization',
        action='store_false',
        help='Disable pre-training visualization preflight checkpoint.'
    )

    parser.add_argument(
        '--preflight-max-val-batches',
        type=int,
        default=0,
        help='Maximum validation batches used for preflight visualization (default: 0, meaning full validation pass).'
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

    parser.add_argument(
        '--wandb-mode',
        type=str,
        choices=['online', 'offline', 'disabled'],
        default='online',
        help='Weights & Biases mode (default: online). Use offline for local-only logging or disabled to skip W&B.'
    )

    parser.add_argument(
        '--wandb-project',
        type=str,
        default='epilepsee-ai',
        help='Weights & Biases project name (default: epilepsee-ai).'
    )

    parser.add_argument(
        '--wandb-run-name',
        type=str,
        default=None,
        help='Optional explicit Weights & Biases run name.'
    )

    # DDP launch compatibility (torch.distributed.launch / torchrun)
    parser.add_argument(
        '--local-rank',
        '--local_rank',
        type=int,
        default=None,
        help='Local rank passed by distributed launchers'
    )

    # ── Safe 3-state classification loss (no GT countdown regression) ─────────
    parser.add_argument(
        '--state-loss',
        action='store_true',
        help=(
            'Use SeizureStateLoss (safe 3-class: interictal / pre-ictal / onset). '
            'GT countdown is used only to bucket labels — no regression is performed, '
            'so the model does NOT learn exact timing from future seizure annotations.'
        )
    )
    parser.add_argument(
        '--onset-threshold-min',
        type=float,
        default=2.0,
        help='Minutes before seizure that define the "onset" bucket (default 2.0).'
    )

    # ── Temporal ring-buffer training ─────────────────────────────────────────
    parser.add_argument(
        '--ring-buffer-size',
        type=int,
        default=0,
        help=(
            'Enable temporal ring-buffer training: expose only the most recent N samples '
            'to the model per epoch, evicting older history. '
            '0 (default) disables windowing and uses the full dataset each epoch. '
            'Simulates deployment conditions where the device has only seen data since it was turned on.'
        )
    )

    parser.add_argument(
        '--optimizer-step-scope',
        type=str,
        choices=['batch', 'patient', 'patients_group'],
        default='batch',
        help='When to apply optimizer step: per batch, per patient boundary, or per N patients.'
    )

    parser.add_argument(
        '--patients-per-step',
        type=int,
        default=1,
        help='Only used when --optimizer-step-scope patients_group; number of consecutive patients per optimizer step.'
    )

    parser.add_argument(
        '--bayes-memory-eval',
        dest='bayes_memory_eval',
        action='store_true',
        default=None,
        help='Enable Bayesian long-sweep memory evaluation when timeline metadata is available.'
    )
    parser.add_argument(
        '--no-bayes-memory-eval',
        dest='bayes_memory_eval',
        action='store_false',
        help='Disable Bayesian long-sweep memory evaluation for ablation runs.'
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


# Per-process memory-mapped signal cache.  Each DataLoader worker forks a
# separate copy so there is no multiprocessing contention.
_MMAP_CACHE: dict = {}


class LazyRealDataset(torch.utils.data.Dataset):
    """Streaming BIDS dataset – stores only window metadata; features are
    computed on-the-fly in __getitem__ from raw float32 signal binary files.

    This eliminates the need to pre-cache hundreds of GB of feature tensors.
    Binary signal files are stored once (22 GB total for SeizeIT2 / ds005873) on the same
    disk as the EDF dataset; feature computation is cheap (~600 µs/window).
    """

    def __init__(
        self,
        rec_indices: np.ndarray,       # (N,) int32 – index into recordings_meta
        end_indices: np.ndarray,       # (N,) int64 – end sample index
        labels: np.ndarray,            # (N,) float32
        sample_end_times_s: np.ndarray,  # (N,) float32
        recording_ids: np.ndarray,     # (N,) str
        recordings_meta: list,         # [{signal_path, fs, seq_samples, n_samples}, ...]
        feature_dim: int,
        subject_ids: Optional[np.ndarray] = None,
    ):
        self.rec_indices = np.asarray(rec_indices, dtype=np.int32)
        self.end_indices = np.asarray(end_indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float32)
        self.recording_ids = np.asarray(recording_ids)
        self.recordings_meta = recordings_meta
        self.feature_dim = feature_dim
        self.subject_ids = subject_ids

        self.preictal_labels = (self.labels >= 0).astype(np.float32)
        self.weights = self._compute_weights(self.labels)

        # Online augmentation flags (mirror SeizureDataset interface)
        self._online_aug_enabled = False
        self._online_aug_probability = 0.0
        self._online_aug_preictal_only = True
        self._online_aug_cfg = None
        self._online_aug_rng = None
        
        # Feature cache: stores recently computed features in RAM for faster access
        # Typical: ~10k features × 600 timesteps × 14 channels × 4 bytes = ~336 MB
        self._feature_cache = {}
        self._feature_cache_max_size = 10000

        logger.info(
            "LazyRealDataset: %d windows, feature_dim=%d, %d unique recordings",
            len(self),
            feature_dim,
            len(recordings_meta),
        )

    # ------------------------------------------------------------------
    def _compute_weights(self, labels: np.ndarray) -> np.ndarray:
        if labels.size == 0:
            return np.array([], dtype=np.float32)
        tau = 60.0
        weights = np.where(labels >= 0, np.exp(-np.maximum(labels, 0.0) * 60 / tau), 0.5)
        return (weights / np.mean(weights)).astype(np.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int):
        rec_idx = int(self.rec_indices[idx])
        end_idx = int(self.end_indices[idx])
        label = float(self.labels[idx])

        meta = self.recordings_meta[rec_idx]
        signal_path: str = meta["signal_path"]
        seq_samples: int = int(meta["seq_samples"])
        n_samples: int = int(meta["n_samples"])

        # Memory-map the raw signal (lazy OS-level caching, no copy)
        if signal_path not in _MMAP_CACHE:
            _MMAP_CACHE[signal_path] = np.memmap(
                signal_path, dtype=np.float32, mode="r", shape=(n_samples,)
            )
        signal = _MMAP_CACHE[signal_path]

        start_idx = max(0, end_idx - seq_samples)
        segment = np.array(signal[start_idx:end_idx], dtype=np.float32)
        if len(segment) < seq_samples:
            segment = np.pad(segment, (seq_samples - len(segment), 0), mode="edge")

        # Check feature cache first
        cache_key = (rec_idx, start_idx, end_idx)
        if cache_key in self._feature_cache:
            features = self._feature_cache[cache_key]
        else:
            # Compute features and cache
            features = _segment_to_feature_matrix(segment, self.feature_dim, target_steps=600)
            
            # Add to cache (simple LRU: remove oldest if cache is full)
            if len(self._feature_cache) >= self._feature_cache_max_size:
                oldest_key = next(iter(self._feature_cache))
                del self._feature_cache[oldest_key]
            
            self._feature_cache[cache_key] = features

        if (
            self._online_aug_enabled
            and self._online_aug_rng is not None
            and (label >= 0 or not self._online_aug_preictal_only)
            and float(self._online_aug_rng.random()) < self._online_aug_probability
        ):
            features = features.copy()
            cfg = self._online_aug_cfg
            if cfg is not None:
                noise_std = float(getattr(cfg, "noise_std", 0.02))
                if noise_std > 0:
                    features += self._online_aug_rng.normal(
                        0, noise_std, features.shape
                    ).astype(np.float32)

        weight = float(self.weights[idx])
        return (
            torch.from_numpy(features),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(weight, dtype=torch.float32),
        )

    # ------------------------------------------------------------------
    @property
    def class_distribution(self) -> dict:
        return {
            "preictal": int(np.sum(self.labels >= 0)),
            "interictal": int(np.sum(self.labels < 0)),
        }

    @property
    def features(self):
        raise AttributeError(
            "LazyRealDataset does not materialise a .features array. "
            "Use __getitem__ / DataLoader for feature access."
        )


def _ensure_raw_signal_cache(
    recording: dict,
    loader: "BIDSDataLoader",
    signal_cache_dir: Path,
) -> Tuple[Path, float, int]:
    """Return path to a raw float32 ECG binary for *recording*.

    If the file does not already exist it is created by loading the EDF and
    writing the first channel as a contiguous float32 array.  The file is
    stored in *signal_cache_dir* (typically co-located with the EDF dataset
    on the same large disk).

    Returns (binary_path, fs, n_samples).
    """
    uid = (
        f"sub-{recording['subject_id']}"
        f"_ses-{recording['session_id']}"
        f"_run-{recording['run_id']}"
    )
    bin_path = signal_cache_dir / f"{uid}_ecg.f32"
    meta_path = signal_cache_dir / f"{uid}_ecg.json"

    if bin_path.exists() and meta_path.exists():
        with open(meta_path) as fh:
            m = json.load(fh)
        return bin_path, float(m["fs"]), int(m["n_samples"])

    # Load EDF and extract first channel
    ecg_data, fs = loader.load_subject_edf(
        recording["subject_id"],
        recording["session_id"],
        "ecg",
        recording["run_id"],
    )
    signal_1d = (ecg_data[0] if ecg_data.ndim > 1 else ecg_data).astype(np.float32)
    n_samples = len(signal_1d)

    # Write raw float32 binary
    tmp = bin_path.with_suffix(".tmp")
    try:
        signal_1d.tofile(str(tmp))
        tmp.rename(bin_path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)
        raise
    finally:
        del signal_1d
        del ecg_data

    with open(meta_path, "w") as fh:
        json.dump({"fs": float(fs), "n_samples": n_samples}, fh)

    return bin_path, float(fs), n_samples


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


def _select_recordings_for_real_mode(recordings: List[dict], max_recordings: int, loader: BIDSDataLoader) -> List[dict]:
    """Select recordings with seizure-first prioritization for better preictal coverage.
    
    Only includes recordings that actually have the required data files.
    """
    if max_recordings and max_recordings > 0:
        candidates = recordings[:]
    else:
        candidates = recordings

    # Filter to only recordings that have ECG data files
    valid_candidates = []
    for recording in candidates:
        try:
            loader.resolve_subject_edf_path(
                recording['subject_id'],
                recording['session_id'],
                'ecg',
                recording['run_id']
            )
            valid_candidates.append(recording)
        except FileNotFoundError:
            continue  # Skip recordings without data files
    
    logger.info(f"Filtered to {len(valid_candidates)} recordings with actual data files out of {len(candidates)} candidates")

    seizure_recs = [recording for recording in valid_candidates if int(recording.get('num_seizures', 0)) > 0]
    nonseizure_recs = [recording for recording in valid_candidates if int(recording.get('num_seizures', 0)) <= 0]

    seizure_recs.sort(key=lambda recording: int(recording.get('num_seizures', 0)), reverse=True)

    selected: List[dict] = []
    if max_recordings and max_recordings > 0:
        n_target = max_recordings
    else:
        n_target = len(valid_candidates)

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
    """Split windows by contiguous recording order to preserve chronology.

    Recordings are ordered by their earliest sample timestamp, then assigned in
    timeline order to train/val/test using sample-count targets. This keeps
    each split temporally coherent and avoids random interleaving.
    """
    del seed  # deterministic timeline split

    y = np.asarray(y, dtype=np.float32)
    recording_ids = np.asarray(recording_ids).astype(str)
    sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float64)

    if y.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    unique_recs = np.unique(recording_ids)
    rec_meta = []
    for rec_id in unique_recs:
        rec_idx = np.where(recording_ids == rec_id)[0]
        if rec_idx.size == 0:
            continue
        rec_times = sample_end_times_s[rec_idx]
        rec_order = rec_idx[np.argsort(rec_times)]
        rec_meta.append({
            'rec_id': rec_id,
            'idx': rec_order.astype(np.int64),
            'n': int(rec_order.size),
            't0': float(np.min(rec_times)),
        })

    if not rec_meta:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    rec_meta.sort(key=lambda row: (row['t0'], row['rec_id']))

    n_total = int(y.size)
    target_train = int(round(float(train_ratio) * n_total))
    target_val = int(round(float(val_ratio) * n_total))
    target_test = max(0, n_total - target_train - target_val)

    splits = {'train': [], 'val': [], 'test': []}
    counts = {'train': 0, 'val': 0, 'test': 0}

    for row in rec_meta:
        # Fill train first, then val, then test while keeping recording integrity.
        if counts['train'] < target_train:
            split_name = 'train'
        elif counts['val'] < target_val:
            split_name = 'val'
        else:
            split_name = 'test'

        splits[split_name].append(row['idx'])
        counts[split_name] += row['n']

    # If due to coarse recording granularity test target remains empty while
    # data exists, keep the tail recording in test for chronological holdout.
    if counts['test'] == 0 and rec_meta:
        tail = rec_meta[-1]['idx']
        for donor in ['val', 'train']:
            if counts[donor] <= tail.size:
                continue
            # remove tail from donor if present
            new_chunks = []
            removed = False
            for chunk in splits[donor]:
                if not removed and np.array_equal(chunk, tail):
                    removed = True
                    continue
                new_chunks.append(chunk)
            if removed:
                splits[donor] = new_chunks
                counts[donor] -= int(tail.size)
                splits['test'].append(tail)
                counts['test'] += int(tail.size)
                break

    def _cat(parts):
        if not parts:
            return np.array([], dtype=np.int64)
        out = np.concatenate(parts).astype(np.int64)
        return _sort_indices_by_timeline(out, recording_ids, sample_end_times_s)

    train_idx = _cat(splits['train'])
    val_idx = _cat(splits['val'])
    test_idx = _cat(splits['test'])

    # If test target was positive but remained empty, carve a tail from val/train.
    if target_test > 0 and test_idx.size == 0:
        if val_idx.size > 1:
            take = max(1, val_idx.size // 5)
            test_idx = val_idx[-take:]
            val_idx = val_idx[:-take]
        elif train_idx.size > 2:
            take = max(1, train_idx.size // 10)
            test_idx = train_idx[-take:]
            train_idx = train_idx[:-take]

    return train_idx, val_idx, test_idx


def _split_indices_by_patient_sequential(
    y: np.ndarray,
    recording_ids: np.ndarray,
    sample_end_times_s: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    seed: int,
):
    """Split by patient (subject) for sequential training, avoiding cross-patient data mixing.

    This ensures each patient appears in only one split (train/val/test), and within
    each split, maintains chronological order within recordings. This prevents
    data leakage and memory issues from mixing unrelated patient samples.
    """
    rng = np.random.default_rng(seed)

    y = np.asarray(y, dtype=np.float32)
    recording_ids = np.asarray(recording_ids)
    sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float64)

    # Extract subject_ids from recording_ids (format: "sub-{subject_id}_ses-{session_id}_run-{run_id}")
    subject_ids = np.array([rec_id.split('_')[0].replace('sub-', '') for rec_id in recording_ids])

    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    # Group recordings by subject
    subject_meta = []
    for subj_id in unique_subjects:
        subj_mask = subject_ids == subj_id
        subj_indices = np.where(subj_mask)[0]
        if subj_indices.size == 0:
            continue

        subj_recordings = recording_ids[subj_mask]
        unique_recs = np.unique(subj_recordings)
        n_samples = int(subj_indices.size)
        has_preictal = bool(np.any(y[subj_indices] >= 0))

        subject_meta.append({
            'subj_id': subj_id,
            'n_samples': n_samples,
            'n_recordings': len(unique_recs),
            'has_preictal': has_preictal,
            'recording_ids': unique_recs,
        })

    if len(subject_meta) == 0:
        empty = np.array([], dtype=np.int64)
        return empty, empty, empty

    # Prioritize subjects with seizures for better preictal coverage
    seizure_subjects = [row for row in subject_meta if row['has_preictal']]
    nonseizure_subjects = [row for row in subject_meta if not row['has_preictal']]
    rng.shuffle(seizure_subjects)
    rng.shuffle(nonseizure_subjects)

    ordered_subjects = seizure_subjects + nonseizure_subjects

    n_total = int(len(y))
    target = {
        'train': float(train_ratio) * n_total,
        'val': float(val_ratio) * n_total,
        'test': max(0.0, (1.0 - float(train_ratio) - float(val_ratio)) * n_total),
    }
    assigned = {'train': 0, 'val': 0, 'test': 0}
    split_subjects = {'train': [], 'val': [], 'test': []}

    for row in ordered_subjects:
        # Greedy assignment by smallest target-fill ratio
        best_split = min(
            ['train', 'val', 'test'],
            key=lambda name: assigned[name] / max(target[name], 1.0)
        )
        split_subjects[best_split].append(row['subj_id'])
        assigned[best_split] += int(row['n_samples'])

    # Ensure no split is empty (for small numbers of subjects)
    if len(unique_subjects) >= 3:
        for need_split in ['train', 'val', 'test']:
            if len(split_subjects[need_split]) > 0:
                continue

            donor = None
            donor_size = -1
            for candidate in ['train', 'val', 'test']:
                if candidate == need_split:
                    continue
                if len(split_subjects[candidate]) <= 1:
                    continue
                if assigned[candidate] > donor_size:
                    donor = candidate
                    donor_size = assigned[candidate]

            if donor is None:
                continue

            # Move the smallest subject from donor to minimize disturbance
            donor_subj = min(split_subjects[donor], key=lambda subj_id: next((row['n_samples'] for row in subject_meta if row['subj_id'] == subj_id), 0))
            split_subjects[donor].remove(donor_subj)
            split_subjects[need_split].append(donor_subj)
            moved_n = next((row['n_samples'] for row in subject_meta if row['subj_id'] == donor_subj), 0)
            assigned[donor] -= moved_n
            assigned[need_split] += moved_n

    def _collect(split_name: str) -> np.ndarray:
        subjects = split_subjects[split_name]
        if len(subjects) == 0:
            return np.array([], dtype=np.int64)

        # Collect all indices for subjects in this split
        split_indices = []
        for subj_id in subjects:
            subj_mask = subject_ids == subj_id
            subj_indices = np.where(subj_mask)[0]
            split_indices.extend(subj_indices)

        idx = np.array(split_indices, dtype=np.int64)
        # Sort by recording and time within each recording for sequential processing
        return _sort_indices_by_timeline(idx, recording_ids, sample_end_times_s)

    train_idx = _collect('train')
    val_idx = _collect('val')
    test_idx = _collect('test')

    logger.info(f"Patient-sequential split: {len(split_subjects['train'])} train subjects, {len(split_subjects['val'])} val subjects, {len(split_subjects['test'])} test subjects")
    logger.info(f"Sample counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    return train_idx, val_idx, test_idx


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
                # Symmetric context around onset (e.g., 60 min => ~-30/+30).
                desired_start_s = onset_time_s - 0.5 * window_seconds
            else:
                onset_time_s = float(rec_times_s[len(rec_times_s) // 2])
                desired_start_s = onset_time_s - 0.5 * window_seconds

            min_start_s = float(rec_times_s[0])
            max_start_s = max(min_start_s, float(rec_times_s[-1]) - window_seconds)
            start_s = float(np.clip(desired_start_s, min_start_s, max_start_s))
            end_s = start_s + window_seconds

            in_window = (rec_times_s >= start_s) & (rec_times_s <= end_s)
            if np.sum(in_window) >= 2:
                selected = rec_indices[in_window]
                # Onset-relative axis: negative before onset, positive after.
                time_minutes = (rec_times_s[in_window] - onset_time_s) / 60.0
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
            time_minutes = (rec_times_s[start:end] - onset_time_s) / 60.0
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

    time_minutes = (selected.astype(np.float64) - float(anchor_idx)) * step_s / 60.0
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


def _configure_long_sweep_online_augmentation(train_dataset, config: Config, args):
    """Enable memory-light augmentation for long-sweep training datasets."""
    if not getattr(args, 'long_sweep_training', False):
        return train_dataset
    if not bool(getattr(config.data, 'augment_preictal', True)):
        logger.info("Long-sweep mode: preictal augmentation disabled (augment_preictal=false)")
        return train_dataset
    if not bool(getattr(config.data, 'online_preictal_augmentation', True)):
        logger.info("Long-sweep mode: online preictal augmentation disabled by config")
        return train_dataset

    prob = float(np.clip(getattr(config.data, 'online_preictal_augmentation_prob', 0.7), 0.0, 1.0))
    if prob <= 0.0:
        logger.info("Long-sweep mode: online preictal augmentation probability is 0.0")
        return train_dataset

    if hasattr(train_dataset, 'enable_online_preictal_augmentation'):
        train_dataset.enable_online_preictal_augmentation(
            config.data,
            seed=int(config.data.random_seed),
            probability=prob,
            preictal_only=True,
        )
        logger.info(
            "Long-sweep mode: enabled online preictal augmentation (prob=%.2f) without expanding dataset in memory",
            prob,
        )
    else:
        logger.warning("Long-sweep mode: train dataset does not support online augmentation")
    return train_dataset


def _ensure_nonempty_splits(
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    long_sweep_training: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ensure val/test splits are non-empty to keep evaluation stable."""
    train_idx = np.asarray(train_idx, dtype=np.int64)
    val_idx = np.asarray(val_idx, dtype=np.int64)
    test_idx = np.asarray(test_idx, dtype=np.int64)

    def _take_tail(src: np.ndarray, count: int) -> Tuple[np.ndarray, np.ndarray]:
        count = int(min(max(count, 0), len(src)))
        if count <= 0:
            return src, np.array([], dtype=np.int64)
        return src[:-count], src[-count:]

    if len(val_idx) == 0 and len(train_idx) > 1:
        n_move = max(1, len(train_idx) // 10)
        train_idx, moved = _take_tail(train_idx, n_move)
        val_idx = moved
        logger.warning("Split fix: val split was empty; moved %d samples from train to val", len(moved))

    if len(test_idx) == 0:
        if len(val_idx) > 1:
            n_move = max(1, len(val_idx) // 10)
            val_idx, moved = _take_tail(val_idx, n_move)
            test_idx = moved
            logger.warning("Split fix: test split was empty; moved %d samples from val to test", len(moved))
        elif len(train_idx) > 1:
            n_move = max(1, len(train_idx) // 10)
            train_idx, moved = _take_tail(train_idx, n_move)
            test_idx = moved
            logger.warning("Split fix: test split was empty; moved %d samples from train to test", len(moved))

    # Keep chronological order in long-sweep mode.
    if long_sweep_training:
        train_idx = np.sort(train_idx)
        val_idx = np.sort(val_idx)
        test_idx = np.sort(test_idx)

    return train_idx, val_idx, test_idx


def _prepare_real_datasets(config: Config, args, loader: BIDSDataLoader, recordings: List[dict], patient_sequential: bool = False):
    """Build lazy streaming datasets from real BIDS ECG recordings.

    Instead of pre-computing and storing feature tensors (which would require
    ~500 GB for the full SeizeIT2 dataset at 1 s stride), this function:

      1. Extracts/caches the raw ECG signal for each recording as a compact
         float32 binary file (~68 MB per recording, 22 GB total for 325 recs).
      2. Computes only window indices and labels – no feature matrices.
      3. Returns LazyRealDataset instances that compute features on-the-fly
         in __getitem__ via memory-mapped binary files.

    Raw signal binary files are stored in {dataset_root}/.ecg_signal_cache/
    (co-located with the EDF files on the large data disk so the home
    partition is not filled).
    """
    logger.info("Preparing REAL dataset (lazy streaming) from SeizeIT2 ECG + events...")
    prep_start = time.time()

    recordings = _select_recordings_for_real_mode(recordings, args.max_recordings, loader)
    seizure_recordings = sum(1 for recording in recordings if int(recording.get('num_seizures', 0)) > 0)
    logger.info(
        "Selected %d recordings for real mode (%d with seizures)",
        len(recordings),
        seizure_recordings,
    )

    from src.models import ModelFactory as _MF
    is_multimodal = _MF.is_multimodal(config.model.model_type)

    ecg_dim = int(config.model.ecg_feature_dim)
    # For now lazy streaming only supports ECG-only models; multimodal can be
    # extended later by adding EEG/motion binary caches.
    feature_dim = ecg_dim

    logger.info("REAL mode (lazy): ECG feature dim=%d", feature_dim)

    sequence_seconds = 120.0
    stride_seconds = float(config.data.feature_step_s)
    preictal_window_s = float(config.data.pre_ictal_window_s)

    # Raw signal cache on the same disk as the dataset (avoids filling /home)
    dataset_root = Path(config.data.dataset_root)
    signal_cache_dir = dataset_root / ".ecg_signal_cache"
    signal_cache_dir.mkdir(parents=True, exist_ok=True)

    # Metadata collected per window (all lightweight)
    all_rec_indices: List[int] = []
    all_end_indices: List[int] = []
    all_labels_list: List[float] = []
    all_times_list: List[float] = []
    all_rec_ids: List[str] = []
    recordings_meta: List[dict] = []  # one entry per accepted recording

    total_samples = 0

    for rec_idx, recording in enumerate(recordings):
        try:
            rec_start = time.time()
            if rec_idx < 5 or (rec_idx + 1) % 10 == 0:
                logger.info(
                    "[REAL PREP] Recording %d/%d: sub-%s ses-%s run-%s",
                    rec_idx + 1,
                    len(recordings),
                    recording.get("subject_id", "?"),
                    recording.get("session_id", "?"),
                    recording.get("run_id", "?"),
                )

            # --- ensure raw signal binary is cached on disk --------------------
            try:
                bin_path, fs, n_samples = _ensure_raw_signal_cache(
                    recording, loader, signal_cache_dir
                )
            except FileNotFoundError as exc:
                logger.warning("Skipping sub-%s run-%s: %s", recording.get("subject_id"), recording.get("run_id"), exc)
                continue

            seq_samples = int(sequence_seconds * fs)
            stride_samples = max(1, int(stride_seconds * fs))

            # --- seizure onsets -----------------------------------------------
            events_df = loader.load_events_tsv(
                recording["subject_id"],
                recording["session_id"],
                recording["run_id"],
            )
            seizure_onsets = _extract_seizure_onsets(events_df)

            # --- generate window end-indices -----------------------------------
            candidate_end_indices = np.arange(
                seq_samples, n_samples, stride_samples, dtype=np.int64
            )
            if len(candidate_end_indices) == 0:
                continue

            # Long-sweep / patient-sequential: keep all chronological windows
            if (
                not (args.long_sweep_training or args.patient_sequential)
                and args.max_samples_per_recording > 0
                and len(candidate_end_indices) > args.max_samples_per_recording
            ):
                pos = np.linspace(
                    0,
                    len(candidate_end_indices) - 1,
                    num=args.max_samples_per_recording,
                    dtype=np.int64,
                )
                selected_end_indices = candidate_end_indices[pos]
            else:
                selected_end_indices = candidate_end_indices

            # --- vectorised label computation ----------------------------------
            end_times = selected_end_indices.astype(np.float64) / fs
            if len(seizure_onsets) > 0:
                dt = seizure_onsets.astype(np.float64)[None, :] - end_times[:, None]  # (N, K)
                dt[dt < 0] = np.inf
                min_dt = dt.min(axis=1)
                labels_arr = np.where(min_dt <= preictal_window_s, min_dt / 60.0, -1.0).astype(np.float32)
            else:
                labels_arr = np.full(len(selected_end_indices), -1.0, dtype=np.float32)

            # --- register recording and append window metadata -----------------
            meta_idx = len(recordings_meta)
            recording_uid = (
                f"sub-{recording['subject_id']}"
                f"_ses-{recording['session_id']}"
                f"_run-{recording['run_id']}"
            )
            recordings_meta.append({
                "signal_path": str(bin_path),
                "fs": float(fs),
                "seq_samples": seq_samples,
                "n_samples": n_samples,
                "recording_uid": recording_uid,
            })

            n_windows = len(selected_end_indices)
            all_rec_indices.extend([meta_idx] * n_windows)
            all_end_indices.extend(selected_end_indices.tolist())
            all_labels_list.extend(labels_arr.tolist())
            all_times_list.extend(end_times.tolist())
            all_rec_ids.extend([recording_uid] * n_windows)
            total_samples += n_windows

            if rec_idx < 5 or (rec_idx + 1) % 10 == 0:
                elapsed = time.time() - prep_start
                avg = elapsed / max(1, rec_idx + 1)
                eta = avg * max(0, len(recordings) - rec_idx - 1)
                logger.info(
                    "[REAL PREP] Done %d/%d | +%d windows (total=%d) | rec_time=%.1fs | ETA=%.1f min",
                    rec_idx + 1, len(recordings),
                    n_windows, total_samples,
                    time.time() - rec_start,
                    eta / 60.0,
                )

        except Exception as error:
            import traceback as _tb
            logger.debug("Skipping recording: %s\n%s", error, _tb.format_exc())
            continue

    if total_samples < 50:
        logger.warning("Real dataset extraction produced too few samples; falling back to dummy mode.")
        return None

    logger.info(
        "[REAL PREP] Completed in %.1f min with %d total windows across %d recordings",
        (time.time() - prep_start) / 60.0,
        total_samples,
        len(recordings_meta),
    )

    # Convert to arrays
    rec_indices_arr = np.array(all_rec_indices, dtype=np.int32)
    end_indices_arr = np.array(all_end_indices, dtype=np.int64)
    labels_arr = np.array(all_labels_list, dtype=np.float32)
    times_arr = np.array(all_times_list, dtype=np.float32)
    recording_ids = np.array(all_rec_ids)

    # --- train/val/test split --------------------------------------------------
    if patient_sequential:
        train_idx, val_idx, test_idx = _split_indices_by_patient_sequential(
            labels_arr,
            recording_ids=recording_ids,
            sample_end_times_s=times_arr,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )
    elif args.long_sweep_training:
        train_idx, val_idx, test_idx = _split_indices_by_recording_timeline(
            labels_arr,
            recording_ids=recording_ids,
            sample_end_times_s=times_arr,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )
    else:
        train_idx, val_idx, test_idx = _split_indices_with_preictal_coverage(
            labels_arr,
            train_ratio=config.data.train_ratio,
            val_ratio=config.data.val_ratio,
            seed=config.data.random_seed,
        )

    train_idx, val_idx, test_idx = _ensure_nonempty_splits(
        train_idx, val_idx, test_idx,
        long_sweep_training=bool(args.long_sweep_training or args.patient_sequential),
    )

    def _make_subject_ids(idx):
        return np.array([r.split("_")[0].replace("sub-", "") for r in recording_ids[idx]])

    subject_ids_train = _make_subject_ids(train_idx) if patient_sequential else None
    subject_ids_val = _make_subject_ids(val_idx) if patient_sequential else None
    subject_ids_test = _make_subject_ids(test_idx) if patient_sequential else None

    def _make_split(idx, subject_ids):
        return LazyRealDataset(
            rec_indices=rec_indices_arr[idx],
            end_indices=end_indices_arr[idx],
            labels=labels_arr[idx],
            sample_end_times_s=times_arr[idx],
            recording_ids=recording_ids[idx],
            recordings_meta=recordings_meta,
            feature_dim=feature_dim,
            subject_ids=subject_ids,
        )

    train_dataset = _make_split(train_idx, subject_ids_train)
    train_dataset = _configure_long_sweep_online_augmentation(train_dataset, config, args)
    val_dataset = _make_split(val_idx, subject_ids_val)
    test_dataset = _make_split(test_idx, subject_ids_test)

    logger.info(
        "Real dataset ready: total=%d, train=%d, val=%d, test=%d",
        total_samples,
        len(train_dataset),
        len(val_dataset),
        len(test_dataset),
    )
    logger.info("Train class distribution: %s", train_dataset.class_distribution)
    logger.info("Val   class distribution: %s", val_dataset.class_distribution)
    logger.info("Test  class distribution: %s", test_dataset.class_distribution)

    return train_dataset, val_dataset, test_dataset


def _prepare_wearable_datasets(config: Config, args, loader: WearableDeviceDataLoader, recordings: List[dict], patient_sequential: bool = False):
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

    # For BIDS data, validate that recordings have actual data files
    # For wearable data, validation is done during get_all_recordings()
    if args.data_source == 'bids':
        recordings = _select_recordings_for_real_mode(recordings, args.max_recordings, loader)
    
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

            # Optional cap per recording for faster iteration.
            # In long-sweep mode we keep the earliest contiguous windows to
            # preserve timeline semantics and avoid random jumps.
            if args.max_samples_per_recording > 0 and len(features) > args.max_samples_per_recording:
                if args.long_sweep_training or args.patient_sequential:
                    keep_positions = np.arange(args.max_samples_per_recording, dtype=np.int64)
                else:
                    keep_positions = np.linspace(
                        0,
                        len(features) - 1,
                        num=args.max_samples_per_recording,
                        dtype=np.int64,
                    )
                features = features[keep_positions]
                labels = labels[keep_positions]
                sample_times_s = sample_times_s[keep_positions]

            if not (args.long_sweep_training or args.patient_sequential):
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
            if max_global > 0:
                remaining = max_global - total_kept
                if remaining <= 0:
                    logger.info(
                        "Reached wearable global sample budget (%d); skipping remaining recordings",
                        max_global,
                    )
                    break

                if len(labels) > remaining:
                    if args.long_sweep_training or args.patient_sequential:
                        # Preserve strict chronology in long-sweep mode.
                        chosen_local = np.arange(remaining, dtype=np.int64)
                    else:
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
    if max_global > 0 and n_before_cap > max_global:
        logger.warning(
            "Wearable dataset has %d samples; applying global cap max_wearable_global_samples=%d",
            n_before_cap,
            max_global,
        )
        if args.long_sweep_training or args.patient_sequential:
            # Keep earliest windows to preserve chronological training flow.
            chosen = np.arange(max_global, dtype=np.int64)
        else:
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

    if args.long_sweep_training or args.patient_sequential:
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

    train_idx, val_idx, test_idx = _ensure_nonempty_splits(
        train_idx,
        val_idx,
        test_idx,
        long_sweep_training=bool(args.long_sweep_training or args.patient_sequential),
    )

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    val_sample_end_times_s = sample_end_times_s[val_idx]
    val_recording_ids = recording_ids[val_idx]
    test_sample_end_times_s = sample_end_times_s[test_idx]
    test_recording_ids = recording_ids[test_idx]
    n_total = len(y)

    if args.long_sweep_training or patient_sequential:
        logger.info("Long-sweep mode: using online preictal augmentation (no offline expansion)")
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
    train_dataset = _configure_long_sweep_online_augmentation(train_dataset, config, args)
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
    cache_root_override = os.environ.get('EPILEPSEE_CACHE_DIR')
    if cache_root_override:
        cache_root = Path(cache_root_override).expanduser()
    else:
        cache_root = Path(config.training.save_dir).expanduser() / 'cache' / 'datasets'
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
        # Bump when long-sweep split/cap/augmentation semantics change.
        'window_sampling_strategy_version': 7,  # lazy streaming format
        'feature_step_s': float(config.data.feature_step_s),
        'pre_ictal_window_s': float(config.data.pre_ictal_window_s),
        'seed': int(config.data.random_seed),
        'train_ratio': float(config.data.train_ratio),
        'val_ratio': float(config.data.val_ratio),
        'augment_preictal': bool(config.data.augment_preictal),
        'augmentation_factor': int(config.data.augmentation_factor) if config.data.augment_preictal else 0,
        'long_sweep_training': bool(getattr(args, 'long_sweep_training', False)),
        'patient_sequential': bool(getattr(args, 'patient_sequential', False)),
    }
    key = hashlib.md5(json.dumps(key_payload, sort_keys=True).encode('utf-8')).hexdigest()[:16]
    return cache_root / f"real_dataset_{key}.npz"


def _save_real_dataset_cache(cache_path: Path, train_dataset, val_dataset, test_dataset) -> None:
    """Persist LazyRealDataset metadata to a compact NPZ cache.

    For LazyRealDataset the cache stores only window indices + labels
    (kilobytes per recording, not gigabytes).  For legacy SeizureDataset the
    original full-features NPZ format is preserved.
    """
    tmp_path = cache_path.with_suffix('.tmp.npz')

    def _s(ds, key):
        v = getattr(ds, key, None)
        return np.array([], dtype=np.float32) if v is None else np.asarray(v)

    if isinstance(train_dataset, LazyRealDataset):
        np.savez_compressed(
            tmp_path,
            # schema tag
            lazy_format=np.array([1], dtype=np.int8),
            # recordings metadata (JSON-encoded)
            recordings_json=np.array([json.dumps(train_dataset.recordings_meta)]),
            feature_dim=np.array([train_dataset.feature_dim], dtype=np.int32),
            # train
            train_rec_indices=train_dataset.rec_indices,
            train_end_indices=train_dataset.end_indices,
            train_labels=train_dataset.labels,
            train_times=train_dataset.sample_end_times_s,
            train_rec_ids=train_dataset.recording_ids,
            train_subject_ids=(np.array([], dtype='<U1') if train_dataset.subject_ids is None else np.asarray(train_dataset.subject_ids)),
            # val
            val_rec_indices=val_dataset.rec_indices,
            val_end_indices=val_dataset.end_indices,
            val_labels=val_dataset.labels,
            val_times=val_dataset.sample_end_times_s,
            val_rec_ids=val_dataset.recording_ids,
            val_subject_ids=(np.array([], dtype='<U1') if val_dataset.subject_ids is None else np.asarray(val_dataset.subject_ids)),
            # test
            test_rec_indices=test_dataset.rec_indices,
            test_end_indices=test_dataset.end_indices,
            test_labels=test_dataset.labels,
            test_times=test_dataset.sample_end_times_s,
            test_rec_ids=test_dataset.recording_ids,
            test_subject_ids=(np.array([], dtype='<U1') if test_dataset.subject_ids is None else np.asarray(test_dataset.subject_ids)),
        )
    else:
        np.savez_compressed(
            tmp_path,
            train_features=train_dataset.features,
            train_labels=train_dataset.labels,
            val_features=val_dataset.features,
            val_labels=val_dataset.labels,
            val_sample_end_times_s=_s(val_dataset, 'sample_end_times_s'),
            val_recording_ids=(np.array([], dtype='<U1') if getattr(val_dataset, 'recording_ids', None) is None else np.asarray(val_dataset.recording_ids)),
            test_features=test_dataset.features,
            test_labels=test_dataset.labels,
            test_sample_end_times_s=_s(test_dataset, 'sample_end_times_s'),
            test_recording_ids=(np.array([], dtype='<U1') if getattr(test_dataset, 'recording_ids', None) is None else np.asarray(test_dataset.recording_ids)),
        )
    os.replace(tmp_path, cache_path)


def _load_real_dataset_cache(cache_path: Path):
    """Load cached real datasets (supports both lazy-metadata and legacy full-features formats)."""
    data = np.load(cache_path, allow_pickle=False)

    if 'lazy_format' in data:
        # --- lazy metadata format -------------------------------------------
        recordings_meta = json.loads(str(data['recordings_json'][0]))
        feature_dim = int(data['feature_dim'][0])

        def _opt_arr(key):
            v = data[key] if key in data else None
            return None if (v is None or len(v) == 0) else v

        def _make(prefix):
            subj = _opt_arr(f'{prefix}_subject_ids')
            return LazyRealDataset(
                rec_indices=data[f'{prefix}_rec_indices'],
                end_indices=data[f'{prefix}_end_indices'],
                labels=data[f'{prefix}_labels'],
                sample_end_times_s=data[f'{prefix}_times'],
                recording_ids=data[f'{prefix}_rec_ids'],
                recordings_meta=recordings_meta,
                feature_dim=feature_dim,
                subject_ids=subj,
            )

        return _make('train'), _make('val'), _make('test')

    # --- legacy full-features format ----------------------------------------
    from src import SeizureDataset

    def _opt(key):
        v = data[key] if key in data else None
        return None if (v is None or len(v) == 0) else v

    train_dataset = SeizureDataset(data['train_features'], data['train_labels'])
    val_dataset = SeizureDataset(
        data['val_features'], data['val_labels'],
        sample_end_times_s=_opt('val_sample_end_times_s'),
        recording_ids=_opt('val_recording_ids'),
    )
    test_dataset = SeizureDataset(
        data['test_features'], data['test_labels'],
        sample_end_times_s=_opt('test_sample_end_times_s'),
        recording_ids=_opt('test_recording_ids'),
    )
    return train_dataset, val_dataset, test_dataset


def _cleanup_temp_real_memmaps(*datasets) -> None:
    """No-op: LazyRealDataset uses no temporary memmaps."""
    pass


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
        patient_sequential = bool(
            args.patient_sequential
            or (str(getattr(config.training, 'training_mode', 'stateless')).lower() == 'stateful'
                and str(getattr(config.training, 'batching_strategy', 'random')).lower() == 'patient_sequential')
        )
        world_size = int(os.environ.get('WORLD_SIZE', '1'))
        rank = int(os.environ.get('RANK', '0'))
        cache_path = _real_cache_path(config, args)

        if cache_path.exists():
            logger.info(f"Loading real dataset from cache: {cache_path}")
            train_dataset, val_dataset, test_dataset = _load_real_dataset_cache(cache_path)
            train_dataset = _configure_long_sweep_online_augmentation(train_dataset, config, args)
            return train_dataset, val_dataset, test_dataset

        if world_size == 1 or rank == 0:
            if args.data_source == 'wearable':
                prepared = _prepare_wearable_datasets(config, args, wearable_loader, recordings, patient_sequential=patient_sequential)
            else:
                prepared = _prepare_real_datasets(config, args, loader, recordings, patient_sequential=patient_sequential)
            if prepared is not None:
                _save_real_dataset_cache(cache_path, *prepared)
                logger.info(f"Saved real dataset cache: {cache_path}")
                _cleanup_temp_real_memmaps(*prepared)
                logger.info(f"Loading real dataset from cache after save: {cache_path}")
                return _load_real_dataset_cache(cache_path)
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

    def make_subject_metadata(num_samples: int, split_name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        num_subjects = max(4, min(16, num_samples // 64))
        subject_ids = np.array(
            [f"{split_name}-sub-{subject_idx % num_subjects:03d}" for subject_idx in range(num_samples)],
            dtype=object,
        )
        recording_ids = np.array(
            [f"{subject_id}-rec-000" for subject_id in subject_ids],
            dtype=object,
        )
        sample_end_times_s = np.zeros(num_samples, dtype=np.float32)
        for subject_id in np.unique(subject_ids):
            subject_mask = subject_ids == subject_id
            sample_end_times_s[subject_mask] = np.arange(np.sum(subject_mask), dtype=np.float32) * float(config.data.feature_step_s)
        return subject_ids, recording_ids, sample_end_times_s

    X_train = np.random.randn(n_train, 600, feature_dim).astype(np.float32)
    y_train = make_labels(n_train, preictal_ratio)
    train_subject_ids, train_recording_ids, train_end_times = make_subject_metadata(n_train, 'train')
    
    X_val = np.random.randn(n_val, 600, feature_dim).astype(np.float32)
    y_val = make_labels(n_val, preictal_ratio)
    val_subject_ids, val_recording_ids, val_end_times = make_subject_metadata(n_val, 'val')
    
    X_test = np.random.randn(n_test, 600, feature_dim).astype(np.float32)
    y_test = make_labels(n_test, preictal_ratio)
    test_subject_ids, test_recording_ids, test_end_times = make_subject_metadata(n_test, 'test')
    
    # Import SeizureDataset
    from src import SeizureDataset
    
    train_dataset = SeizureDataset(
        X_train,
        y_train,
        subject_ids=train_subject_ids,
        sample_end_times_s=train_end_times,
        recording_ids=train_recording_ids,
    )
    val_dataset = SeizureDataset(
        X_val,
        y_val,
        subject_ids=val_subject_ids,
        sample_end_times_s=val_end_times,
        recording_ids=val_recording_ids,
    )
    test_dataset = SeizureDataset(
        X_test,
        y_test,
        subject_ids=test_subject_ids,
        sample_end_times_s=test_end_times,
        recording_ids=test_recording_ids,
    )
    
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

    if _OPTIONAL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Training dependencies failed to import. "
            "Install project requirements (for example via environment.yml) "
            "and retry."
        ) from _OPTIONAL_IMPORT_ERROR

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
    if args.patient_sequential:
        config.training.long_sweep_training = True
        config.training.use_weighted_sampling = False
    if args.training_mode is not None:
        config.training.training_mode = str(args.training_mode)
    if args.batching_strategy is not None:
        config.training.batching_strategy = str(args.batching_strategy)
    config.training.optimizer_step_scope = str(args.optimizer_step_scope)
    config.training.patients_per_step = max(1, int(args.patients_per_step))
    if args.augment_preictal is not None:
        config.data.augment_preictal = bool(args.augment_preictal)
    if args.online_aug_prob is not None:
        config.data.online_preictal_augmentation_prob = float(np.clip(args.online_aug_prob, 0.0, 1.0))
    if args.bayes_memory_eval is not None:
        config.evaluation.enable_bayesian_memory_eval = bool(args.bayes_memory_eval)
    config.training.preflight_visualization = bool(getattr(args, 'preflight_visualization', True))
    config.training.preflight_max_val_batches = int(getattr(args, 'preflight_max_val_batches', 0))
    if args.dataset_root:
        config.data.dataset_root = args.dataset_root
    elif args.data_source == 'wearable':
        root_hint = str(config.data.dataset_root).lower()
        looks_like_bids_root = ('ds005873' in root_hint) or ('seizeit2' in root_hint)
        if looks_like_bids_root:
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

    # ── Loss type ─────────────────────────────────────────────────────────────
    # --state-loss: safe 3-class classification, no countdown regression.
    # Stores the choice in config.loss so Trainer.train() picks it up.
    if getattr(args, 'state_loss', False):
        config.loss.loss_type = 'state'
        config.loss.onset_threshold_min = float(getattr(args, 'onset_threshold_min', 2.0))
        logger.info(
            "Loss mode: SeizureStateLoss (3-class, no GT countdown regression). "
            "Onset window: %.1f min.", config.loss.onset_threshold_min
        )
    else:
        config.loss.loss_type = 'countdown'

    # Device
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    
    if args.no_cuda:
        device = torch.device('cpu')
        config.training.distributed = False
    elif world_size > 1:
        # In DDP mode, use GPU for each local rank
        device = torch.device(f'cuda:{local_rank}')
        # If launched with torchrun, force distributed mode on regardless of YAML default.
        config.training.distributed = True
    elif torch.cuda.is_available():
        # Single GPU mode
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
        config.training.distributed = False
    
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

    # ── Ring buffer ───────────────────────────────────────────────────────────
    # Wrap train_dataset in a TemporalRingBufferDataset so only the most recent
    # N samples are visible per epoch, evicting older data as training advances.
    ring_buffer_size = int(getattr(args, 'ring_buffer_size', 0))
    if ring_buffer_size > 0 and len(train_dataset) > ring_buffer_size:
        from src.data_loader import TemporalRingBufferDataset
        train_dataset = TemporalRingBufferDataset(
            train_dataset, ring_buffer_size=ring_buffer_size
        )
        logger.info(
            "Ring-buffer training enabled: window=%d samples out of %d total. "
            "Window advances ~10%% per epoch.",
            ring_buffer_size,
            len(train_dataset._dataset),
        )
    elif ring_buffer_size > 0:
        logger.info(
            "Ring-buffer size %d >= dataset size %d — windowing disabled.",
            ring_buffer_size, len(train_dataset),
        )

    # Create trainer
    if args.wandb_mode in ('online', 'offline'):
        os.environ['WANDB_MODE'] = args.wandb_mode
    use_wandb = args.wandb_mode != 'disabled'

    trainer = Trainer(
        config,
        device=device,
        use_wandb=use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    
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
    bayes_enabled = bool(getattr(config.evaluation, 'enable_bayesian_memory_eval', True))
    if bayes_enabled and test_times is not None and test_rec_ids is not None:
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

    # Per-patient difficulty analytics for triaging hard/easy subjects and
    # selecting representative examples for inspection.
    test_subject_ids = getattr(test_dataset, 'subject_ids', None)
    patient_report = _compute_patient_report(
        prediction_payload=prediction_payload,
        subject_ids=(None if test_subject_ids is None else np.asarray(test_subject_ids)),
        threshold=float(config.loss.detection_threshold),
        countdown_max=float(config.model.output_countdown_max),
        top_k=10,
        top_examples=30,
    )
    patient_report_path = trainer.save_dir / "patient_difficulty_report.json"
    with open(patient_report_path, "w", encoding="utf-8") as fh:
        json.dump(_to_jsonable(patient_report), fh, indent=2)

    patient_csv_path = trainer.save_dir / "patient_difficulty_report.csv"
    patient_rows = patient_report.get('patients', []) if isinstance(patient_report, dict) else []
    if patient_rows:
        fieldnames = [
            'patient_id', 'n_samples', 'n_preictal', 'n_interictal',
            'difficulty_score', 'alert_balanced_accuracy', 'alert_sensitivity',
            'alert_specificity', 'alert_precision', 'alert_f1',
            'mean_prob_abs_error', 'preictal_countdown_mae',
        ]
        with open(patient_csv_path, 'w', newline='', encoding='utf-8') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in patient_rows:
                writer.writerow({k: row.get(k, None) for k in fieldnames})

    # Append patient analytics metadata into results artifact.
    results['patient_report'] = {
        'n_patients': int(patient_report.get('n_patients', 0)) if isinstance(patient_report, dict) else 0,
        'hardest_patient': (
            patient_report.get('hardest_patients', [{}])[0] if isinstance(patient_report, dict) and patient_report.get('hardest_patients') else None
        ),
        'easiest_patient': (
            patient_report.get('easiest_patients', [{}])[0] if isinstance(patient_report, dict) and patient_report.get('easiest_patients') else None
        ),
        'json_path': str(patient_report_path),
        'csv_path': str(patient_csv_path),
    }
    with open(results_path, "w", encoding="utf-8") as file_handle:
        json.dump(_to_jsonable(results), file_handle, indent=2)

    hardest_patients = patient_report.get('hardest_patients', []) if isinstance(patient_report, dict) else []
    easiest_patients = patient_report.get('easiest_patients', []) if isinstance(patient_report, dict) else []
    if hardest_patients:
        logger.info(
            "Hardest test patient: %s | difficulty=%.4f | bal_acc=%.4f",
            str(hardest_patients[0].get('patient_id', 'unknown')),
            float(hardest_patients[0].get('difficulty_score', 0.0)),
            float(hardest_patients[0].get('alert_balanced_accuracy', 0.0)),
        )
    if easiest_patients:
        logger.info(
            "Easiest test patient: %s | difficulty=%.4f | bal_acc=%.4f",
            str(easiest_patients[0].get('patient_id', 'unknown')),
            float(easiest_patients[0].get('difficulty_score', 0.0)),
            float(easiest_patients[0].get('alert_balanced_accuracy', 0.0)),
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
    if isinstance(test_dataset, LazyRealDataset):
        # Lazy dataset: materialise only what the visualiser needs by computing
        # features for the first 2000 test windows (enough for a representative
        # panel) without loading all windows into RAM.
        n_vis = min(len(test_dataset), 2000)
        _vis_feats = np.stack([
            test_dataset[i][0].numpy() for i in range(n_vis)
        ])
        features_arr = _vis_feats
        logger.info("LazyRealDataset: materialised %d/%d test windows for visualisation", n_vis, len(test_dataset))
    else:
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

    # Guard against index-space mismatch when visualization uses a partially
    # materialized proxy (e.g., 2000 windows materialized out of a larger set).
    panel_indices = np.asarray(panel_indices, dtype=np.int64)
    panel_time_minutes = np.asarray(panel_time_minutes, dtype=np.float64)
    proxy_len = int(len(signal_proxy))
    valid_panel_mask = (panel_indices >= 0) & (panel_indices < proxy_len)
    invalid_count = int(np.sum(~valid_panel_mask))
    if invalid_count > 0:
        max_requested = int(panel_indices.max()) if panel_indices.size > 0 else -1
        logger.warning(
            "Dropping %d invalid panel indices (max=%d, proxy_len=%d)",
            invalid_count,
            max_requested,
            proxy_len,
        )
        panel_indices = panel_indices[valid_panel_mask]
        panel_time_minutes = panel_time_minutes[valid_panel_mask]

    if panel_indices.size == 0:
        logger.warning(
            "No valid panel indices remain after filtering; using fallback tail window."
        )
        fallback_points = min(
            proxy_len,
            max(1, int(round(float(args.panel_window_minutes) * 60.0 / max(float(config.data.feature_step_s), 1e-6)))),
        )
        panel_indices = np.arange(proxy_len - fallback_points, proxy_len, dtype=np.int64)
        panel_time_minutes = (
            (panel_indices - panel_indices[0]).astype(np.float64) * float(config.data.feature_step_s) / 60.0
        )

    panel_effective_hz = 0.0
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

    # For sparse BIDS panel timelines (caused by per-recording sample caps),
    # rebuild the panel from dense raw ECG and EEG samples from the selected
    # recording/time window, then align predictions by interpolation.
    dense_panel_used = False
    dense_panel_step_s = float(config.data.feature_step_s)

    # Extract predictions for selected windows
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

    # Extract full feature windows (not just middle sample) for denser signal representation.
    # features_arr shape: (N_windows, T_samples, N_features)
    # panel_indices indexes into the first dimension (window indices).
    # This extracts shape (n_panel, T_samples, N_features), then concatenates along time.
    n_panel = len(panel_indices)
    window_time_samples = features_arr.shape[1]
    
    if n_panel > 0 and window_time_samples > 1:
        # Expand predictions to match expanded signal (repeat each value window_time_samples times)
        panel_true_countdown = np.repeat(panel_true_countdown, window_time_samples)
        panel_pred_countdown = np.repeat(panel_pred_countdown, window_time_samples)
        panel_pred_preictal = np.repeat(panel_pred_preictal, window_time_samples)
        if panel_pred_smooth is not None:
            panel_pred_smooth = np.repeat(panel_pred_smooth, window_time_samples)
    
    if n_panel > 0 and window_time_samples > 1:
        # Extract full windows and reshape to (n_panel * T_samples, N_features)
        panel_features_full = features_arr[panel_indices]  # (n_panel, T_samples, N_features)
        panel_features_flat = panel_features_full.reshape(n_panel * window_time_samples, -1)
        
        # Reconstruct dense time axis accounting for intra-window sampling
        # Each window spans feature_step_s seconds with window_time_samples samples.
        intra_window_dt_s = float(config.data.feature_step_s) / max(window_time_samples - 1, 1)
        panel_time_dense_minutes = []
        for i, win_idx in enumerate(panel_indices):
            win_time = panel_time_minutes[i] if i < len(panel_time_minutes) else 0.0
            # Generate T samples within this window, offset by intra-window times
            intra_times_s = np.arange(window_time_samples) * intra_window_dt_s
            intra_times_min = intra_times_s / 60.0
            panel_time_dense_minutes.extend(win_time + intra_times_min)
        panel_time_dense_minutes = np.asarray(panel_time_dense_minutes, dtype=np.float64)
        
        # Extract dense signals per channel
        panel_ecg_raw = panel_features_flat[:, best_ch].astype(np.float32)
        panel_ppg_raw = panel_features_flat[:, 0].astype(np.float32)
        panel_eda_raw = panel_features_flat[:, 2].astype(np.float32) if features_arr.shape[-1] > 2 else None
        
        if config.model.model_type in ('multimodal', 'multimodal_transformer'):
            ecg_dim = int(config.model.ecg_feature_dim)
            eeg_dim = int(config.model.eeg_feature_dim)
            if features_arr.shape[-1] >= ecg_dim + max(1, eeg_dim):
                panel_eeg_raw = panel_features_flat[:, ecg_dim].astype(np.float32)
            else:
                panel_eeg_raw = None
        else:
            panel_eeg_raw = None
        
        # Update panel_time_minutes to dense version for visualization
        panel_time_minutes = panel_time_dense_minutes
        
        logger.info(
            "Expanded sparse panel from %d points to %d dense samples "
            "(intra-window dt=%.3f s, effective ~%.2f Hz)",
            n_panel,
            len(panel_ecg_raw),
            intra_window_dt_s,
            1.0 / max(intra_window_dt_s, 1e-6),
        )
        # Also expand HR and HRV to match the expanded signal
        panel_hr_raw = np.repeat(panel_hr_full[panel_indices], window_time_samples).astype(np.float32)
        panel_hrv_raw = np.repeat(panel_hrv_full[panel_indices], window_time_samples).astype(np.float32)
    else:
        # Fallback to single-sample per window if sparse structure
        panel_ecg_raw = signal_proxy[panel_indices]
        panel_ppg_raw = panel_ppg_full[panel_indices] if panel_ppg_full is not None else panel_ecg_raw
        panel_eda_raw = panel_eda_full[panel_indices] if panel_eda_full is not None else None
        panel_eeg_raw = panel_eeg_full[panel_indices] if panel_eeg_full is not None else None
        panel_hr_raw = panel_hr_full[panel_indices]
        panel_hrv_raw = panel_hrv_full[panel_indices]
    
    panel_emg_raw = None
    panel_mov_raw = None
    panel_adxl_raw = None
    eeg_signal_label = "EEG proxy"

    rec_ids_meta = getattr(test_dataset, 'recording_ids', None)
    times_meta = getattr(test_dataset, 'sample_end_times_s', None)
    panel_loader = None
    if data_source == "bids" and BIDSDataLoader is not None:
        try:
            panel_loader = BIDSDataLoader(config.data)
        except Exception as panel_loader_exc:
            logger.warning("Could not initialize BIDS loader for dense panel modalities: %s", str(panel_loader_exc))
    can_dense_rebuild = (
        data_source == "bids"
        and isinstance(test_dataset, LazyRealDataset)
        and rec_ids_meta is not None
        and times_meta is not None
        and len(panel_indices) >= 2
    )
    logger.info(
        "Panel dense rebuild eval | data_source=%s | LazyReal=%s | rec_meta=%s | times_meta=%s | panel_indices=%d | loader=%s",
        data_source,
        isinstance(test_dataset, LazyRealDataset),
        rec_ids_meta is not None,
        times_meta is not None,
        len(panel_indices),
        'ok' if panel_loader is not None else 'missing',
    )

    if can_dense_rebuild:
        # Snapshot sparse arrays so we can restore them if the dense path
        # fails mid-way (preventing a size mismatch between panel arrays).
        _sparse_panel_pred_preictal = panel_pred_preictal
        _sparse_panel_pred_countdown = panel_pred_countdown
        _sparse_panel_true_countdown = panel_true_countdown
        _sparse_panel_pred_smooth = panel_pred_smooth
        _sparse_panel_time_minutes = panel_time_minutes
        try:
            rec_ids_meta_arr = np.asarray(rec_ids_meta)
            times_meta_arr = np.asarray(times_meta, dtype=np.float64)

            rec_ids_panel = rec_ids_meta_arr[panel_indices]
            times_panel_abs_s = times_meta_arr[panel_indices]
            unique_rec, rec_counts = np.unique(rec_ids_panel, return_counts=True)
            chosen_rec = str(unique_rec[int(np.argmax(rec_counts))])

            # Prefer recordings with richer modalities (EEG/EMG/MOV) for
            # multimodal panel interpretability.
            if panel_loader is not None:
                modality_rank = []
                for rec_id, freq in zip(unique_rec, rec_counts):
                    rec_name = str(rec_id)
                    try:
                        parts = rec_name.split('_')
                        s_id = parts[0].replace('sub-', '')
                        sess_id = parts[1].replace('ses-', '')
                        r_id = int(parts[2].replace('run-', ''))
                    except Exception:
                        modality_rank.append((-1, int(freq), rec_name))
                        continue

                    score = 0
                    for dtype in ('eeg', 'emg', 'mov'):
                        try:
                            panel_loader.resolve_subject_edf_path(s_id, session_id=sess_id, datatype=dtype, run_id=r_id)
                            score += 1
                        except Exception:
                            continue
                    modality_rank.append((score, int(freq), rec_name))

                if modality_rank:
                    modality_rank.sort(key=lambda x: (x[0], x[1]), reverse=True)
                    chosen_score, chosen_freq, chosen_name = modality_rank[0]
                    chosen_rec = str(chosen_name)
                    logger.info(
                        "Dense panel recording preference: %s (modalities=%d, panel_windows=%d)",
                        chosen_rec,
                        chosen_score,
                        chosen_freq,
                    )

                    # If panel-selected recordings have no multimodal support,
                    # fall back to a modality-rich recording from the test set.
                    if chosen_score <= 0:
                        global_recs, global_counts = np.unique(rec_ids_meta_arr, return_counts=True)
                        global_rank = []
                        for rec_id, rec_n in zip(global_recs, global_counts):
                            rec_name = str(rec_id)
                            if int(rec_n) < 2:
                                continue
                            try:
                                parts = rec_name.split('_')
                                s_id = parts[0].replace('sub-', '')
                                sess_id = parts[1].replace('ses-', '')
                                r_id = int(parts[2].replace('run-', ''))
                            except Exception:
                                continue

                            score = 0
                            for dtype in ('eeg', 'emg', 'mov'):
                                try:
                                    panel_loader.resolve_subject_edf_path(s_id, session_id=sess_id, datatype=dtype, run_id=r_id)
                                    score += 1
                                except Exception:
                                    continue
                            global_rank.append((score, int(rec_n), rec_name))

                        if global_rank:
                            global_rank.sort(key=lambda x: (x[0], x[1]), reverse=True)
                            best_score, best_count, best_rec = global_rank[0]
                            if best_score > chosen_score:
                                chosen_rec = str(best_rec)
                                # Re-center dense panel window within chosen recording.
                                rec_all_times = np.sort(times_meta_arr[rec_ids_meta_arr == chosen_rec])
                                if rec_all_times.size >= 2:
                                    center_s = float(rec_all_times[rec_all_times.size // 2])
                                    half_window_s = max(60.0, float(args.panel_window_minutes) * 30.0)
                                    start_s = max(float(rec_all_times[0]), center_s - half_window_s)
                                    end_s = min(float(rec_all_times[-1]), center_s + half_window_s)
                                    if end_s <= start_s:
                                        start_s = float(rec_all_times[0])
                                        end_s = float(rec_all_times[-1])
                                    times_panel_abs_s = np.array([start_s, end_s], dtype=np.float64)
                                logger.info(
                                    "Dense panel fallback recording: %s (modalities=%d, windows=%d)",
                                    chosen_rec,
                                    best_score,
                                    best_count,
                                )
            parts = chosen_rec.split('_')
            subject_id = parts[0].replace('sub-', '')
            session_id = parts[1].replace('ses-', '')
            run_id = int(parts[2].replace('run-', ''))

            rec_mask_all = rec_ids_meta_arr == chosen_rec
            rec_all_idx = np.where(rec_mask_all)[0]
            if rec_all_idx.size >= 2:
                rec_all_order = np.argsort(times_meta_arr[rec_all_idx])
                rec_all_idx = rec_all_idx[rec_all_order]
                rec_all_times_s = times_meta_arr[rec_all_idx]

                start_s = float(np.min(times_panel_abs_s))
                end_s = float(np.max(times_panel_abs_s))
                dense_panel_step_s = 1.0
                dense_times_abs_s = np.arange(start_s, end_s + dense_panel_step_s * 0.5, dense_panel_step_s, dtype=np.float64)

                if dense_times_abs_s.size >= 10:
                    def _interp_from_sparse(values: np.ndarray) -> np.ndarray:
                        src = np.asarray(values, dtype=np.float32)[rec_all_idx]
                        return np.interp(
                            dense_times_abs_s,
                            rec_all_times_s,
                            src,
                            left=float(src[0]),
                            right=float(src[-1]),
                        ).astype(np.float32)

                    panel_pred_preictal = _interp_from_sparse(prediction_payload['pred_preictal'])
                    panel_pred_countdown = _interp_from_sparse(prediction_payload['pred_countdown'])
                    if bayes_sweep_payload is not None:
                        panel_pred_smooth = _interp_from_sparse(bayes_sweep_payload['fused_preictal_smooth'])
                    else:
                        panel_pred_smooth = _causal_smooth(panel_pred_preictal)

                    # Recompute countdown labels from event onsets on dense times.
                    try:
                        if panel_loader is not None:
                            events_df = panel_loader.load_events_tsv(subject_id, session_id, run_id)
                            seizure_onsets = _extract_seizure_onsets(events_df)
                        else:
                            seizure_onsets = np.array([], dtype=np.float32)

                        if seizure_onsets.size > 0:
                            dt = seizure_onsets.astype(np.float64)[None, :] - dense_times_abs_s[:, None]
                            dt[dt < 0] = np.inf
                            min_dt = dt.min(axis=1)
                            panel_true_countdown = np.where(
                                min_dt <= float(config.data.pre_ictal_window_s),
                                min_dt / 60.0,
                                -1.0,
                            ).astype(np.float32)
                        else:
                            panel_true_countdown = _interp_from_sparse(prediction_payload['true_countdown'])
                    except Exception:
                        panel_true_countdown = _interp_from_sparse(prediction_payload['true_countdown'])

                    # Dense ECG signal from raw cache.
                    rec_meta = None
                    for meta in getattr(test_dataset, 'recordings_meta', []):
                        if str(meta.get('recording_uid', '')) == chosen_rec:
                            rec_meta = meta
                            break

                    if rec_meta is not None:
                        fs = float(rec_meta.get('fs', 250.0))
                        n_samples = int(rec_meta.get('n_samples', 0))
                        signal_path = str(rec_meta.get('signal_path', ''))

                        if signal_path and n_samples > 0:
                            if signal_path not in _MMAP_CACHE:
                                _MMAP_CACHE[signal_path] = np.memmap(
                                    signal_path, dtype=np.float32, mode='r', shape=(n_samples,)
                                )
                            ecg_signal = _MMAP_CACHE[signal_path]
                            ecg_idx = np.clip((dense_times_abs_s * fs).astype(np.int64), 0, n_samples - 1)
                            panel_ecg_raw = np.asarray(ecg_signal[ecg_idx], dtype=np.float32)
                            panel_ppg_raw = panel_ecg_raw.copy()

                            # Derive BPM from dense raw ECG when SciPy is available.
                            bpm_from_ecg = None
                            try:
                                from scipy.signal import find_peaks as _find_peaks

                                seg_pad = int(max(3.0 * fs, 1.0))
                                seg_start = max(0, int(ecg_idx.min()) - seg_pad)
                                seg_end = min(n_samples, int(ecg_idx.max()) + seg_pad + 1)
                                ecg_seg = np.asarray(ecg_signal[seg_start:seg_end], dtype=np.float32)

                                if ecg_seg.size > int(fs * 20):
                                    ecg_std = float(np.std(ecg_seg))
                                    prominence = max(ecg_std * 0.35, 1e-7)
                                    min_dist = max(1, int(0.25 * fs))
                                    peaks, _ = _find_peaks(ecg_seg, distance=min_dist, prominence=prominence)

                                    if peaks.size >= 3:
                                        impulses = np.zeros(ecg_seg.size, dtype=np.float32)
                                        impulses[peaks] = 1.0
                                        win = max(1, int(20.0 * fs))
                                        beat_count = np.convolve(impulses, np.ones(win, dtype=np.float32), mode='same')
                                        bpm_seg = beat_count * (60.0 * fs / float(win))
                                        local_idx = np.clip(ecg_idx - seg_start, 0, len(bpm_seg) - 1)
                                        bpm_from_ecg = np.asarray(bpm_seg[local_idx], dtype=np.float32)
                            except Exception:
                                bpm_from_ecg = None

                            if bpm_from_ecg is not None:
                                panel_hr_raw = bpm_from_ecg
                                k = max(3, int(round(30.0 / max(dense_panel_step_s, 1e-6))))
                                kern = np.ones(k, dtype=np.float32) / float(k)
                                mean_hr = np.convolve(panel_hr_raw, kern, mode='same')
                                mean2_hr = np.convolve(panel_hr_raw ** 2, kern, mode='same')
                                panel_hrv_raw = np.sqrt(np.clip(mean2_hr - mean_hr ** 2, 0.0, None)).astype(np.float32)
                            else:
                                # Fallback when peak detection is unavailable.
                                k = max(3, int(round(30.0 / max(dense_panel_step_s, 1e-6))))
                                kernel = np.ones(k, dtype=np.float32) / float(k)
                                mean_sig = np.convolve(panel_ecg_raw, kernel, mode='same')
                                mean2_sig = np.convolve(panel_ecg_raw ** 2, kernel, mode='same')
                                panel_hr_raw = (60.0 + 25.0 * np.convolve(np.abs(panel_ecg_raw), kernel, mode='same')).astype(np.float32)
                                panel_hrv_raw = np.sqrt(np.clip(mean2_sig - mean_sig ** 2, 0.0, None)).astype(np.float32)

                    # Dense EEG signal from BIDS EEG stream if available.
                    try:
                        if panel_loader is not None:
                            eeg_data, eeg_fs = panel_loader.load_subject_edf(subject_id, session_id, 'eeg', run_id)
                            eeg_arr = np.asarray(eeg_data)
                            if eeg_arr.ndim > 1:
                                ch_std = np.std(eeg_arr, axis=1)
                                eeg_ch = int(np.argmax(ch_std))
                                eeg_1d = eeg_arr[eeg_ch]
                                eeg_signal_label = f"EEG (ch{eeg_ch}, dense raw)"
                            else:
                                eeg_1d = eeg_arr
                                eeg_signal_label = "EEG (dense raw)"
                            eeg_idx = np.clip((dense_times_abs_s * float(eeg_fs)).astype(np.int64), 0, len(eeg_1d) - 1)
                            panel_eeg_raw = np.asarray(eeg_1d[eeg_idx], dtype=np.float32)
                    except Exception as eeg_exc:
                        logger.warning("Dense panel EEG extraction skipped for %s: %s", chosen_rec, str(eeg_exc))

                    # Dense EMG and MOV signals (for muscle spasm / motion context).
                    try:
                        if panel_loader is not None:
                            emg_data, emg_fs = panel_loader.load_subject_edf(subject_id, session_id, 'emg', run_id)
                            emg_arr = np.asarray(emg_data)
                            if emg_arr.ndim > 1:
                                emg_ch = int(np.argmax(np.std(emg_arr, axis=1)))
                                emg_1d = emg_arr[emg_ch]
                            else:
                                emg_1d = emg_arr
                            emg_idx = np.clip((dense_times_abs_s * float(emg_fs)).astype(np.int64), 0, len(emg_1d) - 1)
                            panel_emg_raw = np.asarray(emg_1d[emg_idx], dtype=np.float32)
                    except Exception as emg_exc:
                        logger.warning("Dense panel EMG extraction skipped for %s: %s", chosen_rec, str(emg_exc))

                    try:
                        if panel_loader is not None:
                            mov_data, mov_fs = panel_loader.load_subject_edf(subject_id, session_id, 'mov', run_id)
                            mov_arr = np.asarray(mov_data)
                            if mov_arr.ndim > 1:
                                # Aggregate multi-axis MOV into a single magnitude-like trace.
                                mov_1d = np.sqrt(np.mean(mov_arr.astype(np.float32) ** 2, axis=0)).astype(np.float32)
                            else:
                                mov_1d = mov_arr.astype(np.float32)
                            mov_idx = np.clip((dense_times_abs_s * float(mov_fs)).astype(np.int64), 0, len(mov_1d) - 1)
                            panel_mov_raw = np.asarray(mov_1d[mov_idx], dtype=np.float32)
                    except Exception as mov_exc:
                        logger.warning("Dense panel MOV extraction skipped for %s: %s", chosen_rec, str(mov_exc))

                    # Build ADXL/motion row input from MOV and/or EMG envelope.
                    if panel_emg_raw is not None:
                        k_env = max(3, int(round(15.0 / max(dense_panel_step_s, 1e-6))))
                        env_kernel = np.ones(k_env, dtype=np.float32) / float(k_env)
                        panel_emg_env = np.convolve(np.abs(panel_emg_raw), env_kernel, mode='same').astype(np.float32)
                    else:
                        panel_emg_env = None

                    if panel_mov_raw is not None and panel_emg_env is not None:
                        mov_m = float(np.mean(panel_mov_raw))
                        mov_s = float(np.std(panel_mov_raw))
                        emg_m = float(np.mean(panel_emg_env))
                        emg_s = float(np.std(panel_emg_env))
                        mov_z = (panel_mov_raw - mov_m) / (mov_s if mov_s > 1e-6 else 1.0)
                        emg_z = (panel_emg_env - emg_m) / (emg_s if emg_s > 1e-6 else 1.0)
                        panel_adxl_raw = (0.6 * mov_z + 0.4 * emg_z).astype(np.float32)
                    elif panel_mov_raw is not None:
                        panel_adxl_raw = panel_mov_raw.astype(np.float32)
                    elif panel_emg_env is not None:
                        panel_adxl_raw = panel_emg_env.astype(np.float32)

                    if panel_eeg_raw is not None and panel_emg_raw is not None and len(panel_eeg_raw) == len(panel_emg_raw):
                        eeg_z = (panel_eeg_raw - float(np.mean(panel_eeg_raw))) / max(float(np.std(panel_eeg_raw)), 1e-6)
                        emg_z = (panel_emg_raw - float(np.mean(panel_emg_raw))) / max(float(np.std(panel_emg_raw)), 1e-6)
                        eeg_emg_corr = float(np.corrcoef(eeg_z, emg_z)[0, 1])
                        logger.info("Dense panel EEG-EMG coupling | corr=%.3f | eeg_std=%.4f | emg_std=%.4f",
                                    eeg_emg_corr,
                                    float(np.std(panel_eeg_raw)),
                                    float(np.std(panel_emg_raw)))

                    logger.info(
                        "Dense panel raw stats | eeg=%s | emg=%s | mov=%s | hr=%s | hrv=%s",
                        None if panel_eeg_raw is None else (float(np.min(panel_eeg_raw)), float(np.max(panel_eeg_raw)), float(np.std(panel_eeg_raw))),
                        None if panel_emg_raw is None else (float(np.min(panel_emg_raw)), float(np.max(panel_emg_raw)), float(np.std(panel_emg_raw))),
                        None if panel_mov_raw is None else (float(np.min(panel_mov_raw)), float(np.max(panel_mov_raw)), float(np.std(panel_mov_raw))),
                        None if panel_hr_raw is None else (float(np.min(panel_hr_raw)), float(np.max(panel_hr_raw)), float(np.std(panel_hr_raw))),
                        None if panel_hrv_raw is None else (float(np.min(panel_hrv_raw)), float(np.max(panel_hrv_raw)), float(np.std(panel_hrv_raw))),
                    )

                    panel_time_minutes = ((dense_times_abs_s - float(dense_times_abs_s[0])) / 60.0).astype(np.float64)
                    dense_panel_used = True
                    logger.info(
                        "Dense panel rebuilt from raw %s: %d -> %d samples (step=%.1fs)",
                        chosen_rec,
                        len(panel_indices),
                        len(panel_time_minutes),
                        dense_panel_step_s,
                    )
        except Exception as dense_exc:
            # Restore sparse arrays to undo any partial updates from the
            # failed dense reconstruction (prevents size mismatches).
            panel_pred_preictal = _sparse_panel_pred_preictal
            panel_pred_countdown = _sparse_panel_pred_countdown
            panel_true_countdown = _sparse_panel_true_countdown
            panel_pred_smooth = _sparse_panel_pred_smooth
            panel_time_minutes = _sparse_panel_time_minutes
            logger.warning("Dense panel rebuild skipped due to error: %s", str(dense_exc))

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
        panel_ecg = panel_ecg_raw.astype(np.float32)
        panel_hr = panel_hr_raw.astype(np.float32)
        panel_hrv = panel_hrv_raw.astype(np.float32)

    panel_ppg = _standardize_for_plot(panel_ppg_raw)
    panel_eda = _standardize_for_plot(panel_eda_raw) if panel_eda_raw is not None else None
    panel_eeg = _standardize_for_plot(panel_eeg_raw) if panel_eeg_raw is not None else None
    panel_adxl = _standardize_for_plot(panel_adxl_raw) if panel_adxl_raw is not None else None

    # Display sampling rate accounts for three cases:
    # 1. Dense panel from raw ECG: 1 / dense_panel_step_s (very high, e.g. 250 Hz)
    # 2. Expanded sparse panel with intra-window samples: window_time_samples / feature_step_s
    # 3. Single-sample sparse panel (fallback): 1 / feature_step_s (very low, e.g. 0.25 Hz)
    if dense_panel_used:
        display_sampling_hz = 1.0 / max(dense_panel_step_s, 1e-6)
    elif window_time_samples > 1 and len(panel_indices) > 0:
        # Intra-window sampling rate from expanded sparse panel
        intra_window_dt_s = float(config.data.feature_step_s) / max(window_time_samples - 1, 1)
        display_sampling_hz = 1.0 / max(intra_window_dt_s, 1e-6)
    else:
        # Fallback: single-sample per window
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
        f"Signal: {'PPG proxy' if data_source == 'wearable' else 'ECG raw amplitude + BPM overlay'}"
    )
    panel_footer += (
        f" | ModalitySources: ECG={'dense_raw' if dense_panel_used else 'window_proxy'}"
        f",EEG={'dense_raw' if panel_eeg_raw is not None else 'missing'}"
        f",EMG={'dense_raw' if panel_emg_raw is not None else 'missing'}"
        f",MOV={'dense_raw' if panel_mov_raw is not None else 'missing'}"
        f",MotionRow={'mov/emg_fusion' if panel_adxl_raw is not None else 'missing'}"
    )
    logger.info(
        "Panel modality sources | ECG=%s EEG=%s EMG=%s MOV=%s MotionRow=%s",
        'dense_raw' if dense_panel_used else 'window_proxy',
        'dense_raw' if panel_eeg_raw is not None else 'missing',
        'dense_raw' if panel_emg_raw is not None else 'missing',
        'dense_raw' if panel_mov_raw is not None else 'missing',
        'mov/emg_fusion' if panel_adxl_raw is not None else 'missing',
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

    # Build token activation waterfall from prediction confidence channels
    panel_token_roll = None
    try:
        n_panel = len(panel_pred_preictal)
        n_channels = 4  # alert_raw, alert_smooth, countdown_imminence, preictal_confidence
        if n_panel >= 2:
            panel_token_roll = np.zeros((n_panel, n_channels), dtype=np.float32)
            
            # Channel 0: Raw alert probability
            alert_raw = np.clip(panel_pred_preictal, 0.0, 1.0)
            for i in range(n_panel):
                panel_token_roll[i, 0] = float(np.mean(alert_raw[:i + 1]))
            
            # Channel 1: Smoothed alert (if available)
            if panel_pred_smooth is not None:
                alert_smooth = np.clip(panel_pred_smooth, 0.0, 1.0)
                for i in range(n_panel):
                    panel_token_roll[i, 1] = float(np.mean(alert_smooth[:i + 1]))
            
            # Channel 2: Countdown imminence (exp decay from predicted countdown)
            tau_min = 5.0  # assume 5-min typical pre-ictal
            countdown_imminence = np.exp(-np.clip(panel_pred_countdown, 0.0, None) / tau_min)
            for i in range(n_panel):
                panel_token_roll[i, 2] = float(np.mean(countdown_imminence[:i + 1]))
            
            # Channel 3: Ground truth preictal window
            gt_preictal = (panel_true_countdown >= 0).astype(np.float32)
            for i in range(n_panel):
                panel_token_roll[i, 3] = float(np.mean(gt_preictal[:i + 1]))
            
            logger.info("Computed token activation waterfall: %d timepoints x %d channels", n_panel, n_channels)
    except Exception as token_exc:
        logger.warning("Token activation build skipped for test panel: %s", str(token_exc))
        panel_token_roll = None

    # Log basic stats of the panel signals so we can see whether they are
    # truly flat or just low-dynamic-range for this selection.
    logger.info(
        "Final panel signal stats | ppg: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "eda: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "eeg: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "emg: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "mov: min=%.3f max=%.3f mean=%.3f std=%.3f | "
        "motion(adxl): min=%.3f max=%.3f mean=%.3f std=%.3f | "
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
        float(panel_emg_raw.min()) if panel_emg_raw is not None else 0.0,
        float(panel_emg_raw.max()) if panel_emg_raw is not None else 0.0,
        float(panel_emg_raw.mean()) if panel_emg_raw is not None else 0.0,
        float(panel_emg_raw.std()) if panel_emg_raw is not None else 0.0,
        float(panel_mov_raw.min()) if panel_mov_raw is not None else 0.0,
        float(panel_mov_raw.max()) if panel_mov_raw is not None else 0.0,
        float(panel_mov_raw.mean()) if panel_mov_raw is not None else 0.0,
        float(panel_mov_raw.std()) if panel_mov_raw is not None else 0.0,
        float(panel_adxl.min()) if panel_adxl is not None else 0.0,
        float(panel_adxl.max()) if panel_adxl is not None else 0.0,
        float(panel_adxl.mean()) if panel_adxl is not None else 0.0,
        float(panel_adxl.std()) if panel_adxl is not None else 0.0,
        float(panel_ecg.min()), float(panel_ecg.max()), float(panel_ecg.mean()), float(panel_ecg.std()),
        float(panel_hr.min()), float(panel_hr.max()), float(panel_hr.mean()), float(panel_hr.std()),
        float(panel_hrv.min()), float(panel_hrv.max()), float(panel_hrv.mean()), float(panel_hrv.std()),
    )

    panel_fig = visualizer.plot_gt_vs_inference_panel(
        ecg_signal=panel_ecg,
        ppg_signal=panel_ppg if data_source == 'wearable' else None,
        eda_signal=panel_eda if data_source == 'wearable' else None,
        eeg_signal=panel_eeg,
        emg_signal=panel_emg_raw,
        mov_signal=panel_mov_raw,
        adxl_series=panel_adxl,
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
        eeg_signal_label=eeg_signal_label,
        signal_label='Wearable signals (PPG+ADXL, z-score)' if data_source == 'wearable' else 'ECG raw amplitude + BPM',
        token_roll=panel_token_roll,
        token_window=len(panel_pred_preictal) if panel_token_roll is not None else None,
        token_labels=['Alert (raw)', 'Alert (smooth)', 'Imminence', 'GT preictal'] if panel_token_roll is not None else None,
        data_source=data_source,
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

    # Automated comprehensive run report (JSON + Markdown) for PI/review.
    report_json_path, report_md_path = _write_comprehensive_run_report(
        save_dir=trainer.save_dir,
        config=config,
        args=args,
        results=results,
        test_metrics=test_metrics,
        patient_report=patient_report,
        threshold_sweep=threshold_sweep,
    )
    logger.info("Comprehensive report written: %s", report_md_path)
    logger.info("Comprehensive report JSON: %s", report_json_path)

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
            sys.executable, "scripts/visualize_token_explainability.py",
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
