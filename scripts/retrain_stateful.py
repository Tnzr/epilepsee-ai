#!/usr/bin/env python3
"""
Retrain ECGCountdownPredictor with Stateful LSTM and Patient-Sequential Batching

This script implements Option A: retraining with temporal continuity that matches deployment requirements.

Key differences from old training:
- Uses TemporalPatientSequenceDataLoader (patient-sequential batching)
- Maintains LSTM hidden state across batches within each patient
- Resets hidden state when patient changes
- No random shuffling (shuffles patient order, not sample order)

Usage:
    python scripts/retrain_stateful.py \\
        --config config/stateful_lstm.yaml \\
        --data-root /mnt/d/Datasets/SeizeIT2 \\
        --output-dir models/stateful_v1 \\
        --num-epochs 16 \\
        --smoke-test  # Optional: run on small subset first
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support, balanced_accuracy_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config
from src.data_loader import BIDSDataLoader, SeizureDataset
from src.stateful_data_loader import TemporalPatientSequenceDataLoader, HiddenStateManager
from src.models import ECGCountdownPredictor
from src.losses import SeizureCountdownLoss
from src.visualization import SignalVisualizer


logger = logging.getLogger(__name__)


def _binary_classification_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute binary classification metrics from probabilities."""
    y_true = np.asarray(y_true).astype(np.int32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob >= float(threshold)).astype(np.int32)

    if y_true.size == 0:
        return {
            'accuracy': 0.0,
            'balanced_accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'confusion_matrix': np.zeros((2, 2), dtype=np.int32),
        }

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average='binary',
        zero_division=0,
    )

    return {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_true, y_pred)),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'confusion_matrix': cm.astype(np.int32),
    }


def _plot_confusion_matrix(cm: np.ndarray, title: str):
    """Build a confusion matrix figure for logging/saving."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    class_names = ['Interictal', 'Pre-ictal']
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel='True label',
        xlabel='Predicted label',
        title=title,
    )

    thresh = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{int(cm[i, j])}",
                ha='center',
                va='center',
                color='white' if cm[i, j] > thresh else 'black',
            )

    fig.tight_layout()
    return fig


def setup_logging(output_dir: Path):
    """Setup logging to file and console."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return log_file


def load_dataset(config: Config, smoke_test: bool = False, max_patients: int = None) -> Tuple:
    """
    Load SeizeIT2 dataset and split into train/val/test.

    For real data, reuses the proven lazy-streaming pipeline from train.py
    which memory-maps pre-cached float32 signal binaries in .ecg_signal_cache/
    instead of loading raw EDF files.

    Returns:
        (train_dataset, val_dataset, test_dataset, None, None)
    """
    logger.info(f"Loading SeizeIT2 dataset from {config.data.dataset_root}")

    if smoke_test:
        # Synthetic data: 5 patients, 100 samples each
        logger.info("SMOKE TEST: Creating synthetic dataset (5 patients, 100 samples/patient)")
        n_patients = 5
        n_samples_per_patient = 100
        n_samples = n_patients * n_samples_per_patient

        all_features = np.random.randn(n_samples, 600, 14).astype(np.float32)
        all_labels = np.random.uniform(-10, 10, n_samples).astype(np.float32)
        all_subject_ids = np.repeat(np.arange(1, n_patients + 1), n_samples_per_patient)
        all_seizure_ids = np.zeros(n_samples)
        all_sample_times = np.linspace(0, 3600, n_samples).astype(np.float32)
        all_recording_ids = np.array([f"rec_{i}" for i in range(n_samples)])

        logger.info(f"Loaded {n_samples} samples from {n_patients} patients")

        unique_subjects = np.unique(all_subject_ids)
        n_train_patients = int(len(unique_subjects) * 0.6)
        n_val_patients = int(len(unique_subjects) * 0.15)
        train_patients = unique_subjects[:n_train_patients]
        val_patients = unique_subjects[n_train_patients:n_train_patients + n_val_patients]
        test_patients = unique_subjects[n_train_patients + n_val_patients:]

        def _make_subset(mask):
            return SeizureDataset(
                all_features[mask], all_labels[mask],
                subject_ids=all_subject_ids[mask],
                seizure_ids=all_seizure_ids[mask],
                sample_end_times_s=all_sample_times[mask],
                recording_ids=all_recording_ids[mask],
            )

        train_dataset = _make_subset(np.isin(all_subject_ids, train_patients))
        val_dataset   = _make_subset(np.isin(all_subject_ids, val_patients))
        test_dataset  = _make_subset(np.isin(all_subject_ids, test_patients))

    else:
        # Real data via lazy streaming (uses .ecg_signal_cache/, no EDF loading)
        import importlib.util, sys as _sys, types as _types

        # Import helper symbols from train.py without executing its __main__ block
        train_spec = importlib.util.spec_from_file_location(
            "_train_helpers",
            Path(__file__).parent / "train.py",
        )
        train_mod = importlib.util.module_from_spec(train_spec)
        train_spec.loader.exec_module(train_mod)

        _prepare_real_datasets = train_mod._prepare_real_datasets

        bids_loader = BIDSDataLoader(config.data)
        recordings = bids_loader.list_all_recordings()
        logger.info(f"Found {len(recordings)} recordings")

        if max_patients is not None:
            seen = set()
            filtered = []
            for r in recordings:
                seen.add(r["subject_id"])
                if len(seen) <= max_patients:
                    filtered.append(r)
            recordings = filtered
            logger.info(f"Capped to {max_patients} patients → {len(recordings)} recordings")

        # Build a minimal args namespace matching what _prepare_real_datasets expects
        import argparse as _ap
        fake_args = _ap.Namespace(
            patient_sequential=True,
            long_sweep_training=False,
            max_recordings=0,
            max_samples_per_recording=0,
        )

        result = _prepare_real_datasets(config, fake_args, bids_loader, recordings)
        if result is None:
            raise RuntimeError("_prepare_real_datasets returned None – too few samples")

        train_dataset, val_dataset, test_dataset = result

    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val:   {len(val_dataset)} samples")
    logger.info(f"Test:  {len(test_dataset)} samples")

    return train_dataset, val_dataset, test_dataset, None, None


def train_epoch_stateful(model, train_loader: TemporalPatientSequenceDataLoader, 
                        criterion, optimizer, device: str, hidden_mgr: HiddenStateManager,
                        epoch: int = 1, num_epochs: int = 1):
    """
    Train one epoch using stateful LSTM.
    
    Args:
        model: ECGCountdownPredictor
        train_loader: TemporalPatientSequenceDataLoader (not PyTorch DataLoader!)
        criterion: Loss function
        optimizer: Optimizer
        device: 'cpu' or 'cuda'
        hidden_mgr: HiddenStateManager for tracking hidden state
    
    Returns:
        (avg_loss, metrics_dict)
    """
    model.train()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []
    last_batch_size = None
    
    # Wrap iterator with tqdm for progress bar
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}/{num_epochs}", unit="batch")
    
    for patient_id, batch_features, batch_labels, batch_weights in pbar:
        # Move to device
        features = torch.from_numpy(batch_features).to(device)
        labels = torch.from_numpy(batch_labels).to(device)
        weights = torch.from_numpy(batch_weights).to(device)
        current_batch_size = features.shape[0]
        
        # Convert countdown labels to binary pre-ictal labels
        pre_ictal_labels = (labels >= 0).float()
        
        # Get current hidden state (None at patient start, accumulated within patient)
        # Also reset if batch size changed (hidden state must match batch size)
        hidden_state = hidden_mgr.get_hidden_state()
        if last_batch_size is not None and last_batch_size != current_batch_size:
            hidden_state = None
        last_batch_size = current_batch_size
        
        # Forward pass with hidden state
        pre_ictal_pred, countdown_pred, hidden_state_new = model(features, hidden_state)
        
        # Compute loss
        # loss function expects: (pre_ictal_pred, countdown_pred, pre_ictal_true, countdown_true, sample_weights)
        loss_dict = criterion(pre_ictal_pred, countdown_pred, pre_ictal_labels, labels, weights)
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Detach hidden state from computation graph after backward to prevent reuse errors
        if hidden_state_new is not None:
            if isinstance(hidden_state_new, tuple):
                hidden_state_new = tuple(h.detach() for h in hidden_state_new)
            else:
                hidden_state_new = hidden_state_new.detach()
        
        # Update hidden state manager
        hidden_mgr.update_for_batch(patient_id, hidden_state_new)
        
        # Track metrics
        total_loss += loss.item()
        batch_count += 1
        all_preds.extend(pre_ictal_pred.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        avg_loss_so_far = total_loss / batch_count
        pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}', 'pred_var': f'{np.var(all_preds[-min(1000, len(all_preds)):]): .4f}' if all_preds else 'N/A'})
        
        # Log to WandB every 100 batches
        if batch_count % 100 == 0:
            wandb.log({
                "batch_loss": avg_loss_so_far,
                "batch_count": batch_count,
            })
    
    avg_loss = total_loss / max(batch_count, 1)
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_variance = np.var(all_preds)
    
    metrics = {
        'loss': avg_loss,
        'pred_variance': pred_variance,
        'batch_count': batch_count,
        'hidden_state_resets': hidden_mgr.stats['patient_resets'],
        'hidden_state_detaches': hidden_mgr.stats['detaches']
    }
    
    return avg_loss, metrics


def val_epoch_stateful(model, val_loader: TemporalPatientSequenceDataLoader,
                       criterion, device: str, epoch: int = 1, num_epochs: int = 1,
                       threshold: float = 0.5):
    """
    Validate one epoch using stateful LSTM.
    
    Args:
        model: ECGCountdownPredictor
        val_loader: TemporalPatientSequenceDataLoader
        criterion: Loss function
        device: 'cpu' or 'cuda'
    
    Returns:
        (avg_loss, metrics_dict)
    """
    model.eval()
    total_loss = 0
    batch_count = 0
    all_preds = []
    all_labels = []
    all_binary_labels = []
    hidden_state = None
    prev_patient = None
    
    # Wrap iterator with tqdm for progress bar
    pbar = tqdm(val_loader, desc=f"Val Epoch {epoch}/{num_epochs}", unit="batch")
    
    with torch.no_grad():
        for patient_id, batch_features, batch_labels, batch_weights in pbar:
            # Reset hidden state when patient changes
            if patient_id != prev_patient:
                hidden_state = None
                prev_patient = patient_id
            
            # Move to device
            features = torch.from_numpy(batch_features).to(device)
            labels = torch.from_numpy(batch_labels).to(device)
            weights = torch.from_numpy(batch_weights).to(device)
            
            # Convert countdown labels to binary pre-ictal labels
            pre_ictal_labels = (labels >= 0).float()
            
            # Forward pass
            pre_ictal_pred, countdown_pred, hidden_state = model(features, hidden_state)
            
            # Compute loss
            # loss function expects: (pre_ictal_pred, countdown_pred, pre_ictal_true, countdown_true, sample_weights)
            loss_dict = criterion(pre_ictal_pred, countdown_pred, pre_ictal_labels, labels, weights)
            loss = loss_dict['total']
            
            # Track metrics
            total_loss += loss.item()
            batch_count += 1
            all_preds.extend(pre_ictal_pred.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_binary_labels.extend(pre_ictal_labels.cpu().numpy())
            
            # Update progress bar
            avg_loss_so_far = total_loss / batch_count
            pbar.set_postfix({'loss': f'{avg_loss_so_far:.4f}'})
    
    avg_loss = total_loss / max(batch_count, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_binary_labels = np.array(all_binary_labels, dtype=np.int32)
    pred_variance = np.var(all_preds)
    cls_metrics = _binary_classification_metrics(all_binary_labels, all_preds, threshold=threshold)
    
    metrics = {
        'loss': avg_loss,
        'pred_variance': pred_variance,
        'batch_count': batch_count,
        'accuracy': cls_metrics['accuracy'],
        'balanced_accuracy': cls_metrics['balanced_accuracy'],
        'precision': cls_metrics['precision'],
        'recall': cls_metrics['recall'],
        'f1': cls_metrics['f1'],
        'confusion_matrix': cls_metrics['confusion_matrix'],
        'pred_probs': all_preds,
        'true_binary': all_binary_labels,
        'true_countdown': all_labels,
    }
    
    return avg_loss, metrics


def main():
    """Main retraining script."""
    parser = argparse.ArgumentParser(description='Retrain ECGCountdownPredictor with Stateful LSTM')
    parser.add_argument('--config', type=str, default='config/stateful_lstm.yaml',
                       help='Configuration file')
    parser.add_argument('--data-root', type=str, required=True,
                       help='SeizeIT2 dataset root')
    parser.add_argument('--output-dir', type=str, default='models/stateful_v1',
                       help='Output directory for models and logs')
    parser.add_argument('--num-epochs', type=int, default=16,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--smoke-test', action='store_true',
                       help='Run smoke test on small subset')
    parser.add_argument('--no-cuda', action='store_true',
                       help='Disable CUDA')
    parser.add_argument('--max-patients', type=int, default=None,
                       help='Maximum number of patients to load (None = all)')
    parser.add_argument('--metric-threshold', type=float, default=0.5,
                       help='Decision threshold for pre-ictal classification metrics')
    parser.add_argument('--viz-every-fraction', type=float, default=0.25,
                       help='Save confusion-matrix and metric snapshots every N fraction of total epochs (default: 0.25)')
    parser.add_argument('--plateau-min-improvement', type=float, default=1e-3,
                       help='Minimum absolute val-loss improvement to count as progress')
    parser.add_argument('--plateau-patience', type=int, default=3,
                       help='Epochs with < min improvement before plateau warning')
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    log_file = setup_logging(output_dir)
    logger.info(f"Starting stateful LSTM retraining")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Initialize WandB
    wandb.init(
        project="epilepsee-ai",
        name=f"stateful-lstm-{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "num_epochs": args.num_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "smoke_test": args.smoke_test,
        },
        dir=str(output_dir)
    )
    logger.info(f"WandB initialized: {wandb.run.name}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    logger.info(f"Using device: {device}")
    if device == 'cuda':
        gpu_count = torch.cuda.device_count()
        logger.info(f"Detected CUDA devices: {gpu_count}")
        if gpu_count >= 2:
            logger.info(
                "Multiple GPUs detected. For synchronized 2-GPU data-parallel training, "
                "use the distributed trainer path: torchrun --nproc_per_node=2 scripts/train.py ..."
            )
    wandb.config.update({"device": device})
    wandb.config.update({
        "metric_threshold": float(args.metric_threshold),
        "viz_every_fraction": float(args.viz_every_fraction),
        "plateau_min_improvement": float(args.plateau_min_improvement),
        "plateau_patience": int(args.plateau_patience),
    })
    
    # Load configuration
    if Path(args.config).exists():
        config = Config.from_yaml(args.config)
    else:
        logger.warning(f"Config file not found: {args.config}, using defaults")
        config = Config()
    
    # Override with command-line arguments
    config.data.dataset_root = args.data_root
    config.training.batch_size = args.batch_size
    config.training.learning_rate = args.learning_rate
    
    # Load dataset
    train_dataset, val_dataset, test_dataset, _, _ = load_dataset(config, smoke_test=args.smoke_test, max_patients=args.max_patients)
    
    # Create loaders (stateful!)
    train_loader = TemporalPatientSequenceDataLoader(
        train_dataset,
        batch_size=args.batch_size,
        allow_partial_batches=False
    )
    val_loader = TemporalPatientSequenceDataLoader(
        val_dataset,
        batch_size=args.batch_size * 2,  # Can use larger batch for eval
        allow_partial_batches=True
    )
    
    logger.info(f"Train loader: {train_loader.get_stats()}")
    logger.info(f"Val loader: {val_loader.get_stats()}")
    
    # Create model
    model = ECGCountdownPredictor(config.model)
    model = model.to(device)
    logger.info(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Loss and optimizer
    criterion = SeizureCountdownLoss(config.loss)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)
    
    # Hidden state manager
    hidden_mgr = HiddenStateManager(device=device, detach_interval=10)
    
    # Visualizer
    visualizer = SignalVisualizer(save_dir=output_dir, upload_to_wandb=True)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    val_bal_acc = []
    val_f1 = []
    plateau_counter = 0
    prev_val_loss = None

    # Build epoch checkpoints for richer evaluation artifacts (e.g., 25/50/75/100%).
    checkpoint_epochs = set([args.num_epochs])
    if args.viz_every_fraction > 0:
        frac = float(args.viz_every_fraction)
        checkpoints = np.arange(frac, 1.0 + 1e-9, frac)
        for c in checkpoints:
            checkpoint_epochs.add(max(1, int(round(c * args.num_epochs))))

    logger.info(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{args.num_epochs} =====")
        
        # Reset hidden state between epochs
        hidden_mgr.reset()
        
        # Train
        train_loss, train_metrics = train_epoch_stateful(
            model, train_loader, criterion, optimizer, device, hidden_mgr,
            epoch=epoch+1, num_epochs=args.num_epochs
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        train_losses.append(train_loss)
        
        # Validate
        val_loss, val_metrics = val_epoch_stateful(
            model, val_loader, criterion, device,
            epoch=epoch+1, num_epochs=args.num_epochs,
            threshold=float(args.metric_threshold)
        )
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(
            "Val Metrics: loss=%.4f, acc=%.4f, bal_acc=%.4f, precision=%.4f, recall=%.4f, f1=%.4f, pred_var=%.6f",
            val_metrics['loss'],
            val_metrics['accuracy'],
            val_metrics['balanced_accuracy'],
            val_metrics['precision'],
            val_metrics['recall'],
            val_metrics['f1'],
            val_metrics.get('pred_variance', 0.0),
        )
        val_losses.append(val_loss)
        val_bal_acc.append(val_metrics['balanced_accuracy'])
        val_f1.append(val_metrics['f1'])

        if prev_val_loss is not None:
            improvement = prev_val_loss - val_loss
            if improvement < float(args.plateau_min_improvement):
                plateau_counter += 1
            else:
                plateau_counter = 0

            if plateau_counter >= int(args.plateau_patience):
                logger.warning(
                    "Validation loss appears to be plateauing (improvement %.6f < %.6f for %d epochs). "
                    "Consider increasing model capacity or tuning LR schedule.",
                    improvement,
                    float(args.plateau_min_improvement),
                    plateau_counter,
                )
        prev_val_loss = val_loss
        
        # Log to WandB
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "train/pred_variance": train_metrics.get('pred_variance', 0),
            "val/loss": val_loss,
            "val/pred_variance": val_metrics.get('pred_variance', 0),
            "val/accuracy": val_metrics.get('accuracy', 0),
            "val/balanced_accuracy": val_metrics.get('balanced_accuracy', 0),
            "val/precision": val_metrics.get('precision', 0),
            "val/recall": val_metrics.get('recall', 0),
            "val/f1": val_metrics.get('f1', 0),
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        # Rich evaluation artifacts at checkpoints (default every 25% of total epochs).
        current_epoch = epoch + 1
        if current_epoch in checkpoint_epochs:
            cm = val_metrics.get('confusion_matrix', None)
            if cm is not None:
                cm_fig = _plot_confusion_matrix(
                    np.asarray(cm),
                    title=f"Validation confusion matrix @ epoch {current_epoch}/{args.num_epochs}",
                )
                cm_path = output_dir / f"confusion_matrix_epoch_{current_epoch:03d}.png"
                cm_fig.savefig(cm_path, dpi=150, bbox_inches='tight')
                plt.close(cm_fig)

                wandb.log({
                    "visualization/confusion_matrix": wandb.Image(str(cm_path)),
                    "visualization/checkpoint_epoch": current_epoch,
                })
                logger.info("Saved checkpoint confusion matrix: %s", cm_path)

            metrics_snapshot = {
                "epoch": current_epoch,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_accuracy": float(val_metrics.get('accuracy', 0.0)),
                "val_balanced_accuracy": float(val_metrics.get('balanced_accuracy', 0.0)),
                "val_precision": float(val_metrics.get('precision', 0.0)),
                "val_recall": float(val_metrics.get('recall', 0.0)),
                "val_f1": float(val_metrics.get('f1', 0.0)),
                "metric_threshold": float(args.metric_threshold),
            }
            metrics_path = output_dir / f"metrics_snapshot_epoch_{current_epoch:03d}.json"
            with open(metrics_path, 'w', encoding='utf-8') as f:
                json.dump(metrics_snapshot, f, indent=2)
        
        # Save checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = output_dir / 'best_model_stateful.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config.to_dict()
            }, ckpt_path)
            logger.info(f"Saved best model: {ckpt_path}")
        
        # Save last model
        ckpt_path = output_dir / 'last_model_stateful.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_loss': val_loss,
            'config': config.to_dict()
        }, ckpt_path)
        
        # Scheduler step
        scheduler.step(val_loss)
    
    logger.info("\nTraining complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Models saved to: {output_dir}")
    
    # Create training curve visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    if val_losses:
        ax.plot(epochs, val_losses, 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss Over Epochs', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save and log to WandB
    curve_path = output_dir / 'training_curves.png'
    plt.savefig(curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    wandb.log({"training_curves": wandb.Image(str(curve_path))})
    logger.info(f"Saved training curves to {curve_path}")

    # Accuracy/F1 trend visualization
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    if val_bal_acc:
        ax2.plot(epochs, val_bal_acc, 'g-', label='Val Balanced Accuracy', linewidth=2)
    if val_f1:
        ax2.plot(epochs, val_f1, 'm-', label='Val F1', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Validation Classification Metrics Over Epochs', fontsize=14)
    ax2.set_ylim(0.0, 1.0)
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()

    metric_curve_path = output_dir / 'validation_classification_metrics.png'
    plt.savefig(metric_curve_path, dpi=150, bbox_inches='tight')
    plt.close(fig2)
    wandb.log({"validation_classification_metrics": wandb.Image(str(metric_curve_path))})
    logger.info(f"Saved validation metric curves to {metric_curve_path}")
    
    # Log final metrics to WandB
    wandb.log({
        "final_best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1] if train_losses else 0,
        "final_val_loss": val_losses[-1] if val_losses else 0,
        "final_val_balanced_accuracy": val_bal_acc[-1] if val_bal_acc else 0,
        "final_val_f1": val_f1[-1] if val_f1 else 0,
        "total_epochs_trained": args.num_epochs
    })
    
    # Save best model as artifact
    best_model_path = output_dir / 'best_model_stateful.pt'
    if best_model_path.exists():
        artifact = wandb.Artifact('best_model_stateful', type='model')
        artifact.add_file(str(best_model_path), name='model.pt')
        wandb.log_artifact(artifact)
        logger.info(f"Logged best model to WandB artifacts")
    
    # Finish WandB run
    wandb.finish()
    logger.info("WandB run finished")


if __name__ == '__main__':
    main()
