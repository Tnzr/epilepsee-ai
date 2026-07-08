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

import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, merge_configs
from src.data_loader import BIDSDataLoader, SeizureDataset
from src.stateful_data_loader import TemporalPatientSequenceDataLoader, HiddenStateManager
from src.models import ECGCountdownPredictor
from src.losses import SeizureCountdownLoss
from src.training import Trainer


logger = logging.getLogger(__name__)


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


def load_dataset(config: Config, smoke_test: bool = False) -> Tuple:
    """
    Load SeizeIT2 dataset and split into train/val/test.
    
    Returns:
        (train_dataset, val_dataset, test_dataset, all_features, all_labels)
    """
    logger.info(f"Loading SeizeIT2 dataset from {config.data.dataset_root}")
    
    bids_loader = BIDSDataLoader(config.data)
    
    # Load full dataset (this is expensive, happens once)
    all_features, all_labels, subject_ids, seizure_ids, sample_times, recording_ids = \
        bids_loader.load_all_data()
    
    logger.info(f"Loaded {len(all_features)} samples from {len(np.unique(subject_ids))} patients")
    
    if smoke_test:
        # For smoke test: use only first 50 samples per patient
        unique_subjects = np.unique(subject_ids)[:5]  # Only first 5 patients
        mask = np.isin(subject_ids, unique_subjects)
        all_features = all_features[mask]
        all_labels = all_labels[mask]
        subject_ids = subject_ids[mask]
        seizure_ids = seizure_ids[mask]
        sample_times = sample_times[mask]
        recording_ids = recording_ids[mask]
        logger.info(f"SMOKE TEST: Using {len(all_features)} samples from {len(unique_subjects)} patients")
    
    # Split data
    n_total = len(all_features)
    n_train = int(n_total * 0.6)
    n_val = int(n_total * 0.15)
    
    # Use deterministic split based on patient ID (stratified)
    unique_subjects = np.unique(subject_ids)
    n_patients = len(unique_subjects)
    n_train_patients = int(n_patients * 0.6)
    n_val_patients = int(n_patients * 0.15)
    
    train_patients = unique_subjects[:n_train_patients]
    val_patients = unique_subjects[n_train_patients:n_train_patients + n_val_patients]
    test_patients = unique_subjects[n_train_patients + n_val_patients:]
    
    train_mask = np.isin(subject_ids, train_patients)
    val_mask = np.isin(subject_ids, val_patients)
    test_mask = np.isin(subject_ids, test_patients)
    
    train_dataset = SeizureDataset(
        all_features[train_mask],
        all_labels[train_mask],
        subject_ids=subject_ids[train_mask],
        seizure_ids=seizure_ids[train_mask],
        sample_end_times_s=sample_times[train_mask],
        recording_ids=recording_ids[train_mask]
    )
    
    val_dataset = SeizureDataset(
        all_features[val_mask],
        all_labels[val_mask],
        subject_ids=subject_ids[val_mask],
        seizure_ids=seizure_ids[val_mask],
        sample_end_times_s=sample_times[val_mask],
        recording_ids=recording_ids[val_mask]
    )
    
    test_dataset = SeizureDataset(
        all_features[test_mask],
        all_labels[test_mask],
        subject_ids=subject_ids[test_mask],
        seizure_ids=seizure_ids[test_mask],
        sample_end_times_s=sample_times[test_mask],
        recording_ids=recording_ids[test_mask]
    )
    
    logger.info(f"Train: {len(train_dataset)} samples")
    logger.info(f"Val:   {len(val_dataset)} samples")
    logger.info(f"Test:  {len(test_dataset)} samples")
    
    return train_dataset, val_dataset, test_dataset, all_features, all_labels


def train_epoch_stateful(model, train_loader: TemporalPatientSequenceDataLoader, 
                        criterion, optimizer, device: str, hidden_mgr: HiddenStateManager):
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
    
    for patient_id, batch_features, batch_labels, batch_weights in train_loader:
        # Move to device
        features = torch.from_numpy(batch_features).to(device)
        labels = torch.from_numpy(batch_labels).to(device)
        weights = torch.from_numpy(batch_weights).to(device)
        
        # Get current hidden state (None at patient start, accumulated within patient)
        hidden_state = hidden_mgr.get_hidden_state()
        
        # Forward pass with hidden state
        pre_ictal_pred, countdown_pred, hidden_state_new = model(features, hidden_state)
        
        # Compute loss
        loss_dict = criterion(pre_ictal_pred, countdown_pred, labels, weights)
        loss = loss_dict['total']
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Update hidden state manager
        hidden_mgr.update_for_batch(patient_id, hidden_state_new)
        
        # Track metrics
        total_loss += loss.item()
        batch_count += 1
        all_preds.extend(pre_ictal_pred.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
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
                       criterion, device: str):
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
    hidden_state = None
    prev_patient = None
    
    with torch.no_grad():
        for patient_id, batch_features, batch_labels, batch_weights in val_loader:
            # Reset hidden state when patient changes
            if patient_id != prev_patient:
                hidden_state = None
                prev_patient = patient_id
            
            # Move to device
            features = torch.from_numpy(batch_features).to(device)
            labels = torch.from_numpy(batch_labels).to(device)
            weights = torch.from_numpy(batch_weights).to(device)
            
            # Forward pass
            pre_ictal_pred, countdown_pred, hidden_state = model(features, hidden_state)
            
            # Compute loss
            loss_dict = criterion(pre_ictal_pred, countdown_pred, labels, weights)
            loss = loss_dict['total']
            
            # Track metrics
            total_loss += loss.item()
            batch_count += 1
            all_preds.extend(pre_ictal_pred.detach().cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / max(batch_count, 1)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    pred_variance = np.var(all_preds)
    
    metrics = {
        'loss': avg_loss,
        'pred_variance': pred_variance,
        'batch_count': batch_count
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
    
    args = parser.parse_args()
    
    # Setup
    output_dir = Path(args.output_dir)
    log_file = setup_logging(output_dir)
    logger.info(f"Starting stateful LSTM retraining")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Log file: {log_file}")
    
    # Device
    device = 'cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu'
    logger.info(f"Using device: {device}")
    
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
    train_dataset, val_dataset, test_dataset, _, _ = load_dataset(config, smoke_test=args.smoke_test)
    
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
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    
    # Hidden state manager
    hidden_mgr = HiddenStateManager(device=device, detach_interval=10)
    
    # Training loop
    best_val_loss = float('inf')
    logger.info(f"Starting training for {args.num_epochs} epochs...")
    
    for epoch in range(args.num_epochs):
        logger.info(f"\n===== Epoch {epoch+1}/{args.num_epochs} =====")
        
        # Reset hidden state between epochs
        hidden_mgr.reset()
        
        # Train
        train_loss, train_metrics = train_epoch_stateful(
            model, train_loader, criterion, optimizer, device, hidden_mgr
        )
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        
        # Validate
        val_loss, val_metrics = val_epoch_stateful(
            model, val_loader, criterion, device
        )
        logger.info(f"Val Loss: {val_loss:.4f}")
        logger.info(f"Val Metrics: {val_metrics}")
        
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


if __name__ == '__main__':
    main()
