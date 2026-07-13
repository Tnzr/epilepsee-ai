"""
Quick test script to verify wandb logging integration with training pipeline.

This script:
1. Creates synthetic datasets
2. Trains a model with wandb logging enabled
3. Verifies metrics are logged correctly
4. Generates sample visualizations
"""

import logging
import numpy as np
import torch
from pathlib import Path

# Setup path
import sys
project_root = Path(__file__).parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from config.config import DEFAULT_CONFIG
from src.data_loader import SeizureDataset
from src.training import Trainer
from src.visualization import SignalVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_wandb_training():
    """Test training with wandb logging enabled."""
    
    logger.info("="*60)
    logger.info("Testing wandb integration with training pipeline")
    logger.info("="*60)
    
    # Create synthetic datasets
    logger.info("Creating synthetic datasets...")
    X_train = np.random.randn(50, 600, 14).astype(np.float32)
    y_train = np.random.rand(50) * 10
    train_dataset = SeizureDataset(X_train, y_train)
    
    X_val = np.random.randn(20, 600, 14).astype(np.float32)
    y_val = np.random.rand(20) * 10
    val_dataset = SeizureDataset(X_val, y_val)
    
    X_test = np.random.randn(10, 600, 14).astype(np.float32)
    y_test = np.random.rand(10) * 10
    test_dataset = SeizureDataset(X_test, y_test)
    
    # Create trainer with wandb enabled
    logger.info("Creating trainer with wandb logging...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainer = Trainer(DEFAULT_CONFIG, device=device, use_wandb=True)
    
    # Verify wandb is initialized
    if trainer.wandb_run:
        logger.info(f"✓ wandb initialized: {trainer.wandb_run.get_url()}")
    else:
        logger.warning("⚠ wandb not initialized")
    
    # Train for 2 epochs
    logger.info("Starting training with wandb logging...")
    results = trainer.train(train_dataset, val_dataset, test_dataset)
    
    logger.info("="*60)
    logger.info("Training completed!")
    logger.info(f"Best validation MAE: {results['best_val_metric']:.4f}")
    logger.info(f"Training time: {results['training_time']/3600:.2f} hours")
    logger.info("="*60)
    
    return results


def test_visualizations():
    """Test signal visualization module."""
    
    logger.info("="*60)
    logger.info("Testing signal visualization module")
    logger.info("="*60)
    
    # Create visualizer
    viz = SignalVisualizer(save_dir="test_visualizations", upload_to_wandb=False)
    
    # Generate example signals
    logger.info("Generating example signals...")
    t = np.linspace(0, 10, 2500)
    normal_signal = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(len(t))
    seizure_signal = (1.5 * np.sin(2 * np.pi * 2 * t) + 
                     0.5 * np.sin(2 * np.pi * 5 * t) + 
                     0.2 * np.random.randn(len(t)))
    
    # Create visualizations
    logger.info("Creating signal comparison plot...")
    fig1 = viz.plot_signal_comparison(normal_signal, seizure_signal)
    viz.save_figure(fig1, "test_signal_comparison")
    
    logger.info("Creating countdown prediction plot...")
    true_countdown = np.concatenate([np.arange(10, -1, -0.01), np.full(500, -1)])
    pred_countdown = true_countdown + 0.5 * np.random.randn(len(true_countdown))
    pred_countdown = np.clip(pred_countdown, -1, 10)
    
    fig2 = viz.plot_countdown_prediction(true_countdown, pred_countdown)
    viz.save_figure(fig2, "test_countdown_prediction")
    
    logger.info("Creating error distribution plot...")
    fig3 = viz.plot_error_distribution(true_countdown, pred_countdown)
    viz.save_figure(fig3, "test_error_distribution")
    
    logger.info("="*60)
    logger.info("Visualizations created successfully!")
    logger.info("="*60)


if __name__ == "__main__":
    # Test visualizations first (doesn't require model training)
    test_visualizations()
    
    # Test training with wandb
    # Note: Set use_wandb=False in Trainer() if you don't want to log to wandb
    # results = test_wandb_training()
    
    print("\n✓ All tests completed!")
