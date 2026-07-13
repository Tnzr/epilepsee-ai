#!/usr/bin/env python
"""
Quick verification test for the training pipeline.
Tests all components with minimal data and 2 epochs.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ['PYTHONPATH'] = str(Path(__file__).parent.parent)

import torch
import numpy as np
import logging

from config.config import Config, DEFAULT_CONFIG
from src import (
    SeizureDataset,
    Trainer,
    Evaluator,
    ModelFactory,
    LossFactory,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model_creation():
    """Test model instantiation."""
    logger.info("="*60)
    logger.info("TEST 1: Model Creation")
    logger.info("="*60)
    
    config = DEFAULT_CONFIG
    
    for model_type in ['ecg_lstm', 'cnn_lstm', 'multimodal']:
        config.model.model_type = model_type
        try:
            model = ModelFactory.create_model(config.model)
            num_params = sum(p.numel() for p in model.parameters())
            logger.info(f"✓ {model_type}: {num_params:,} parameters")
        except Exception as e:
            logger.error(f"✗ {model_type} failed: {str(e)}")
            return False
    
    return True


def test_loss_creation():
    """Test loss function instantiation."""
    logger.info("\n" + "="*60)
    logger.info("TEST 2: Loss Function Creation")
    logger.info("="*60)
    
    config = DEFAULT_CONFIG
    
    try:
        loss_fn = LossFactory.create_loss(config.loss)
        logger.info(f"✓ Loss function created: {loss_fn.__class__.__name__}")
        
        # Test forward pass
        batch_size = 8
        pred_class = torch.rand(batch_size)
        pred_countdown = torch.rand(batch_size) * 10
        target_class = (torch.rand(batch_size) > 0.5).float()
        target_countdown = torch.rand(batch_size) * 10
        sample_weights = torch.ones(batch_size)
        
        loss_dict = loss_fn(pred_class, pred_countdown, target_class, target_countdown, sample_weights)
        logger.info(f"✓ Loss forward pass successful: {loss_dict['total'].item():.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Loss creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_dataset_creation():
    """Test dataset creation."""
    logger.info("\n" + "="*60)
    logger.info("TEST 3: Dataset Creation")
    logger.info("="*60)
    
    # Create small dummy dataset
    n_samples = 100
    n_timesteps = 600
    n_features = 14
    
    X = np.random.randn(n_samples, n_timesteps, n_features).astype(np.float32)
    y = np.clip(np.random.randn(n_samples) * 3 + 5, -1, 10).astype(np.float32)
    
    try:
        dataset = SeizureDataset(X, y)
        logger.info(f"✓ Dataset created: {len(dataset)} samples")
        
        # Test data loading
        features, label, weight = dataset[0]
        logger.info(f"✓ Sample shape: {features.shape}, label: {label:.2f}, weight: {weight:.2f}")
        
        # Test class distribution
        dist = dataset.class_distribution
        logger.info(f"✓ Class distribution: {dist}")
        
        return dataset
    except Exception as e:
        logger.error(f"✗ Dataset creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_training_epoch():
    """Test single training epoch."""
    logger.info("\n" + "="*60)
    logger.info("TEST 4: Training Epoch")
    logger.info("="*60)
    
    # Setup
    config = DEFAULT_CONFIG
    config.model.model_type = 'ecg_lstm'
    config.training.batch_size = 16
    config.training.num_epochs = 2
    config.training.distributed = False  # No DDP for quick test
    config.training.num_workers = 0  # Avoid multiprocessing issues
    
    # Create small datasets
    n_train = 100
    n_val = 30
    
    X_train = np.random.randn(n_train, 600, 14).astype(np.float32)
    y_train = np.clip(np.random.randn(n_train) * 3 + 5, -1, 10).astype(np.float32)
    
    X_val = np.random.randn(n_val, 600, 14).astype(np.float32)
    y_val = np.clip(np.random.randn(n_val) * 3 + 5, -1, 10).astype(np.float32)
    
    train_dataset = SeizureDataset(X_train, y_train)
    val_dataset = SeizureDataset(X_val, y_val)
    
    logger.info(f"Train: {len(train_dataset)} samples, Val: {len(val_dataset)} samples")
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    try:
        # Create trainer
        trainer = Trainer(config, device=device)
        logger.info("✓ Trainer created")
        
        # Train for 2 epochs
        results = trainer.train(train_dataset, val_dataset)
        
        logger.info(f"✓ Training completed!")
        logger.info(f"  - Epochs: {len(results['history']['train_loss'])}")
        logger.info(f"  - Final train loss: {results['history']['train_loss'][-1]:.4f}")
        logger.info(f"  - Final val MAE: {results['history']['val_mae'][-1]:.4f}")
        logger.info(f"  - Best val MAE: {results['best_val_metric']:.4f}")
        logger.info(f"  - Training time: {results['training_time']:.2f}s")
        
        return True
    except Exception as e:
        logger.error(f"✗ Training failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluation():
    """Test evaluation metrics."""
    logger.info("\n" + "="*60)
    logger.info("TEST 5: Evaluation")
    logger.info("="*60)
    
    config = DEFAULT_CONFIG
    config.model.model_type = 'ecg_lstm'
    
    # Create small test dataset
    n_test = 50
    X_test = np.random.randn(n_test, 600, 14).astype(np.float32)
    y_test = np.clip(np.random.randn(n_test) * 3 + 5, -1, 10).astype(np.float32)
    test_dataset = SeizureDataset(X_test, y_test)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        # Create model
        model = ModelFactory.create_model(config.model)
        model = model.to(device)
        model.eval()
        
        # Create evaluator
        evaluator = Evaluator(config)
        logger.info("✓ Evaluator created")
        
        # Evaluate
        metrics = evaluator.evaluate(model, test_dataset, device)
        
        logger.info(f"✓ Evaluation completed!")
        logger.info(f"  - MAE: {metrics['mae']:.4f}")
        logger.info(f"  - RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  - MEDAE: {metrics['medae']:.4f}")
        logger.info(f"  - Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"  - Sensitivity @ 5min: {metrics.get('sensitivity_5min', 0):.4f}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Evaluation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all verification tests."""
    logger.info("\n" + "#"*60)
    logger.info("#  TRAINING PIPELINE VERIFICATION TEST")
    logger.info("#"*60 + "\n")
    
    results = {}
    
    # Test 1: Model creation
    results['model_creation'] = test_model_creation()
    
    # Test 2: Loss creation
    results['loss_creation'] = test_loss_creation()
    
    # Test 3: Dataset creation
    dataset = test_dataset_creation()
    results['dataset_creation'] = dataset is not None
    
    # Test 4: Training epoch
    results['training'] = test_training_epoch()
    
    # Test 5: Evaluation
    results['evaluation'] = test_evaluation()
    
    # Summary
    logger.info("\n" + "="*60)
    logger.info("VERIFICATION SUMMARY")
    logger.info("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        logger.info(f"{test_name.ljust(30)}: {status}")
        if not passed:
            all_passed = False
    
    logger.info("="*60)
    
    if all_passed:
        logger.info("\n🎉 ALL TESTS PASSED! Training pipeline is functional.")
        return 0
    else:
        logger.error("\n❌ SOME TESTS FAILED. Check errors above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
