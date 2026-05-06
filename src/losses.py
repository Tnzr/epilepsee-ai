"""
Loss functions for multi-task seizure countdown prediction.

Combines:
1. Binary classification loss (pre-ictal detection)
2. Regression loss (countdown prediction)
3. Optional ranking loss (monotonic decreasing constraint)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict

from config.config import LossConfig


class SeizureCountdownLoss(nn.Module):
    """
    Multi-task loss combining classification and regression with temporal weighting.
    
    L_total = α * L_classification + β * L_regression + γ * L_ranking
    
    where:
    - L_classification: Binary cross-entropy for pre-ictal detection
    - L_regression: Weighted MSE with time-based importance weights
    - L_ranking: Ensures countdown decreases monotonically over time
    """
    
    def __init__(self, config: LossConfig):
        """Initialize loss function.
        
        Args:
            config: LossConfig with loss weights and settings
        """
        super().__init__()
        self.config = config
        
        # Loss weights
        self.alpha = config.classification_weight
        self.beta = config.regression_weight
        self.gamma = config.ranking_weight
        
        # Normalize weights
        total = self.alpha + self.beta + self.gamma + 1e-6
        self.alpha /= total
        self.beta /= total
        self.gamma /= total
        
        # Classification loss
        self.classification_loss_type = getattr(config, "classification_loss_type", "bce")
        self.bce_loss = nn.BCELoss(reduction='mean')
        # Focal loss instance (used when classification_loss_type == "focal")
        self.focal_loss = FocalLoss(
            alpha=getattr(config, "focal_alpha", 0.25),
            gamma=getattr(config, "focal_gamma", 2.0),
        )
        
        # Regression loss type
        self.regression_loss_type = config.regression_loss
        if self.regression_loss_type == "smoothl1":
            self.regression_loss = nn.SmoothL1Loss(reduction='none', beta=0.5)
        else:  # "mse" or "weighted_mse"
            self.regression_loss = nn.MSELoss(reduction='none')
        
        # Temporal weighting
        self.weight_tau = config.weight_tau
        self.use_class_weighting = getattr(config, 'use_class_weighting', False)
        self.classification_positive_weight = getattr(config, 'classification_positive_weight', None)
        self.max_positive_weight = 10.0
    
    def forward(self, pre_ictal_pred: torch.Tensor, countdown_pred: torch.Tensor,
                pre_ictal_true: torch.Tensor, countdown_true: torch.Tensor,
                sample_weights: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass computing all loss components.
        
        Args:
            pre_ictal_pred: (batch,) predicted pre-ictal probabilities [0, 1]
            countdown_pred: (batch,) predicted countdown in minutes [0, 10]
            pre_ictal_true: (batch,) ground truth binary labels {0, 1}
            countdown_true: (batch,) ground truth countdown in minutes
            sample_weights: (batch,) temporal importance weights
        
        Returns:
            Dictionary with loss components: 'total', 'classification', 'regression', 'ranking'
        """
        # 1. Classification loss: BCE or focal BCE (optionally class-weighted)
        # Always clamp predicted probabilities away from 0/1 to avoid
        # log(0) → inf inside BCE when the model saturates.
        pred = torch.clamp(pre_ictal_pred, 1e-6, 1 - 1e-6)

        if self.classification_loss_type == "focal":
            # Focal loss variant for heavy imbalance. Class weighting is
            # encoded via focal_alpha in the config rather than
            # per-batch pos_weight.
            bce_loss = self.focal_loss(pred, pre_ictal_true)
        elif self.use_class_weighting:
            if self.classification_positive_weight is not None:
                pos_weight = torch.tensor(
                    float(self.classification_positive_weight),
                    device=pred.device,
                    dtype=pred.dtype,
                )
            else:
                pos_count = torch.sum(pre_ictal_true)
                neg_count = torch.sum(1.0 - pre_ictal_true)
                pos_weight = (neg_count + 1e-6) / (pos_count + 1e-6)

            pos_weight = torch.clamp(pos_weight, min=1.0, max=self.max_positive_weight)

            bce_per_sample = -(
                pos_weight * pre_ictal_true * torch.log(pred)
                + (1.0 - pre_ictal_true) * torch.log(1.0 - pred)
            )
            bce_loss = torch.mean(bce_per_sample)
        else:
            bce_loss = self.bce_loss(pred, pre_ictal_true)
        
        # 2. Regression loss: Weighted MSE (only on preictal samples)
        # Create mask that zeros out interictal samples but keeps gradients flowing
        preictal_mask = (pre_ictal_true > 0.5).float()  # Binary mask as float for multiplication
        
        # Compute regression loss on all samples (for gradient flow)
        if self.regression_loss_type in ["weighted_mse", "mse"]:
            raw_loss = self.regression_loss(countdown_pred, countdown_true)
        else:
            raw_loss = self.regression_loss(countdown_pred, countdown_true)
        
        # Apply preictal mask and sample weights (zeros out interictal, emphasizes late preictal)
        weighted_loss = raw_loss * preictal_mask * sample_weights
        
        # Normalize by number of preictal samples (avoid division by zero)
        num_preictal = torch.sum(preictal_mask) + 1e-8
        regression_loss = torch.sum(weighted_loss) / num_preictal
        
        # 3. Ranking loss: Penalize if countdown increases (should decrease over time)
        ranking_loss = self._compute_ranking_loss(countdown_pred, countdown_true)
        
        # Total loss
        total_loss = (self.alpha * bce_loss + 
                     self.beta * regression_loss + 
                     self.gamma * ranking_loss)
        
        return {
            'total': total_loss,
            'classification': bce_loss,
            'regression': regression_loss,
            'ranking': ranking_loss,
        }
    
    def _compute_ranking_loss(self, countdown_pred: torch.Tensor, 
                             countdown_true: torch.Tensor) -> torch.Tensor:
        """Compute ranking loss to enforce monotonic countdown.
        
        Penalizes if predicted countdown increases when it shouldn't.
        Only applies to consecutive preictal samples.
        
        Args:
            countdown_pred: (batch,) predicted countdown
            countdown_true: (batch,) true countdown
        
        Returns:
            Scalar ranking loss
        """
        if self.gamma == 0 or len(countdown_pred) < 2:
            return torch.tensor(0.0, device=countdown_pred.device)
        
        # Compute differences (should be negative for countdown decreasing)
        countdown_diffs = countdown_pred[1:] - countdown_pred[:-1]
        
        # Create preictal mask for consecutive pairs
        preictal_mask = ((countdown_true[:-1] >= 0) & (countdown_true[1:] >= 0)).float()
        
        # Penalize positive differences (increasing countdown) only on preictal pairs
        ranking_penalty = F.relu(countdown_diffs) * preictal_mask
        
        # Normalize by number of preictal pairs
        num_preictal_pairs = torch.sum(preictal_mask) + 1e-8
        
        return torch.sum(ranking_penalty) / num_preictal_pairs


class WeightedMSELoss(nn.Module):
    """
    Time-weighted MSE loss that penalizes late errors more heavily.
    
    Weights: w(t) = exp(-(T-t)/τ)
    where T = seizure time, t = current time, τ = time constant
    """
    
    def __init__(self, tau: float = 60.0):
        """Initialize weighted MSE loss.
        
        Args:
            tau: Time constant in seconds (default 60s = 1 minute)
        """
        super().__init__()
        self.tau = tau
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor,
                weights: torch.Tensor) -> torch.Tensor:
        """Compute weighted MSE.
        
        Args:
            pred: (batch,) predictions
            target: (batch,) targets
            weights: (batch,) temporal weights
        
        Returns:
            Scalar weighted MSE loss
        """
        mse = (pred - target) ** 2
        weighted_mse = mse * weights
        return torch.mean(weighted_mse)


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance (pre-ictal vs inter-ictal).
    
    L = -α * (1 - p_t)^γ * log(p_t)
    
    where:
    - α: class balancing weight
    - γ: focusing parameter (higher = more focus on hard negatives)
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """Initialize focal loss.
        
        Args:
            alpha: Weighting factor [0, 1]
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.
        
        Args:
            pred: (batch,) predicted probabilities
            target: (batch,) binary targets
        
        Returns:
            Scalar focal loss
        """
        # Clamp to avoid log(0) when probabilities saturate.
        eps = 1e-6
        pred = torch.clamp(pred, eps, 1.0 - eps)

        # Standard focal loss for binary classification:
        #   FL = - alpha_t * (1 - p_t)^gamma * log(p_t)
        # where p_t is the model-assigned probability of the true class and
        # alpha_t balances positive vs negative examples.
        p_t = pred * target + (1.0 - pred) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        focal_weight = (1.0 - p_t) ** self.gamma

        focal_loss = -alpha_t * focal_weight * torch.log(p_t)
        return focal_loss.mean()


class LossFactory:
    """Factory for creating loss functions."""
    
    @staticmethod
    def create_loss(config: LossConfig) -> nn.Module:
        """Create loss function based on config.
        
        Args:
            config: LossConfig
        
        Returns:
            Instantiated loss module
        """
        return SeizureCountdownLoss(config)


if __name__ == "__main__":
    from config.config import DEFAULT_CONFIG
    
    # Create loss function
    loss_fn = SeizureCountdownLoss(DEFAULT_CONFIG.loss)
    
    # Create sample data
    batch_size = 32
    pre_ictal_pred = torch.sigmoid(torch.randn(batch_size))
    countdown_pred = torch.clamp(torch.randn(batch_size) * 2 + 5, 0, 10)
    pre_ictal_true = torch.randint(0, 2, (batch_size,)).float()
    countdown_true = torch.rand(batch_size) * 10
    weights = torch.ones(batch_size)
    
    # Compute loss
    loss_dict = loss_fn(pre_ictal_pred, countdown_pred, pre_ictal_true, countdown_true, weights)
    
    print("Loss components:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value.item():.4f}")
