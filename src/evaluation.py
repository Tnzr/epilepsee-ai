"""
Evaluation module with comprehensive metrics for seizure countdown prediction.

Metrics:
- Regression: MAE, MEDAE, RMSE
- Clinical: Sensitivity @ lead times, False positive rate
- Stability: Prediction jitter
- Classification: Accuracy, AUROC, F1-score
"""

import logging
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix, roc_curve, precision_recall_curve, auc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

from config.config import Config, EvaluationConfig
from src.data_loader import SeizureDataset
from src.models import ModelFactory


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Causal prediction smoother (zero-retraining, deploy on top of any model)
# ---------------------------------------------------------------------------

def causal_smooth_predictions(
    pred_probs: np.ndarray,
    rise_alpha: float = 0.50,
    fall_alpha: float = 0.12,
    streak_threshold: float = 0.30,
    streak_window: int = 8,
    streak_max_bonus: float = 0.15,
) -> np.ndarray:
    """Causal asymmetric EMA with alarm-streak confidence bonus.

    Each raw per-window prediction is treated as a noisy instantaneous
    estimate. This smoother runs causally (no future lookahead) and:

    * Uses a **fast EMA** (rise_alpha) when the raw alarm goes up — alarm
      escalation should not be delayed.
    * Uses a **slow EMA** (fall_alpha) when the raw alarm drops — brief
      true alarms must not be suppressed by a single quiet window.
    * Adds a small **streak bonus** when K consecutive windows are above
      a threshold, rewarding sustained activity.

    Args:
        pred_probs:       Raw per-window alarm probabilities, shape (N,).
        rise_alpha:       EMA weight for alarm escalation  (fast, 0–1).
        fall_alpha:       EMA weight for alarm decay        (slow, 0–1).
        streak_threshold: Raw prob above this is an 'active' window.
        streak_window:    Consecutive active windows needed for full bonus.
        streak_max_bonus: Max probability bonus added when streak is full.

    Returns:
        Smoothed alarm probabilities, shape (N,), values in [0, 1].
    """
    n = len(pred_probs)
    if n == 0:
        return np.array([], dtype=np.float32)

    smoothed = np.empty(n, dtype=np.float32)
    state = float(pred_probs[0])
    consecutive_above = 0

    for i in range(n):
        p = float(pred_probs[i])
        # Asymmetric EMA — fast up, slow down
        alpha = rise_alpha if p >= state else fall_alpha
        state = alpha * p + (1.0 - alpha) * state

        # Streak counter
        if p >= streak_threshold:
            consecutive_above += 1
        else:
            consecutive_above = 0

        # Smooth streak bonus: ramps linearly 0 → streak_max_bonus
        streak_frac = min(consecutive_above / max(streak_window, 1), 1.0)
        bonus = streak_frac * streak_max_bonus
        smoothed[i] = float(np.clip(state + bonus, 0.0, 1.0))

    return smoothed


class Evaluator:
    """
    Comprehensive evaluation framework for seizure prediction models.
    """
    
    def __init__(self, config: Config):
        """Initialize evaluator.
        
        Args:
            config: Master Config object
        """
        self.config = config
        self.eval_config = config.evaluation

    def _detection_threshold(self) -> float:
        """Read current detection threshold from config."""
        return float(getattr(self.config.loss, 'detection_threshold', 0.5))

    def _forward_model(self, model: torch.nn.Module, features: torch.Tensor):
        """Forward pass wrapper for single-modal and multimodal models."""
        if ModelFactory.is_multimodal(self.config.model.model_type):
            ecg_dim = self.config.model.ecg_feature_dim
            eeg_dim = self.config.model.eeg_feature_dim
            motion_dim = self.config.model.motion_feature_dim

            ecg_x = features[:, :, :ecg_dim]
            eeg_x = features[:, :, ecg_dim:ecg_dim + eeg_dim]
            motion_x = features[:, :, ecg_dim + eeg_dim:ecg_dim + eeg_dim + motion_dim]
            outputs = model(ecg_x, eeg_x, motion_x)
        else:
            outputs = model(features)

        if isinstance(outputs, dict):
            pre_ictal_pred = None
            countdown_pred = None
            for key in ('pre_ictal_pred', 'classification', 'pre_ictal_logits'):
                if key in outputs and outputs[key] is not None:
                    pre_ictal_pred = outputs[key]
                    break
            for key in ('countdown_pred', 'regression', 'countdown'):
                if key in outputs and outputs[key] is not None:
                    countdown_pred = outputs[key]
                    break
            if pre_ictal_pred is None or countdown_pred is None:
                raise ValueError(
                    "Model forward dict output must include pre-ictal and countdown tensors. "
                    f"Available keys: {sorted(outputs.keys())}"
                )
            return pre_ictal_pred, countdown_pred

        if isinstance(outputs, (tuple, list)):
            if len(outputs) < 2:
                raise ValueError(
                    "Model forward must return at least two outputs: "
                    "(pre_ictal_pred, countdown_pred)"
                )
            return outputs[0], outputs[1]

        raise ValueError(
            "Model forward returned unsupported output type. Expected tuple/list/dict with "
            "pre-ictal and countdown predictions."
        )
    
    def evaluate(self, model: torch.nn.Module, dataset: SeizureDataset,
                device: torch.device) -> Dict[str, float]:
        """Comprehensive evaluation on dataset.
        
        Args:
            model: Trained model
            dataset: SeizureDataset
            device: torch device
        
        Returns:
            Dictionary with all metrics
        """
        model.eval()
        
        all_pred_preictal = []
        all_pred_countdown = []
        all_true_preictal = []
        all_true_countdown = []
        
        with torch.no_grad():
            # Create dataloader
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=64, shuffle=False)
            
            for batch in loader:
                features = batch[0].to(device).float()
                labels = batch[1].to(device).float()
                
                # Predict
                pre_ictal_pred, countdown_pred = self._forward_model(model, features)
                
                # Store predictions
                all_pred_preictal.extend(pre_ictal_pred.cpu().numpy())
                all_pred_countdown.extend(countdown_pred.cpu().numpy())
                
                # Store labels
                all_true_preictal.extend((labels >= 0).cpu().numpy())
                all_true_countdown.extend(labels.cpu().numpy())
        
        all_pred_preictal = np.array(all_pred_preictal)
        all_pred_countdown = np.array(all_pred_countdown)
        all_true_preictal = np.array(all_true_preictal, dtype=bool)
        all_true_countdown = np.array(all_true_countdown)
        
        # Compute metrics
        metrics = {}
        
        # Regression metrics (pre-ictal only)
        preictal_mask = all_true_countdown >= 0
        if np.sum(preictal_mask) > 0:
            metrics.update(self._compute_regression_metrics(
                all_pred_countdown[preictal_mask],
                all_true_countdown[preictal_mask]
            ))
            metrics.update(self._compute_countdown_bucket_metrics(
                all_pred_countdown[preictal_mask],
                all_true_countdown[preictal_mask]
            ))
        
        # Classification metrics
        metrics.update(self._compute_classification_metrics(
            all_pred_preictal,
            all_true_preictal
        ))
        
        # Clinical metrics
        metrics.update(self._compute_clinical_metrics(
            all_pred_countdown,
            all_true_countdown
        ))
        
        # Stability metrics
        metrics['prediction_stability'] = self._compute_stability(all_pred_countdown)

        # Longitudinal Bayesian memory metrics (when timeline metadata exists).
        sample_end_times_s = getattr(dataset, 'sample_end_times_s', None)
        recording_ids = getattr(dataset, 'recording_ids', None)
        bayes_enabled = bool(getattr(self.eval_config, 'enable_bayesian_memory_eval', True))
        if bayes_enabled and sample_end_times_s is not None and recording_ids is not None:
            try:
                times_arr = np.asarray(sample_end_times_s, dtype=np.float64)
                rec_arr = np.asarray(recording_ids).astype(str)
                if len(times_arr) == len(all_true_countdown) and len(rec_arr) == len(all_true_countdown):
                    bayes_payload = self.simulate_bayesian_long_sweep(
                        pred_preictal=all_pred_preictal,
                        pred_countdown=all_pred_countdown,
                        true_countdown=all_true_countdown,
                        sample_end_times_s=times_arr,
                        recording_ids=rec_arr,
                    )
                    metrics.update(bayes_payload.get('metrics', {}))
                else:
                    logger.warning(
                        "Skipping Bayesian memory metrics due to metadata length mismatch "
                        "(pred=%d, times=%d, rec=%d)",
                        len(all_true_countdown),
                        len(times_arr),
                        len(rec_arr),
                    )
            except Exception as bayes_error:
                logger.warning("Bayesian memory metrics skipped due to error: %s", str(bayes_error))
        
        return metrics

    def collect_predictions(self, model: torch.nn.Module, dataset: SeizureDataset,
                           device: torch.device) -> Dict[str, np.ndarray]:
        """Run inference and return arrays for downstream plotting/export."""
        model.eval()

        all_pred_preictal = []
        all_pred_countdown = []
        all_true_countdown = []

        with torch.no_grad():
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=64, shuffle=False)

            for batch in loader:
                features = batch[0].to(device).float()
                labels = batch[1].to(device).float()

                if ModelFactory.is_multimodal(self.config.model.model_type):
                    ecg_dim = self.config.model.ecg_feature_dim
                    eeg_dim = self.config.model.eeg_feature_dim
                    motion_dim = self.config.model.motion_feature_dim
                    ecg_x = features[:, :, :ecg_dim]
                    eeg_x = features[:, :, ecg_dim:ecg_dim + eeg_dim]
                    motion_x = features[:, :, ecg_dim + eeg_dim:ecg_dim + eeg_dim + motion_dim]
                    pre_ictal_pred, countdown_pred = model(ecg_x, eeg_x, motion_x)
                else:
                    pre_ictal_pred, countdown_pred = model(features)

                all_pred_preictal.extend(pre_ictal_pred.detach().cpu().numpy())
                all_pred_countdown.extend(countdown_pred.detach().cpu().numpy())
                all_true_countdown.extend(labels.detach().cpu().numpy())

        true_countdown = np.array(all_true_countdown)
        pred_preictal = np.array(all_pred_preictal, dtype=np.float32)
        pred_preictal_smooth = causal_smooth_predictions(pred_preictal)
        payload = {
            "pred_preictal": pred_preictal,
            "pred_preictal_smooth": pred_preictal_smooth,
            "pred_countdown": np.array(all_pred_countdown),
            "true_preictal": (true_countdown >= 0).astype(np.int32),
            "true_countdown": true_countdown,
        }

        sample_end_times_s = getattr(dataset, 'sample_end_times_s', None)
        recording_ids = getattr(dataset, 'recording_ids', None)
        if sample_end_times_s is not None:
            payload['sample_end_times_s'] = np.asarray(sample_end_times_s, dtype=np.float64)
        if recording_ids is not None:
            payload['recording_ids'] = np.asarray(recording_ids).astype(str)

        return payload

    @staticmethod
    def _compute_ece(pred: np.ndarray, true: np.ndarray, n_bins: int = 10) -> float:
        """Compute Expected Calibration Error for binary predictions."""
        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(true).astype(np.int32)

        if pred.size == 0 or true.size == 0:
            return 0.0

        bins = np.linspace(0.0, 1.0, num=max(2, int(n_bins) + 1), dtype=np.float32)
        ece = 0.0
        n_total = float(pred.size)

        for i in range(len(bins) - 1):
            left = float(bins[i])
            right = float(bins[i + 1])
            if i == len(bins) - 2:
                mask = (pred >= left) & (pred <= right)
            else:
                mask = (pred >= left) & (pred < right)
            count = int(np.sum(mask))
            if count == 0:
                continue
            conf = float(np.mean(pred[mask]))
            acc = float(np.mean(true[mask]))
            ece += (count / n_total) * abs(acc - conf)

        return float(ece)

    def simulate_bayesian_long_sweep(
        self,
        pred_preictal: np.ndarray,
        pred_countdown: np.ndarray,
        true_countdown: np.ndarray,
        sample_end_times_s: np.ndarray,
        recording_ids: np.ndarray,
        decay: float = 0.995,
        fusion_alpha: float = 0.65,
        n_countdown_bins: int = 8,
        n_prob_bins: int = 4,
    ) -> Dict[str, Any]:
        """Run chronological per-recording Bayesian memory simulation.

        This applies an online token-count + transition-memory model over test
        predictions to emulate day-long watch operation with persistent state.
        """
        pred_preictal = np.asarray(pred_preictal, dtype=np.float32)
        pred_countdown = np.asarray(pred_countdown, dtype=np.float32)
        true_countdown = np.asarray(true_countdown, dtype=np.float32)
        sample_end_times_s = np.asarray(sample_end_times_s, dtype=np.float64)
        recording_ids = np.asarray(recording_ids).astype(str)

        n = len(pred_preictal)
        if not (n == len(pred_countdown) == len(true_countdown) == len(sample_end_times_s) == len(recording_ids)):
            raise ValueError("simulate_bayesian_long_sweep expects equal-length arrays")

        if n == 0:
            return {
                'fused_preictal': np.array([], dtype=np.float32),
                'fused_preictal_smooth': np.array([], dtype=np.float32),
                'memory_risk': np.array([], dtype=np.float32),
                'uncertainty': np.array([], dtype=np.float32),
                'token_id': np.array([], dtype=np.int32),
                'timeline_order_idx': np.array([], dtype=np.int64),
                'metrics': {},
            }

        order_idx = np.lexsort((sample_end_times_s, recording_ids))
        max_countdown = float(max(1e-6, self.config.model.output_countdown_max))

        n_countdown_bins = max(2, int(n_countdown_bins))
        n_prob_bins = max(2, int(n_prob_bins))
        token_count = (n_countdown_bins + 1) * n_prob_bins

        fused_preictal = np.zeros(n, dtype=np.float32)
        memory_risk = np.zeros(n, dtype=np.float32)
        uncertainty = np.zeros(n, dtype=np.float32)
        token_id_arr = np.zeros(n, dtype=np.int32)

        current_rec = None
        c = np.zeros(token_count, dtype=np.float64)
        A = np.zeros((token_count, token_count), dtype=np.float64)
        N = 0.0
        prev_token = -1

        for idx in order_idx:
            rec_id = recording_ids[idx]
            if current_rec != rec_id:
                current_rec = rec_id
                c.fill(0.0)
                A.fill(0.0)
                N = 0.0
                prev_token = -1

            p_raw = float(np.clip(pred_preictal[idx], 0.0, 1.0))
            cd_raw = float(pred_countdown[idx])

            if cd_raw < 0:
                countdown_bin = 0
            else:
                frac = min(max(cd_raw / max_countdown, 0.0), 0.999999)
                countdown_bin = 1 + int(frac * n_countdown_bins)

            prob_bin = min(int(p_raw * n_prob_bins), n_prob_bins - 1)
            token_id = countdown_bin * n_prob_bins + prob_bin
            token_id_arr[idx] = int(token_id)

            if N > 0:
                p_hist = float((c[token_id] + 1.0) / (N + token_count))
            else:
                p_hist = 1.0 / float(token_count)

            if prev_token >= 0:
                row_sum = float(np.sum(A[prev_token]))
                p_trans = float((A[prev_token, token_id] + 1.0) / (row_sum + token_count))
            else:
                p_trans = p_hist

            mem_risk = float(np.clip(0.5 * p_hist + 0.5 * p_trans, 0.0, 1.0))
            fused = float(np.clip(fusion_alpha * p_raw + (1.0 - fusion_alpha) * mem_risk, 0.0, 1.0))
            unc = float(np.sqrt(max(fused * (1.0 - fused), 0.0) / max(N + 2.0, 1.0)))

            memory_risk[idx] = mem_risk
            fused_preictal[idx] = fused
            uncertainty[idx] = unc

            c *= float(decay)
            A *= float(decay)
            N *= float(decay)

            c[token_id] += 1.0
            N += 1.0
            if prev_token >= 0:
                A[prev_token, token_id] += 1.0
            prev_token = int(token_id)

        # Smooth fused predictions per recording so continuity stays local.
        fused_preictal_smooth = np.zeros_like(fused_preictal)
        unique_recs = np.unique(recording_ids)
        for rec_id in unique_recs:
            rec_mask = recording_ids == rec_id
            rec_idx = np.where(rec_mask)[0]
            rec_order = rec_idx[np.argsort(sample_end_times_s[rec_idx])]
            rec_smooth = causal_smooth_predictions(fused_preictal[rec_order])
            fused_preictal_smooth[rec_order] = rec_smooth

        true_preictal = (true_countdown >= 0).astype(np.int32)
        cls_metrics = self._compute_classification_metrics(fused_preictal_smooth, true_preictal)
        metrics = {f'bayes_{k}': float(v) for k, v in cls_metrics.items()}
        metrics['bayes_ece'] = float(self._compute_ece(fused_preictal_smooth, true_preictal, n_bins=10))
        metrics['bayes_brier'] = float(np.mean((fused_preictal_smooth - true_preictal.astype(np.float32)) ** 2))

        return {
            'fused_preictal': fused_preictal.astype(np.float32),
            'fused_preictal_smooth': fused_preictal_smooth.astype(np.float32),
            'memory_risk': memory_risk.astype(np.float32),
            'uncertainty': uncertainty.astype(np.float32),
            'token_id': token_id_arr.astype(np.int32),
            'timeline_order_idx': order_idx.astype(np.int64),
            'metrics': metrics,
        }

    def select_optimal_threshold(self,
                                 pred_preictal: np.ndarray,
                                 true_preictal: np.ndarray,
                                 objective: str = "balanced_accuracy",
                                 min_sensitivity: float = 0.0,
                                 max_fpr: float = 1.0,
                                 num_thresholds: int = 81) -> Dict[str, float]:
        """Sweep thresholds and select operating point.

        Args:
            pred_preictal: Predicted pre-ictal probabilities [0,1].
            true_preictal: Ground-truth binary labels {0,1}.
            objective: "balanced_accuracy" or "f1".
            min_sensitivity: Minimum required sensitivity.
            max_fpr: Maximum allowed false-positive rate.
            num_thresholds: Number of thresholds in [0.1, 0.9].

        Returns:
            Dict with selected threshold + metrics at that threshold.
        """
        pred = np.asarray(pred_preictal, dtype=np.float32)
        true = np.asarray(true_preictal).astype(np.int32)

        if pred.size == 0 or true.size == 0 or np.unique(true).size < 2:
            threshold = float(self._detection_threshold())
            return {
                "threshold": threshold,
                "objective": 0.0,
                "accuracy": 0.0,
                "f1": 0.0,
                "sensitivity": 0.0,
                "specificity": 0.0,
                "fpr": 1.0,
            }

        thresholds = np.linspace(0.10, 0.90, num=max(3, int(num_thresholds)), dtype=np.float32)
        best = None

        for thr in thresholds:
            pred_bin = (pred >= thr).astype(np.int32)
            tn, fp, fn, tp = confusion_matrix(true, pred_bin, labels=[0, 1]).ravel()

            sensitivity = tp / max(1, tp + fn)
            specificity = tn / max(1, tn + fp)
            fpr = fp / max(1, fp + tn)
            acc = float(np.mean(pred_bin == true))
            f1 = float(f1_score(true, pred_bin, zero_division=0))
            balanced = 0.5 * (sensitivity + specificity)

            if sensitivity < float(min_sensitivity) or fpr > float(max_fpr):
                continue

            objective_value = balanced if objective == "balanced_accuracy" else f1
            candidate = {
                "threshold": float(thr),
                "objective": float(objective_value),
                "accuracy": float(acc),
                "f1": float(f1),
                "sensitivity": float(sensitivity),
                "specificity": float(specificity),
                "fpr": float(fpr),
            }

            if best is None or candidate["objective"] > best["objective"]:
                best = candidate

        if best is None:
            # Fallback to unconstrained best objective
            for thr in thresholds:
                pred_bin = (pred >= thr).astype(np.int32)
                tn, fp, fn, tp = confusion_matrix(true, pred_bin, labels=[0, 1]).ravel()
                sensitivity = tp / max(1, tp + fn)
                specificity = tn / max(1, tn + fp)
                fpr = fp / max(1, fp + tn)
                acc = float(np.mean(pred_bin == true))
                f1 = float(f1_score(true, pred_bin, zero_division=0))
                balanced = 0.5 * (sensitivity + specificity)
                objective_value = balanced if objective == "balanced_accuracy" else f1
                candidate = {
                    "threshold": float(thr),
                    "objective": float(objective_value),
                    "accuracy": float(acc),
                    "f1": float(f1),
                    "sensitivity": float(sensitivity),
                    "specificity": float(specificity),
                    "fpr": float(fpr),
                }
                if best is None or candidate["objective"] > best["objective"]:
                    best = candidate

        return best

    def plot_classification_and_countdown(self,
                                          pred_preictal: np.ndarray,
                                          pred_countdown: np.ndarray,
                                          true_preictal: np.ndarray,
                                          true_countdown: np.ndarray,
                                          save_path: Optional[Path] = None) -> None:
        """Plot combined classification and countdown performance overview."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        unique_classes = np.unique(true_preictal.astype(int))

        # ROC curve
        ax = axes[0, 0]
        if unique_classes.size > 1:
            fpr, tpr, _ = roc_curve(true_preictal, pred_preictal)
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
            ax.plot([0, 1], [0, 1], 'k--', alpha=0.6)
        else:
            ax.text(0.5, 0.5, 'ROC unavailable\n(single class in y_true)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('Classification ROC')
        if unique_classes.size > 1:
            ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)

        # Precision-Recall curve
        ax = axes[0, 1]
        if unique_classes.size > 1:
            precision, recall, _ = precision_recall_curve(true_preictal, pred_preictal)
            pr_auc = auc(recall, precision)
            ax.plot(recall, precision, label=f'AUC = {pr_auc:.3f}')
        else:
            ax.text(0.5, 0.5, 'PR unavailable\n(single class in y_true)',
                    ha='center', va='center', transform=ax.transAxes)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Classification Precision-Recall')
        if unique_classes.size > 1:
            ax.legend(loc='lower left')
        ax.grid(True, alpha=0.3)

        # Confusion matrix
        ax = axes[1, 0]
        threshold = self._detection_threshold()
        pred_binary = (pred_preictal >= threshold).astype(int)
        cm = confusion_matrix(true_preictal, pred_binary, labels=[0, 1])
        im = ax.imshow(cm, cmap='Blues')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Interictal', 'Preictal'])
        ax.set_yticklabels(['Interictal', 'Preictal'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'Confusion Matrix @{threshold:.2f}')
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # Countdown regression scatter
        ax = axes[1, 1]
        mask = true_countdown >= 0
        if np.any(mask):
            ax.scatter(true_countdown[mask], pred_countdown[mask], alpha=0.4, s=12)
            lim_low = min(float(np.min(true_countdown[mask])), float(np.min(pred_countdown[mask])))
            lim_high = max(float(np.max(true_countdown[mask])), float(np.max(pred_countdown[mask])))
            ax.plot([lim_low, lim_high], [lim_low, lim_high], 'r--', alpha=0.7)
            ax.set_xlabel('True Countdown (min)')
            ax.set_ylabel('Pred Countdown (min)')
            ax.set_title('Countdown Regression (Preictal)')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No preictal samples', ha='center', va='center')
            ax.set_axis_off()

        plt.tight_layout()
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved combined performance figure to {save_path}")
            plt.close(fig)
        else:
            plt.show()
    
    def _compute_regression_metrics(self, pred: np.ndarray,
                                   true: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics.
        
        Args:
            pred: Predicted countdown
            true: True countdown
        
        Returns:
            Dictionary with MAE, MEDAE, RMSE
        """
        errors = np.abs(pred - true)
        
        return {
            'mae': float(np.mean(errors)),
            'medae': float(np.median(errors)),
            'rmse': float(np.sqrt(np.mean((pred - true) ** 2))),
            'mae_std': float(np.std(errors)),
        }

    def _compute_countdown_bucket_metrics(self, pred: np.ndarray,
                                          true: np.ndarray) -> Dict[str, float]:
        """Compute regression metrics over clinically meaningful countdown buckets.

        Buckets come from config and are interpreted as contiguous lead-time ranges
        in minutes before seizure, e.g. [0,2), [2,5), [5,10].

        Args:
            pred: Predicted countdown values for preictal samples.
            true: True countdown values for preictal samples.

        Returns:
            Flat metric dictionary with per-bucket counts/MAE/RMSE/MEDAE.
        """
        metrics = {}
        bucket_edges = list(getattr(self.eval_config, 'countdown_metric_buckets', [0.0, 2.0, 5.0, 10.0]))

        if len(bucket_edges) < 2:
            return metrics

        pred = np.asarray(pred, dtype=np.float32)
        true = np.asarray(true, dtype=np.float32)

        for idx in range(len(bucket_edges) - 1):
            low = float(bucket_edges[idx])
            high = float(bucket_edges[idx + 1])

            if idx == len(bucket_edges) - 2:
                mask = (true >= low) & (true <= high)
                bucket_label = f"{low:.0f}_{high:.0f}min"
            else:
                mask = (true >= low) & (true < high)
                bucket_label = f"{low:.0f}_{high:.0f}min"

            count = int(np.sum(mask))
            metrics[f'countdown_bucket_{bucket_label}_count'] = count

            if count == 0:
                continue

            bucket_pred = pred[mask]
            bucket_true = true[mask]
            bucket_errors = np.abs(bucket_pred - bucket_true)

            metrics[f'countdown_bucket_{bucket_label}_mae'] = float(np.mean(bucket_errors))
            metrics[f'countdown_bucket_{bucket_label}_medae'] = float(np.median(bucket_errors))
            metrics[f'countdown_bucket_{bucket_label}_rmse'] = float(
                np.sqrt(np.mean((bucket_pred - bucket_true) ** 2))
            )

        return metrics
    
    def _compute_classification_metrics(self, pred: np.ndarray,
                                       true: np.ndarray) -> Dict[str, float]:
        """Compute classification metrics.
        
        Args:
            pred: Predicted pre-ictal probabilities
            true: True pre-ictal labels
        
        Returns:
            Dictionary with Acc, AUROC, F1, etc.
        """
        # Binary predictions (configurable threshold)
        threshold = self._detection_threshold()
        pred_binary = (pred >= threshold).astype(int)
        
        # Accuracy
        accuracy = np.mean(pred_binary == true)
        
        # AUROC
        if np.unique(true).size > 1:
            try:
                auroc = float(roc_auc_score(true, pred))
                if not np.isfinite(auroc):
                    auroc = 0.0
            except Exception:
                auroc = 0.0
        else:
            auroc = 0.0
        
        # F1-score
        f1 = f1_score(true, pred_binary, zero_division=0)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(true, pred_binary, labels=[0, 1]).ravel()
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        
        return {
            'accuracy': float(accuracy),
            'auroc': float(auroc),
            'f1_score': float(f1),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'precision': float(precision),
        }
    
    def _compute_clinical_metrics(self, pred: np.ndarray,
                                 true: np.ndarray) -> Dict[str, float]:
        """Compute clinical metrics (sensitivity @ lead times, FPR).
        
        Args:
            pred: Predicted countdown
            true: True countdown
        
        Returns:
            Dictionary with clinical metrics
        """
        metrics = {}
        
        # Sensitivity @ different lead times
        for lead_time in self.eval_config.lead_time_thresholds:
            # True positives: correctly predicted ≥ lead_time ahead
            tp_mask = (true >= lead_time) & (pred >= lead_time)
            # False negatives: missed (true ≥ but pred <)
            fn_mask = (true >= lead_time) & (pred < lead_time)
            
            tp_count = np.sum(tp_mask)
            fn_count = np.sum(fn_mask)
            
            if (tp_count + fn_count) > 0:
                sensitivity = tp_count / (tp_count + fn_count)
            else:
                sensitivity = 0.0
            
            metrics[f'sensitivity_@{lead_time}min'] = float(sensitivity)
        
        # False positive rate (inter-ictal epochs)
        interictal_mask = true < 0
        fp_interictal = np.sum((pred[interictal_mask] > 5) if np.sum(interictal_mask) > 0 else 0)
        total_interictal = np.sum(interictal_mask)
        
        if total_interictal > 0:
            # Convert to false alarms per 8 hours (assuming 1-second resolution, 250 Hz)
            fpr_per_8h = (fp_interictal / total_interictal) * (8 * 3600)
            metrics['fpr_per_8hours'] = float(fpr_per_8h)
        
        return metrics
    
    def _compute_stability(self, pred: np.ndarray, window: int = 30) -> float:
        """Compute prediction stability (jitter).
        
        Args:
            pred: Predicted countdown sequence
            window: Window size for stability computation
        
        Returns:
            Median prediction std over windows
        """
        if len(pred) <= window:
            return 0.0
        
        stabilities = []
        for i in range(len(pred) - window):
            window_std = np.std(pred[i:i+window])
            stabilities.append(window_std)
        
        if len(stabilities) == 0:
            return 0.0

        return float(np.median(stabilities))
    
    def print_results(self, metrics: Dict[str, float]) -> None:
        """Pretty print metrics.
        
        Args:
            metrics: Dictionary of metrics
        """
        print("="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        
        # Group metrics by category
        print("\nREGRESSION METRICS:")
        for key in ['mae', 'medae', 'rmse', 'mae_std']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

        bucket_metric_keys = [key for key in metrics if key.startswith('countdown_bucket_')]
        if bucket_metric_keys:
            print("\nCOUNTDOWN BUCKET METRICS:")
            bucket_prefixes = sorted(set(
                key.rsplit('_', 1)[0] for key in bucket_metric_keys if not key.endswith('_count')
            ))
            for prefix in bucket_prefixes:
                count_key = f"{prefix}_count"
                mae_key = f"{prefix}_mae"
                medae_key = f"{prefix}_medae"
                rmse_key = f"{prefix}_rmse"
                bucket_name = prefix.replace('countdown_bucket_', '')
                count = int(metrics.get(count_key, 0))
                if count <= 0:
                    print(f"  {bucket_name}: no samples")
                else:
                    print(
                        f"  {bucket_name}: n={count} | "
                        f"mae={metrics.get(mae_key, 0.0):.4f} | "
                        f"medae={metrics.get(medae_key, 0.0):.4f} | "
                        f"rmse={metrics.get(rmse_key, 0.0):.4f}"
                    )
        
        print("\nCLASSIFICATION METRICS:")
        for key in ['accuracy', 'auroc', 'f1_score', 'sensitivity', 'specificity', 'precision']:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")
        
        print("\nCLINICAL METRICS:")
        for key in metrics:
            if 'sensitivity_@' in key or 'fpr_' in key:
                print(f"  {key}: {metrics[key]:.4f}")
        
        print("\nSTABILITY:")
        print(f"  prediction_stability: {metrics.get('prediction_stability', 0):.4f}")
        
        print("="*60)
    
    def plot_results(self, predictions: np.ndarray, targets: np.ndarray,
                    save_path: Optional[Path] = None) -> None:
        """Plot prediction vs target countdown.
        
        Args:
            predictions: Predicted countdown
            targets: True countdown
            save_path: Path to save figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Prediction vs target (for pre-ictal only)
        preictal_mask = targets >= 0
        ax = axes[0, 0]
        ax.scatter(targets[preictal_mask], predictions[preictal_mask], alpha=0.5)
        ax.plot([0, 10], [0, 10], 'r--', label='Perfect prediction')
        ax.set_xlabel('True Countdown (min)')
        ax.set_ylabel('Predicted Countdown (min)')
        ax.set_title('Prediction vs Target (Pre-ictal)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Error distribution
        errors = predictions[preictal_mask] - targets[preictal_mask]
        ax = axes[0, 1]
        ax.hist(errors, bins=30, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Prediction Error (min)')
        ax.set_ylabel('Frequency')
        ax.set_title('Error Distribution')
        ax.axvline(x=0, color='r', linestyle='--')
        ax.grid(True, alpha=0.3)
        
        # Countdown sequence
        ax = axes[1, 0]
        sample_idx = np.arange(0, min(500, len(predictions)))
        ax.plot(sample_idx, predictions[sample_idx], label='Predicted', alpha=0.7)
        ax.plot(sample_idx, targets[sample_idx], label='True', alpha=0.7)
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Countdown (min)')
        ax.set_title('Countdown Sequence (First 500 samples)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Absolute error vs true countdown
        abs_errors = np.abs(errors)
        ax = axes[1, 1]
        ax.scatter(targets[preictal_mask], abs_errors, alpha=0.5)
        ax.set_xlabel('True Countdown (min)')
        ax.set_ylabel('Absolute Error (min)')
        ax.set_title('Error vs Countdown')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved figure to {save_path}")
        else:
            plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config.config import DEFAULT_CONFIG
    
    # Create evaluator
    evaluator = Evaluator(DEFAULT_CONFIG)
    
    # Create dummy predictions and targets
    pred_preictal = np.random.rand(100)
    pred_countdown = np.random.rand(100) * 10
    true_preictal = (np.random.rand(100) > 0.5).astype(bool)
    true_countdown = np.concatenate([np.random.rand(70) * 10, -np.ones(30)])
    
    # Compute metrics
    metrics = evaluator._compute_regression_metrics(
        pred_countdown[true_countdown >= 0],
        true_countdown[true_countdown >= 0]
    )
    metrics.update(evaluator._compute_classification_metrics(pred_preictal, true_preictal))
    metrics.update(evaluator._compute_clinical_metrics(pred_countdown, true_countdown))
    
    # Print results
    evaluator.print_results(metrics)
