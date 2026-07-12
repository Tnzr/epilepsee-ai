"""
Signal visualization utilities for seizure detection and analysis.

Features:
- Plot raw ECG signals (normal vs seizure)
- Visualize model predictions over time
- Compare predicted vs actual countdown
- Signal quality assessment
- Wandb integration for experiment tracking
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import importlib
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
from sklearn.metrics import confusion_matrix

def _import_wandb_safely():
    """Import real wandb package even when local ./wandb folder shadows it."""
    try:
        import wandb as _wandb
        if hasattr(_wandb, 'init') and hasattr(_wandb, 'Image'):
            return _wandb, True
    except Exception:
        pass

    try:
        import os
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


logger = logging.getLogger(__name__)


class SignalVisualizer:
    """Visualize physiological signals and model predictions."""
    
    def __init__(self, save_dir: Optional[str] = None, upload_to_wandb: bool = True):
        """Initialize visualizer.
        
        Args:
            save_dir: Directory to save plots (optional)
            upload_to_wandb: Whether to upload plots to wandb
        """
        self.save_dir = Path(save_dir) if save_dir else None
        if self.save_dir:
            self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.upload_to_wandb = upload_to_wandb and HAS_WANDB
    
    def plot_signal_comparison(self, 
                               normal_signal: np.ndarray,
                               seizure_signal: np.ndarray,
                               sampling_rate: float = 250,
                               signal_name: str = "ECG",
                               figsize: Tuple[int, int] = (14, 6)) -> plt.Figure:
        """Plot normal vs seizure signals side by side.
        
        Args:
            normal_signal: (timesteps,) normal ECG signal
            seizure_signal: (timesteps,) seizure ECG signal
            sampling_rate: Sampling rate in Hz
            signal_name: Name of signal (ECG, EEG, EMG)
            figsize: Figure size (width, height)
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(2, 1, figsize=figsize)
        
        # Time axis in seconds
        time_normal = np.arange(len(normal_signal)) / sampling_rate
        time_seizure = np.arange(len(seizure_signal)) / sampling_rate
        
        # Plot normal signal
        axes[0].plot(time_normal, normal_signal, color='steelblue', linewidth=1, alpha=0.8)
        axes[0].set_ylabel('Amplitude (μV)', fontsize=11)
        axes[0].set_title(f'Normal {signal_name} Signal', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].axhline(y=0, color='k', linewidth=0.5)
        
        # Plot seizure signal
        axes[1].plot(time_seizure, seizure_signal, color='crimson', linewidth=1, alpha=0.8)
        axes[1].set_ylabel('Amplitude (μV)', fontsize=11)
        axes[1].set_xlabel('Time (seconds)', fontsize=11)
        axes[1].set_title(f'Seizure {signal_name} Signal', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].axhline(y=0, color='k', linewidth=0.5)
        
        # Add statistics
        normal_stats = f"μ={normal_signal.mean():.1f}, σ={normal_signal.std():.1f}"
        seizure_stats = f"μ={seizure_signal.mean():.1f}, σ={seizure_signal.std():.1f}"
        
        axes[0].text(0.98, 0.05, normal_stats, transform=axes[0].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].text(0.98, 0.05, seizure_stats, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='bottom', horizontalalignment='right',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        return fig
    
    def plot_countdown_prediction(self,
                                  true_countdown: np.ndarray,
                                  pred_countdown: np.ndarray,
                                  pre_ictal_true: Optional[np.ndarray] = None,
                                  pre_ictal_pred: Optional[np.ndarray] = None,
                                  sampling_rate: float = 250,
                                  figsize: Tuple[int, int] = (14, 8)) -> plt.Figure:
        """Plot predicted countdown vs true countdown over time.
        
        Args:
            true_countdown: (timesteps,) true countdown in minutes (-1 for inter-ictal)
            pred_countdown: (timesteps,) predicted countdown in minutes
            pre_ictal_true: (timesteps,) binary true pre-ictal labels
            pre_ictal_pred: (timesteps,) predicted pre-ictal probabilities [0, 1]
            sampling_rate: Sampling rate in Hz
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        time_axis = np.arange(len(true_countdown)) / sampling_rate / 60  # Convert to minutes
        
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        gs = GridSpec(2 if pre_ictal_true is not None else 1, 1, figure=fig, hspace=0.3)
        
        # Countdown plot
        ax1 = fig.add_subplot(gs[0])
        
        # Mark pre-ictal and inter-ictal regions
        preictal_mask = true_countdown >= 0
        interictal_mask = true_countdown < 0
        
        ax1.scatter(time_axis[interictal_mask], true_countdown[interictal_mask], 
                   alpha=0.3, s=20, color='gray', label='True (inter-ictal)')
        ax1.scatter(time_axis[preictal_mask], true_countdown[preictal_mask],
                   alpha=0.6, s=30, color='green', marker='o', label='True (pre-ictal)')
        
        ax1.plot(time_axis[preictal_mask], pred_countdown[preictal_mask],
                color='red', linewidth=2, alpha=0.7, label='Predicted')
        
        ax1.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Countdown (minutes)', fontsize=11)
        ax1.set_title('Countdown Prediction: True vs Predicted', fontsize=12, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-2, 11)
        
        # Pre-ictal classification plot
        if pre_ictal_true is not None and pre_ictal_pred is not None:
            ax2 = fig.add_subplot(gs[1])
            
            ax2.scatter(time_axis[interictal_mask], pre_ictal_true[interictal_mask],
                       alpha=0.4, s=20, color='blue', label='True')
            ax2.scatter(time_axis[preictal_mask], pre_ictal_true[preictal_mask],
                       alpha=0.6, s=30, color='blue', marker='o', label='True (pre-ictal)')
            
            ax2.plot(time_axis, pre_ictal_pred, color='orange', linewidth=2, 
                    alpha=0.7, label='Predicted probability')
            
            ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Decision threshold')
            ax2.set_ylabel('Pre-ictal Probability', fontsize=11)
            ax2.set_xlabel('Time (minutes)', fontsize=11)
            ax2.set_title('Pre-ictal Classification', fontsize=12, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(-0.1, 1.1)
        else:
            ax1.set_xlabel('Time (minutes)', fontsize=11)
        
        return fig
    
    def plot_feature_importance(self,
                               features: np.ndarray,
                               feature_names: List[str],
                               predicted_labels: np.ndarray,
                               figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """Plot feature values colored by prediction confidence.
        
        Args:
            features: (timesteps, num_features) feature matrix
            feature_names: List of feature names
            predicted_labels: (timesteps,) predicted countdown or pre-ictal probability
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Normalize features for visualization
        features_norm = (features - features.min(axis=0)) / (features.max(axis=0) - features.min(axis=0) + 1e-6)
        
        # Create heatmap
        im = ax.imshow(features_norm.T, aspect='auto', cmap='viridis', interpolation='nearest')
        
        ax.set_ylabel('Features', fontsize=11)
        ax.set_xlabel('Time step', fontsize=11)
        ax.set_title('Normalized Feature Values Over Time', fontsize=12, fontweight='bold')
        ax.set_yticks(range(len(feature_names)))
        ax.set_yticklabels(feature_names, fontsize=9)
        
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Normalized Value', fontsize=10)
        
        plt.tight_layout()
        return fig
    
    def plot_error_distribution(self,
                               true_values: np.ndarray,
                               pred_values: np.ndarray,
                               metric_name: str = "Countdown",
                               figsize: Tuple[int, int] = (12, 5)) -> plt.Figure:
        """Plot prediction error distribution.
        
        Args:
            true_values: True values
            pred_values: Predicted values
            metric_name: Name of metric being plotted
            figsize: Figure size
        
        Returns:
            matplotlib Figure object
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Calculate errors
        errors = pred_values - true_values
        
        # Error histogram
        axes[0].hist(errors, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero error')
        axes[0].axvline(x=np.mean(errors), color='green', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.2f}')
        axes[0].set_xlabel(f'{metric_name} Error (predicted - true)', fontsize=11)
        axes[0].set_ylabel('Frequency', fontsize=11)
        axes[0].set_title('Error Distribution', fontsize=12, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Predicted vs True scatter
        axes[1].scatter(true_values, pred_values, alpha=0.5, s=30, color='steelblue')
        
        # Identity line
        min_val = min(true_values.min(), pred_values.min())
        max_val = max(true_values.max(), pred_values.max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')
        
        axes[1].set_xlabel(f'True {metric_name}', fontsize=11)
        axes[1].set_ylabel(f'Predicted {metric_name}', fontsize=11)
        axes[1].set_title('Predicted vs True', fontsize=12, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
        
        # Add statistics
        mae = np.mean(np.abs(errors))
        rmse = np.sqrt(np.mean(errors ** 2))
        stats_text = f"MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nMean error: {np.mean(errors):.3f}"
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig

    def plot_confusion_matrix(self,
                              true_preictal: np.ndarray,
                              pred_preictal_prob: np.ndarray,
                              threshold: float = 0.5,
                              split_label: str = '',
                              figsize: Tuple[int, int] = (6, 5)) -> plt.Figure:
        """Plot confusion matrix for pre-ictal classification."""
        pred_binary = (pred_preictal_prob >= threshold).astype(int)
        cm = confusion_matrix(true_preictal.astype(int), pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        total = int(cm.sum())
        n_preictal = int(tp + fn)
        n_interictal = int(tn + fp)

        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(cm, cmap='Blues')

        # Cell labels with row-wise percentages (relative to GT class samples)
        interictal_total = int(tn + fp)
        preictal_total = int(fn + tp)
        tn_pct = 100.0 * tn / max(interictal_total, 1)
        fp_pct = 100.0 * fp / max(interictal_total, 1)
        fn_pct = 100.0 * fn / max(preictal_total, 1)
        tp_pct = 100.0 * tp / max(preictal_total, 1)

        labels_mat = [
            [f'TN\n{tn}\n({tn_pct:.1f}%)', f'FP\n{fp}\n({fp_pct:.1f}%)'],
            [f'FN\n{fn}\n({fn_pct:.1f}%)', f'TP\n{tp}\n({tp_pct:.1f}%)'],
        ]
        for row_idx in range(2):
            for col_idx in range(2):
                ax.text(col_idx, row_idx, labels_mat[row_idx][col_idx],
                        ha='center', va='center', color='black', fontsize=10)

        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Interictal', 'Preictal'])
        ax.set_yticklabels(['Interictal', 'Preictal'])
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Ground Truth')
        prefix = f'{split_label} — ' if split_label else ''
        ax.set_title(
            f'{prefix}Confusion Matrix @ {threshold:.2f}\n'
            f'n={total}  ({n_preictal} preictal / {n_interictal} interictal)\n'
            f'Percentages are row-normalized by GT class',
            fontweight='bold',
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        return fig

    def plot_gt_vs_inference_panel(self,
                                   ecg_signal: np.ndarray,
                                   hr_series: np.ndarray,
                                   hrv_series: np.ndarray,
                                   true_countdown: np.ndarray,
                                   pred_countdown: np.ndarray,
                                   pred_preictal_prob: np.ndarray,
                                   ppg_signal: Optional[np.ndarray] = None,
                                   eda_signal: Optional[np.ndarray] = None,
                                   eeg_signal: Optional[np.ndarray] = None,
                                   emg_signal: Optional[np.ndarray] = None,
                                   mov_signal: Optional[np.ndarray] = None,
                                   adxl_series: Optional[np.ndarray] = None,
                                   pred_preictal_smooth: Optional[np.ndarray] = None,
                                   train_cm_true_preictal: Optional[np.ndarray] = None,
                                   train_cm_pred_preictal_prob: Optional[np.ndarray] = None,
                                   cm_true_countdown: Optional[np.ndarray] = None,
                                   cm_pred_preictal_prob: Optional[np.ndarray] = None,
                                   sampling_rate: float = 1.0,
                                   detection_threshold: float = 0.5,
                                   time_axis_minutes: Optional[np.ndarray] = None,
                                   title_prefix: str = '',
                                   footer_text: str = '',
                                   signal_label: str = 'ECG proxy',
                                   eeg_signal_label: str = 'EEG proxy',
                                   pred_onset_prob: Optional[np.ndarray] = None,
                                   pred_preictal_only_prob: Optional[np.ndarray] = None,
                                   gt_onset: Optional[np.ndarray] = None,
                                   gt_preictal_only: Optional[np.ndarray] = None,
                                   visualization_mode: str = 'countdown',
                                   figsize: Tuple[int, int] = (12, 9),
                                   token_roll: Optional[np.ndarray] = None,
                                   token_window: Optional[int] = None,
                                   token_vmax: float = 1.0,
                                   token_labels: Optional[List[str]] = None,
                                   context_energy: Optional[np.ndarray] = None,
                                   local_accuracy: Optional[np.ndarray] = None,
                                   gt_event_mask: Optional[np.ndarray] = None,
                                   data_source: str = 'bids') -> plt.Figure:
        """Create panel with multimodal traces, ADXL+HRV, optional EEG subplot, and inference.

        Layout (left column):
          Row 0: PPG + EDA overlay
          Row 1: ADXL (motion) + HRV dual axis
          Row 2: EEG (only when eeg_signal is provided)
          Row -2: Smoothed alarm + countdown references
          Row -1: Inference heatmap
        """
        n_samples = len(true_countdown)
        if time_axis_minutes is not None:
            x_minutes = np.asarray(time_axis_minutes, dtype=np.float64)
        else:
            x_minutes = np.arange(n_samples) / sampling_rate / 60.0
        gt_preictal = (true_countdown >= 0).astype(int)
        gt_onset = np.asarray(gt_onset, dtype=np.int32) if gt_onset is not None and len(gt_onset) == n_samples else np.zeros(n_samples, dtype=np.int32)
        gt_preictal_only = np.asarray(gt_preictal_only, dtype=np.int32) if gt_preictal_only is not None and len(gt_preictal_only) == n_samples else np.maximum(gt_preictal - gt_onset, 0)
        if gt_event_mask is not None and len(gt_event_mask) == n_samples:
            gt_events = (np.asarray(gt_event_mask, dtype=np.int32) > 0).astype(np.int32)
        else:
            gt_events = np.zeros(n_samples, dtype=np.int32)
            if n_samples > 0:
                gt_events[0] = int(gt_preictal[0] > 0)
            if n_samples > 1:
                gt_events[1:] = ((gt_preictal[1:] > 0) & (gt_preictal[:-1] <= 0)).astype(np.int32)

        local_acc = (
            np.clip(np.asarray(local_accuracy, dtype=np.float32), 0.0, 1.0)
            if local_accuracy is not None and len(local_accuracy) == n_samples
            else None
        )
        pred_onset_prob = np.asarray(pred_onset_prob, dtype=np.float32) if pred_onset_prob is not None and len(pred_onset_prob) == n_samples else None
        pred_preictal_only_prob = np.asarray(pred_preictal_only_prob, dtype=np.float32) if pred_preictal_only_prob is not None and len(pred_preictal_only_prob) == n_samples else None
        state_mode = str(visualization_mode).lower() == 'state' and pred_onset_prob is not None and pred_preictal_only_prob is not None
        pred_binary = (pred_preictal_prob >= detection_threshold).astype(int)
        context_energy = np.asarray(context_energy, dtype=np.float32) if context_energy is not None and len(context_energy) == n_samples else None

        has_eeg = eeg_signal is not None and len(eeg_signal) == n_samples
        add_token_row = token_roll is not None or context_energy is not None
        n_left_rows = (5 if has_eeg else 4) + (2 if add_token_row else 0)
        fig_h = figsize[1] + (2 if has_eeg else 0) + (4 if add_token_row else 0)

        fig = plt.figure(figsize=(figsize[0], fig_h), constrained_layout=True)
        panel_title = 'GT vs Inference Panel (ECG/BPM | EMG/MOV/HRV | EEG | Heads | Heatmap)'
        if title_prefix:
            panel_title = f'{title_prefix} | {panel_title}'
        fig.suptitle(panel_title, fontsize=11, fontweight='bold')
        # Removed right-column confusion matrix; using single column for better heatmap visibility
        grid = GridSpec(n_left_rows, 1, figure=fig, width_ratios=[1.0], hspace=0.18, wspace=0.15)

        # Row indices — shift alarm/heatmap down by 1 when EEG subplot is present
        row_adxl   = 1
        row_eeg    = 2 if has_eeg else None
        row_alarm  = 3 if has_eeg else 2
        row_heat   = 4 if has_eeg else 3
        row_token_ctx = n_left_rows - 2 if add_token_row else None
        row_token  = n_left_rows - 1 if add_token_row else None

        # ── Left-1: PPG + EDA overlay ──────────────────────────────────────────
        ax_signal = fig.add_subplot(grid[0, 0])

        def _normalize_signal(sig: np.ndarray) -> np.ndarray:
            sig = np.asarray(sig, dtype=np.float32)
            s_min, s_max = float(np.nanmin(sig)), float(np.nanmax(sig))
            if np.isnan(s_min) or np.isnan(s_max) or s_max <= s_min:
                return np.zeros_like(sig, dtype=np.float32)
            return (sig - s_min) / (s_max - s_min + 1e-8)

        def _contrast_scale(sig: np.ndarray) -> np.ndarray:
            arr = np.asarray(sig, dtype=np.float32)
            if arr.size == 0:
                return arr
            lo = float(np.nanpercentile(arr, 5.0))
            hi = float(np.nanpercentile(arr, 95.0))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
                lo = float(np.nanmin(arr))
                hi = float(np.nanmax(arr))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-8:
                return np.zeros_like(arr, dtype=np.float32)
            return np.clip((arr - lo) / (hi - lo), 0.0, 1.0)

        # Use explicit data_source from caller; avoid call-stack heuristics that can
        # misclassify runs and apply the wrong y-axis semantics.
        is_wearable = str(data_source).lower() == 'wearable'

        def _has_valid_signal(signal: Optional[np.ndarray]) -> bool:
            if signal is None:
                return False
            if len(signal) != n_samples:
                return False
            arr = np.asarray(signal, dtype=np.float32)
            if not np.any(np.isfinite(arr)):
                return False
            if np.allclose(arr, 0.0):
                return False
            return True

        def _place_panel_legend(ax: plt.Axes,
                                handles=None,
                                labels=None,
                                fontsize: int = 8) -> None:
            if handles is None or labels is None:
                handles, labels = ax.get_legend_handles_labels()
            if not handles:
                return
            ncol = min(3, max(1, len(labels)))
            ax.legend(
                handles,
                labels,
                loc='lower left',
                bbox_to_anchor=(0.0, 1.02, 1.0, 0.2),
                mode='expand',
                ncol=ncol,
                borderaxespad=0.0,
                fontsize=fontsize,
                framealpha=0.9,
            )

        def _compute_token_context_summary(token_roll_arr: np.ndarray) -> Dict[str, np.ndarray]:
            tok = np.asarray(token_roll_arr, dtype=np.float32)
            if tok.ndim != 2 or tok.shape[0] == 0 or tok.shape[1] == 0:
                empty = np.zeros(0, dtype=np.float32)
                return {
                    'dominant_strength': empty,
                    'dominant_share': empty,
                    'mean_strength': empty,
                    'certainty': empty,
                    'entropy': empty,
                    'transition_density': empty,
                    'dominant_index': np.zeros(0, dtype=np.int32),
                }

            clipped = np.clip(tok, 0.0, None)
            row_sum = np.sum(clipped, axis=1, keepdims=True)
            norm = np.divide(clipped, np.maximum(row_sum, 1e-6))
            dominant_index = np.argmax(clipped, axis=1).astype(np.int32)
            dominant_strength = np.max(clipped, axis=1).astype(np.float32)
            dominant_share = np.max(norm, axis=1).astype(np.float32)
            mean_strength = np.mean(clipped, axis=1).astype(np.float32)

            entropy = -np.sum(norm * np.log(norm + 1e-12), axis=1)
            entropy = (entropy / np.log(max(2, tok.shape[1]))).astype(np.float32)
            certainty = np.clip(1.0 - entropy, 0.0, 1.0).astype(np.float32)

            switch_events = np.zeros(tok.shape[0], dtype=np.float32)
            if tok.shape[0] > 1:
                switch_events[1:] = (dominant_index[1:] != dominant_index[:-1]).astype(np.float32)
            window = max(3, min(25, tok.shape[0] // 20 if tok.shape[0] >= 20 else 3))
            kernel = np.ones(window, dtype=np.float32) / float(window)
            transition_density = np.convolve(switch_events, kernel, mode='same').astype(np.float32)

            return {
                'dominant_strength': dominant_strength,
                'dominant_share': dominant_share,
                'mean_strength': mean_strength,
                'certainty': certainty,
                'entropy': entropy,
                'transition_density': transition_density,
                'dominant_index': dominant_index,
            }

        def _rolling_mean(series: np.ndarray, window_samples: int) -> np.ndarray:
            arr = np.asarray(series, dtype=np.float32)
            if arr.size == 0:
                return arr
            w = int(max(1, window_samples))
            if w <= 1:
                return arr.copy()
            kernel = np.ones(w, dtype=np.float32) / float(w)
            return np.convolve(arr, kernel, mode='same').astype(np.float32)

        has_eda = _has_valid_signal(eda_signal)
        has_hr = hr_series is not None and len(hr_series) == n_samples and np.any(np.isfinite(hr_series))

        if is_wearable and has_hr and has_eda:
            # Plot wearable HR only when enough samples are physiologically plausible.
            hr_arr = np.asarray(hr_series, dtype=np.float32)
            finite = np.isfinite(hr_arr)
            plausible = finite & (hr_arr >= 20.0) & (hr_arr <= 240.0)
            plausible_ratio = float(np.mean(plausible)) if hr_arr.size > 0 else 0.0
            if plausible_ratio >= 0.2:
                hr_plot = np.where(plausible, hr_arr, np.nan)
            else:
                logger.warning('Skipping wearable HR overlay: insufficient plausible BPM samples in panel (ratio=%.2f)', plausible_ratio)
                hr_plot = np.full_like(hr_arr, np.nan, dtype=np.float32)

            ax_signal.plot(x_minutes, hr_plot, color='black',
                           linewidth=1.2, alpha=0.85, label='HR (bpm)')
            ax_signal.set_ylabel('HR (bpm)', fontsize=9, color='black')
            ax_signal.tick_params(axis='y', labelcolor='black')
            ax_signal2 = ax_signal.twinx()
            ax_signal2.plot(x_minutes, eda_signal, color='seagreen',
                            linewidth=1.0, alpha=0.75, label='EDA (raw)')
            ax_signal2.set_ylabel('EDA (raw)', fontsize=9, color='seagreen')
            ax_signal2.tick_params(axis='y', labelcolor='seagreen')
            # Add legends for both axes
            lines1, labels1 = ax_signal.get_legend_handles_labels()
            lines2, labels2 = ax_signal2.get_legend_handles_labels()
            _place_panel_legend(ax_signal, lines1 + lines2, labels1 + labels2, fontsize=8)
            if np.any(np.isfinite(hr_plot)):
                p1 = float(np.nanpercentile(hr_plot, 1.0))
                p99 = float(np.nanpercentile(hr_plot, 99.0))
                y_lo = max(20.0, p1 - 5.0)
                y_hi = min(220.0, max(y_lo + 10.0, p99 + 5.0))
                ax_signal.set_ylim(y_lo, y_hi)
            ax_signal2.set_ylim(np.nanmin(eda_signal), np.nanmax(eda_signal))
        else:
            # Non-wearable (BIDS): always prioritize raw ECG amplitude as the
            # primary trace and overlay BPM on a secondary axis to avoid
            # ambiguity between raw signal and processed heart-rate trends.
            primary_signal = np.asarray(ecg_signal, dtype=np.float32)
            finite_primary = np.isfinite(primary_signal)
            if np.any(finite_primary):
                centered = primary_signal - float(np.nanmedian(primary_signal[finite_primary]))
                lo = float(np.nanpercentile(centered[finite_primary], 1.0))
                hi = float(np.nanpercentile(centered[finite_primary], 99.0))
                if not np.isfinite(lo) or not np.isfinite(hi) or (hi - lo) < 1e-6:
                    spread = float(np.nanstd(centered[finite_primary]))
                    spread = max(spread, 1e-3)
                    lo = -3.0 * spread
                    hi = 3.0 * spread
                ecg_display = np.clip(centered, lo, hi)
                y_pad = max((hi - lo) * 0.05, 1e-3)
            else:
                ecg_display = np.zeros_like(primary_signal, dtype=np.float32)
                lo, hi, y_pad = -1.0, 1.0, 0.1

            ax_signal.plot(x_minutes, ecg_display, color='black',
                           linewidth=1.2, alpha=0.9, label='ECG centered amplitude')
            ax_signal.set_ylim(lo - y_pad, hi + y_pad)
            ax_signal.set_ylabel('ECG centered amplitude', fontsize=9, color='black')
            ax_signal.tick_params(axis='y', labelcolor='black')

            bpm_axis = None
            if hr_series is not None and len(hr_series) == len(primary_signal):
                hr_arr = np.asarray(hr_series, dtype=np.float32)
                finite = np.isfinite(hr_arr)
                plausible = finite & (hr_arr >= 20.0) & (hr_arr <= 240.0)
                plausible_ratio = float(np.mean(plausible)) if hr_arr.size > 0 else 0.0

                if plausible_ratio >= 0.2:
                    hr_plot = np.where(plausible, hr_arr, np.nan)
                    bpm_axis = ax_signal.twinx()
                    bpm_axis.plot(
                        x_minutes,
                        hr_plot,
                        color='dodgerblue',
                        linewidth=1.0,
                        alpha=0.85,
                        label='Estimated BPM',
                    )
                    p1 = float(np.nanpercentile(hr_plot, 1.0))
                    p99 = float(np.nanpercentile(hr_plot, 99.0))
                    y_lo = max(20.0, p1 - 5.0)
                    y_hi = min(220.0, max(y_lo + 10.0, p99 + 5.0))
                    bpm_axis.set_ylim(y_lo, y_hi)
                    bpm_axis.set_ylabel('Heart rate (BPM)', fontsize=9, color='dodgerblue')
                    bpm_axis.tick_params(axis='y', labelcolor='dodgerblue')
                elif str(data_source).lower() == 'bids' and np.any(finite):
                    hr_plot = np.where(finite, hr_arr, np.nan)
                    bpm_axis = ax_signal.twinx()
                    bpm_axis.plot(
                        x_minutes,
                        hr_plot,
                        color='dodgerblue',
                        linewidth=1.0,
                        alpha=0.85,
                        label='Estimated HR proxy',
                    )
                    y_lo = float(np.nanpercentile(hr_plot[np.isfinite(hr_plot)], 1.0)) if np.any(np.isfinite(hr_plot)) else 0.0
                    y_hi = float(np.nanpercentile(hr_plot[np.isfinite(hr_plot)], 99.0)) if np.any(np.isfinite(hr_plot)) else 1.0
                    if y_hi <= y_lo:
                        y_lo -= 1.0
                        y_hi += 1.0
                    bpm_axis.set_ylim(y_lo, y_hi)
                    bpm_axis.set_ylabel('Estimated HR proxy', fontsize=9, color='dodgerblue')
                    bpm_axis.tick_params(axis='y', labelcolor='dodgerblue')
                    logger.info('Showing BIDS HR proxy overlay with ratio=%.2f', plausible_ratio)
                else:
                    logger.warning('Skipping BPM overlay: insufficient plausible BPM samples in panel (ratio=%.2f)', plausible_ratio)

            if has_eda:
                ax_signal.plot(x_minutes, eda_signal, color='seagreen',
                               linewidth=1.0, alpha=0.65, label='EDA proxy (raw)')

            lines1, labels1 = ax_signal.get_legend_handles_labels()
            if bpm_axis is not None:
                lines2, labels2 = bpm_axis.get_legend_handles_labels()
                _place_panel_legend(ax_signal, lines1 + lines2, labels1 + labels2, fontsize=8)
            else:
                _place_panel_legend(ax_signal, fontsize=8)
        # Fill preictal/predicted regions (always on main axis)
        ax_signal.fill_between(x_minutes, ax_signal.get_ylim()[0], ax_signal.get_ylim()[1], where=gt_preictal > 0,
                               color='green', alpha=0.12, label='GT preictal')
        ax_signal.fill_between(x_minutes, ax_signal.get_ylim()[0], ax_signal.get_ylim()[1], where=pred_binary > 0,
                               color='orange', alpha=0.10, label='Predicted preictal')
        ax_signal.set_title('HR / EDA with GT + predicted preictal regions' if is_wearable else 'ECG raw amplitude + BPM with GT + predicted preictal regions', fontweight='bold', fontsize=10)
        ax_signal.grid(True, alpha=0.25)

        # ── Left-2: ADXL (motion) + HRV dual axis ──────────────────────────────
        ax_adxl = fig.add_subplot(grid[row_adxl, 0], sharex=ax_signal)

        has_hrv = hrv_series is not None and len(hrv_series) == n_samples and np.any(np.isfinite(hrv_series))
        hrv_clean = np.asarray(hrv_series, dtype=np.float32) if has_hrv else np.zeros(n_samples, dtype=np.float32)

        has_emg = emg_signal is not None and len(emg_signal) == n_samples
        has_mov = mov_signal is not None and len(mov_signal) == n_samples
        if has_emg:
            emg_clean = np.asarray(emg_signal, dtype=np.float32)
            ax_adxl.plot(
                x_minutes,
                emg_clean,
                color='darkorange',
                linewidth=1.25,
                alpha=0.9,
                label='EMG envelope (raw)',
            )
        if has_mov:
            mov_clean = np.asarray(mov_signal, dtype=np.float32)
            ax_adxl.plot(
                x_minutes,
                mov_clean,
                color='sienna',
                linewidth=1.1,
                alpha=0.75,
                linestyle='--',
                label='MOV sensor (raw)',
            )
        if (not has_emg and not has_mov) and adxl_series is not None and len(adxl_series) == n_samples:
            adxl_clean = np.asarray(adxl_series, dtype=np.float32)
            ax_adxl.plot(x_minutes, adxl_clean, color='darkorange',
                         linewidth=1.3, alpha=0.9, label='Motion proxy (raw)')
        ax_adxl.set_ylabel('EMG/MOV', color='darkorange', fontsize=9)
        ax_adxl.tick_params(axis='y', labelcolor='darkorange')

        ax_adxl.grid(True, alpha=0.25)

        ax_hrv2 = ax_adxl.twinx()
        hrv_label = 'HRV (ms)' if str(data_source).lower() == 'bids' else 'HRV (native)'
        if has_hrv:
            ax_hrv2.plot(x_minutes, hrv_clean, color='teal', linewidth=1.3, alpha=0.85, label=hrv_label)
        ax_hrv2.set_ylabel(hrv_label, color='teal', fontsize=9)
        ax_hrv2.tick_params(axis='y', labelcolor='teal')
        ax_adxl.set_title('EMG/MOV + HRV (dual axis)', fontweight='bold', fontsize=10)

        adxl_lines, adxl_lbls = ax_adxl.get_legend_handles_labels()
        hrv_lines, hrv_lbls = ax_hrv2.get_legend_handles_labels()
        _place_panel_legend(ax_adxl, adxl_lines + hrv_lines, adxl_lbls + hrv_lbls, fontsize=8)

        # ── Left-EEG (optional): dedicated EEG subplot ─────────────────────────
        ax_eeg_plot = None
        if has_eeg:
            ax_eeg_plot = fig.add_subplot(grid[row_eeg, 0], sharex=ax_signal)
            eeg_raw = np.asarray(eeg_signal, dtype=np.float32)
            ax_eeg_plot.plot(x_minutes, eeg_raw, color='purple', linewidth=1.0, alpha=0.85,
                             label=f'{eeg_signal_label} (raw)')
            y0, y1 = float(np.nanmin(eeg_raw)), float(np.nanmax(eeg_raw))
            if not np.isfinite(y0) or not np.isfinite(y1) or y1 <= y0:
                y0, y1 = -1.0, 1.0
            ax_eeg_plot.fill_between(x_minutes, y0, y1, where=gt_preictal > 0,
                                     color='green', alpha=0.08)
            ax_eeg_plot.set_title(f'{eeg_signal_label} (raw)', fontweight='bold', fontsize=10)
            ax_eeg_plot.set_ylabel('EEG amplitude', color='purple', fontsize=9)
            ax_eeg_plot.tick_params(axis='y', labelcolor='purple')
            ax_eeg_plot.set_ylim(y0, y1)
            ax_eeg_plot.grid(True, alpha=0.25)
            _place_panel_legend(ax_eeg_plot, fontsize=8)

        # ── Left-alarm: smoothed alarm + countdown references ───────────────────
        ax_smooth = fig.add_subplot(grid[row_alarm, 0], sharex=ax_signal)
        has_smooth = pred_preictal_smooth is not None and len(pred_preictal_smooth) == len(pred_preictal_prob)
        if has_smooth:
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_preictal_prob, 0.0, 1.0),
                color='orange',
                linewidth=1.8,
                alpha=0.75,
                linestyle='--',
                label='Alarm prob (raw)',
            )
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_preictal_smooth, 0.0, 1.0),
                color='goldenrod',
                linewidth=2.0,
                alpha=0.95,
                label='Alarm prob (smoothed)',
            )
        else:
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_preictal_prob, 0.0, 1.0),
                color='peru',
                linewidth=1.8,
                alpha=0.95,
                label='Alarm prob',
            )
        ax_smooth.axhline(y=detection_threshold, color='red', linewidth=1.0,
                          linestyle='--', alpha=0.7, label=f'Threshold={detection_threshold:.2f}')
        ax_smooth.fill_between(x_minutes, 0.0, 1.0, where=gt_preictal > 0,
                               color='green', alpha=0.08, label='GT preictal window')

        if state_mode:
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_preictal_only_prob, 0.0, 1.0),
                color='royalblue',
                linewidth=1.5,
                alpha=0.90,
                linestyle='-',
                label='Preictal prob',
            )
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_onset_prob, 0.0, 1.0),
                color='crimson',
                linewidth=1.6,
                alpha=0.90,
                linestyle=':',
                label='Onset prob',
            )
            ax_smooth.fill_between(x_minutes, 0.0, 1.0, where=gt_preictal_only > 0,
                                   color='cornflowerblue', alpha=0.10, label='GT preictal-only')
            ax_smooth.fill_between(x_minutes, 0.0, 1.0, where=gt_onset > 0,
                                   color='crimson', alpha=0.10, label='GT onset')
            ax_smooth.set_title('State probabilities', fontweight='bold', fontsize=10)
        else:
            tau_min = max(0.5, float(np.nanpercentile(np.clip(true_countdown[true_countdown >= 0], 0.0, None), 50)) if np.any(true_countdown >= 0) else 2.0)
            imminent_prob = np.exp(-np.clip(pred_countdown.astype(np.float32), 0.0, None) / tau_min)
            imminent_prob = np.clip(imminent_prob, 0.0, 1.0)
            ax_smooth.plot(
                x_minutes,
                imminent_prob,
                color='crimson',
                linewidth=1.5,
                alpha=0.9,
                linestyle=':',
                label='Imminent risk prob (countdown head)',
            )

        event_indices = np.where(gt_events > 0)[0]
        if event_indices.size > 3:
            event_indices = event_indices[-1:]
        for idx_ev, ev in enumerate(event_indices.tolist()):
            x_ev = float(x_minutes[ev])
            label = 'Past event' if idx_ev == 0 else None
            ax_smooth.axvline(x=x_ev, color='magenta', linestyle='-', linewidth=1.1, alpha=0.7, label=label)

        if np.any(gt_preictal > 0):
            preictal_indices = np.where(gt_preictal > 0)[0]
            onset_idx = preictal_indices[np.argmin(np.abs(true_countdown[preictal_indices]))]
            onset_x = x_minutes[onset_idx]

            # Draw GT seizure onset indicator in multiple panels so it is
            # not obscured by legends in any single subplot.
            ax_smooth.axvline(x=onset_x, color='purple', linestyle='--', linewidth=1.6,
                              label='GT seizure onset')
            ax_signal.axvline(x=onset_x, color='purple', linestyle='--', linewidth=1.0, alpha=0.9)
            ax_adxl.axvline(x=onset_x, color='purple', linestyle='--', linewidth=1.0, alpha=0.9)
            if ax_eeg_plot is not None:
                ax_eeg_plot.axvline(x=onset_x, color='purple', linestyle='--', linewidth=1.0, alpha=0.9)

        ax_smooth.set_ylim(0.0, 1.0)
        ax_smooth.set_ylabel('Alarm prob')
        if not state_mode:
            ax_smooth.set_title('Head probabilities', fontweight='bold', fontsize=10)
        ax_smooth.grid(True, alpha=0.25)
        legend_lines, legend_labels = ax_smooth.get_legend_handles_labels()
        _place_panel_legend(ax_smooth, legend_lines, legend_labels, fontsize=7)

        # ── Left-heat: inference heatmap ────────────────────────────────────────
        ax_heat = fig.add_subplot(grid[row_heat, 0], sharex=ax_signal)

        if state_mode:
            heatmap_rows = [
                gt_preictal.astype(np.float32),
                gt_preictal_only.astype(np.float32),
                gt_onset.astype(np.float32),
                np.clip(pred_preictal_prob, 0.0, 1.0),
                np.clip(pred_preictal_only_prob, 0.0, 1.0),
                np.clip(pred_onset_prob, 0.0, 1.0),
            ]
            ytick_labels = ['GT Active', 'GT Preictal', 'GT Onset', 'Alert Prob', 'Preictal Prob', 'Onset Prob']
            ytick_pos = [0.5 + idx for idx in range(len(heatmap_rows))]
            n_rows = len(heatmap_rows)
        else:
            alert_raw = np.clip(pred_preictal_prob, 0.0, 1.0)
            alert_smooth = np.clip(pred_preictal_smooth, 0.0, 1.0) if has_smooth else np.clip(pred_preictal_prob, 0.0, 1.0)
            no_risk_prob = np.clip(1.0 - alert_raw, 0.0, 1.0)
            tau_min = max(0.5, float(np.nanpercentile(np.clip(true_countdown[true_countdown >= 0], 0.0, None), 50)) if np.any(true_countdown >= 0) else 2.0)
            imminent_prob = np.exp(-np.clip(pred_countdown.astype(np.float32), 0.0, None) / tau_min)
            imminent_prob = np.clip(imminent_prob, 0.0, 1.0)
            heatmap_rows = [
                no_risk_prob,
                alert_raw,
                alert_smooth,
                imminent_prob,
            ]
            ytick_labels = ['No-risk Prob', 'Alarm Prob', 'Smoothed Alarm', 'Imminent Prob']
            ytick_pos = [0.5 + idx for idx in range(len(heatmap_rows))]
            n_rows = len(heatmap_rows)

        heatmap = np.vstack(heatmap_rows)
        im = ax_heat.imshow(
            heatmap,
            aspect='auto',
            cmap='viridis',
            interpolation='nearest',
            extent=[x_minutes[0], x_minutes[-1] if len(x_minutes) > 1 else 1.0, 0, n_rows],
            origin='lower',
            vmin=0,
            vmax=1,
        )
        ax_heat.set_yticks(ytick_pos)
        ax_heat.set_yticklabels(ytick_labels, fontsize=8)

        ax_heat.set_xlabel('Time (minutes)')
        if state_mode:
            ax_heat.set_title('State-mode inference heatmap', fontweight='bold', fontsize=10)
        else:
            ax_heat.set_title('Inference heatmap (probability heads)', fontweight='bold', fontsize=10)
        fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.01)

        # ── Left-token context + waterfall ────────────────────────────────────
        if add_token_row:
            token_summary = _compute_token_context_summary(
                token_roll if token_roll is not None else np.zeros((n_samples, 1), dtype=np.float32)
            )
            ax_token_ctx = fig.add_subplot(grid[row_token_ctx, 0], sharex=ax_signal)
            if context_energy is not None:
                # Instantaneous latent context energy is not currently reliable as a
                # standalone waveform trace; keep the multi-timescale cascades instead.
                ctx = np.asarray(context_energy, dtype=np.float32)
                if np.any(np.isfinite(ctx)):
                    pass
            ax_token_ctx.plot(
                x_minutes,
                token_summary['dominant_share'],
                color='darkmagenta',
                linewidth=1.6,
                alpha=0.9,
                label='Dominant token share',
            )
            ax_token_ctx.plot(
                x_minutes,
                token_summary['mean_strength'],
                color='slateblue',
                linewidth=1.2,
                alpha=0.85,
                linestyle='--',
                label='Mean token strength',
            )
            ax_token_ctx.set_ylabel('Context / token strength', color='darkmagenta', fontsize=9)
            ax_token_ctx.tick_params(axis='y', labelcolor='darkmagenta')
            ax_token_ctx.set_ylim(0.0, max(1.0, float(np.nanmax(token_summary['mean_strength'])) * 1.05 if token_summary['mean_strength'].size else 1.0))
            ax_token_ctx.grid(True, alpha=0.25)

            ax_token_ctx2 = ax_token_ctx.twinx()
            ax_token_ctx2.plot(
                x_minutes,
                token_summary['certainty'],
                color='black',
                linewidth=1.2,
                alpha=0.8,
                label='Token certainty (1 - entropy)',
            )
            ax_token_ctx2.plot(
                x_minutes,
                token_summary['transition_density'],
                color='tomato',
                linewidth=1.1,
                alpha=0.75,
                linestyle=':',
                label='Token transition density',
            )

            # Multi-timescale context cascade (minutes → hours → day proxy).
            if context_energy is not None and np.any(np.isfinite(context_energy)) and len(x_minutes) >= 2:
                ctx = np.asarray(context_energy, dtype=np.float32)
                valid_dx = np.diff(np.asarray(x_minutes, dtype=np.float64))
                valid_dx = valid_dx[np.isfinite(valid_dx) & (valid_dx > 0)]
                dt_min = float(np.median(valid_dx)) if valid_dx.size > 0 else 1.0 / 60.0
                dt_min = max(dt_min, 1e-5)

                ctx_lo = float(np.nanpercentile(ctx, 5.0))
                ctx_hi = float(np.nanpercentile(ctx, 95.0))
                if ctx_hi <= ctx_lo:
                    ctx_hi = ctx_lo + 1.0
                ctx_norm = np.clip((ctx - ctx_lo) / (ctx_hi - ctx_lo), 0.0, 1.0)

                cascade_specs = [
                    ('Context 1m', 1.0, '#2ca25f', '-'),
                    ('Context 15m', 15.0, '#238b45', '--'),
                    ('Context 3h', 180.0, '#006d2c', '-.'),
                    ('Context 24h', 1440.0, '#00441b', ':'),
                ]
                for label, span_min, color, style in cascade_specs:
                    window_samples = int(round(span_min / dt_min))
                    if window_samples <= 1:
                        cascade = ctx_norm
                    else:
                        cascade = _rolling_mean(ctx_norm, min(window_samples, len(ctx_norm)))
                    ax_token_ctx2.plot(
                        x_minutes,
                        cascade,
                        color=color,
                        linewidth=1.05,
                        alpha=0.70,
                        linestyle=style,
                        label=f'{label} cascade',
                    )

            ax_token_ctx2.set_ylabel('Certainty / transitions', color='black', fontsize=9)
            ax_token_ctx2.tick_params(axis='y', labelcolor='black')
            ax_token_ctx2.set_ylim(0.0, 1.0)
            ax_token_ctx.set_title('Token context summary', fontweight='bold', fontsize=10)

            ctx_lines, ctx_labels = ax_token_ctx.get_legend_handles_labels()
            ctx2_lines, ctx2_labels = ax_token_ctx2.get_legend_handles_labels()
            _place_panel_legend(ax_token_ctx, ctx_lines + ctx2_lines, ctx_labels + ctx2_labels, fontsize=7)

            if token_roll is not None:
                ax_token = fig.add_subplot(grid[row_token, 0], sharex=ax_signal)
                im_token = ax_token.imshow(
                    token_roll.T,
                    aspect='auto',
                    origin='lower',
                    interpolation='nearest',
                    cmap='magma',
                    vmin=0,
                    vmax=token_vmax,
                    extent=[x_minutes[0], x_minutes[-1] if len(x_minutes) > 1 else 1.0, 0, token_roll.shape[1]],
                )
                ax_token.set_ylabel('Token ID')
                ax_token.set_xlabel('Time (minutes)')
                title = 'Token Activation Waterfall'
                if token_window is not None:
                    title += f' (window={token_window})'
                ax_token.set_title(title, fontweight='bold', fontsize=10)
                if token_labels is not None and len(token_labels) == token_roll.shape[1]:
                    yticks = np.arange(0.5, token_roll.shape[1] + 0.5, 1.0)
                    if len(token_labels) > 16:
                        step = int(np.ceil(len(token_labels) / 16.0))
                        yticks = yticks[::step]
                        show_labels = token_labels[::step]
                    else:
                        show_labels = token_labels
                    ax_token.set_yticks(yticks)
                    ax_token.set_yticklabels(show_labels, fontsize=7)
                fig.colorbar(im_token, ax=ax_token, fraction=0.035, pad=0.01, label='Rolling occupancy')

        # Confusion matrices removed for better visualization clarity
        # (Confusion matrix analysis now available in separate metrics files)
        
        def _resolve_cm_inputs(default_true: np.ndarray,
                               default_prob: np.ndarray,
                               full_true_countdown: Optional[np.ndarray],
                               full_pred_prob: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
            if full_true_countdown is not None and full_pred_prob is not None:
                true_arr = np.asarray(full_true_countdown)
                unique_vals = np.unique(true_arr)
                if np.all(np.isin(unique_vals, [0, 1])):
                    resolved_true = true_arr.astype(int)
                else:
                    resolved_true = (true_arr >= 0).astype(int)
                resolved_prob = np.asarray(full_pred_prob, dtype=np.float32)
            else:
                resolved_true = np.asarray(default_true, dtype=int)
                resolved_prob = np.asarray(default_prob, dtype=np.float32)
            return resolved_true, resolved_prob

        def _draw_cm(ax: plt.Axes,
                     true_labels: np.ndarray,
                     pred_probs: np.ndarray,
                     split_label: str):
            cm_pred_binary = (pred_probs >= detection_threshold).astype(int)
            cm = confusion_matrix(true_labels, cm_pred_binary, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            total = int(cm.sum())
            im_local = ax.imshow(cm, cmap='Blues')
            
            # Cell labels with row-wise percentages (relative to GT class samples)
            interictal_total = int(tn + fp)
            preictal_total = int(fn + tp)
            tn_pct = 100.0 * tn / max(interictal_total, 1)
            fp_pct = 100.0 * fp / max(interictal_total, 1)
            fn_pct = 100.0 * fn / max(preictal_total, 1)
            tp_pct = 100.0 * tp / max(preictal_total, 1)
            
            text_labels = [
                [f'TN\n{tn}\n({tn_pct:.1f}%)', f'FP\n{fp}\n({fp_pct:.1f}%)'],
                [f'FN\n{fn}\n({fn_pct:.1f}%)', f'TP\n{tp}\n({tp_pct:.1f}%)']
            ]
            for row_idx in range(2):
                for col_idx in range(2):
                    ax.text(col_idx, row_idx, text_labels[row_idx][col_idx],
                            ha='center', va='center', color='black', fontsize=9)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Interictal', 'Preictal'])
            ax.set_yticklabels(['Interictal', 'Preictal'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Ground Truth')
            ax.set_title(
                f'{split_label} confusion matrix\n'
                f'th={detection_threshold:.2f}  n={total}  pos={int(true_labels.sum())}  (row-normalized %)',
                fontweight='bold',
                fontsize=10,
            )
            return im_local

        # val_true_preictal, val_pred_probs = _resolve_cm_inputs(
        #     gt_preictal,
        #     pred_preictal_prob,
        #     cm_true_countdown,
        #     cm_pred_preictal_prob,
        # )
        #
        # if train_cm_true_preictal is not None and train_cm_pred_preictal_prob is not None:
        #     right_grid = grid[:, 1].subgridspec(2, 1, hspace=0.28)
        #     train_true_preictal, train_pred_probs = _resolve_cm_inputs(
        #         gt_preictal,
        #         pred_preictal_prob,
        #         train_cm_true_preictal.astype(int),
        #         train_cm_pred_preictal_prob,
        #     )
        #     ax_cm_train = fig.add_subplot(right_grid[0, 0])
        #     _draw_cm(ax_cm_train, train_true_preictal, train_pred_probs, 'Train')
        #
        #     ax_cm_val = fig.add_subplot(right_grid[1, 0])
        #     im_cm_val = _draw_cm(ax_cm_val, val_true_preictal, val_pred_probs, 'Validation')
        #     fig.colorbar(im_cm_val, ax=[ax_cm_train, ax_cm_val], fraction=0.035, pad=0.02)
        # else:
        #     ax_cm = fig.add_subplot(grid[:, 1])
        #     im_cm = _draw_cm(ax_cm, val_true_preictal, val_pred_probs, 'Confusion')
        #     fig.colorbar(im_cm, ax=ax_cm, fraction=0.035, pad=0.02)

        # Transparent metadata: report effective sampling shown on the x-axis.
        if len(x_minutes) > 1:
            dt_seconds = float(np.nanmedian(np.diff(x_minutes)) * 60.0)
            effective_hz = 1.0 / max(dt_seconds, 1e-6)
        else:
            effective_hz = float(max(sampling_rate, 1e-6))
            dt_seconds = 1.0 / effective_hz

        sampling_meta = f"Sampling: {effective_hz:.3f} Hz ({dt_seconds:.2f} s/point)"
        if footer_text:
            footer_combined = f"{footer_text} | {sampling_meta}"
        else:
            footer_combined = sampling_meta
        fig.text(0.5, 0.01, footer_combined, ha='center', va='bottom', fontsize=9, color='dimgray')

        return fig
    
    def save_figure(self, fig: plt.Figure, name: str, step: Optional[int] = None) -> Optional[str]:
        """Save figure to disk and optionally upload to wandb.
        
        Args:
            fig: matplotlib Figure object
            name: Figure name (without extension)
            step: Optional step/epoch number for wandb indexing
        
        Returns:
            Path to saved file or None
        """
        filepath = None
        
        # Save to disk
        if self.save_dir:
            filepath = self.save_dir / f"{name}.png"
            fig.savefig(filepath, dpi=150, bbox_inches='tight')
            logger.info(f"Saved figure: {filepath}")
        
        # Upload to wandb (do not force step; trainer also logs batch metrics)
        if self.upload_to_wandb and HAS_WANDB:
            try:
                log_payload = {
                    "visualizations/epoch_panel": wandb.Image(fig),
                }
                if step is not None:
                    log_payload["visualizations/epoch"] = step
                wandb.log(log_payload)
            except Exception as e:
                logger.warning(f"Failed to upload to wandb: {str(e)}")
        
        plt.close(fig)
        return filepath
    
    def create_signal_summary(self,
                             normal_signals: Dict[str, np.ndarray],
                             seizure_signals: Dict[str, np.ndarray],
                             sampling_rates: Dict[str, float]) -> None:
        """Create comparison plots for all available signals.
        
        Args:
            normal_signals: Dict of signal_name -> signal_array
            seizure_signals: Dict of signal_name -> signal_array
            sampling_rates: Dict of signal_name -> sampling_rate
        """
        for signal_name in normal_signals.keys():
            if signal_name not in seizure_signals:
                continue
            
            fig = self.plot_signal_comparison(
                normal_signals[signal_name],
                seizure_signals[signal_name],
                sampling_rate=sampling_rates.get(signal_name, 250),
                signal_name=signal_name.upper()
            )
            
            self.save_figure(fig, f"signal_comparison_{signal_name}")


def create_training_visualization(wandb_run=None):
    """Create visualization helper for training with optional wandb integration.
    
    Args:
        wandb_run: Optional wandb run object
    
    Returns:
        SignalVisualizer instance
    """
    return SignalVisualizer(
        save_dir="visualizations",
        upload_to_wandb=wandb_run is not None
    )


if __name__ == "__main__":
    # Example usage
    import logging
    logging.basicConfig(level=logging.INFO)
    
    # Create visualizer
    viz = SignalVisualizer(save_dir="visualizations")
    
    # Generate example signals
    np.random.seed(42)
    t = np.linspace(0, 10, 2500)  # 10 seconds at 250 Hz
    
    normal_signal = 0.5 * np.sin(2 * np.pi * 1 * t) + 0.1 * np.random.randn(len(t))
    seizure_signal = 1.5 * np.sin(2 * np.pi * 2 * t) + 0.5 * np.sin(2 * np.pi * 5 * t) + 0.2 * np.random.randn(len(t))
    
    # Create comparison figure
    fig1 = viz.plot_signal_comparison(normal_signal, seizure_signal)
    viz.save_figure(fig1, "example_signal_comparison")
    
    # Create countdown prediction figure
    true_countdown = np.concatenate([np.arange(10, -1, -0.01), np.full(500, -1)])
    pred_countdown = true_countdown + 0.5 * np.random.randn(len(true_countdown))
    pred_countdown = np.clip(pred_countdown, -1, 10)
    
    fig2 = viz.plot_countdown_prediction(true_countdown, pred_countdown)
    viz.save_figure(fig2, "example_countdown_prediction")
    
    # Create error distribution figure
    fig3 = viz.plot_error_distribution(true_countdown, pred_countdown)
    viz.save_figure(fig3, "example_error_distribution")
    
    print("Example visualizations created!")
