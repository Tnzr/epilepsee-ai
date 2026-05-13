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
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
from sklearn.metrics import confusion_matrix

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


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
                                   figsize: Tuple[int, int] = (12, 9)) -> plt.Figure:
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
        pred_onset_prob = np.asarray(pred_onset_prob, dtype=np.float32) if pred_onset_prob is not None and len(pred_onset_prob) == n_samples else None
        pred_preictal_only_prob = np.asarray(pred_preictal_only_prob, dtype=np.float32) if pred_preictal_only_prob is not None and len(pred_preictal_only_prob) == n_samples else None
        state_mode = str(visualization_mode).lower() == 'state' and pred_onset_prob is not None and pred_preictal_only_prob is not None
        pred_binary = (pred_preictal_prob >= detection_threshold).astype(int)

        has_eeg = eeg_signal is not None and len(eeg_signal) == n_samples
        n_left_rows = 5 if has_eeg else 4
        fig_h = figsize[1] + (2 if has_eeg else 0)

        fig = plt.figure(figsize=(figsize[0], fig_h), constrained_layout=True)
        panel_title = 'GT vs Inference Panel (PPG/EDA | ADXL/HRV | EEG | Alarm | Heatmap)'
        if title_prefix:
            panel_title = f'{title_prefix} | {panel_title}'
        fig.suptitle(panel_title, fontsize=11, fontweight='bold')
        grid = GridSpec(n_left_rows, 2, figure=fig, width_ratios=[3.0, 1.35], hspace=0.18, wspace=0.15)

        # Row indices — shift alarm/heatmap down by 1 when EEG subplot is present
        row_adxl   = 1
        row_eeg    = 2 if has_eeg else None
        row_alarm  = 3 if has_eeg else 2
        row_heat   = 4 if has_eeg else 3

        # ── Left-1: PPG + EDA overlay ──────────────────────────────────────────
        ax_signal = fig.add_subplot(grid[0, 0])

        def _normalize_signal(sig: np.ndarray) -> np.ndarray:
            sig = np.asarray(sig, dtype=np.float32)
            s_min, s_max = float(np.nanmin(sig)), float(np.nanmax(sig))
            if np.isnan(s_min) or np.isnan(s_max) or s_max <= s_min:
                return np.zeros_like(sig, dtype=np.float32)
            return (sig - s_min) / (s_max - s_min + 1e-8)

        primary_signal = ppg_signal if ppg_signal is not None and len(ppg_signal) == len(ecg_signal) else ecg_signal
        ax_signal.plot(x_minutes, _normalize_signal(primary_signal), color='black',
                       linewidth=1.2, alpha=0.85, label=f'{signal_label} (norm)')

        if eda_signal is not None and len(eda_signal) == len(primary_signal):
            ax_signal.plot(x_minutes, _normalize_signal(eda_signal), color='seagreen',
                           linewidth=1.0, alpha=0.75, label='EDA proxy (norm)')

        y_min, y_max = 0.0, 1.0
        ax_signal.fill_between(x_minutes, y_min, y_max, where=gt_preictal > 0,
                               color='green', alpha=0.12, label='GT preictal')
        ax_signal.fill_between(x_minutes, y_min, y_max, where=pred_binary > 0,
                               color='orange', alpha=0.10, label='Predicted preictal')
        ax_signal.set_title('PPG / EDA with GT + predicted preictal regions', fontweight='bold', fontsize=10)
        ax_signal.set_ylabel('Signal (normalized)', fontsize=9)
        ax_signal.set_ylim(y_min, y_max)
        ax_signal.grid(True, alpha=0.25)
        ax_signal.legend(loc='upper right', fontsize=8)

        # ── Left-2: ADXL (motion) + HRV dual axis ──────────────────────────────
        ax_adxl = fig.add_subplot(grid[row_adxl, 0], sharex=ax_signal)
        hrv_clean = np.asarray(hrv_series, dtype=np.float32)

        if adxl_series is not None and len(adxl_series) == n_samples:
            adxl_clean = np.asarray(adxl_series, dtype=np.float32)
            adxl_range = float(np.nanmax(adxl_clean)) - float(np.nanmin(adxl_clean))
            adxl_label = 'ADXL motion (raw)' 
            ax_adxl.plot(x_minutes, adxl_clean, color='darkorange',
                         linewidth=1.3, alpha=0.9, label=adxl_label)
            ax_adxl.set_ylabel('ADXL (raw)', color='darkorange', fontsize=9)
            ax_adxl.tick_params(axis='y', labelcolor='darkorange')
        else:
            ax_adxl.set_ylabel('ADXL (no data)', fontsize=9)

        ax_adxl.grid(True, alpha=0.25)

        ax_hrv2 = ax_adxl.twinx()
        hrv_range = float(np.nanmax(hrv_clean)) - float(np.nanmin(hrv_clean))
        if hrv_range < 5.0 and hrv_range > 0:
            hrv_scaled = 10.0 + (hrv_clean - float(np.nanmin(hrv_clean))) / (hrv_range + 1e-8) * 50.0
            hrv_label = 'HRV (rescaled, ms)'
        else:
            hrv_scaled = hrv_clean
            hrv_label = 'HRV (native, ms)'
        ax_hrv2.plot(x_minutes, hrv_scaled, color='teal', linewidth=1.3, alpha=0.85, label=hrv_label)
        ax_hrv2.set_ylabel(hrv_label, color='teal', fontsize=9)
        ax_hrv2.tick_params(axis='y', labelcolor='teal')
        ax_adxl.set_title('ADXL motion + HRV (dual axis)', fontweight='bold', fontsize=10)

        adxl_lines, adxl_lbls = ax_adxl.get_legend_handles_labels()
        hrv_lines, hrv_lbls = ax_hrv2.get_legend_handles_labels()
        ax_adxl.legend(adxl_lines + hrv_lines, adxl_lbls + hrv_lbls, loc='upper right', fontsize=8)

        # ── Left-EEG (optional): dedicated EEG subplot ─────────────────────────
        ax_eeg_plot = None
        if has_eeg:
            ax_eeg_plot = fig.add_subplot(grid[row_eeg, 0], sharex=ax_signal)
            eeg_norm = _normalize_signal(eeg_signal)
            ax_eeg_plot.plot(x_minutes, eeg_norm, color='purple', linewidth=1.0, alpha=0.85,
                             label=f'{eeg_signal_label} (norm)')
            ax_eeg_plot.fill_between(x_minutes, 0.0, 1.0, where=gt_preictal > 0,
                                     color='green', alpha=0.08)
            ax_eeg_plot.set_title(eeg_signal_label, fontweight='bold', fontsize=10)
            ax_eeg_plot.set_ylabel('EEG (norm)', color='purple', fontsize=9)
            ax_eeg_plot.tick_params(axis='y', labelcolor='purple')
            ax_eeg_plot.set_ylim(0.0, 1.0)
            ax_eeg_plot.grid(True, alpha=0.25)
            ax_eeg_plot.legend(loc='upper right', fontsize=8)

        # ── Left-alarm: smoothed alarm + countdown references ───────────────────
        ax_smooth = fig.add_subplot(grid[row_alarm, 0], sharex=ax_signal)
        has_smooth = pred_preictal_smooth is not None and len(pred_preictal_smooth) == len(pred_preictal_prob)
        if has_smooth:
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_preictal_smooth, 0.0, 1.0),
                color='goldenrod',
                linewidth=2.0,
                alpha=0.95,
                label='Smoothed alarm',
            )
        else:
            ax_smooth.plot(
                x_minutes,
                np.clip(pred_preictal_prob, 0.0, 1.0),
                color='peru',
                linewidth=1.8,
                alpha=0.95,
                label='Raw alarm (smooth unavailable)',
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
            positive_gt = true_countdown[true_countdown >= 0]
            observed_gt_max = float(np.max(positive_gt)) if positive_gt.size > 0 else 0.0
            if observed_gt_max <= 1.5:
                countdown_ref_max = 1.0
            else:
                countdown_ref_max = max(30.0, observed_gt_max)

            gt_countdown_ref = np.full_like(true_countdown, 1.0, dtype=np.float32)
            gt_preictal_mask = true_countdown >= 0
            if np.any(gt_preictal_mask):
                preictal_max = 10.0
                gt_countdown_ref[gt_preictal_mask] = true_countdown[gt_preictal_mask] / preictal_max
            if countdown_ref_max > 0:
                gt_countdown_ref_norm = np.clip(gt_countdown_ref / countdown_ref_max, 0.0, 1.0)
            else:
                gt_countdown_ref_norm = np.zeros_like(gt_countdown_ref, dtype=np.float32)

            pred_scale = max(countdown_ref_max, float(np.max(np.abs(pred_countdown))) + 1e-6)
            pred_countdown_ref = np.clip(pred_countdown.astype(np.float32) / pred_scale, 0.0, 1.0)
            ax_smooth.plot(
                x_minutes,
                gt_countdown_ref_norm,
                color='forestgreen',
                linewidth=1.8,
                alpha=0.9,
                linestyle='-',
                label='GT countdown ref (continuous)',
            )
            ax_smooth.plot(
                x_minutes,
                pred_countdown_ref,
                color='crimson',
                linewidth=1.6,
                alpha=0.85,
                linestyle=':',
                label='Pred countdown ref',
            )

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
            ax_smooth.set_title('Smoothed alarm + countdown references', fontweight='bold', fontsize=10)
        ax_smooth.grid(True, alpha=0.25)
        ax_smooth.legend(loc='upper right', fontsize=7)

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
            pred_countdown_norm = np.clip(pred_countdown / pred_scale, 0.0, 1.0)
            gt_proximity = gt_countdown_ref_norm
            heatmap_rows = [
                gt_proximity,
                np.clip(pred_preictal_prob, 0.0, 1.0),
                pred_countdown_norm,
            ]
            ytick_labels = ['GT Countdown (cont)', 'Raw Prob', 'Pred Countdown']
            ytick_pos    = [0.5, 1.5, 2.5]
            n_rows = 3

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
            ax_heat.set_title('Inference heatmap (GT processed + raw + countdown)', fontweight='bold', fontsize=10)
        fig.colorbar(im, ax=ax_heat, fraction=0.035, pad=0.01)

        # Right column: train + validation confusion matrices for epoch views,
        # or a single full-set confusion matrix for final/test views.

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

        val_true_preictal, val_pred_probs = _resolve_cm_inputs(
            gt_preictal,
            pred_preictal_prob,
            cm_true_countdown,
            cm_pred_preictal_prob,
        )

        if train_cm_true_preictal is not None and train_cm_pred_preictal_prob is not None:
            right_grid = grid[:, 1].subgridspec(2, 1, hspace=0.28)
            train_true_preictal, train_pred_probs = _resolve_cm_inputs(
                gt_preictal,
                pred_preictal_prob,
                train_cm_true_preictal.astype(int),
                train_cm_pred_preictal_prob,
            )
            ax_cm_train = fig.add_subplot(right_grid[0, 0])
            _draw_cm(ax_cm_train, train_true_preictal, train_pred_probs, 'Train')

            ax_cm_val = fig.add_subplot(right_grid[1, 0])
            im_cm_val = _draw_cm(ax_cm_val, val_true_preictal, val_pred_probs, 'Validation')
            fig.colorbar(im_cm_val, ax=[ax_cm_train, ax_cm_val], fraction=0.035, pad=0.02)
        else:
            ax_cm = fig.add_subplot(grid[:, 1])
            im_cm = _draw_cm(ax_cm, val_true_preictal, val_pred_probs, 'Confusion')
            fig.colorbar(im_cm, ax=ax_cm, fraction=0.035, pad=0.02)

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
