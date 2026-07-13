#!/usr/bin/env python
"""
Wearable Device Dataset Sample Visualizer - Comprehensive Version

Visualizes ALL signal streams from a patient's wearable device recording
including PPG, ADPD, ADXL (with magnitude), EDA, Temperature, Battery, Pedometer, and SQI.
Seizure events marked with before/after context windows.

Features:
- Automatic detection and parsing of ALL signals (PPG, ADPD, ADXL, AGC, EDA, Temp, Battery, Pedometer, SQI)
- Multi-signal visualization with seizure event markers
- Configurable time windows (2-5 min before, 1-2 min after seizure)
- Signal statistics and quality metrics  
- Seizure region highlighting across all signals
- ADXL magnitude computation (total acceleration)
- Multiple output formats (PNG)

Usage:
    # Visualize with key signals only
    python scripts/visualize_wearable_sample_v2.py --patient 2

        # Include synchronized EEG EDF overlay (patient #2 attached EEG folder)
        python scripts/visualize_wearable_sample_v2.py --patient 2 \
            --eeg-dir /media/tnzr/HDD11/Datasets/WearablwDevice-Oregon/#2/EEG
    
    # Generate comprehensive all-signals plot
    python scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals
    
    # All seizures with extended window
    python scripts/visualize_wearable_sample_v2.py --patient 2 --all-seizures --all-signals --before-min 5 --after-min 2
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from datetime import timedelta

try:
    from tqdm.auto import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config import Config, DEFAULT_CONFIG
from src.data_loader import WearableDeviceDataLoader


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def progress_iter(iterable, desc: str, total: Optional[int] = None):
    """Return tqdm iterator when available; otherwise return iterable unchanged."""
    if HAS_TQDM:
        return tqdm(iterable, desc=desc, total=total, leave=False)
    return iterable


class WearableDatasetVisualizer:
    """Visualize wearable device signals with seizure context."""
    
    def __init__(self, config: Config, dataset_root: Optional[str] = None, 
                 master_db_path: Optional[str] = None,
                 eeg_edf_path: Optional[str] = None,
                 eeg_metadata_path: Optional[str] = None,
                 eeg_channel: Optional[str] = None):
        """Initialize visualizer.
        
        Args:
            config: Config object with preprocessing params
            dataset_root: Path to WearableDevice-Oregon dataset
            master_db_path: Path to master seizure annotations file
            eeg_edf_path: Optional EDF path for synchronized EEG overlay
            eeg_metadata_path: Optional metadata JSON path for EEG export context
            eeg_channel: Optional EEG channel name to plot (default: first EEG channel)
        """
        self.config = config
        self.loader = WearableDeviceDataLoader(config, dataset_root, master_db_path)
        self.recordings = self.loader.get_all_recordings()
        self.eeg_edf_path = Path(eeg_edf_path) if eeg_edf_path else None
        self.eeg_metadata_path = Path(eeg_metadata_path) if eeg_metadata_path else None
        self.eeg_channel = eeg_channel
        self._eeg_raw = None
        self._eeg_info: Optional[Dict[str, Any]] = None
        self._eeg_meta: Optional[Dict[str, Any]] = None
        logger.info(f"Loaded {len(self.recordings)} recordings from database")

        if self.eeg_edf_path is not None:
            logger.info(f"EEG EDF configured: {self.eeg_edf_path}")
            if self.eeg_metadata_path is not None:
                logger.info(f"EEG metadata configured: {self.eeg_metadata_path}")

    @staticmethod
    def _to_naive_timestamp(value: Any) -> Optional[pd.Timestamp]:
        """Convert datetime-like values to timezone-naive pandas Timestamp."""
        if value is None:
            return None
        try:
            ts = pd.Timestamp(value)
            if ts.tzinfo is not None:
                ts = ts.tz_convert(None)
            return ts
        except Exception:
            return None

    def _load_eeg_metadata_start(self) -> Optional[pd.Timestamp]:
        """Read EEG export metadata and infer recording start timestamp when available."""
        if self.eeg_metadata_path is None or not self.eeg_metadata_path.exists():
            return None
        try:
            with open(self.eeg_metadata_path, 'r', encoding='utf-8') as handle:
                metadata = json.load(handle)

            events = metadata.get('Events', [])
            start_events = [
                e for e in events
                if str(e.get('Message', '')).strip().lower() == 'start recording'
            ]
            candidate_events = start_events if start_events else events
            timestamps_us = [
                int(e.get('Timestamp'))
                for e in candidate_events
                if e.get('Timestamp') is not None
            ]
            if not timestamps_us:
                return None

            start_us = min(timestamps_us)
            return pd.Timestamp(start_us, unit='us')
        except Exception as error:
            logger.debug(f"Failed to parse EEG metadata start timestamp: {error}")
            return None

    def _ensure_eeg_metadata(self) -> Optional[Dict[str, Any]]:
        """Parse EEG metadata for recording start and seizure-like event anchors."""
        if self._eeg_meta is not None:
            return self._eeg_meta
        if self.eeg_metadata_path is None or not self.eeg_metadata_path.exists():
            return None

        try:
            with open(self.eeg_metadata_path, 'r', encoding='utf-8') as handle:
                metadata = json.load(handle)

            events = metadata.get('Events', [])
            start_us = None
            seizure_us: List[int] = []

            for event in events:
                ts = event.get('Timestamp')
                if ts is None:
                    continue
                try:
                    ts_us = int(ts)
                except Exception:
                    continue

                message = str(event.get('Message', '')).strip().lower()
                if message == 'start recording':
                    start_us = ts_us if start_us is None else min(start_us, ts_us)

                # Common seizure annotations observed in exported metadata.
                if (
                    'estart' in message
                    or message == 'sz'
                    or '@seizuredetected' in message
                    or 'seizure' in message
                ):
                    seizure_us.append(ts_us)

            if start_us is None:
                all_ts = [int(e['Timestamp']) for e in events if e.get('Timestamp') is not None]
                if all_ts:
                    start_us = min(all_ts)

            seizure_us = sorted(set(seizure_us))
            self._eeg_meta = {
                'start_us': start_us,
                'seizure_us': seizure_us,
            }
            return self._eeg_meta
        except Exception as error:
            logger.debug(f"Failed to parse EEG metadata anchors: {error}")
            self._eeg_meta = None
            return None

    def _ensure_eeg_loaded(self) -> bool:
        """Lazy-load EDF metadata and keep a handle for windowed extraction."""
        if self.eeg_edf_path is None:
            return False
        if self._eeg_raw is not None and self._eeg_info is not None:
            return True
        if not self.eeg_edf_path.exists():
            logger.warning(f"EEG EDF file not found: {self.eeg_edf_path}")
            return False

        try:
            import mne
        except ImportError:
            logger.warning("MNE-Python is required for EDF EEG loading. Install with: pip install mne")
            return False

        try:
            raw = mne.io.read_raw_edf(str(self.eeg_edf_path), preload=False, verbose='ERROR')
            fs = float(raw.info.get('sfreq', 0.0))
            if not np.isfinite(fs) or fs <= 0:
                logger.warning(f"Invalid EEG sampling rate in EDF: {fs}")
                return False

            picks = mne.pick_types(raw.info, eeg=True)
            if len(picks) == 0:
                picks = np.array([0])

            chosen_idx = int(picks[0])
            if self.eeg_channel:
                ch_map = {str(ch): i for i, ch in enumerate(raw.ch_names)}
                if self.eeg_channel in ch_map:
                    chosen_idx = ch_map[self.eeg_channel]
                else:
                    logger.warning(
                        "Requested EEG channel '%s' not found. Using '%s'.",
                        self.eeg_channel,
                        raw.ch_names[chosen_idx],
                    )

            edf_start = self._to_naive_timestamp(raw.info.get('meas_date'))
            meta_start = self._load_eeg_metadata_start()
            start_time = edf_start or meta_start
            if start_time is None:
                logger.warning("Could not infer EEG recording start time from EDF/meas_date or metadata.")
                return False

            duration_s = float(raw.n_times) / fs
            self._eeg_raw = raw
            self._eeg_info = {
                'fs': fs,
                'channel_index': chosen_idx,
                'channel_name': raw.ch_names[chosen_idx],
                'start_time': start_time,
                'duration_s': duration_s,
            }
            logger.info(
                "Loaded EEG EDF '%s' channel='%s' fs=%.2fHz start=%s duration=%.1fs",
                self.eeg_edf_path.name,
                self._eeg_info['channel_name'],
                fs,
                start_time,
                duration_s,
            )
            return True
        except Exception as error:
            logger.warning(f"Failed to load EEG EDF {self.eeg_edf_path}: {error}")
            return False

    def _load_eeg_window(
        self,
        seizure_time: pd.Timestamp,
        before_minutes: float,
        after_minutes: float,
        n_target: int,
        eeg_anchor_index: Optional[int] = None,
    ) -> Optional[np.ndarray]:
        """Extract seizure-aligned EEG segment and resample to requested length."""
        if not self._ensure_eeg_loaded() or self._eeg_info is None or self._eeg_raw is None:
            return None

        fs = float(self._eeg_info['fs'])
        start_time = self._eeg_info['start_time']
        before_s = float(before_minutes) * 60.0
        after_s = float(after_minutes) * 60.0

        seizure_time = self._to_naive_timestamp(seizure_time)
        if seizure_time is None:
            return None

        window_start_abs = seizure_time - pd.Timedelta(seconds=before_s)
        window_end_abs = seizure_time + pd.Timedelta(seconds=after_s)

        start_offset_s = (window_start_abs - start_time).total_seconds()
        end_offset_s = (window_end_abs - start_time).total_seconds()
        eeg_total_s = float(self._eeg_info['duration_s'])

        use_metadata_anchor = (end_offset_s <= 0 or start_offset_s >= eeg_total_s)

        if use_metadata_anchor:
            meta = self._ensure_eeg_metadata()
            seizure_anchors = meta.get('seizure_us', []) if meta else []
            meta_start_us = meta.get('start_us') if meta else None

            if seizure_anchors and meta_start_us is not None:
                anchor_offsets_s = [(ts - meta_start_us) / 1_000_000.0 for ts in seizure_anchors]
                in_range_offsets = [s for s in anchor_offsets_s if 0.0 <= s <= eeg_total_s]

                # Prefer explicit seizure index from wearable selection when available.
                if eeg_anchor_index is not None and 0 <= eeg_anchor_index < len(anchor_offsets_s):
                    anchor_offset_s = float(anchor_offsets_s[eeg_anchor_index])
                else:
                    anchor_offset_s = float(anchor_offsets_s[0])

                # If selected anchor is out of EDF bounds, fall back to a valid
                # metadata anchor within recording duration when possible.
                if not (0.0 <= anchor_offset_s <= eeg_total_s) and in_range_offsets:
                    if eeg_anchor_index is not None:
                        anchor_offset_s = min(
                            in_range_offsets,
                            key=lambda val: abs(val - float(anchor_offsets_s[min(max(eeg_anchor_index, 0), len(anchor_offsets_s)-1)])),
                        )
                    else:
                        anchor_offset_s = in_range_offsets[0]

                # Final safety: clamp to valid timeline and carve a non-empty
                # seizure-centered interval near the nearest edge.
                anchor_offset_s = float(np.clip(anchor_offset_s, 0.0, eeg_total_s))
                start_offset_s = max(0.0, anchor_offset_s - before_s)
                end_offset_s = min(eeg_total_s, anchor_offset_s + after_s)

                if end_offset_s <= start_offset_s:
                    if eeg_total_s <= 0:
                        logger.warning("EEG metadata fallback failed: EDF duration is non-positive.")
                        return None
                    # Expand to nearest available edge window.
                    edge_start = max(0.0, eeg_total_s - (before_s + after_s))
                    edge_end = eeg_total_s
                    start_offset_s, end_offset_s = edge_start, edge_end

                logger.info(
                    "EEG absolute alignment missed; using metadata seizure anchor #%s at %.1fs in EDF timeline.",
                    str(eeg_anchor_index if eeg_anchor_index is not None else 0),
                    anchor_offset_s,
                )
            else:
                logger.warning("Requested seizure window does not overlap EEG timeline and no metadata seizure anchors were found.")
                return None

        start_offset_s = max(0.0, start_offset_s)
        end_offset_s = min(eeg_total_s, end_offset_s)
        if end_offset_s <= start_offset_s:
            logger.warning("EEG window collapsed after alignment; unable to extract segment.")
            return None

        start_idx = int(round(start_offset_s * fs))
        end_idx = int(round(end_offset_s * fs))
        if end_idx <= start_idx:
            return None

        data = self._eeg_raw.get_data(
            picks=[int(self._eeg_info['channel_index'])],
            start=start_idx,
            stop=end_idx,
        )[0].astype(np.float32)
        if data.size == 0:
            return None

        if data.size != n_target:
            data = np.interp(
                np.linspace(0.0, 1.0, n_target),
                np.linspace(0.0, 1.0, data.size),
                data,
            ).astype(np.float32)

        return data
    
    def get_patient_recordings(self, patient_num: int) -> List[Dict]:
        """Get all recordings for a patient."""
        patient_recs = [r for r in self.recordings if r['subject_num'] == patient_num]
        logger.info(f"Patient {patient_num}: {len(patient_recs)} recordings")
        return patient_recs
    
    def get_seizure_recordings(self, patient_num: int) -> List[Dict]:
        """Get recordings with seizures for a patient."""
        patient_recs = self.get_patient_recordings(patient_num)
        seizure_recs = [r for r in patient_recs if r['has_seizure']]
        logger.info(f"Patient {patient_num}: {len(seizure_recs)} recordings with seizures")
        return seizure_recs
    
    def _infer_sampling_rate_from_timestamps(self, df: pd.DataFrame) -> float:
        """Infer sampling rate from timestamps in first column."""
        ts_col = df.columns[0]
        ts_values = pd.to_numeric(df[ts_col], errors='coerce').dropna().values
        
        if len(ts_values) >= 3:
            diffs = np.diff(ts_values)
            diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
            if len(diffs) > 0:
                median_dt = float(np.median(diffs))
                if median_dt > 5.0:  # likely milliseconds
                    return 1000.0 / median_dt
                else:  # likely seconds
                    return 1.0 / median_dt
        return 1.0

    @staticmethod
    def _first_matching_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
        """Find first column whose normalized name contains any candidate token."""
        lowered = {str(col).strip().lower(): col for col in df.columns}
        for token in candidates:
            token = token.lower()
            for col_lower, original in lowered.items():
                if token in col_lower:
                    return original
        return None

    @staticmethod
    def _to_numeric_array(series: pd.Series) -> np.ndarray:
        """Convert a pandas series to finite float32 numpy array."""
        values = pd.to_numeric(series, errors='coerce').fillna(0.0).values.astype(np.float32)
        return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling std for a 1D signal."""
        if values.size == 0:
            return values.astype(np.float32)
        win = max(3, int(window))
        return (
            pd.Series(values)
            .rolling(window=win, min_periods=max(2, win // 4))
            .std()
            .fillna(0.0)
            .values
            .astype(np.float32)
        )
    
    def extract_signal_around_seizure(self, 
                                     recording: Dict,
                                     seizure_time: pd.Timestamp,
                                     before_minutes: float = 3.0,
                                     after_minutes: float = 1.5,
                                     signal_types: List[str] = None,
                                     include_eeg: bool = False,
                                     eeg_anchor_index: Optional[int] = None) -> Dict:
        """Extract signals around a seizure event."""
        if signal_types is None:
            signal_types = ['ppg', 'adxl', 'eda', 'temperature', 'agc']
        
        if recording['recording_dir'] is None:
            raise ValueError(f"No recording directory for {recording['subject_id']}")
        
        # Load signals
        logger.info(f"Loading signals from {recording['recording_dir'].name}")
        signals_dict = self.loader.load_signals_from_recording(
            recording['recording_dir'], 
            signal_types=signal_types
        )
        
        if not signals_dict:
            raise ValueError("No signals loaded from recording")
        
        # Convert timestamps to seconds from recording start
        recording_start = recording['start_datetime']
        seizure_offset_s = (seizure_time - recording_start).total_seconds()
        
        # Determine minimum sampling rate
        min_fs = min(signals_dict[sig][1] for sig in signals_dict)
        
        # Calculate window
        before_s = before_minutes * 60.0
        after_s = after_minutes * 60.0
        window_start_s = seizure_offset_s - before_s
        window_end_s = seizure_offset_s + after_s
        
        window_start_idx = int(window_start_s * min_fs)
        window_end_idx = int(window_end_s * min_fs)
        seizure_idx = int(seizure_offset_s * min_fs)
        
        logger.info(f"Seizure offset: {seizure_offset_s:.1f}s, "
                   f"window: [{window_start_s:.1f}s, {window_end_s:.1f}s]")
        
        # Extract windows for each signal
        extracted_signals = {}
        signal_items = list(signals_dict.items())
        for sig_type, (sig_data, fs) in progress_iter(signal_items, desc="Extracting signals", total=len(signal_items)):
            sig_data = np.asarray(sig_data, dtype=np.float32)

            # Ensure all streams are converted to 1D before interpolation.
            if sig_data.ndim > 1:
                if sig_type == 'adxl' and sig_data.shape[1] >= 3:
                    # Use XYZ vector norm for acceleration streams.
                    sig_data = np.linalg.norm(sig_data[:, :3], axis=1)
                else:
                    # For other multi-column streams (e.g., AGC settings), collapse to mean trace.
                    sig_data = np.nanmean(sig_data, axis=1)

            sig_data = np.ravel(sig_data).astype(np.float32)
            if sig_data.size == 0:
                logger.warning(f"  {sig_type}: empty signal, skipping")
                continue

            if not np.isfinite(fs) or fs <= 0:
                logger.warning(f"  {sig_type}: invalid sampling rate {fs}, using min_fs={min_fs:.2f}")
                fs = min_fs
            
            # Resample to common rate
            if fs != min_fs:
                scale = min_fs / fs
                new_length = max(1, int(sig_data.shape[0] * scale))
                x_old = np.linspace(0, 1, sig_data.shape[0])
                x_new = np.linspace(0, 1, new_length)
                sig_data = np.interp(x_new, x_old, sig_data)
            
            # Extract window
            window_data = sig_data[max(0, window_start_idx):min(len(sig_data), window_end_idx)]
            if window_data.size == 0:
                logger.warning(f"  {sig_type}: no samples in requested seizure window, skipping")
                continue

            extracted_signals[sig_type] = window_data
            logger.info(f"  {sig_type}: {len(window_data)} samples @ {min_fs:.1f} Hz")

        if not extracted_signals:
            raise ValueError("No non-empty signal windows available for requested seizure interval")
        
        # Create time axis relative to seizure
        n_samples = max(len(v) for v in extracted_signals.values())
        time_axis = np.arange(n_samples) / min_fs - before_s
        seizure_idx_in_window = int(before_s * min_fs)

        eeg_channel_name = None
        if include_eeg:
            eeg_window = self._load_eeg_window(
                seizure_time=seizure_time,
                before_minutes=before_minutes,
                after_minutes=after_minutes,
                n_target=n_samples,
                eeg_anchor_index=eeg_anchor_index,
            )
            if eeg_window is not None:
                extracted_signals['eeg'] = eeg_window
                eeg_channel_name = self._eeg_info['channel_name'] if self._eeg_info else None
                logger.info(
                    "  eeg: %d samples from EDF channel '%s'",
                    len(eeg_window),
                    eeg_channel_name or 'unknown',
                )
            elif self.eeg_edf_path is not None:
                logger.info("  eeg: unavailable for this seizure window")
        
        return {
            'signals': extracted_signals,
            'sampling_rate': min_fs,
            'time_axis': time_axis,
            'seizure_idx': seizure_idx_in_window,
            'recording_metadata': recording,
            'seizure_time': seizure_time,
            'before_minutes': before_minutes,
            'after_minutes': after_minutes,
            'eeg_channel_name': eeg_channel_name,
            'eeg_anchor_index': eeg_anchor_index,
        }
    
    def plot_seizure_signals(self, 
                            extracted_data: Dict,
                            figsize: Tuple[int, int] = (16, 12),
                            title: Optional[str] = None) -> plt.Figure:
        """Plot PPG and key wearable device signals around seizure event."""
        signals = extracted_data['signals']
        time_axis = extracted_data['time_axis']
        seizure_idx = extracted_data['seizure_idx']
        seizure_time = extracted_data['seizure_time']
        recording_meta = extracted_data['recording_metadata']
        before_min = extracted_data['before_minutes']
        after_min = extracted_data['after_minutes']
        
        if title is None:
            title = (f"{recording_meta['subject_id']}/{recording_meta['watch_id']} - "
                    f"PPG & Wearable Signals During Seizure - {seizure_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        n_signals = len(signals)
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(n_signals, 1, figure=fig, hspace=0.3)
        
        colors = {
            'ppg': '#1f77b4', 'adxl': '#ff7f0e', 'eda': '#2ca02c',
            'temperature': '#d62728', 'agc': '#9467bd', 'eeg': '#7f3c8d',
        }
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
        
        for idx, (sig_type, sig_data) in enumerate(sorted(signals.items())):
            ax = fig.add_subplot(gs[idx])
            color = colors.get(sig_type, '#1f77b4')

            n_plot = min(len(time_axis), len(sig_data))
            if n_plot == 0:
                logger.warning(f"Skipping plot for {sig_type}: empty data")
                continue

            ax.plot(time_axis[:n_plot], sig_data[:n_plot], color=color, linewidth=1.0, alpha=0.8, label=sig_type.upper())
            
            seizure_time_s = time_axis[min(seizure_idx, n_plot - 1)]
            ax.axvline(x=seizure_time_s, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Seizure Onset')
            
            after_window = min(after_min * 60.0, 120.0)
            ax.axvspan(seizure_time_s, seizure_time_s + after_window, 
                      alpha=0.1, color='red', label='Seizure Window')
            
            warning_window_start = seizure_time_s - 60.0
            warning_window_end = seizure_time_s - 30.0
            if warning_window_start >= time_axis[0]:
                ax.axvspan(max(warning_window_start, time_axis[0]), 
                          min(warning_window_end, time_axis[-1]),
                          alpha=0.1, color='orange', label='Pre-seizure Window')
            
            if sig_type == 'eeg':
                eeg_name = extracted_data.get('eeg_channel_name') or 'EEG'
                ylabel = f"EEG ({eeg_name})"
            else:
                ylabel = sig_type.upper()

            ax.set_ylabel(ylabel, fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)
            
            if sig_data.size > 0:
                sig_mean = np.nanmean(sig_data)
                sig_std = np.nanstd(sig_data)
                stats_text = f"μ={sig_mean:.2f}, σ={sig_std:.2f}, min={np.nanmin(sig_data):.2f}, max={np.nanmax(sig_data):.2f}"
                ax.text(0.01, 0.95, stats_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Time relative to seizure (seconds)', fontsize=11)
        time_ticks = np.arange(int(time_axis[0]), int(time_axis[-1]), 60)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f"{int(t//60):+d}m" if t % 60 == 0 else f"{int(t)}s" for t in time_ticks])
        
        meta_text = (
            f"Recording: {recording_meta['recording_dir'].name if recording_meta.get('recording_dir') else 'N/A'}\n"
            f"Start: {recording_meta['start_datetime'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"End: {recording_meta['end_datetime'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Window: {before_min:.1f} min before → {after_min:.1f} min after\n"
            f"Sampling rate: {extracted_data['sampling_rate']:.1f} Hz"
        )
        fig.text(0.02, 0.02, meta_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
        
        plt.tight_layout()
        return fig
    
    def plot_all_signals_comprehensive(self,
                                      extracted_data: Dict,
                                      figsize: Tuple[int, int] = (18, 14),
                                      title: Optional[str] = None) -> plt.Figure:
        """Plot ALL available signals from CSV folder."""
        recording_meta = extracted_data['recording_metadata']
        time_axis = extracted_data['time_axis']
        seizure_idx = extracted_data['seizure_idx']
        seizure_time = extracted_data['seizure_time']
        before_min = extracted_data['before_minutes']
        after_min = extracted_data['after_minutes']
        recording_dir = recording_meta['recording_dir']
        
        if title is None:
            title = (f"{recording_meta['subject_id']}/{recording_meta['watch_id']} - "
                    f"ALL Wearable Device Signals - {seizure_time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        csv_dir = self.loader._resolve_csv_dir(recording_dir)
        if csv_dir is None:
            logger.warning(f"No CSV directory found")
            return self.plot_seizure_signals(extracted_data, figsize, title)
        
        # Load ALL signals comprehensively
        all_signals = {}
        
        # PPG stream (HR + confidence + derivatives)
        ppg_files = list(csv_dir.glob('*PPGAppStream.csv'))
        if ppg_files:
            try:
                ppg_df = pd.read_csv(ppg_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                ppg_fs = self._infer_sampling_rate_from_timestamps(ppg_df)

                hr_col = self._first_matching_column(ppg_df, ['hr', 'heart rate'])
                conf_col = self._first_matching_column(ppg_df, ['confidence'])
                state_col = self._first_matching_column(ppg_df, ['libstate', 'state'])

                if hr_col is not None:
                    hr = self._to_numeric_array(ppg_df[hr_col])
                    all_signals['PPG HR (bpm)'] = (hr, ppg_fs)

                    # Derivatives and HRV proxy to avoid overly flat single-trace interpretation.
                    hr_d1 = np.gradient(hr).astype(np.float32)
                    all_signals['PPG dHR/dt (bpm/sample)'] = (hr_d1, ppg_fs)

                    hrv_proxy = self._rolling_std(hr, window=max(5, int(ppg_fs * 30.0)))
                    all_signals['PPG HRV Proxy (rolling std)'] = (hrv_proxy, ppg_fs)

                if conf_col is not None:
                    confidence = self._to_numeric_array(ppg_df[conf_col])
                    all_signals['PPG Confidence'] = (confidence, ppg_fs)

                # LibState is typically constant in these exports, so omit it from
                # the comprehensive plot to reduce clutter.
            except Exception as e:
                logger.debug(f"Failed to load PPG: {e}")
        
        # ADPD and ADXL
        adpd_files = list(csv_dir.glob('*ADPD_ADXL_CombinedData_CSV.csv'))
        if adpd_files:
            try:
                # This export usually has a multi-row preamble. Skip to the row that
                # starts with Epoch Delta TS / D1 / S1 / ... / X / Y / Z.
                wanted_cols = {'d1', 's1', 'd2', 's2', 'x', 'y', 'z', 'magnitude'}
                df = pd.read_csv(
                    adpd_files[0],
                    skiprows=4,
                    engine='python',
                    on_bad_lines='skip',
                    usecols=lambda c: str(c).strip().lower() in wanted_cols,
                )
                clean_cols = [str(col).strip().lower() for col in df.columns]
                col_map = {str(col).strip().lower(): col for col in df.columns}
                
                adpd_fs = 100.0

                # ADPD channels often contain richer optical waveform dynamics.
                # The D1/D2 channels are consistently flat in inspected Oregon runs,
                # so keep only S1/S2 in the comprehensive visualization.
                for key, out_name in [
                    ('s1', 'ADPD Ch1 S1'),
                    ('s2', 'ADPD Ch2 S2'),
                ]:
                    if key in clean_cols:
                        col = col_map[key]
                        sig = self._to_numeric_array(df[col])
                        all_signals[out_name] = (sig, adpd_fs)
                
                # ADXL
                x_col = col_map.get('x')
                y_col = col_map.get('y')
                z_col = col_map.get('z')
                
                if x_col and y_col and z_col:
                    x = self._to_numeric_array(df[x_col])
                    y = self._to_numeric_array(df[y_col])
                    z = self._to_numeric_array(df[z_col])

                    # Total acceleration magnitude
                    magnitude = np.sqrt(x**2 + y**2 + z**2).astype(np.float32)
                    all_signals['ADXL Magnitude (Total Accel)'] = (magnitude, 50.0)
            except Exception as e:
                logger.debug(f"Failed to load ADPD/ADXL: {e}")
        
        # EDA
        eda_files = list(csv_dir.glob('*EDAAppStream.csv'))
        if eda_files:
            try:
                df = pd.read_csv(eda_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                eda_fs = self._infer_sampling_rate_from_timestamps(df)
                imp_col = self._first_matching_column(df, ['imp module'])
                adm_col = self._first_matching_column(df, ['adm module'])
                phase_col = self._first_matching_column(df, ['imp phase'])

                if imp_col is not None:
                    sig = self._to_numeric_array(df[imp_col])
                    all_signals['EDA Impedance (Ohms)'] = (sig, eda_fs)
                if adm_col is not None:
                    sig = self._to_numeric_array(df[adm_col])
                    all_signals['EDA Admittance'] = (sig, eda_fs)
                if phase_col is not None:
                    sig = self._to_numeric_array(df[phase_col])
                    all_signals['EDA Phase (rad)'] = (sig, eda_fs)
            except Exception as e:
                logger.debug(f"Failed to load EDA: {e}")
        
        # Temperature
        temp_files = list(csv_dir.glob('*TemperatureStream.csv'))
        if temp_files:
            try:
                df = pd.read_csv(temp_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                if 'Temperature' in df.columns:
                    sig = pd.to_numeric(df['Temperature'], errors='coerce').fillna(0.0).values.astype(np.float32)
                    all_signals['Temperature (°C)'] = (sig, self._infer_sampling_rate_from_timestamps(df))
            except Exception as e:
                logger.debug(f"Failed to load Temperature: {e}")
        
        # AGC
        agc_files = list(csv_dir.glob('*AGCAppStream.csv'))
        if agc_files:
            try:
                df = pd.read_csv(agc_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                numeric_cols = df.select_dtypes(include=[np.number]).columns[1:]
                if len(numeric_cols) > 0:
                    sig = df[numeric_cols[0]].astype(np.float32).fillna(0.0).values
                    all_signals['AGC Setting'] = (sig, self._infer_sampling_rate_from_timestamps(df))
            except Exception as e:
                logger.debug(f"Failed to load AGC: {e}")
        
        # Battery
        batt_files = list(csv_dir.glob('*BatteryStream.csv'))
        if batt_files:
            try:
                df = pd.read_csv(batt_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                if 'Battery' in df.columns:
                    sig = pd.to_numeric(df['Battery'], errors='coerce').fillna(0.0).values.astype(np.float32)
                    all_signals['Battery (%)'] = (sig, self._infer_sampling_rate_from_timestamps(df))
            except Exception as e:
                logger.debug(f"Failed to load Battery: {e}")
        
        # Pedometer
        ped_files = list(csv_dir.glob('*PedometerAppStream.csv'))
        if ped_files:
            try:
                df = pd.read_csv(ped_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                if 'Steps' in df.columns:
                    sig = pd.to_numeric(df['Steps'], errors='coerce').fillna(0.0).values.astype(np.float32)
                    all_signals['Steps'] = (sig, self._infer_sampling_rate_from_timestamps(df))
            except Exception as e:
                logger.debug(f"Failed to load Pedometer: {e}")
        
        # SQI
        sqi_files = list(csv_dir.glob('*SQIStream.csv'))
        if sqi_files:
            try:
                df = pd.read_csv(sqi_files[0], skiprows=2, engine='python', on_bad_lines='skip')
                if 'SQI' in df.columns:
                    sig = pd.to_numeric(df['SQI'], errors='coerce').fillna(0.0).values.astype(np.float32)
                    all_signals['SQI (Quality)'] = (sig, self._infer_sampling_rate_from_timestamps(df))
            except Exception as e:
                logger.debug(f"Failed to load SQI: {e}")

        # Optional synchronized EEG from external EDF attachment.
        n_target = len(time_axis)
        eeg_window = self._load_eeg_window(
            seizure_time=seizure_time,
            before_minutes=before_min,
            after_minutes=after_min,
            n_target=n_target,
            eeg_anchor_index=extracted_data.get('eeg_anchor_index'),
        )
        if eeg_window is not None:
            eeg_name = self._eeg_info['channel_name'] if self._eeg_info else 'EEG'
            all_signals[f"EEG ({eeg_name})"] = (eeg_window, extracted_data['sampling_rate'])
        
        if not all_signals:
            logger.warning("No comprehensive signals loaded")
            return self.plot_seizure_signals(extracted_data, figsize, title)
        
        # Use the canonical time_axis from extracted_data so all signals share the same
        # x-range as the key-signals plot.
        target_time_axis = time_axis          # shape: (N,)  relative to seizure (seconds)
        n_target = len(target_time_axis)
        target_fs = extracted_data['sampling_rate']
        seizure_offset_s = (seizure_time - recording_meta['start_datetime']).total_seconds()
        before_s = before_min * 60.0
        after_s = after_min * 60.0

        # First pass: window + resample every signal to exactly n_target points.
        # Skip signals whose window is empty.
        windowed_signals = {}  # name -> ndarray of length n_target
        for sig_name, (sig_data_orig, fs) in sorted(all_signals.items()):
            sig_data = np.ravel(sig_data_orig).astype(np.float32)
            if sig_data.size == 0:
                logger.warning(f"Skipping comprehensive plot for {sig_name}: empty data")
                continue
            if not np.isfinite(fs) or fs <= 0:
                fs = target_fs

            # Some streams (e.g., EEG overlay) are already windowed to n_target.
            if sig_data.size == n_target:
                window = sig_data
            else:
                # Extract the seizure window at the signal's native rate.
                w_start = max(0, int((seizure_offset_s - before_s) * fs))
                w_end = min(len(sig_data), int((seizure_offset_s + after_s) * fs))
                window = sig_data[w_start:w_end]
            if window.size == 0:
                logger.warning(f"Skipping comprehensive plot for {sig_name}: empty data")
                continue

            # Resample to exactly n_target points so it aligns with target_time_axis
            if len(window) != n_target:
                window = np.interp(
                    np.linspace(0, 1, n_target),
                    np.linspace(0, 1, len(window)),
                    window
                ).astype(np.float32)

            windowed_signals[sig_name] = window

        if not windowed_signals:
            logger.warning("No comprehensive signal windows available")
            return self.plot_seizure_signals(extracted_data, figsize, title)

        filtered_signals = {}
        for sig_name, windowed_sig in windowed_signals.items():
            sig_std = float(np.nanstd(windowed_sig))
            sig_range = float(np.nanmax(windowed_sig) - np.nanmin(windowed_sig))
            if sig_std == 0.0 or sig_range == 0.0:
                logger.info(f"Skipping comprehensive plot for {sig_name}: constant signal")
                continue
            filtered_signals[sig_name] = windowed_sig

        if not filtered_signals:
            logger.warning("All comprehensive signals were constant after filtering")
            return self.plot_seizure_signals(extracted_data, figsize, title)

        # Create figure only after we know which signals are valid
        n_signals = len(filtered_signals)
        dynamic_height = max(figsize[1], 2.1 * n_signals + 2.5)
        fig = plt.figure(figsize=(figsize[0], dynamic_height))
        gs = GridSpec(n_signals, 1, figure=fig, hspace=0.35)

        colors_map = {
            'PPG': '#e74c3c', 'ADPD': '#3498db', 'ADXL': '#f39c12',
            'Magnitude': '#e67e22', 'EDA': '#2ecc71', 'Temperature': '#e91e63',
            'AGC': '#9b59b6', 'Battery': '#34495e', 'Steps': '#16a085', 'SQI': '#95a5a6',
            'EEG': '#7f3c8d',
        }
        
        def get_color(signal_name):
            for key, color in colors_map.items():
                if key in signal_name:
                    return color
            return '#95a5a6'
        
        fig.suptitle(title, fontsize=14, fontweight='bold', y=0.985)
        
        seizure_time_s = time_axis[seizure_idx]
        
        for idx, (sig_name, windowed_sig) in enumerate(sorted(filtered_signals.items())):
            ax = fig.add_subplot(gs[idx])
            color = get_color(sig_name)

            ax.plot(target_time_axis, windowed_sig, color=color, linewidth=1.0, alpha=0.85, label=sig_name)

            ax.axvline(x=seizure_time_s, color='red', linestyle='--', linewidth=2.5, alpha=0.8, label='Seizure')

            after_window = min(after_min * 60.0, 120.0)
            ax.axvspan(seizure_time_s, seizure_time_s + after_window,
                      alpha=0.12, color='red', label='Seizure Period')

            warning_start = seizure_time_s - 60.0
            warning_end = seizure_time_s - 30.0
            if warning_start >= target_time_axis[0]:
                ax.axvspan(max(warning_start, target_time_axis[0]),
                          min(warning_end, target_time_axis[-1]),
                          alpha=0.08, color='orange', label='Pre-seizure')

            ax.set_ylabel(sig_name, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle=':')
            ax.legend(loc='upper right', fontsize=8)

            sig_mean = np.nanmean(windowed_sig)
            sig_std  = np.nanstd(windowed_sig)
            stats_text = (f"\u03bc={sig_mean:.2f}, \u03c3={sig_std:.2f}, "
                          f"range=[{np.nanmin(windowed_sig):.2f}, {np.nanmax(windowed_sig):.2f}]")
            ax.text(0.01, 0.95, stats_text, transform=ax.transAxes,
                   fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.6))

        ax = fig.axes[-1]
        ax.set_xlabel('Time relative to seizure (seconds)', fontsize=11, fontweight='bold')
        time_ticks = np.arange(int(target_time_axis[0]), int(target_time_axis[-1]), 60)
        ax.set_xticks(time_ticks)
        ax.set_xticklabels([f"{int(t//60):+d}m" if t % 60 == 0 else f"{int(t)}s" for t in time_ticks], fontsize=9)

        meta_text = (
            f"Patient: {recording_meta['subject_id']}/{recording_meta['watch_id']}\n"
            f"Recording: {recording_meta['recording_dir'].name}\n"
            f"Start: {recording_meta['start_datetime'].strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Window: {before_min:.1f}min before \u2192 {after_min:.1f}min after\n"
            f"Signals: {n_signals} streams aligned @ {target_fs:.1f} Hz"
        )
        fig.text(0.01, 0.01, meta_text, fontsize=9, verticalalignment='bottom',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.75))

        fig.subplots_adjust(top=0.94, bottom=0.07)
        return fig

    def save_figure(self, fig: plt.Figure, output_path: Path, dpi: int = 150) -> None:
        """Save figure to disk."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
        logger.info(f"Saved figure to {output_path}")
        plt.close(fig)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Visualize wearable device signals around seizure events',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # View first seizure for patient #2 (key signals)
  python scripts/visualize_wearable_sample_v2.py --patient 2

    # Overlay synchronized EEG EDF for patient #2
    python scripts/visualize_wearable_sample_v2.py --patient 2 \
        --eeg-dir /media/tnzr/HDD11/Datasets/WearablwDevice-Oregon/#2/EEG
  
  # Generate comprehensive ALL-signals plot
  python scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals
  
  # Custom patient and time window
  python scripts/visualize_wearable_sample_v2.py --patient 5 --before-min 5 --after-min 2
  
  # All seizures with all signals
  python scripts/visualize_wearable_sample_v2.py --patient 2 --all-seizures --all-signals
        """
    )
    
    parser.add_argument('--patient', type=int, default=2, help='Patient number (default: 2)')
    parser.add_argument('--before-min', type=float, default=3.0, help='Minutes before seizure (default: 3.0)')
    parser.add_argument('--after-min', type=float, default=1.5, help='Minutes after seizure (default: 1.5)')
    parser.add_argument('--seizure-index', type=int, default=0, help='Seizure index (default: 0)')
    parser.add_argument('--all-seizures', action='store_true', help='All seizures for patient')
    parser.add_argument('--all-signals', action='store_true', help='Generate comprehensive ALL-signals plot')
    parser.add_argument('--output-dir', type=str, default='./visualizations/wearable/', help='Output directory')
    parser.add_argument('--dataset-root', type=str, default=None, help='Dataset root')
    parser.add_argument('--eeg-dir', type=str, default=None,
                        help='Directory containing synchronized EEG EDF + optional Metadata.json')
    parser.add_argument('--eeg-edf', type=str, default=None,
                        help='Path to synchronized EEG EDF file (overrides --eeg-dir autodiscovery)')
    parser.add_argument('--eeg-metadata', type=str, default=None,
                        help='Optional Metadata.json path for EEG export timing context')
    parser.add_argument('--eeg-channel', type=str, default=None,
                        help='Optional EEG channel name to plot (default: first EEG channel in EDF)')
    parser.add_argument('--dpi', type=int, default=150, help='DPI (default: 150)')
    
    args = parser.parse_args()
    
    
    config = Config(DEFAULT_CONFIG)

    eeg_dir = Path(args.eeg_dir) if args.eeg_dir else None
    eeg_edf_path = Path(args.eeg_edf) if args.eeg_edf else None
    eeg_metadata_path = Path(args.eeg_metadata) if args.eeg_metadata else None

    if eeg_dir is None and eeg_edf_path is None:
        default_root = Path(args.dataset_root) if args.dataset_root else Path(os.environ.get('WEARABLE_DATASET_ROOT', 'data/wearable'))
        candidate_eeg_dir = default_root / f"#{args.patient}" / 'EEG'
        if candidate_eeg_dir.exists():
            eeg_dir = candidate_eeg_dir
            logger.info(f"Auto-detected EEG directory: {eeg_dir}")

    if eeg_edf_path is None and eeg_dir is not None and eeg_dir.exists():
        discovered_edf = sorted(eeg_dir.glob('*.edf'))
        if discovered_edf:
            eeg_edf_path = discovered_edf[0]

    if eeg_metadata_path is None and eeg_dir is not None and eeg_dir.exists():
        candidate_meta = eeg_dir / 'Metadata.json'
        if candidate_meta.exists():
            eeg_metadata_path = candidate_meta
    
    try:
        visualizer = WearableDatasetVisualizer(
            config,
            dataset_root=args.dataset_root,
            eeg_edf_path=str(eeg_edf_path) if eeg_edf_path else None,
            eeg_metadata_path=str(eeg_metadata_path) if eeg_metadata_path else None,
            eeg_channel=args.eeg_channel,
        )
    except FileNotFoundError as e:
        logger.error(f"Failed: {e}")
        sys.exit(1)
    
    seizure_recordings = visualizer.get_seizure_recordings(args.patient)
    
    if not seizure_recordings:
        logger.error(f"No seizure recordings for patient {args.patient}")
        sys.exit(1)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    seizure_events = []
    for rec in seizure_recordings:
        if args.all_seizures:
            for local_idx, seizure_time in enumerate(rec['seizure_times']):
                seizure_events.append((rec, seizure_time, local_idx))
        else:
            if rec['num_seizures'] > args.seizure_index:
                seizure_time = rec['seizure_times'][args.seizure_index]
                seizure_events.append((rec, seizure_time, args.seizure_index))
                break
    
    if not seizure_events:
        logger.error(f"No seizure events found")
        sys.exit(1)
    
    logger.info(f"Visualizing {len(seizure_events)} seizure event(s)")
    
    event_iter = progress_iter(
        list(enumerate(seizure_events)),
        desc="Processing seizure events",
        total=len(seizure_events)
    )
    for event_idx, (recording, seizure_time, seizure_local_index) in event_iter:
        try:
            logger.info(f"\n[{event_idx+1}/{len(seizure_events)}] Extracting signals...")
            
            extracted_data = visualizer.extract_signal_around_seizure(
                recording, seizure_time,
                before_minutes=args.before_min,
                after_minutes=args.after_min,
                include_eeg=(eeg_edf_path is not None),
                eeg_anchor_index=seizure_local_index,
            )
            
            # Plot key signals
            logger.info(f"Plotting key signals...")
            fig = visualizer.plot_seizure_signals(extracted_data)
            
            timestamp = seizure_time.strftime('%Y%m%d_%H%M%S')
            filename = f"patient_{args.patient:02d}_seizure_{event_idx:02d}_{timestamp}.png"
            output_path = output_dir / filename
            visualizer.save_figure(fig, output_path, dpi=args.dpi)
            
            # Plot all signals if requested
            if args.all_signals:
                logger.info(f"Generating comprehensive ALL-signals plot...")
                fig_all = visualizer.plot_all_signals_comprehensive(extracted_data)
                filename_all = f"patient_{args.patient:02d}_seizure_{event_idx:02d}_{timestamp}_ALL_SIGNALS.png"
                output_path_all = output_dir / filename_all
                visualizer.save_figure(fig_all, output_path_all, dpi=args.dpi)
            
        except Exception as e:
            logger.error(f"Failed on seizure event {event_idx}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    logger.info(f"\n✓ Complete! Saved to {output_dir}")


if __name__ == '__main__':
    main()
