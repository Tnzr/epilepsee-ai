"""
Signal preprocessing module for ECG, EEG, EMG, and motion data.
Handles filtering, artifact detection, and feature normalization.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from scipy import signal
from scipy.signal import find_peaks, butter, sosfilt

from config.config import DataConfig


logger = logging.getLogger(__name__)


class ECGPreprocessor:
    """
    ECG signal preprocessing and RR interval extraction.
    
    Pipeline:
    1. Bandpass filtering (0.5-40 Hz)
    2. R-peak detection
    3. RR interval calculation
    4. Ectopic beat removal
    """
    
    def __init__(self, config: DataConfig):
        """Initialize ECG preprocessor.
        
        Args:
            config: DataConfig with ECG filter specifications
        """
        self.fs = config.sampling_rate_high
        self.lowcut = config.ecg_lowcut
        self.highcut = config.ecg_highcut
        
        # Design bandpass filter once
        self.sos = butter(4, [self.lowcut, self.highcut], btype='bandpass', 
                         output='sos', fs=self.fs)
    
    def bandpass_filter(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Apply bandpass filter to raw ECG.
        
        Args:
            ecg_signal: (N,) raw ECG signal
        
        Returns:
            (N,) filtered ECG signal
        """
        if len(ecg_signal) == 0:
            raise ValueError("Empty ECG signal")
        
        filtered = sosfilt(self.sos, ecg_signal)
        logger.debug(f"Filtered ECG: shape={filtered.shape}")
        return filtered
    
    def detect_r_peaks(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Detect R-peaks in ECG signal using peak finding.
        
        Args:
            ecg_signal: (N,) filtered ECG signal
        
        Returns:
            (M,) array of R-peak sample indices
        """
        if len(ecg_signal) == 0:
            return np.array([], dtype=int)
        
        # Find peaks (R-peaks are positive peaks)
        # Distance: minimum samples between peaks (prevents multiple peaks for single QRS)
        min_distance = int(0.4 * self.fs)  # Minimum 0.4s between beats (max HR ~150 bpm)
        
        # Find peaks using adaptive threshold
        signal_std = np.std(ecg_signal)
        threshold = signal_std * 2.0
        
        r_peaks, _ = find_peaks(ecg_signal, distance=min_distance, height=threshold)
        
        logger.debug(f"Detected {len(r_peaks)} R-peaks")
        return r_peaks
    
    def extract_rr_intervals(self, ecg_signal: np.ndarray) -> np.ndarray:
        """Extract RR intervals from ECG.
        
        Args:
            ecg_signal: (N,) raw ECG signal
        
        Returns:
            (M-1,) RR intervals in seconds
        """
        # Preprocess
        filtered = self.bandpass_filter(ecg_signal)
        
        # Detect R-peaks
        r_peaks = self.detect_r_peaks(filtered)
        
        if len(r_peaks) < 2:
            logger.warning("Not enough R-peaks detected for RR intervals")
            return np.array([])
        
        # Convert to times (seconds)
        r_peak_times = r_peaks / self.fs
        
        # Calculate RR intervals
        rr_intervals = np.diff(r_peak_times)
        
        # Remove ectopic beats (keep 0.4-1.2 seconds)
        valid_mask = (rr_intervals >= 0.4) & (rr_intervals <= 1.2)
        rr_clean = rr_intervals[valid_mask]
        
        removed = np.sum(~valid_mask)
        if removed > 0:
            logger.info(f"Removed {removed} ectopic beats ({removed/len(rr_intervals)*100:.1f}%)")
        
        return rr_clean
    
    def compute_quality_metrics(self, ecg_signal: np.ndarray) -> dict:
        """Compute ECG signal quality metrics.
        
        Args:
            ecg_signal: (N,) raw ECG signal
        
        Returns:
            Dictionary with quality metrics
        """
        filtered = self.bandpass_filter(ecg_signal)
        r_peaks = self.detect_r_peaks(filtered)
        
        # Expected number of peaks (assume ~70 bpm baseline)
        expected_peaks = len(ecg_signal) / self.fs / (60 / 70)
        
        metrics = {
            'num_peaks': len(r_peaks),
            'expected_peaks': expected_peaks,
            'detection_sensitivity': len(r_peaks) / max(expected_peaks, 1),
            'signal_snr': np.max(filtered) / (np.std(filtered) + 1e-6),
        }
        
        return metrics


class MotionDetector:
    """
    Motion artifact detection using accelerometer data.
    Identifies windows with excessive motion.
    """
    
    def __init__(self, config: DataConfig):
        """Initialize motion detector.
        
        Args:
            config: DataConfig with motion threshold
        """
        self.fs = config.sampling_rate_motion
        self.threshold = config.motion_threshold  # m/s²
    
    def compute_acceleration_magnitude(self, acc_data: np.ndarray) -> np.ndarray:
        """Compute magnitude of acceleration vector.
        
        Args:
            acc_data: (3, N) - 3-axis acceleration
        
        Returns:
            (N,) magnitude in m/s²
        """
        if acc_data.shape[0] != 3:
            raise ValueError(f"Expected 3 acceleration axes, got {acc_data.shape[0]}")
        
        magnitude = np.linalg.norm(acc_data, axis=0)
        return magnitude
    
    def detect_artifact_epochs(self, acc_data: np.ndarray, 
                               window_s: float = 1.0) -> np.ndarray:
        """Detect epochs with motion artifacts.
        
        Args:
            acc_data: (3, N) - acceleration data
            window_s: Window size for artifact detection
        
        Returns:
            (N,) binary mask where 1 = artifact
        """
        magnitude = self.compute_acceleration_magnitude(acc_data)
        
        # Compute RMS in sliding window
        window_samples = int(window_s * self.fs)
        artifact_mask = np.zeros(len(magnitude), dtype=bool)
        
        for start in range(0, len(magnitude) - window_samples, 1):
            end = start + window_samples
            window_rms = np.sqrt(np.mean(magnitude[start:end] ** 2))
            
            if window_rms > self.threshold:
                artifact_mask[start:end] = True
        
        artifact_percentage = np.sum(artifact_mask) / len(artifact_mask) * 100
        logger.info(f"Motion artifacts: {artifact_percentage:.1f}% of signal")
        
        return artifact_mask.astype(np.uint8)


class SignalNormalizer:
    """
    Per-subject signal normalization for consistent feature scales.
    """
    
    def __init__(self, method: str = "zscore"):
        """Initialize normalizer.
        
        Args:
            method: "zscore" or "minmax"
        """
        self.method = method
        self.mean__ = None
        self.std_ = None
        self.min_ = None
        self.max_ = None
    
    def fit(self, signal_data: np.ndarray) -> None:
        """Compute normalization statistics.
        
        Args:
            signal_data: (N, F) - N samples, F features
        """
        if self.method == "zscore":
            self.mean_ = np.mean(signal_data, axis=0)
            self.std_ = np.std(signal_data, axis=0)
        elif self.method == "minmax":
            self.min_ = np.min(signal_data, axis=0)
            self.max_ = np.max(signal_data, axis=0)
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
    
    def transform(self, signal_data: np.ndarray) -> np.ndarray:
        """Apply normalization.
        
        Args:
            signal_data: (N, F) - N samples, F features
        
        Returns:
            (N, F) normalized signal
        """
        if self.method == "zscore":
            if self.mean_ is None or self.std_ is None:
                raise RuntimeError("Normalizer not fitted. Call fit() first.")
            return (signal_data - self.mean_) / (self.std_ + 1e-6)
        
        elif self.method == "minmax":
            if self.min_ is None or self.max_ is None:
                raise RuntimeError("Normalizer not fitted. Call fit() first.")
            denom = self.max_ - self.min_ + 1e-6
            return (signal_data - self.min_) / denom
    
    def fit_transform(self, signal_data: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(signal_data)
        return self.transform(signal_data)


class SignalProcessor:
    """
    Unified signal processor combining ECG preprocessing and motion detection.
    """
    
    def __init__(self, config: DataConfig):
        """Initialize signal processor.
        
        Args:
            config: DataConfig with all signal specifications
        """
        self.config = config
        self.ecg_preprocessor = ECGPreprocessor(config)
        self.motion_detector = MotionDetector(config)
        self.normalizer = SignalNormalizer(method="zscore")
    
    def process_ecg(self, ecg_signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Process raw ECG and extract RR intervals.
        
        Args:
            ecg_signal: (N,) raw ECG signal (1 channel)
        
        Returns:
            Tuple of (rr_intervals, quality_metrics)
        """
        # Ensure 1D
        if ecg_signal.ndim > 1:
            ecg_signal = ecg_signal[0]
        
        rr_intervals = self.ecg_preprocessor.extract_rr_intervals(ecg_signal)
        quality = self.ecg_preprocessor.compute_quality_metrics(ecg_signal)
        
        return rr_intervals, quality
    
    def detect_motion_artifacts(self, motion_data: np.ndarray) -> np.ndarray:
        """Detect motion artifact epochs.
        
        Args:
            motion_data: (6, N) - 3 ACC + 3 GYR channels
        
        Returns:
            (N,) binary artifact mask
        """
        if motion_data.shape[0] < 3:
            logger.warning("Insufficient motion channels")
            return np.zeros(motion_data.shape[1], dtype=np.uint8)
        
        # Use only accelerometer data (first 3 channels)
        acc_data = motion_data[:3, :]
        
        # Upsample to match high-frequency signals
        upsampled = self._upsample_motion(acc_data, self.config.sampling_rate_high)
        
        artifact_mask = self.motion_detector.detect_artifact_epochs(upsampled)
        return artifact_mask
    
    def _upsample_motion(self, motion_data: np.ndarray, target_fs: int) -> np.ndarray:
        """Upsample motion data to match high-frequency signals.
        
        Args:
            motion_data: (3, N) - motion data at 25 Hz
            target_fs: Target sampling rate (250 Hz)
        
        Returns:
            (3, M) - upsampled motion data
        """
        source_fs = self.config.sampling_rate_motion
        
        if source_fs >= target_fs:
            return motion_data
        
        ratio = target_fs / source_fs
        new_length = int(motion_data.shape[1] * ratio)
        
        upsampled = np.zeros((motion_data.shape[0], new_length))
        for i in range(motion_data.shape[0]):
            upsampled[i, :] = np.interp(
                np.arange(new_length) / ratio,
                np.arange(len(motion_data[i])),
                motion_data[i]
            )
        
        return upsampled


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config.config import DEFAULT_CONFIG
    
    # Create sample ECG data
    fs = 250
    duration = 10  # seconds
    t = np.arange(0, duration, 1/fs)
    # Simulate ECG: baseline + QRS complex repeated
    ecg = 0.1 * np.sin(2 * np.pi * 0.5 * t)  # Baseline
    for beat in range(int(duration * 70 / 60)):  # ~70 bpm
        beat_time = beat * (60 / 70)
        beat_idx = int(beat_time * fs)
        if beat_idx < len(ecg):
            ecg[beat_idx:beat_idx+50] += 1.0 * np.exp(-np.arange(50) / 10)  # QRS-like
    
    # Test ECG processor
    processor = SignalProcessor(DEFAULT_CONFIG.data)
    rr_intervals, quality = processor.process_ecg(ecg)
    
    print(f"RR intervals: mean={np.mean(rr_intervals):.3f}s, std={np.std(rr_intervals):.3f}s")
    print(f"Quality metrics: {quality}")
