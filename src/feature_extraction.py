"""
Feature extraction module for ECG, EEG, and motion signals.
Computes heart rate, HRV, frequency domain, and entropy features.
"""

import logging
from typing import Dict, Tuple, Optional
import numpy as np
from scipy import stats
from scipy.signal import welch, periodogram
from scipy.fft import fft

from config.config import DataConfig


logger = logging.getLogger(__name__)


class HRVCalculator:
    """
    Heart Rate Variability (HRV) feature extraction.
    
    Computes:
    - Time domain: RMSSD, SDNN, pNN50
    - Frequency domain: LF, HF, LF/HF ratio
    - Entropy metrics
    """
    
    def __init__(self, config: DataConfig):
        """Initialize HRV calculator.
        
        Args:
            config: DataConfig with signal specifications
        """
        self.fs = config.sampling_rate_high
        self.window_s = config.feature_window_s
    
    def compute_time_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Compute time-domain HRV features.
        
        Args:
            rr_intervals: (N,) RR intervals in seconds
        
        Returns:
            Dictionary with time-domain HRV features
        """
        if len(rr_intervals) < 2:
            return {
                'hr_mean': 0.0,
                'hr_std': 0.0,
                'rmssd': 0.0,
                'sdnn': 0.0,
                'pnn50': 0.0,
            }
        
        # Convert to heart rate (bpm)
        hr = 60.0 / rr_intervals
        
        # RR interval differences
        rr_diffs = np.diff(rr_intervals)
        
        features = {
            'hr_mean': float(np.mean(hr)),
            'hr_std': float(np.std(hr)),
            'hr_min': float(np.min(hr)),
            'hr_max': float(np.max(hr)),
            # RR interval features
            'rr_mean': float(np.mean(rr_intervals)),
            'rr_std': float(np.std(rr_intervals)),
            # RMSSD: root mean square of successive differences
            'rmssd': float(np.sqrt(np.mean(rr_diffs ** 2))),
            # SDNN: standard deviation of all RR intervals
            'sdnn': float(np.std(rr_intervals)),
            # pNN50: percentage of successive RR intervals > 50ms
            'pnn50': float(np.sum(np.abs(rr_diffs) > 0.05) / len(rr_diffs) * 100),
        }
        
        return features
    
    def compute_frequency_domain(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Compute frequency-domain HRV features (power spectral density).
        
        Args:
            rr_intervals: (N,) RR intervals in seconds
        
        Returns:
            Dictionary with frequency-domain HRV features
        """
        if len(rr_intervals) < 32:  # Need minimum for FFT
            return {
                'lf_power': 0.0,
                'hf_power': 0.0,
                'lf_hf_ratio': 0.0,
                'lf_norm': 0.0,
                'hf_norm': 0.0,
            }
        
        # Interpolate RR intervals to uniform sampling
        # Typical RR sampling is 1 Hz or higher
        hr_fs = max(1.0, 1.0 / np.mean(rr_intervals))
        
        # Resample to 4 Hz (standard for HRV)
        target_fs = 4.0
        t_original = np.cumsum(rr_intervals)
        t_original = np.concatenate([[0], t_original])
        rr_concat = np.concatenate([[rr_intervals[0]], rr_intervals])
        
        t_uniform = np.arange(0, t_original[-1], 1/target_fs)
        rr_uniform = np.interp(t_uniform, t_original, rr_concat)
        
        # Compute PSD using Welch method
        nperseg = min(256, len(rr_uniform) // 2)
        freq, power = welch(rr_uniform, fs=target_fs, nperseg=nperseg)
        
        # Define bands (standard IEC definitions)
        vlf_mask = (freq >= 0.003) & (freq < 0.04)   # VLF
        lf_mask = (freq >= 0.04) & (freq < 0.15)      # LF
        hf_mask = (freq >= 0.15) & (freq < 0.40)      # HF
        
        lf_power = np.sum(power[lf_mask])
        hf_power = np.sum(power[hf_mask])
        vlf_power = np.sum(power[vlf_mask])
        
        total_power = lf_power + hf_power + vlf_power
        
        features = {
            'lf_power': float(lf_power),
            'hf_power': float(hf_power),
            'vlf_power': float(vlf_power),
            'lf_hf_ratio': float(lf_power / (hf_power + 1e-6)),
            'lf_norm': float(lf_power / (total_power + 1e-6)),
            'hf_norm': float(hf_power / (total_power + 1e-6)),
        }
        
        return features
    
    def compute_entropy(self, rr_intervals: np.ndarray, order: int = 2) -> Dict[str, float]:
        """Compute entropy-based HRV features.
        
        Args:
            rr_intervals: (N,) RR intervals
            order: Embedding dimension
        
        Returns:
            Dictionary with entropy features
        """
        if len(rr_intervals) < order + 1:
            return {
                'sample_entropy': 0.0,
                'permutation_entropy': 0.0,
            }
        
        # Sample entropy approximation
        sample_ent = self._sample_entropy(rr_intervals, order)
        
        # Detrended fluctuation analysis (approximate)
        dfa = self._detrended_fluctuation_analysis(rr_intervals)
        
        features = {
            'sample_entropy': float(sample_ent),
            'dfa_alpha': float(dfa),
        }
        
        return features
    
    def _sample_entropy(self, signal: np.ndarray, order: int) -> float:
        """Compute sample entropy (simplified)."""
        N = len(signal)
        
        # Count template matches
        matches_m = 0
        matches_m1 = 0
        r = 0.15 * np.std(signal)  # Threshold
        
        for i in range(N - order):
            for j in range(i + 1, N - order):
                if np.max(np.abs(signal[i:i+order] - signal[j:j+order])) < r:
                    matches_m += 1
                if np.max(np.abs(signal[i:i+order+1] - signal[j:j+order+1])) < r:
                    matches_m1 += 1
        
        return -np.log((matches_m1 + 1e-6) / (matches_m + 1e-6))
    
    def _detrended_fluctuation_analysis(self, signal: np.ndarray) -> float:
        """Compute DFA exponent (alpha)."""
        # Simplified DFA
        y = np.cumsum(signal - np.mean(signal))
        
        window_sizes = [10, 20, 40, 80]
        fluct = []
        
        for w in window_sizes:
            if w > len(y) // 2:
                continue
            
            # Fit trends and compute fluctuation
            n_segments = len(y) // w
            flu = 0
            
            for seg in range(n_segments):
                segment = y[seg*w:(seg+1)*w]
                trend = np.polyfit(np.arange(len(segment)), segment, 2)
                poly = np.poly1d(trend)
                residual = segment - poly(np.arange(len(segment)))
                flu += np.mean(residual ** 2)
            
            fluct.append(np.sqrt(flu / n_segments))
        
        # Fit log-log relationship
        if len(fluct) > 1:
            log_ws = np.log(window_sizes[:len(fluct)])
            log_fluct = np.log(np.array(fluct))
            slope = np.polyfit(log_ws, log_fluct, 1)[0]
            return slope
        
        return 0.5


class InstabilityExtractor:
    """
    Cardiac instability features indicating pre-ictal autonomic changes.
    """
    
    def compute_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Compute cardiac instability indicators.
        
        Args:
            rr_intervals: (N,) RR intervals
        
        Returns:
            Dictionary with instability features
        """
        if len(rr_intervals) < 2:
            return {
                'rr_acceleration': 0.0,
                'rr_jitter': 0.0,
                'rr_complexity': 0.0,
            }
        
        # RR interval derivatives
        rr_diffs = np.diff(rr_intervals)
        rr_diffs2 = np.diff(rr_diffs)  # Second derivative
        
        features = {
            # Mean absolute change in RR intervals
            'rr_acceleration': float(np.mean(np.abs(rr_diffs))),
            # Standard deviation of RR changes
            'rr_jitter': float(np.std(rr_diffs)),
            # Second derivative (curvature)
            'rr_curvature': float(np.mean(np.abs(rr_diffs2))),
            # Complexity (variance of changes)
            'rr_complexity': float(np.std(np.abs(rr_diffs))),
            # Trend (linear regression slope)
            'rr_trend': float(np.polyfit(np.arange(len(rr_intervals)), rr_intervals, 1)[0]),
        }
        
        return features


class SpectralFeatureExtractor:
    """
    Spectral feature extraction for EEG/EMG signals.
    """
    
    def __init__(self, config: DataConfig):
        """Initialize spectral extractor.
        
        Args:
            config: DataConfig with signal specifications
        """
        self.fs = config.sampling_rate_high
    
    def compute_bands(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute power spectral density in standard EEG bands.
        
        Args:
            signal: (N,) time series signal
        
        Returns:
            Dictionary with band power features
        """
        if len(signal) < 64:
            return {
                'delta': 0.0,
                'theta': 0.0,
                'alpha': 0.0,
                'beta': 0.0,
            }
        
        # Compute PSD
        nperseg = min(256, len(signal) // 2)
        freq, power = welch(signal, fs=self.fs, nperseg=nperseg)
        
        # Band definitions
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
        }
        
        features = {}
        for band_name, (low, high) in bands.items():
            mask = (freq >= low) & (freq < high)
            features[f'{band_name}_power'] = float(np.sum(power[mask]))
            features[f'{band_name}_norm'] = float(np.sum(power[mask]) / np.sum(power))
        
        return features
    
    def compute_spectral_features(self, signal: np.ndarray) -> Dict[str, float]:
        """Compute additional spectral features.
        
        Args:
            signal: (N,) time series
        
        Returns:
            Dictionary with spectral features
        """
        if len(signal) < 64:
            return {
                'spectral_centroid': 0.0,
                'spectral_entropy': 0.0,
                'spectral_flatness': 0.0,
            }
        
        nperseg = min(256, len(signal) // 2)
        freq, power = welch(signal, fs=self.fs, nperseg=nperseg)
        
        # Spectral centroid
        centroid = np.sum(freq * power) / (np.sum(power) + 1e-6)
        
        # Spectral entropy (normalized)
        norm_power = power / (np.sum(power) + 1e-6)
        entropy = -np.sum(norm_power * np.log(norm_power + 1e-10))
        entropy_norm = entropy / np.log(len(freq))
        
        # Spectral flatness (geometric mean / arithmetic mean)
        geom_mean = np.exp(np.mean(np.log(power + 1e-12)))
        arith_mean = np.mean(power)
        flatness = geom_mean / (arith_mean + 1e-6)
        
        return {
            'spectral_centroid': float(centroid),
            'spectral_entropy': float(entropy_norm),
            'spectral_flatness': float(flatness),
        }


class FeatureExtractor:
    """
    Unified feature extraction for all signals.
    """
    
    def __init__(self, config: DataConfig):
        """Initialize feature extractor.
        
        Args:
            config: DataConfig
        """
        self.config = config
        self.hrv_calc = HRVCalculator(config)
        self.instability_extractor = InstabilityExtractor()
        self.spectral_extractor = SpectralFeatureExtractor(config)
    
    def extract_ecg_features(self, rr_intervals: np.ndarray) -> Dict[str, float]:
        """Extract all ECG-derived features.
        
        Args:
            rr_intervals: (N,) RR intervals
        
        Returns:
            Dictionary with all ECG features (14-dim)
        """
        features = {}
        
        # Time domain
        features.update(self.hrv_calc.compute_time_domain(rr_intervals))
        
        # Frequency domain
        features.update(self.hrv_calc.compute_frequency_domain(rr_intervals))
        
        # Entropy
        features.update(self.hrv_calc.compute_entropy(rr_intervals))
        
        # Instability
        features.update(self.instability_extractor.compute_features(rr_intervals))
        
        return features
    
    def extract_eeg_features(self, eeg_signal: np.ndarray) -> Dict[str, float]:
        """Extract EEG features.
        
        Args:
            eeg_signal: (N,) EEG signal
        
        Returns:
            Dictionary with EEG features (~6-8 dim)
        """
        features = {}
        
        # Spectral bands
        features.update(self.spectral_extractor.compute_bands(eeg_signal))
        
        # Spectral characteristics
        features.update(self.spectral_extractor.compute_spectral_features(eeg_signal))
        
        return features
    
    def get_feature_names(self, modality: str = "ecg") -> list:
        """Get list of feature names for a modality.
        
        Args:
            modality: "ecg" or "eeg"
        
        Returns:
            List of feature names
        """
        if modality == "ecg":
            # Time domain
            names = ['hr_mean', 'hr_std', 'hr_min', 'hr_max',
                    'rr_mean', 'rr_std', 'rmssd', 'sdnn', 'pnn50']
            # Frequency domain
            names += ['lf_power', 'hf_power', 'vlf_power', 'lf_hf_ratio', 
                     'lf_norm', 'hf_norm']
            # Entropy
            names += ['sample_entropy', 'dfa_alpha']
            # Instability
            names += ['rr_acceleration', 'rr_jitter', 'rr_curvature', 
                     'rr_complexity', 'rr_trend']
            return names
        
        elif modality == "eeg":
            names = ['delta_power', 'theta_power', 'alpha_power', 'beta_power',
                    'delta_norm', 'theta_norm', 'alpha_norm', 'beta_norm',
                    'spectral_centroid', 'spectral_entropy', 'spectral_flatness']
            return names
        
        else:
            raise ValueError(f"Unknown modality: {modality}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    from config.config import DEFAULT_CONFIG
    
    # Create sample RR intervals
    rr = 0.857 + np.random.normal(0, 0.05, 100)  # ~70 bpm with noise
    rr = np.clip(rr, 0.4, 1.2)
    
    # Extract features
    extractor = FeatureExtractor(DEFAULT_CONFIG.data)
    ecg_features = extractor.extract_ecg_features(rr)
    
    print("ECG Features:")
    for key, value in ecg_features.items():
        print(f"  {key}: {value:.4f}")
    
    print(f"\nTotal ECG features: {len(ecg_features)}")
