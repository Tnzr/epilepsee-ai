"""
Data loading module for BIDS-formatted SeizeIT2 seizure dataset.
Handles EDF file reading, event parsing, and dataset construction.
"""

import os
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import numpy as np
import pandas as pd
from collections import defaultdict
import json
import csv

# Try importing neurophysiology libraries
try:
    import mne
    HAS_MNE = True
except ImportError:
    HAS_MNE = False

try:
    from bids import BIDSLayout
    HAS_BIDS = True
except ImportError:
    HAS_BIDS = False

try:
    import pyedf
    HAS_PYEDF = True
except ImportError:
    HAS_PYEDF = False

from config.config import DataConfig


logger = logging.getLogger(__name__)


# ============================================================================
# Data Augmentation Utilities for Preictal Samples
# ============================================================================

def time_warp(features: np.ndarray, rate: float) -> np.ndarray:
    """Warp time dimension by resampling.
    
    Args:
        features: (time_steps, feature_dim) array
        rate: Warping rate (0.9 = compress 10%, 1.1 = stretch 10%)
    
    Returns:
        Warped features with same shape
    """
    original_length = features.shape[0]
    warped_length = int(original_length * rate)
    
    # Resample each feature channel
    warped = np.zeros_like(features)
    for feat_idx in range(features.shape[1]):
        x_orig = np.linspace(0, 1, original_length)
        x_warp = np.linspace(0, 1, warped_length)
        warped_feat = np.interp(x_warp, x_orig, features[:, feat_idx])
        
        # Resample back to original length
        x_final = np.linspace(0, 1, original_length)
        warped[:, feat_idx] = np.interp(x_final, np.linspace(0, 1, warped_length), warped_feat)
    
    return warped.astype(np.float32)


def amplitude_scale(features: np.ndarray, scale: float) -> np.ndarray:
    """Scale amplitude of features.
    
    Args:
        features: (time_steps, feature_dim) array
        scale: Scaling factor (0.9 = 90%, 1.1 = 110%)
    
    Returns:
        Scaled features
    """
    return (features * scale).astype(np.float32)


def add_gaussian_noise(features: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to features.
    
    Args:
        features: (time_steps, feature_dim) array
        noise_level: Standard deviation of noise relative to signal std
    
    Returns:
        Noisy features
    """
    noise = np.random.normal(0, noise_level, features.shape)
    return (features + noise).astype(np.float32)


def time_shift(features: np.ndarray, shift: int) -> np.ndarray:
    """Shift features in time (circular shift).
    
    Args:
        features: (time_steps, feature_dim) array
        shift: Number of timesteps to shift (negative = backward, positive = forward)
    
    Returns:
        Shifted features
    """
    return np.roll(features, shift, axis=0).astype(np.float32)


def augment_preictal_sample(features: np.ndarray, label: float, 
                           config: DataConfig, rng: np.random.Generator) -> List[Tuple[np.ndarray, float]]:
    """Generate multiple augmented versions of a preictal sample.
    
    Args:
        features: (time_steps, feature_dim) original sample
        label: Countdown label
        config: DataConfig with augmentation parameters
        rng: Random number generator for reproducibility
    
    Returns:
        List of (augmented_features, label) tuples
    """
    augmented = [(features.copy(), label)]  # Include original
    
    # Calculate how many augmentations we need
    total_needed = config.augmentation_factor
    
    aug_types = [
        ('time_warp', config.aug_time_warp_rates),
        ('amplitude', config.aug_amplitude_scales),
        ('noise', config.aug_noise_levels),
        ('shift', config.aug_time_shifts),
    ]
    
    # Cycle through augmentation types
    aug_count = 1  # Already have original
    while aug_count < total_needed:
        for aug_type, params in aug_types:
            if aug_count >= total_needed:
                break
            
            param = rng.choice(params)
            
            if aug_type == 'time_warp':
                aug_features = time_warp(features, param)
            elif aug_type == 'amplitude':
                aug_features = amplitude_scale(features, param)
            elif aug_type == 'noise':
                aug_features = add_gaussian_noise(features, param)
            elif aug_type == 'shift':
                aug_features = time_shift(features, int(param))
            
            augmented.append((aug_features, label))
            aug_count += 1
    
    return augmented[:total_needed]


class BIDSDataLoader:
    """
    Loads physiological data from BIDS-formatted SeizeIT2 dataset.
    
    Features:
    - Reads EDF files using MNE-Python
    - Parses events.tsv for seizure annotations
    - Extracts ECG, EEG, EMG, and motion signals
    - Handles multiple runs per subject
    """
    
    def __init__(self, config: DataConfig):
        """Initialize BIDS dataset loader.
        
        Args:
            config: DataConfig with dataset paths and specifications
        """
        self.config = config
        self.dataset_root = Path(config.dataset_root)
        
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_root}")
        
        # Load BIDS layout
        if HAS_BIDS:
            self.bids_layout = BIDSLayout(str(self.dataset_root), validate=config.bids_validate)
        else:
            logger.warning("pybids not available, using fallback directory scanning")
            self.bids_layout = None

        # Cache resolved EDF paths so selection and loading do not repeatedly
        # scan the same directories or reopen files just to prove they exist.
        self._edf_path_cache: Dict[Tuple[str, str, str, Optional[int]], Path] = {}
        
        logger.info(f"Initialized BIDS dataset loader from {self.dataset_root}")

    def resolve_subject_edf_path(
        self,
        subject_id: str,
        session_id: str = "01",
        datatype: str = "ecg",
        run_id: Optional[int] = None,
    ) -> Path:
        """Resolve the EDF path for a subject/run without loading file contents."""
        cache_key = (str(subject_id), str(session_id), str(datatype), run_id)
        cached = self._edf_path_cache.get(cache_key)
        if cached is not None:
            return cached

        subject_dir = self.dataset_root / f"sub-{subject_id}" / f"ses-{session_id}" / datatype

        if not subject_dir.exists():
            raise FileNotFoundError(f"Subject data not found: {subject_dir}")

        edf_files = sorted(subject_dir.glob("*.edf"))
        if not edf_files:
            raise FileNotFoundError(f"No EDF files found in {subject_dir}")

        if run_id is not None:
            edf_file = next((f for f in edf_files if f"run-{run_id:02d}" in f.name), None)
            if edf_file is None:
                raise FileNotFoundError(f"Run {run_id} not found in {subject_dir}")
        else:
            edf_file = edf_files[0]

        self._edf_path_cache[cache_key] = edf_file
        return edf_file
    
    def get_subjects(self) -> List[str]:
        """Get list of all subject IDs."""
        if self.bids_layout:
            return self.bids_layout.get_subjects()
        else:
            # Fallback: scan directories
            subjects = []
            for item in self.dataset_root.iterdir():
                if item.is_dir() and item.name.startswith('sub-'):
                    subjects.append(item.name.replace('sub-', ''))
            return sorted(subjects)
    
    def load_subject_edf(self, subject_id: str, session_id: str = "01", 
                         datatype: str = "ecg", run_id: Optional[int] = None) -> Tuple[np.ndarray, float]:
        """Load a single EDF file for a subject.
        
        Args:
            subject_id: Subject ID (e.g., "001")
            session_id: Session ID (default: "01")
            datatype: "ecg", "eeg", "emg", or "mov"
            run_id: Run number (optional, loads first if None)
        
        Returns:
            Tuple of (signal_data, sampling_rate)
        """
        if not HAS_MNE:
            raise ImportError("MNE-Python required for EDF loading. Install: pip install mne")
        
        edf_file = self.resolve_subject_edf_path(
            subject_id,
            session_id=session_id,
            datatype=datatype,
            run_id=run_id,
        )
        
        # Load with MNE
        try:
            raw = mne.io.read_raw_edf(str(edf_file), preload=True)
            signal_data = raw.get_data()
            sampling_rate = raw.info['sfreq']
            
            logger.info(f"Loaded {edf_file.name}: shape={signal_data.shape}, fs={sampling_rate} Hz")
            return signal_data, sampling_rate
        
        except Exception as e:
            logger.error(f"Failed to load {edf_file}: {str(e)}")
            raise
    
    def load_events_tsv(self, subject_id: str, session_id: str = "01", run_id: int = 1) -> pd.DataFrame:
        """Load events.tsv for a subject run.
        
        Args:
            subject_id: Subject ID (e.g., "001")
            session_id: Session ID
            run_id: Run number
        
        Returns:
            DataFrame with columns: onset, duration, trial_type, [other metadata]
        """
        # Build path (subject_id is already a string from get_subjects)
        session_dir = self.dataset_root / f"sub-{subject_id}" / f"ses-{session_id}"
        
        # Events files are typically in the eeg/ subdirectory in BIDS format
        eeg_dir = session_dir / "eeg"
        
        # Try to find events file  
        events_files = []
        if eeg_dir.exists():
            events_files = list(eeg_dir.glob(f"*run-{run_id:02d}_events.tsv"))
        
        if not events_files:
            # Try any subdirectory
            events_files = list(session_dir.glob(f"*/*run-{run_id:02d}_events.tsv"))
        
        if not events_files:
            # Try any events file in eeg directory
            if eeg_dir.exists():
                events_files = list(eeg_dir.glob("*_events.tsv"))
        
        if not events_files:
            logger.debug(f"No events.tsv found for sub-{subject_id}")
            return pd.DataFrame()
        
        events_file = events_files[0]
        
        try:
            events_df = pd.read_csv(events_file, sep='\t')
            logger.debug(f"Loaded {events_file.name}: {len(events_df)} events")
            return events_df
        except Exception as e:
            logger.error(f"Failed to load {events_file}: {str(e)}")
            return pd.DataFrame()
    
    def load_subject_session(self, subject_id: str, session_id: str = "01", 
                            run_id: int = 1) -> Dict[str, Tuple[np.ndarray, float]]:
        """Load all signal modalities for a subject session run.
        
        Args:
            subject_id: Subject ID
            session_id: Session ID
            run_id: Run number
        
        Returns:
            Dictionary with keys: 'ecg', 'eeg', 'emg', 'mov' 
            Values: (signal_data, sampling_rate) tuples
        """
        signals = {}
        
        for datatype in ['ecg', 'eeg', 'emg', 'mov']:
            try:
                signal_data, fs = self.load_subject_edf(subject_id, session_id, datatype, run_id)
                signals[datatype] = (signal_data, fs)
            except Exception as e:
                logger.warning(f"Could not load {datatype} for sub-{subject_id}: {str(e)}")
        
        # Load events
        events = self.load_events_tsv(subject_id, session_id, run_id)
        signals['events'] = events
        
        return signals
    
    def list_all_recordings(self) -> List[Dict]:
        """List all available subject-session-run combinations.
        
        Returns:
            List of dicts with keys: subject_id, session_id, run_id, num_seizures, events_file
        """
        recordings = []
        
        # Scan all events.tsv files directly to capture all seizures
        for subject_id in self.get_subjects():
            # subject_id is already a string like "001" from get_subjects()
            session_dir = self.dataset_root / f"sub-{subject_id}"
            
            if not session_dir.exists():
                continue
            
            for session_folder in sorted(session_dir.iterdir()):
                if not session_folder.name.startswith('ses-'):
                    continue
                
                session_id = session_folder.name.replace('ses-', '')
                
                # Find all events.tsv files (typically in eeg/ subdirectory)
                eeg_dir = session_folder / 'eeg'
                events_files = []
                
                if eeg_dir.exists():
                    events_files = sorted(eeg_dir.glob("*_events.tsv"))
                
                # Also check other subdirectories as fallback
                if not events_files:
                    events_files = sorted(session_folder.glob("*/*_events.tsv"))
                
                # Process each events file
                for events_file in events_files:
                    # Extract run number from filename
                    # Format: sub-001_ses-01_task-szMonitoring_run-04_events.tsv
                    run_match = [s for s in events_file.name.split('_') if 'run-' in s]
                    if run_match:
                        run_id = int(run_match[0].replace('run-', ''))
                    else:
                        # If no run number, skip or assign default
                        continue
                    
                    # Load and count seizures
                    try:
                        events_df = pd.read_csv(events_file, sep='\t')
                        
                        if not events_df.empty:
                            # SeizeIT2 uses 'eventType' column with 'sz' prefix for seizures
                            if 'eventType' in events_df.columns:
                                num_seizures = len(events_df[events_df['eventType'].str.startswith('sz', na=False)])
                            elif 'trial_type' in events_df.columns:
                                num_seizures = len(events_df[events_df['trial_type'].str.contains('seizure', case=False, na=False)])
                            else:
                                num_seizures = 0
                        else:
                            num_seizures = 0
                        
                        recordings.append({
                            'subject_id': subject_id,
                            'session_id': session_id,
                            'run_id': run_id,
                            'num_seizures': num_seizures,
                            'events_file': str(events_file),
                        })
                        
                    except Exception as e:
                        logger.debug(f"Could not process {events_file}: {str(e)}")
                        continue
        
        logger.info(f"Found {len(recordings)} total recordings with {sum(r['num_seizures'] for r in recordings)} seizures")
        return recordings


class SubjectDataCache:
    """Cache preprocessed subject data to avoid repeated loading."""
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize cache.
        
        Args:
            cache_dir: Directory for storing cache files (optional)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._memory_cache = {}
    
    def get(self, key: str) -> Optional[Dict]:
        """Get cached data."""
        return self._memory_cache.get(key)
    
    def set(self, key: str, data: Dict) -> None:
        """Cache data in memory."""
        self._memory_cache[key] = data
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._memory_cache.clear()


class SeizureDataset:
    """
    PyTorch-compatible dataset for seizure prediction.
    
    Stores (features, countdown_label) pairs and handles:
    - Seizure/non-seizure balancing
    - Temporal windowing
    - Feature normalization per subject
    """
    
    def __init__(self, features: np.ndarray, labels: np.ndarray,
                 subject_ids: Optional[np.ndarray] = None,
                 seizure_ids: Optional[np.ndarray] = None,
                 sample_end_times_s: Optional[np.ndarray] = None,
                 recording_ids: Optional[np.ndarray] = None):
        """Initialize dataset.
        
        Args:
            features: (N, T, F) - N samples, T timesteps, F features
            labels: (N,) - countdown labels in minutes
            subject_ids: (N,) - subject ID for each sample
            seizure_ids: (N,) - seizure ID for each sample
            sample_end_times_s: (N,) - sample end-time in seconds within recording
            recording_ids: (N,) - recording identifier for each sample
        """
        # Avoid unconditional copies for very large real datasets.
        self.features = np.asarray(features, dtype=np.float32)
        self.labels = np.asarray(labels, dtype=np.float32)
        self.subject_ids = subject_ids
        self.seizure_ids = seizure_ids
        self.sample_end_times_s = None if sample_end_times_s is None else np.asarray(sample_end_times_s, dtype=np.float32)
        self.recording_ids = recording_ids
        
        # Compute pre-ictal labels (binary: seizure approaching?)
        self.preictal_labels = (labels >= 0).astype(np.float32)
        
        # Compute sample weights (late predictions more important)
        self.weights = self._compute_weights(labels)

        # Optional memory-light online augmentation (applied in __getitem__).
        self._online_aug_enabled = False
        self._online_aug_probability = 0.0
        self._online_aug_preictal_only = True
        self._online_aug_cfg = None
        self._online_aug_rng = None
        
        logger.info(f"Dataset: {len(self)} samples, {features.shape[1]} timesteps, {features.shape[2]} features")
    
    def _compute_weights(self, labels: np.ndarray) -> np.ndarray:
        """Compute temporal importance weights."""
        if labels.size == 0:
            return np.array([], dtype=np.float32)
        tau = 60.0  # seconds
        weights = np.where(
            labels >= 0,
            np.exp(-np.maximum(labels, 0.0) * 60 / tau),
            0.5
        )
        weights = weights / np.mean(weights)
        return weights.astype(np.float32)
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get single sample.
        
        Returns:
            (features, countdown_label, weight)
        """
        features = self.features[idx]
        label = self.labels[idx]

        if (
            self._online_aug_enabled
            and self._online_aug_rng is not None
            and (label >= 0 or not self._online_aug_preictal_only)
            and float(self._online_aug_rng.random()) < self._online_aug_probability
        ):
            # Copy before augmentation so memory-mapped/cached arrays stay immutable.
            features = self._apply_online_augmentation(np.array(features, dtype=np.float32, copy=True))

        return features, label, self.weights[idx]

    def enable_online_preictal_augmentation(
        self,
        config: DataConfig,
        seed: int = 42,
        probability: float = 0.7,
        preictal_only: bool = True,
    ) -> None:
        """Enable stochastic augmentation during sample fetch.

        This keeps chronology and dataset size unchanged while improving
        training diversity for long-sweep runs.
        """
        self._online_aug_enabled = True
        self._online_aug_cfg = config
        self._online_aug_rng = np.random.default_rng(int(seed))
        self._online_aug_probability = float(np.clip(probability, 0.0, 1.0))
        self._online_aug_preictal_only = bool(preictal_only)

    def _apply_online_augmentation(self, features: np.ndarray) -> np.ndarray:
        """Apply one random augmentation transform to a sample."""
        cfg = self._online_aug_cfg
        rng = self._online_aug_rng
        if cfg is None or rng is None:
            return features.astype(np.float32, copy=False)

        ops = []
        if getattr(cfg, 'aug_time_warp_rates', None):
            ops.append('time_warp')
        if getattr(cfg, 'aug_amplitude_scales', None):
            ops.append('amplitude')
        if getattr(cfg, 'aug_noise_levels', None):
            ops.append('noise')
        if getattr(cfg, 'aug_time_shifts', None):
            ops.append('shift')

        if not ops:
            return features.astype(np.float32, copy=False)

        op = ops[int(rng.integers(0, len(ops)))]
        if op == 'time_warp':
            rate = float(rng.choice(np.asarray(cfg.aug_time_warp_rates, dtype=np.float32)))
            return time_warp(features, rate)
        if op == 'amplitude':
            scale = float(rng.choice(np.asarray(cfg.aug_amplitude_scales, dtype=np.float32)))
            return amplitude_scale(features, scale)
        if op == 'noise':
            noise_level = float(rng.choice(np.asarray(cfg.aug_noise_levels, dtype=np.float32)))
            noise = rng.normal(0.0, noise_level, features.shape)
            return (features + noise).astype(np.float32)

        shift = int(rng.choice(np.asarray(cfg.aug_time_shifts, dtype=np.int32)))
        return time_shift(features, shift)
    
    def get_subject_masks(self) -> Dict[str, np.ndarray]:
        """Get boolean masks for each subject."""
        if self.subject_ids is None:
            return {}
        
        masks = {}
        for subject_id in np.unique(self.subject_ids):
            masks[str(subject_id)] = self.subject_ids == subject_id
        
        return masks
    
    def get_seizure_masks(self) -> Dict[str, np.ndarray]:
        """Get boolean masks for each seizure."""
        if self.seizure_ids is None:
            return {}
        
        masks = {}
        for seizure_id in np.unique(self.seizure_ids):
            masks[str(seizure_id)] = self.seizure_ids == seizure_id
        
        return masks
    
    @property
    def class_distribution(self) -> Dict[str, int]:
        """Get count of pre-ictal vs inter-ictal samples."""
        return {
            'preictal': int(np.sum(self.preictal_labels)),
            'interictal': int(np.sum(1 - self.preictal_labels)),
        }


class TemporalRingBufferDataset:
    """Sliding-window view over a temporally-sorted SeizureDataset.

    Simulates deployment conditions: the model only has access to the
    most recent ``ring_buffer_size`` samples, older history is evicted.
    This prevents the model from implicitly memorising cross-session
    population statistics and keeps memory proportional to the window
    rather than the full recording corpus.

    Usage pattern (mirrors a ring-buffer / circular queue):

        ds_sorted = SeizureDataset(features, labels, weights)   # time-sorted
        ring = TemporalRingBufferDataset(ds_sorted, ring_buffer_size=2048)
        for epoch in range(epochs):
            ring.advance_to_epoch(epoch, step=512)
            loader = DataLoader(ring, ...)
            for batch in loader:
                ...   # train on current window

    Args:
        dataset:          A SeizureDataset whose samples are ordered in time.
        ring_buffer_size: Number of most-recent samples to expose at a time.
                          Set to 0 (or len(dataset)) to disable windowing.
        initial_offset:   Starting index (default 0 = beginning of recording).
    """

    def __init__(
        self,
        dataset: 'SeizureDataset',
        ring_buffer_size: int = 0,
        initial_offset: int = 0,
    ):
        self._dataset = dataset
        total = len(dataset)
        if ring_buffer_size <= 0 or ring_buffer_size >= total:
            self._buf_size = total
        else:
            self._buf_size = int(ring_buffer_size)
        self._offset = max(0, min(int(initial_offset), total - self._buf_size))

    # ── Window control ────────────────────────────────────────────────────────

    def advance_to_epoch(self, epoch: int, step: int = 0) -> None:
        """Slide the window forward by ``step * epoch`` samples.

        Clamps at the end of the dataset so the last window stays fully
        in bounds.  When ``step == 0`` the window does not move (all
        epochs see the same slice — useful for small datasets).
        """
        if step <= 0:
            return
        max_offset = max(0, len(self._dataset) - self._buf_size)
        self._offset = min(max_offset, int(epoch) * int(step))

    def set_offset(self, offset: int) -> None:
        """Manually set the starting sample index of the current window."""
        max_offset = max(0, len(self._dataset) - self._buf_size)
        self._offset = max(0, min(int(offset), max_offset))

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return self._buf_size

    def __getitem__(self, idx: int):
        """Return the sample at *local* index ``idx`` within the current window."""
        if idx < 0 or idx >= self._buf_size:
            raise IndexError(f"Index {idx} out of ring buffer window [{self._offset}, {self._offset + self._buf_size})")
        return self._dataset[self._offset + idx]

    # ── Passthrough metadata ──────────────────────────────────────────────────

    @property
    def preictal_labels(self) -> np.ndarray:
        """Boolean/float pre-ictal labels for samples in the current window."""
        return self._dataset.preictal_labels[self._offset: self._offset + self._buf_size]

    @property
    def labels(self) -> np.ndarray:
        """Raw countdown labels for samples in the current window."""
        return self._dataset.labels[self._offset: self._offset + self._buf_size]

    @property
    def class_distribution(self) -> Dict[str, int]:
        lbl = self.preictal_labels
        return {
            'preictal': int(np.sum(lbl)),
            'interictal': int(len(lbl) - np.sum(lbl)),
        }

    def enable_online_preictal_augmentation(self, *args, **kwargs) -> None:
        """Delegate to underlying dataset."""
        self._dataset.enable_online_preictal_augmentation(*args, **kwargs)


class WearableDeviceDataLoader:
    """
    Loads wearable device data from Oregon Health & Science University dataset.
    
    Features:
    - Reads multi-stream CSV files (PPG, accelerometer, EDA, temperature)
    - Parses master Excel database for seizure annotations
    - Creates preictal/interictal labels based on seizure times
    - Handles epoch-based timestamps from wearable devices
    """
    
    def __init__(self, config: DataConfig, 
                 dataset_root: Optional[str] = None,
                 master_db_path: Optional[str] = None):
        """Initialize wearable device loader.
        
        Args:
            config: DataConfig with preprocessing parameters
            dataset_root: Path to WearableDevice-Oregon dataset root
            master_db_path: Path to master seizure annotations Excel file
        """
        self.config = config
        
        # Set default paths — prefer WEARABLE_DATASET_ROOT env var over hardcoded fallback
        if dataset_root is None:
            dataset_root = os.environ.get(
                'WEARABLE_DATASET_ROOT',
                str(getattr(config, 'dataset_root', 'data/wearable'))
            )
        if master_db_path is None:
            master_db_path = os.path.join(dataset_root, "Wearable Seizre Detection Master Database.xlsx")
        
        self.dataset_root = Path(dataset_root)
        self.master_db_path = Path(master_db_path)
        
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Dataset not found: {self.dataset_root}")
        if not self.master_db_path.exists():
            raise FileNotFoundError(f"Master database not found: {self.master_db_path}")
        
        # Load seizure annotations
        self.seizure_db = self._load_seizure_database()
        self._recording_inventory = self._build_recording_inventory()
        logger.info(f"Loaded seizure database with {len(self.seizure_db)} recordings")
    
    def _load_seizure_database(self) -> pd.DataFrame:
        """Load and parse the master Excel database.
        
        Returns:
            DataFrame with seizure annotations and recording metadata
        """
        try:
            df = pd.read_excel(self.master_db_path)
            
            # Rename columns for consistency
            df.columns = [col.strip().lower().replace(' ', '_').replace('#', 'num') 
                         for col in df.columns]
            
            logger.info(f"Master DB columns: {list(df.columns)}")
            
            # Track recordings with seizures
            seizure_recordings = df[df['seizure_(1_or_0)'] == 1]
            logger.info(f"Found {len(seizure_recordings)} recordings with seizures out of {len(df)}")
            
            return df
        except Exception as e:
            logger.error(f"Failed to load master database: {e}")
            raise
    
    def _parse_summary_datetime(self, value: str) -> Optional[pd.Timestamp]:
        """Parse summary timestamps like '11:38:37 2022-10-10'."""
        if value is None or (isinstance(value, float) and pd.isna(value)):
            return None
        text = str(value).strip()
        if not text:
            return None
        for fmt in ("%H:%M:%S %Y-%m-%d", "%H:%M %Y-%m-%d", "%I:%M:%S %Y-%m-%d", "%I:%M %Y-%m-%d"):
            try:
                return pd.Timestamp(pd.to_datetime(text, format=fmt))
            except Exception:
                continue
        try:
            return pd.Timestamp(pd.to_datetime(text))
        except Exception:
            return None

    def _resolve_csv_dir(self, recording_dir: Path) -> Optional[Path]:
        """Return CSV/CVS directory for a wearable recording root."""
        for dirname in ("CSV", "CVS"):
            candidate = recording_dir / dirname
            if candidate.exists() and candidate.is_dir():
                return candidate
        return None

    def _build_recording_inventory(self) -> List[Dict]:
        """Scan wearable dataset and collect per-recording start/end timestamps from summary files."""
        inventory: List[Dict] = []
        for summary_file in sorted(self.dataset_root.rglob('*Summary.csv')):
            try:
                rows = []
                with open(summary_file, newline='') as handle:
                    reader = csv.reader(handle)
                    for idx, row in enumerate(reader):
                        rows.append(row)
                        if idx >= 20:
                            break

                summary_map = {}
                for row in rows:
                    if not row:
                        continue
                    key = str(row[0]).strip().lower()
                    value = str(row[1]).strip() if len(row) > 1 else ""
                    summary_map[key] = value

                start_dt = self._parse_summary_datetime(summary_map.get('ppg start date and time'))
                end_dt = self._parse_summary_datetime(summary_map.get('ppg end date and time'))

                parent = summary_file.parent
                recording_dir = parent.parent if parent.name.upper() in ('CSV', 'CVS') else parent
                csv_dir = self._resolve_csv_dir(recording_dir)
                if csv_dir is None:
                    continue

                inventory.append({
                    'recording_dir': recording_dir,
                    'csv_dir': csv_dir,
                    'summary_file': summary_file,
                    'start_datetime': start_dt,
                    'end_datetime': end_dt,
                })
            except Exception as e:
                logger.debug(f"Failed to index summary file {summary_file}: {e}")

        logger.info(f"Indexed {len(inventory)} wearable recording summaries")
        return inventory

    def _find_recording_directory(self, subject_num: int, watch_num: int,
                                  start_datetime: pd.Timestamp,
                                  end_datetime: Optional[pd.Timestamp] = None) -> Optional[Path]:
        """Find the wearable recording directory best matching annotation timestamps."""
        if not self._recording_inventory:
            return None

        best_dir = None
        best_score = None
        for item in self._recording_inventory:
            inv_start = item.get('start_datetime')
            inv_end = item.get('end_datetime')
            if inv_start is None:
                continue

            start_delta = abs((inv_start - start_datetime).total_seconds())
            end_delta = 0.0
            if end_datetime is not None and inv_end is not None:
                end_delta = abs((inv_end - end_datetime).total_seconds())

            score = start_delta + end_delta
            if best_score is None or score < best_score:
                best_score = score
                best_dir = item['recording_dir']

        if best_dir is not None:
            logger.debug(
                "Matched wearable recording for S%s/W%s -> %s (score=%.1fs)",
                subject_num,
                watch_num,
                best_dir,
                0.0 if best_score is None else best_score,
            )
        return best_dir
    
    def load_signals_from_recording(self, recording_dir: Path, 
                                   signal_types: List[str] = None) -> Dict[str, Tuple[np.ndarray, float]]:
        """Load multiple signal streams from a wearable device recording.
        
        Args:
            recording_dir: Path to recording directory
            signal_types: List of signal types to load (ppg, adxl, eda, temperature, pedometer, agc)
        
        Returns:
            Dictionary with keys: signal type
            Values: (signal_array, sampling_rate) tuples
        """
        if signal_types is None:
            signal_types = ['ppg', 'adxl', 'eda', 'temperature']
        
        signals = {}
        csv_dir = self._resolve_csv_dir(recording_dir)
        
        if csv_dir is None:
            raise FileNotFoundError(f"CSV/CVS directory not found: {recording_dir}")
        
        # Map signal types to expected CSV patterns (with wildcards for device IDs)
        signal_patterns = {
            'ppg': '*PPGAppStream.csv',
            'adxl': '*ADPD_ADXL_CombinedData_CSV.csv',
            'eda': '*EDAAppStream.csv',
            'temperature': '*TemperatureStream.csv',
            'pedometer': '*PedometerAppStream.csv',
            'agc': '*AGCAppStream.csv',
            'sqi': '*SQIStream.csv',
        }
        
        for signal_type in signal_types:
            pattern = signal_patterns.get(signal_type, f"*{signal_type}*.csv")
            csv_files = list(csv_dir.glob(pattern))
            
            if not csv_files:
                logger.debug(f"No {signal_type} signal found in {csv_dir}")
                continue
            
            csv_file = csv_files[0]
            logger.debug(f"Loading {signal_type} from {csv_file.name}")
            
            try:
                signal_data, fs = self._load_csv_signal(csv_file, signal_type)
                signals[signal_type] = (signal_data, fs)
            except Exception as e:
                logger.error(f"Failed to load {signal_type} signal: {e}")
        
        return signals
    
    def _load_csv_signal(self, csv_file: Path, signal_type: str) -> Tuple[np.ndarray, float]:
        """Load a single CSV signal file.
        
        Args:
            csv_file: Path to CSV file
            signal_type: Type of signal (ppg, adxl, eda, etc.)
        
        Returns:
            Tuple of (signal_array, sampling_rate_hz)
        """
        # Read CSV with header parsing; use robust parser for malformed wearable exports.
        df = pd.read_csv(csv_file, skiprows=2, engine='python', on_bad_lines='skip')
        
        # Extract sampling rate from header if available
        with open(csv_file) as f:
            lines = f.readlines()
            # Look for ODR (Output Data Rate) in header
            fs = 50.0  # Default to 50Hz
            for line in lines[:10]:
                if 'ODR' in line.upper() or 'hz' in line.lower():
                    try:
                        parts = line.split(',')
                        for i, part in enumerate(parts):
                            if 'hz' in part.lower():
                                fs = float(parts[i-1].strip())
                                break
                    except:
                        pass

        # Prefer empirical timestamp-based fs when an epoch delta column is present.
        # Many wearable app streams report ODR in the header, but exported rows are
        # downsampled summaries (e.g., ~1 Hz heart-rate stream instead of raw 50 Hz PPG).
        ts_col = None
        for candidate in df.columns:
            candidate_lower = str(candidate).strip().lower()
            if 'epoch delta ts' in candidate_lower or candidate_lower.startswith('epoch'):
                ts_col = candidate
                break

        if ts_col is not None:
            ts_values = pd.to_numeric(df[ts_col], errors='coerce').dropna().values
            if len(ts_values) >= 3:
                diffs = np.diff(ts_values)
                diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
                if len(diffs) > 0:
                    median_dt = float(np.median(diffs))
                    if median_dt > 5.0:  # likely milliseconds
                        inferred_fs = 1000.0 / median_dt
                    else:  # likely seconds
                        inferred_fs = 1.0 / median_dt
                    if np.isfinite(inferred_fs) and inferred_fs > 0:
                        fs = float(inferred_fs)
        
        # Parse columns based on signal type
        if signal_type == 'ppg':
            # PPG columns: Epoch Delta TS, LibState, HR, Confidence, Debug Info
            # Most useful columns are HR and LibState
            if 'HR' in df.columns:
                signal = pd.to_numeric(df['HR'], errors='coerce').fillna(0.0).values.astype(np.float32)
            else:
                signal = pd.to_numeric(df[df.columns[2]], errors='coerce').fillna(0.0).values.astype(np.float32)  # Default to 3rd column
                
        elif signal_type == 'adxl':
            # ADPD/ADXL combined: check for acceleration columns
            # Try to find X, Y, Z columns
            acc_cols = [col for col in df.columns if any(x in col.upper() for x in ['X', 'Y', 'Z', 'ACC'])]
            if acc_cols:
                signal = df[acc_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)
            else:
                signal = df[df.columns[3:]].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)  # Skip timestamp columns
                
        elif signal_type == 'eda':
            # EDA (Electrodermal Activity) column
            if 'EDA' in df.columns:
                signal = pd.to_numeric(df['EDA'], errors='coerce').fillna(0.0).values.astype(np.float32)
            else:
                signal = pd.to_numeric(df[df.columns[2]], errors='coerce').fillna(0.0).values.astype(np.float32)
                
        elif signal_type == 'temperature':
            # Temperature column
            if 'Temp' in df.columns or 'Temperature' in df.columns:
                temp_col = [col for col in df.columns if 'temp' in col.lower()][0]
                signal = pd.to_numeric(df[temp_col], errors='coerce').fillna(0.0).values.astype(np.float32)
            else:
                signal = pd.to_numeric(df[df.columns[2]], errors='coerce').fillna(0.0).values.astype(np.float32)
        else:
            # Default: use all numeric columns except timestamp
            numeric_cols = df.select_dtypes(include=[np.number]).columns[1:]
            signal = df[numeric_cols].apply(pd.to_numeric, errors='coerce').fillna(0.0).values.astype(np.float32)

        signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        
        logger.debug(f"Loaded {signal_type}: shape={signal.shape}")
        return signal, fs
    
    def _parse_time_string(self, time_obj) -> Tuple[int, int, int]:
        """Parse time from string or time object.
        
        Args:
            time_obj: Time string (HH:MM:SS or HH:MM) or datetime.time object
        
        Returns:
            Tuple of (hour, minute, second)
        """
        if time_obj is None:
            return 0, 0, 0
        
        # Handle time objects
        if hasattr(time_obj, 'hour'):
            return int(time_obj.hour), int(time_obj.minute), int(time_obj.second)
        
        # Handle strings
        if not isinstance(time_obj, str):
            return 0, 0, 0
        
        try:
            parts = time_obj.strip().split(':')
            hour = int(parts[0]) if len(parts) > 0 else 0
            minute = int(parts[1]) if len(parts) > 1 else 0
            second = int(parts[2]) if len(parts) > 2 else 0
            return hour, minute, second
        except:
            return 0, 0, 0
    
    def get_all_recordings(self) -> List[Dict]:
        """Get list of all available recordings with metadata.
        
        Returns:
            List of dicts with: subject_id, watch_id, start_time, end_time, 
                               has_seizure, num_seizures, seizure_times, recording_dir
        """
        recordings = []
        
        for idx, row in self.seizure_db.iterrows():
            subject_num = row['subject_num']
            watch_num = row['watch_num']
            start_date = pd.Timestamp(row['start_date']).replace(hour=0, minute=0, second=0)
            start_time = row['start_time']
            
            # Combine date and time
            hour, minute, second = self._parse_time_string(start_time)
            start_datetime = start_date.replace(hour=hour, minute=minute, second=second)
            
            end_date = pd.Timestamp(row['stop_date']).replace(hour=0, minute=0, second=0)
            end_time = row['stop_time']
            
            hour, minute, second = self._parse_time_string(end_time)
            end_datetime = end_date.replace(hour=hour, minute=minute, second=second)
            
            # Parse seizure information
            has_seizure = row['seizure_(1_or_0)'] == 1
            seizure_times = []
            
            if has_seizure:
                # Parse up to multiple seizures
                for seizure_num in range(1, 3):  # Support up to 2 seizures
                    # Note: Column normalization creates inconsistent names:
                    # 'Seizure # 1 Date' -> 'seizure_num_1_date' (underscore after num)
                    # 'Seizure #1 Time' -> 'seizure_num1_time' (no underscore)
                    date_col = f'seizure_num_{seizure_num}_date'
                    time_col = f'seizure_num{seizure_num}_time'
                    
                    if date_col not in row.index:
                        continue
                    
                    date_value = row[date_col]
                    if pd.isna(date_value):
                        continue
                    
                    seizure_date = pd.Timestamp(date_value)
                    
                    if time_col not in row.index:
                        continue
                    
                    seizure_time = row[time_col]
                    if pd.isna(seizure_time):
                        continue
                    
                    # Parse time (can be string or time object)
                    hour, minute, second = self._parse_time_string(seizure_time)
                    seizure_datetime = seizure_date.replace(hour=hour, minute=minute, second=second)
                    seizure_times.append(seizure_datetime)
            
            # Find recording directory
            recording_dir = self._find_recording_directory(subject_num, watch_num, start_datetime, end_datetime)
            
            recording = {
                'subject_id': f"S{subject_num}",
                'watch_id': f"W{watch_num}",
                'subject_num': subject_num,
                'watch_num': watch_num,
                'start_datetime': start_datetime,
                'end_datetime': end_datetime,
                'has_seizure': has_seizure,
                'num_seizures': len(seizure_times),
                'seizure_times': seizure_times,
                'recording_dir': recording_dir,
            }
            
            recordings.append(recording)
        
        logger.info(f"Found {len(recordings)} recordings, {sum(1 for r in recordings if r['has_seizure'])} with seizures")
        return recordings
    
    def extract_features_from_recording(self, recording: Dict, 
                                       signal_types: List[str] = None,
                                       window_s: float = 60.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract windowed features from a recording.
        
        Args:
            recording: Recording dict from get_all_recordings()
            signal_types: Signal types to use
            window_s: Window size in seconds
        
        Returns:
            Tuple of (features, labels, sample_times_s)
            - features: (N, T, F) - N samples, T timesteps, F features
            - labels: (N,) - -1 (interictal) or minutes-before-seizure (preictal)
            - sample_times_s: (N,) - Sample time in seconds since recording start
        """
        if recording['recording_dir'] is None:
            logger.warning(f"Skipping {recording['subject_id']}/{recording['watch_id']}: no directory found")
            return None, None, None
        
        if signal_types is None:
            signal_types = ['ppg', 'adxl', 'eda', 'temperature']
        
        # Load signals
        signals = self.load_signals_from_recording(recording['recording_dir'], signal_types)
        
        if not signals:
            logger.warning(f"No signals loaded for {recording['subject_id']}/{recording['watch_id']}")
            return None, None, None
        
        # Use a common time base at the slowest valid sampling rate, but keep
        # the *longest* available duration so shorter auxiliary streams do not
        # truncate seizure windows near the end of a recording.
        valid_fs = [float(s[1]) for s in signals.values() if np.isfinite(s[1]) and float(s[1]) > 0]
        if not valid_fs:
            logger.warning(
                f"No valid sampling rates for {recording.get('subject_id', '?')}/{recording.get('watch_id', '?')}"
            )
            return None, None, None

        min_fs = min(valid_fs)

        resampled = {}
        target_len = 0
        for sig_type, (sig_data, fs) in signals.items():
            arr = np.asarray(sig_data)
            if arr.ndim > 1:
                arr = np.nanmean(arr, axis=1)
            arr = np.ravel(arr).astype(np.float32)
            if arr.size <= 1:
                continue

            fs = float(fs)
            if not np.isfinite(fs) or fs <= 0:
                fs = min_fs

            # Resample each stream to min_fs using linear interpolation on time.
            if abs(fs - min_fs) > 1e-6:
                duration_s = (arr.size - 1) / fs
                new_len = max(2, int(round(duration_s * min_fs)) + 1)
                x_old = np.linspace(0.0, duration_s, num=arr.size, endpoint=True)
                x_new = np.linspace(0.0, duration_s, num=new_len, endpoint=True)
                arr = np.interp(x_new, x_old, arr).astype(np.float32)

            resampled[sig_type] = arr
            target_len = max(target_len, arr.size)

        if not resampled or target_len <= 1:
            return None, None, None
        
        # Build feature matrix in the exact requested signal order. Missing
        # streams are represented as zeros to keep channel indexing stable.
        if not resampled:
            return None, None, None

        feature_arrays = []
        for sig in signal_types:
            if sig in resampled:
                arr = resampled[sig]
                if arr.size < target_len:
                    pad_width = target_len - arr.size
                    arr = np.pad(arr, (0, pad_width), mode='edge')
                else:
                    arr = arr[:target_len]
                feature_arrays.append(arr.astype(np.float32, copy=False))
            else:
                feature_arrays.append(np.zeros(target_len, dtype=np.float32))

        all_features = np.column_stack(feature_arrays)  # (T, F)
        
        # Create windows
        window_samples = max(1, int(round(window_s * min_fs)))
        step_samples = max(1, int(round(self.config.feature_step_s * min_fs)))
        
        num_windows = (all_features.shape[0] - window_samples) // step_samples + 1
        windows = []
        labels = []
        sample_times = []
        
        for i in range(num_windows):
            start_idx = i * step_samples
            end_idx = start_idx + window_samples
            
            if end_idx > all_features.shape[0]:
                break
            
            window = all_features[start_idx:end_idx]
            windows.append(window)
            
            # Calculate time of window center
            center_idx = (start_idx + end_idx) // 2
            center_time_s = center_idx / min_fs
            sample_times.append(center_time_s)
            
            # Assign label based on seizure proximity
            label = self._compute_seizure_label(recording, center_time_s)
            labels.append(label)
        
        if windows:
            features = np.array(windows, dtype=np.float32)  # (N, T, F)
            labels = np.array(labels, dtype=np.float32)  # (N,)
            sample_times = np.array(sample_times, dtype=np.float32)  # (N,)

            # Drop "flat" windows that carry no meaningful signal (e.g.,
            # segments where ADPD/ADXL/PPG streams are stuck at a single
            # value for the entire window). These arise from corrupted
            # exports and otherwise dominate the visualization as constant
            # lines.
            per_window_std = features.reshape(features.shape[0], -1).std(axis=1)
            flat_mask = per_window_std <= 1e-6
            if np.any(flat_mask):
                keep_mask = ~flat_mask
                kept = int(np.sum(keep_mask))
                dropped = int(np.sum(flat_mask))
                if kept == 0:
                    logger.warning(
                        "Wearable recording %s/%s: all %d windows are flat; skipping recording",
                        recording.get('subject_id', '?'),
                        recording.get('watch_id', '?'),
                        dropped,
                    )
                    return None, None, None

                features = features[keep_mask]
                labels = labels[keep_mask]
                sample_times = sample_times[keep_mask]
                logger.info(
                    "Wearable recording %s/%s: dropped %d flat windows, kept %d",
                    recording.get('subject_id', '?'),
                    recording.get('watch_id', '?'),
                    dropped,
                    kept,
                )

            return features, labels, sample_times
        
        return None, None, None
    
    def _compute_seizure_label(self, recording: Dict, time_s: float) -> float:
        """Compute label for a sample based on proximity to seizures.
        
        Args:
            recording: Recording dict
            time_s: Time in seconds since recording start
        
        Returns:
            -1 for interictal, or minutes before seizure onset for preictal
        """
        if not recording['has_seizure']:
            return -1.0
        
        # Convert recording times to seconds since start
        start_dt = recording['start_datetime']
        pre_ictal_window_s = self.config.pre_ictal_window_s
        
        for seizure_dt in recording['seizure_times']:
            # Time of seizure relative to recording start
            seizure_time_s = (seizure_dt - start_dt).total_seconds()
            
            # Check if sample is in preictal window
            time_to_seizure_s = seizure_time_s - time_s
            
            if 0 <= time_to_seizure_s <= pre_ictal_window_s:
                # Return countdown in minutes
                return time_to_seizure_s / 60.0
        
        return -1.0


if __name__ == "__main__":
    # Add project root to path for standalone testing
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    logging.basicConfig(level=logging.INFO)
    
    # Test data loader
    from config.config import DEFAULT_CONFIG
    
    loader = BIDSDataLoader(DEFAULT_CONFIG.data)
    subjects = loader.get_subjects()
    print(f"Found {len(subjects)} subjects")
    
    recordings = loader.list_all_recordings()
    print(f"Total recordings: {len(recordings)}")
    print(f"Total seizures: {sum(r['num_seizures'] for r in recordings)}")
