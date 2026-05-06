# WearableDeviceDataLoader Integration Guide

## Overview

The `WearableDeviceDataLoader` class enables integration of wearable device seizure data from Oregon Health & Science University into the Epilepsee-AI project. This loader handles:

- **Multi-stream physiological signals** (PPG, accelerometer, EDA, temperature)
- **Seizure annotations** from Excel master database
- **Temporal labeling** with countdown to seizure onset
- **Preprocessing** and feature extraction from wearable CSV data

## Key Differences from BIDS Loader

| Aspect | BIDS (ds005873) | Wearable (Oregon) |
|--------|---|---|
| **Signal Format** | EDF files | CSV streams |
| **Sensor Types** | ECG, EEG, EMG, accelerometer | PPG, accelerometer, EDA, temperature |
| **Wear Location** | Chest (ECG), head (EEG) | Wrist (smartwatch) |
| **Annotation Format** | events.tsv files | Excel master database |
| **Device Variability** | Consistent recording setup | Different devices per subject |
| **Seizure Timing** | Precise event markers | Absolute timestamps |

## Dataset Structure

```
WearableDevice-Oregon/
├── #1/, #2/, #3/, ...        # Numbered dataset folders
│   └── Recording folders
│       ├── CSV/               # Signal CSV files
│       │   ├── *PPGAppStream.csv
│       │   ├── *ADPD_ADXL_CombinedData_CSV.csv
│       │   ├── *EDAAppStream.csv
│       │   ├── *TemperatureStream.csv
│       │   └── ... (other signals)
│       └── JSON/              # Raw device data
├── Wearable Seizure Detection Master Database.xlsx  # Seizure annotations
└── AWT Output Guide.pptx      # Documentation
```

## Usage Examples

### Basic Initialization

```python
from config.config import DEFAULT_CONFIG
from src.data_loader import WearableDeviceDataLoader

# Initialize with default paths
loader = WearableDeviceDataLoader(DEFAULT_CONFIG.data)

# Or specify custom paths
loader = WearableDeviceDataLoader(
    DEFAULT_CONFIG.data,
    dataset_root="/path/to/WearableDevice-Oregon",
    master_db_path="/path/to/master_database.xlsx"
)
```

### Get All Available Recordings

```python
# Retrieve metadata for all recordings
recordings = loader.get_all_recordings()

# Filter seizure recordings
seizure_recordings = [r for r in recordings if r['has_seizure']]

print(f"Total recordings: {len(recordings)}")
print(f"Recordings with seizures: {len(seizure_recordings)}")

# Inspect a recording
rec = seizure_recordings[0]
print(f"Subject: {rec['subject_id']}, Watch: {rec['watch_id']}")
print(f"Duration: {rec['start_datetime']} to {rec['end_datetime']}")
print(f"Seizure times: {rec['seizure_times']}")
```

### Load Signals from a Recording

```python
# Load available signals from a recording
signals = loader.load_signals_from_recording(
    rec['recording_dir'],
    signal_types=['ppg', 'eda', 'temperature']  # Available: ppg, adxl, eda, temperature, agc, sqi, pedometer
)

# Signals returned as dict with (signal_array, sampling_rate) tuples
for signal_type, (signal_data, fs) in signals.items():
    print(f"{signal_type}: shape={signal_data.shape}, fs={fs} Hz")
```

### Extract Temporal Features

```python
# Extract windowed features with seizure labels
features, labels, sample_times = loader.extract_features_from_recording(
    rec,
    signal_types=['ppg', 'eda'],
    window_s=60.0  # 60-second windows
)

# Features: (num_samples, timesteps, num_features)
# Labels: -1 for interictal, >0 for minutes-to-seizure
# sample_times: time of each sample (seconds since recording start)

print(f"Features shape: {features.shape}")
print(f"Preictal samples: {(labels >= 0).sum()}")
print(f"Interictal samples: {(labels < 0).sum()}")
```

### Build PyTorch Dataset

```python
import numpy as np
from src.data_loader import SeizureDataset

# Collect features from all recordings
all_features = []
all_labels = []
all_subject_ids = []

for rec in seizure_recordings[: 3]:  # Use first 3 seizure recordings
    if rec['recording_dir'] is None:
        continue
    
    features, labels, _ = loader.extract_features_from_recording(rec)
    if features is not None:
        all_features.append(features)
        all_labels.append(labels)
        all_subject_ids.extend([rec['subject_id']] * len(labels))

# Stack and create dataset
if all_features:
    features = np.concatenate(all_features)  # (N, T, F)
    labels = np.concatenate(all_labels)      # (N,)
    subject_ids = np.array(all_subject_ids)  # (N,)
    
    dataset = SeizureDataset(
        features=features,
        labels=labels,
        subject_ids=subject_ids
    )
    
    print(f"Dataset: {len(dataset)} samples")
    print(f"Class distribution: {dataset.class_distribution}")
```

## Signal Types

The wearable devices record multiple physiological signals:

### PPG (Photoplethysmography)
- **Source**: RGB LED sensors on watch
- **Channels**: HR (heart rate), confidence metrics
- **Sampling Rate**: 50 Hz
- **Utility**: Heart rate variability, pulse morphology for seizure detection

### ADXL (Accelerometer)
- **Source**: 3-axis accelerometer in watch
- **Channels**: X, Y, Z acceleration + magnitude
- **Sampling Rate**: ~50 Hz
- **Utility**: Movement detection, seizure motor activity

### EDA (Electrodermal Activity)
- **Source**: Electrodes on watch surface
- **Channels**: Skin conductance level/response
- **Sampling Rate**: ~15 Hz
- **Utility**: Autonomic nervous system activity, stress response

### Temperature
- **Source**: Temperature sensor in watch
- **Sampling Rate**: ~1 Hz
- **Utility**: Thermoregulation changes during seizures

### Others
- **AGC**: Automatic gain control metrics
- **SQI**: Signal quality index
- **Pedometer**: Step count and activity level

## Label Format

Labels indicate relationship to seizure onset:

- **`label = -1`**: Interictal period (not in preictal window)
- **`label ≥ 0`**: Preictal period (minutes until seizure)
- **`label = 5.0`**: Sample is ~5 minutes before seizure
- **`label = 0.5`**: Sample is ~30 seconds before seizure

The preictal window is configurable via `DataConfig.pre_ictal_window_s` (default: 600 seconds = 10 minutes).

## Configuration

Configure data loading in `config/config.py`:

```python
@dataclass
class DataConfig:
    # Wearable device paths
    dataset_root: str = "$WEARABLE_DATASET_ROOT"
    
    # Feature extraction
    feature_window_s: float = 60.0   # 60-second windows
    feature_step_s: float = 1.0      # 1-second stride (overlap)
    
    # Seizure labeling
    pre_ictal_window_s: float = 600.0   # 10-minute preictal window
    
    # Augmentation (for class imbalance)
    augment_preictal: bool = True
    augmentation_factor: int = 15
```

## Implementation Notes

### Time Parsing
The Excel master database stores times in mixed formats (strings and time objects). The loader handles both:
- String format: "HH:MM:SS" or "HH:MM"
- Python time objects: `datetime.time(8, 30, 0)`

### Recording Matching
Since the device IDs don't directly map to subject/watch numbers, the loader uses:
1. Date-based matching (looks for recording folders with matching dates)
2. Fallback to first available recording (assumes chronological ordering)

### CSV Column Parsing
Different CSV files have different formats (variable headers, inconsistent fields). The loader:
- Skips metadata header rows (first 2-3 lines)
- Extracts columns by name (flexible matching)
- Handles missing columns gracefully

### Signal Resampling
Signals at different sampling rates are automatically:
1. Resampled to lowest common rate
2. Truncated to common length
3. Stacked into feature vectors

## Known Limitations & Future Improvements

### Current Limitations
1. **No automatic subject-to-device mapping**: Manual review needed for precise device assignment
2. **ADPD CSV parsing errors**: Some combined data files have malformed headers
3. **No event markers**: Unlike BIDS format, no precise event onset times within recordings
4. **Limited metadata**: No demographic info, medication status, etc. from wearables

### Future Enhancements
1. **Robust CSV parsing**: Handle variable header formats more robustly
2. **Device synchronization**: Align multiple devices if subjects wore multiple watches
3. **Signal quality filtering**: Exclude low-quality segments based on SQI
4. **Circadian modeling**: Account for time-of-day effects on seizure risk
5. **Multimodal fusion**: Combine wearable signals with concurrent EEG (if available)

## Troubleshooting

### No recordings found
- Verify `dataset_root` points to correct folder
- Check that CSV subdirectories exist and contain signal files
- Ensure master database Excel file is accessible

### Seizure times not parsing
- Check Excel file for consistent date/time formatting
- Verify column name normalization in debug logs
- Some rows may have NaN seizure times (interictal recordings)

### No signals loaded
- Confirm CSV files match expected naming patterns (*PPGAppStream.csv, etc.)
- Check that data file isn't corrupted or truncated
- See logs for specific parsing errors

### Out of memory errors
- Reduce `feature_window_s` to use shorter windows
- Process recordings one at a time instead of batching all
- Use `signal_types=['ppg']` to load only essential signals

## References

- **Dataset**: SeizeIT2 Wearable Seizure Detection Challenge (Oregon Health & Science University)
- **Device**: Empatica E4 wearable sensors (smartwatch)
- **Paper**: [Details on device and methodology]

## See Also

- `BIDSDataLoader`: For EDF-based seizure datasets
- `SeizureDataset`: PyTorch dataset wrapper
- `preprocessing.py`: Signal preprocessing utilities
- `feature_extraction.py`: HRV, spectral, and stability features
