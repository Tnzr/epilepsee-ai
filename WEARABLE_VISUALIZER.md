# Wearable Device Dataset Visualizer

A comprehensive tool for visualizing multi-signal wearable device recordings with automatic seizure event detection and contextual time windows.

## Features

✅ **Multi-Signal Support**  
- PPG (Photoplethysmography/Heart Rate)
- ADPD/ADXL (Accelerometer - acceleration magnitude)
- EDA (Electrodermal Activity)
- Temperature
- AGC (Automatic Gain Control)
- Any other available CSV signal streams

✅ **Seizure Event Visualization**  
- Automatic seizure time detection from master database
- Configurable pre-seizure and post-seizure time windows
- Color-coded seizure event markers
- Pre-seizure warning window highlighting (30-60s before onset)

✅ **Signal Quality Metrics**  
- Per-signal statistics (mean, std dev, min, max)
- Automatic sampling rate detection from timestamp deltas
- Multi-channel signal handling (e.g., XYZ accelerometer → magnitude)
- Robust CSV parsing for malformed wearable exports

✅ **Flexible Filtering**  
- Patient/subject number selection
- Single seizure or all seizures per recording
- Custom time window selection
- Recording-level filtering

## Installation

The visualizer uses the existing `WearableDeviceDataLoader` from the Epilepsee-AI project. No additional installation needed beyond the project dependencies.

Required dependencies (already in environment.yml):
- pandas
- numpy
- matplotlib

## Usage

### Quick Start (Default Patient #2)

```bash
# View first seizure for patient #2 (3 min before, 1.5 min after)
make visualize-wearable

# Extended view (5 min before, 2 min after)
make visualize-wearable-multi

# All seizures for patient #2
make visualize-wearable-all
```

### Command Line

```bash
# Default: patient 2, first seizure, 3 min before, 1.5 min after
python scripts/visualize_wearable_sample.py

# Custom patient and time window
python scripts/visualize_wearable_sample.py --patient 5 --before-min 5 --after-min 2

# Visualize all seizures for a patient
python scripts/visualize_wearable_sample.py --patient 3 --all-seizures

# Specific seizure within a recording
python scripts/visualize_wearable_sample.py --patient 2 --seizure-index 1

# Custom output directory
python scripts/visualize_wearable_sample.py --patient 2 --output-dir ./my_plots/

# High resolution output
python scripts/visualize_wearable_sample.py --patient 2 --dpi 300
```

### Advanced Options

```bash
python scripts/visualize_wearable_sample.py --help
```

Options:
- `--patient`: Patient/subject number (default: 2)
- `--before-min`: Minutes before seizure (default: 3.0)
- `--after-min`: Minutes after seizure (default: 1.5)
- `--seizure-index`: If recording has multiple seizures, which one to show (default: 0)
- `--all-seizures`: Visualize all seizures for the patient (creates multiple plots)
- `--output-dir`: Directory to save PNG files (default: `./visualizations/wearable/`)
- `--dataset-root`: Override dataset root path (advanced debugging)
- `--dpi`: Resolution for saved plots (default: 150)

## Output Files

Visualizations are saved as PNG files with naming convention:

```
patient_XX_seizure_YY_YYYYMMDD_HHMMSS.png
```

Where:
- `XX` = Patient number (zero-padded, e.g., 02)
- `YY` = Seizure index in that recording (zero-padded)
- `YYYYMMDD_HHMMSS` = Seizure event timestamp

Example:
```
patient_02_seizure_00_20220510_143022.png
patient_02_seizure_01_20220511_095415.png
```

## Visualization Components

Each plot includes:

1. **Multi-Signal Subplots** (stacked vertically)
   - One subplot per signal type
   - Signal name and color-coded trace
   - Per-signal statistics box (mean, std, min, max)

2. **Event Markers**
   - 🔴 **Red dashed line**: Seizure onset time
   - 🔴 **Red shade**: Post-seizure window (for context)
   - 🟠 **Orange shade**: Pre-seizure warning window (30-60s before onset)

3. **Time Axis**
   - Relative to seizure onset (0 = seizure starts)
   - Negative values = before seizure
   - Positive values = after seizure
   - Labeled in minutes and seconds

4. **Recording Metadata Box**
   - Recording directory name
   - Recording start/end times
   - Time window configuration
   - Sampling rate in Hz

## Common Use Cases

### Review seizure characteristics for a single patient
```bash
python scripts/visualize_wearable_sample.py --patient 2 --all-seizures
```

### Create publication-ready figures
```bash
python scripts/visualize_wearable_sample.py --patient 2 --dpi 300 --before-min 4 --after-min 2
```

### Explore extended pre-seizure patterns
```bash
python scripts/visualize_wearable_sample.py --patient 2 --before-min 10 --after-min 1
```

### Generate batch visualizations for multiple patients
```bash
for patient in 2 3 5 7; do
    python scripts/visualize_wearable_sample.py --patient $patient --before-min 3 --after-min 1.5
done
```

## Troubleshooting

### "Dataset not found: $WEARABLE_DATASET_ROOT"

The external drive isn't mounted. Options:
1. Mount the drive: `mount /dev/sdX $DATASETS_MOUNT/` (administrator)
2. Override path: `--dataset-root /path/to/WearablwDevice-Oregon`

```bash
python scripts/visualize_wearable_sample.py --patient 2 --dataset-root /new/path/to/dataset
```

### "No seizure recordings found for patient X"

Patient X either doesn't exist or has no recordings with seizure annotations. Check:
1. Patient number is correct (check database)
2. Master seizure database is accessible

### Plots have missing signals

Not all recordings have all signal types. The visualizer only plots available signals. Check the log output:
```
  ppg: 1200 samples @ 50.0 Hz
  adxl: 600 samples @ 25.0 Hz
  # (missing eda, temperature, etc.)
```

## For Developers

### Extending signal types

In `src/data_loader.py`, `WearableDeviceDataLoader.load_signals_from_recording()`:
- Add pattern to `signal_patterns` dict
- Implement parsing in `_load_csv_signal()` method

### Customizing visualization

Edit `WearableDatasetVisualizer.plot_seizure_signals()` in the script:
- Modify `colors` dict for different signal colors
- Adjust `figsize` for larger/smaller plots
- Add custom annotations or overlays

### Dataset format

The visualizer expects:
- Master database: Excel file with seizure annotations and recording metadata
- Signal data: CSV files indexed by recording directory with standardized Stream naming
- Timestamps: Either in header (ODR) or "Epoch Delta TS" column

Refer to `src/data_loader.py` for detailed CSV parsing logic.

## Performance Notes

- **Typical runtime**: 2-5 seconds per seizure event
- **Memory usage**: ~100-500 MB depending on signal length and sampling rate
- **Bottleneck**: CSV file I/O (can be slow for large multi-minute recordings)

For batch processing many seizures:
```bash
for seizure in {0..10}; do
    python scripts/visualize_wearable_sample.py --patient 2 --seizure-index $seizure &
done
wait
```

## References

- Wearable Dataset: Oregon Health & Science University
- Signal Types: Empatica E4 wearable device specification
- Feature Extraction: Based on `src/feature_extraction.py`
