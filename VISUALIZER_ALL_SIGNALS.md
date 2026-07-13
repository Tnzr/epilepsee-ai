# Wearable Visualizer - Comprehensive All-Signals Update

## Summary of Changes

The wearable device visualizer has been significantly enhanced to provide comprehensive visualization of **ALL available signals** from the wearable device recordings.

## What Changed

### 1. **New Comprehensive Visualizer** (`scripts/visualize_wearable_sample_v2.py`)

A completely rewritten and improved version with:

✅ **All signals automatically detected and visualized:**
- PPG (Heart Rate)
- ADPD channels (Ch1 D1, Ch2 D2)  
- ADXL X, Y, Z acceleration vectors
- **ADXL Magnitude** (total acceleration computed as √(x² + y² + z²))
- EDA (Electrodermal Activity / Impedance)
- Temperature
- AGC (Automatic Gain Control)
- Battery Level
- Pedometer (Steps)
- SQI (Signal Quality Index)

✅ **Title updated:** Now clearly says "PPG & Wearable Signals" instead of ECG

✅ **Two visualization modes:**
1. **Key signals only** (default) - PPG, ADPD, ADXL, EDA, Temperature, AGC
2. **Comprehensive ALL-SIGNALS** (with `--all-signals`) - All 10+ signal types

✅ **Seizure highlighting across all signals:**
- Red dashed line at seizure onset (time = 0)
- Red shaded region showing post-seizure period (1-2 min)
- Orange shaded region showing pre-seizure warning window (30-60s before)
- Per-signal statistics showing mean, std, min, max values

✅ **Multi-channel acceleration handling:**
- Individual X, Y, Z acceleration traces
- **Total acceleration magnitude** automatically computed
- All resampled to common sampling rate

✅ **Output files:**
```
patient_02_seizure_00_20220510_143022.png                    # Key signals
patient_02_seizure_00_20220510_143022_ALL_SIGNALS.png         # All signals (with --all-signals)
```

### 2. **Updated Makefile Targets**

```bash
make visualize-wearable              # Patient 2 key signals (3min/1.5min windows)
make visualize-wearable-multi        # Patient 2 with ALL signals, extended windows (5min/2min)
make visualize-wearable-all          # All seizures for patient 2 with ALL signals
make visualize-wearable-all-signals  # Patient 2 ALL-signals comprehensive figure
make visualize-wearable-all-signals-extended  # ALL signals with extended 5min/2min windows
```

### 3. **Command-Line Usage**

```bash
# Key signals only
python scripts/visualize_wearable_sample_v2.py --patient 2

# Comprehensive ALL-signals plot
python scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals

# Extended time windows
python scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals --before-min 5 --after-min 2

# All seizures
python scripts/visualize_wearable_sample_v2.py --patient 2 --all-seizures --all-signals

# Batch visualization
for patient in 2 3 5 7; do
    python scripts/visualize_wearable_sample_v2.py --patient $patient --all-signals
done
```

## Visualization Features

Each comprehensive ALL-signals plot includes:

### Layout
- **Stacked multi-panel figure** with one subplot per signal type
- **Color-coded signals** for easy identification:
  - Red: PPG (primary signal)
  - Blue: ADPD
  - Orange: ADXL individual axes, Dark Orange: Magnitude
  - Green: EDA
  - Pink: Temperature
  - Purple: AGC
  - Gray: Battery, Teal: Steps, Light Gray: SQI

### Signal Markers & Annotations
- 🔴 **Red dashed line**: Seizure onset (time = 0)
- 🔴 **Red shaded area**: Seizure period (post-onset window)
- 🟠 **Orange shaded area**: Pre-seizure warning window (30-60s before)

### Statistics & Metadata
- **Per-signal box** showing: mean (μ), std dev (σ), min, max
- **Recording metadata box** showing:
  - Patient/watch ID
  - Recording directory name
  - Start/end timestamps
  - Time window configuration
  - Number of active signal streams

### Time Axis
- Relative to seizure onset
- Negative = before seizure, Positive = after seizure
- Labeled in minutes and seconds

## What Signals Are Available

From the sample recording `$WEARABLE_DATASET_ROOT/#2/9-19 to 9-21/CVS/`:

| Signal Type | CSV File | Key Columns | Sampling Rate |
|---|---|---|---|
| PPG (Heart Rate) | `*PPGAppStream.csv` | HR, Confidence | 50 Hz |
| ADPD/ADXL | `*ADPD_ADXL_Combined.csv` | D1, D2, X, Y, Z | 50 Hz (ADXL), 100 Hz (ADPD) |
| EDA | `*EDAAppStream.csv` | IMP Module(Ohms) | ~1 Hz |
| Temperature | `*TemperatureStream.csv` | Temperature | ~1 Hz |
| AGC | `*AGCAppStream.csv` | AGC Type, Settings | ~1 Hz |
| Battery | `*BatteryStream.csv` | Battery (%) | ~5 Hz |
| Pedometer | `*PedometerAppStream.csv` | Steps | ~1 Hz |
| SQI | `*SQIStream.csv` | SQI (Quality: 0-1) | ~1 Hz |

## Key Improvements Over Previous Version

| Feature | Previous | New (v2) |
|---|---|---|
| Signals shown | 5-6 | All 10+ available |
| Title clarity | "Seizure Events" | "PPG & Wearable Signals" |
| ADXL magnitude | No | ✓ Auto-computed |
| All signals plot | No | ✓ With --all-signals flag |
| Signal identification | Basic | Color-coded by type |
| Seizure highlighting | Markers only | Markers + shaded regions |
| Automatic detection | CSV directory scan | All CSV files auto-detected |
| Resampling | Limited | All signals to common rate |

## Example Output Files

Patient #2 Seizure visualization generates:

```
visualizations/wearable/
├── patient_02_seizure_00_20220510_143022.png
│   └── Key signals: PPG, ADPD, ADXL, EDA, Temperature
├── patient_02_seizure_00_20220510_143022_ALL_SIGNALS.png
│   └── Comprehensive: All 10+ signal types with magnitude
├── patient_02_seizure_01_20220511_095415.png
└── patient_02_seizure_01_20220511_095415_ALL_SIGNALS.png
```

Each PNG is ~2-10 MB (depends on DPI and signal duration)

## Quick Start

```bash
# Recommended: Comprehensive visualization with extended window
make visualize-wearable-all-signals-extended

# Or manually:
python scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals --before-min 5 --after-min 2
```

## Testing

```bash
# Verify script syntax
python -m py_compile scripts/visualize_wearable_sample_v2.py

# Check CLI help
python scripts/visualize_wearable_sample_v2.py --help

# Quick test run (will print "No seizure recordings" if dataset not mounted - expected)
python scripts/visualize_wearable_sample_v2.py --patient 2
```

## Notes

- **Dataset requirement:** External HDD with WearableDevice-Oregon dataset must be mounted at `$WEARABLE_DATASET_ROOT`
- **Output directory:** Visualizations saved to `./visualizations/wearable/` by default
- **Resolution:** Default 150 DPI, use `--dpi 300` for publication-quality figures
- **Sampling rates:** All signals automatically resampled to minimum common rate for alignment
- **Seizure samples:** Highlighted with red shading across entire duration in all signals simultaneously

## Files Changed/Created

### Created
- `scripts/visualize_wearable_sample_v2.py` - New comprehensive visualizer
- `VISUALIZER_ALL_SIGNALS.md` - This documentation

### Updated
- `Makefile` - Updated targets to use v2 script with --all-signals option
- Documentation references (old `visualize_wearable_sample.py` still available for reference)

## Future Enhancements

Potential additions:
- Interactive HTML plots (Plotly)
- Spectral analysis (FFT, wavelet)
- Multi-seizure comparison
- Feature correlations
- Animation showing temporal progression
- Export to CSV with time-aligned samples
