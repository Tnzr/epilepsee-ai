# Longitudinal Inference Validation Guide

**Purpose:** Validate seizure countdown model in realistic deployment scenario (continuous inference with temporal context).

**Key Principle:** Model is LSTM-based and accumulates temporal context. Testing must respect this by running inference continuously from patient data start → through time → capturing seizure events in context.

---

## Why This Matters

### ❌ Naive Approach (WRONG for this model):
1. Find seizure onset time T
2. Extract data from T-30 min to T+30 min  
3. Run inference on this isolated window
4. Check predictions near T

**Problem:** LSTM hidden state is uninitialized. Missing 24-72 hours of prior context that patient normally accumulates. Results won't reflect real deployment.

### ✅ Realistic Approach (RIGHT):
1. Load patient's full recording from start
2. Run inference continuously from beginning forward
3. When reaching seizure time T, examine how model's state evolved to that point
4. Measure: advance warning time, false alarm rate throughout prior context, etc.

**Result:** Captures real temporal dependencies that LSTM learned.

---

## Data Preparation

### Step 1: Identify Validation Patients
```bash
# Find patients with seizure events
python3 << 'EOF'
import json

# Load SeizeIT2 events
with open('/mnt/d/Datasets/SeizeIT2/events.json') as f:
    events = json.load(f)

# Group by patient, filter for seizure events
patients_to_seizures = {}
for event in events:
    if event.get('seizure'):
        patient_id = event['participant_id']
        if patient_id not in patients_to_seizures:
            patients_to_seizures[patient_id] = []
        patients_to_seizures[patient_id].append({
            'onset': event['onset'],
            'duration': event.get('duration'),
            'id': event.get('id')
        })

# Print summary
print("Patients with seizure events:")
for patient_id in sorted(patients_to_seizures.keys()):
    n_seizures = len(patients_to_seizures[patient_id])
    print(f"  {patient_id}: {n_seizures} seizures")

# Save for validation
import json
with open('validation_patients.json', 'w') as f:
    json.dump(patients_to_seizures, f, indent=2)
EOF
```

**Output:** `validation_patients.json` - maps each patient to their seizure event times

### Step 2: Verify Data Continuity
```bash
# For each patient, check recording duration and gaps
python3 << 'EOF'
import os
import json
from pathlib import Path

dataset_root = Path('/mnt/d/Datasets/SeizeIT2')

with open('validation_patients.json') as f:
    patients = json.load(f)

print("\nData availability for validation:")
for patient_id in sorted(patients.keys()):
    patient_dir = dataset_root / f'sub-{patient_id:03d}'
    if patient_dir.exists():
        # Count session recordings
        ses_dirs = sorted([d for d in patient_dir.iterdir() if d.name.startswith('ses-')])
        print(f"\n{patient_id}:")
        print(f"  Sessions: {len(ses_dirs)}")
        for ses_dir in ses_dirs:
            # Count recording files
            eeg_files = list(ses_dir.glob('eeg/*.eeg'))
            ecg_files = list(ses_dir.glob('ecg/*.ecg'))
            print(f"    {ses_dir.name}: {len(eeg_files)} EEG, {len(ecg_files)} ECG files")
    else:
        print(f"{patient_id}: NOT FOUND")
EOF
```

---

## Continuous Inference Generation

### Step 3: Implement Continuous Inference Function

**Key considerations:**
- Initialize LSTM hidden state at data start (all zeros)
- Process data sequentially, window by window
- Track hidden state evolution (for debugging/visualization)
- Record predictions and timestamps
- Correlate with known seizure event times

**Python Implementation Sketch:**
```python
import torch
import numpy as np
from pathlib import Path

def run_continuous_patient_inference(
    model_path: str,
    patient_id: str,
    dataset_root: str,
    output_dir: str,
    prior_context_hours: int = 48
):
    """
    Run continuous inference for entire patient recording.
    
    Args:
        model_path: Path to best_model.pt
        patient_id: Patient identifier (e.g., '001')
        dataset_root: SeizeIT2 root directory
        output_dir: Where to save results
        prior_context_hours: How many hours of prior context to show in viz
    
    Returns:
        Saves:
        - {patient_id}_inference_trace.json: All predictions, times, hidden states
        - {patient_id}_inference_trace.png: Full timeline visualization
        - {patient_id}_seizure_neighborhood.png: Zoomed views around seizures
        - {patient_id}_temporal_metrics.json: Detection timing stats
    """
    
    # 1. Load model
    model = load_model(model_path)
    model.eval()
    
    # 2. Load patient data sequentially
    patient_dir = Path(dataset_root) / f'sub-{patient_id:03d}'
    all_samples = []
    all_timestamps = []
    
    # Load chronologically ordered recordings
    for session_dir in sorted(patient_dir.glob('ses-*')):
        for recording_file in sorted(session_dir.glob('eeg/*.eeg')):
            # Load ECG/EEG data
            samples, timestamps = load_bids_recording(recording_file)
            all_samples.append(samples)
            all_timestamps.append(timestamps)
    
    # Concatenate into continuous timeline
    patient_data = np.concatenate(all_samples, axis=0)
    patient_times = np.concatenate(all_timestamps, axis=0)
    
    # 3. Run continuous inference
    predictions = []
    hidden_states = []
    classification_outputs = []
    regression_outputs = []
    
    # Initialize LSTM hidden state (patient baseline)
    hidden_state = None
    
    # Sliding window inference
    window_size = 960  # Samples (ECG features padded)
    stride = 1  # One-sample stride for continuous inference
    
    for i in range(0, len(patient_data) - window_size, stride):
        window = torch.FloatTensor(patient_data[i:i+window_size])
        
        with torch.no_grad():
            output, hidden_state = model(
                window.unsqueeze(0),
                hidden_state=hidden_state
            )
        
        # Extract predictions
        classification_prob = output['classification'].item()  # P(preictal)
        regression_countdown = output['regression'].item()  # Minutes to seizure
        
        predictions.append({
            'timestamp': patient_times[i + window_size // 2],
            'classification': classification_prob,
            'regression': regression_countdown,
            'hidden_state': hidden_state.detach().cpu().numpy() if hidden_state is not None else None
        })
        
        # Store for visualization
        hidden_states.append(hidden_state.detach().cpu().numpy())
        classification_outputs.append(classification_prob)
        regression_outputs.append(regression_countdown)
    
    # 4. Save inference trace
    import json
    with open(f'{output_dir}/{patient_id}_inference_trace.json', 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        trace_data = {
            'patient_id': patient_id,
            'predictions': [{
                'timestamp': str(p['timestamp']),
                'classification': float(p['classification']),
                'regression': float(p['regression']),
            } for p in predictions]
        }
        json.dump(trace_data, f, indent=2)
    
    # 5. Visualize full timeline
    visualize_continuous_inference(
        predictions,
        patient_times,
        patient_id,
        output_dir
    )
    
    return predictions
```

### Step 4: Generate Visualizations

**Full Timeline Visualization:**
```
Timeline: Patient 001, 72 hours of continuous recording

Panel 1 (top):    ECG signal (background)
Panel 2:          Classification probability (preictal likelihood over time)
Panel 3:          Regression output (countdown minutes)
Panel 4:          Smoothed alarm (threshold-based detection)
Panel 5 (bottom): Seizure event annotations (yellow bars at known onset times)

Visual inspection:
- Does classification spike BEFORE seizure onset?
- How far in advance? (e.g., 5 min, 15 min, 1 hour?)
- Are there false alarms in interictal periods?
- How does alarm evolve throughout patient's baseline state?
```

**Seizure Neighborhood Visualization (Zoomed):**
```
For each seizure event:

Panel layout (temporal):
  [-24 hrs prior baseline] ... [-1 hour] ... [SEIZURE TIME] ... [+1 hour] ... [+24 hrs after]
                  ↓              ↓                ↓                  ↓           ↓
    Background context    Escalation phase   ONSET (T=0)    Recovery phase   Stabilization

Rows:
1. ECG/EEG trace with local context
2. Classification probability (does it spike smoothly, or suddenly?)
3. Regression countdown (does it approach zero smoothly?)
4. Token activation heatmap (what is model attending to?)
5. LSTM hidden state (magnitude and composition over time)

Metrics overlaid:
- Time-to-detection: ___ min before onset
- Peak confidence: ___ % preictal prob
- False alarms in prior 24 hrs: ___
```

---

## Metrics to Extract

### Per-Patient Metrics
```json
{
  "patient_id": "001",
  "total_recording_hours": 72.5,
  
  "seizure_events": [
    {
      "event_id": "seizure_1",
      "onset_time": "2024-03-15T14:32:00Z",
      "duration_minutes": 2.5,
      "time_to_detection": 6.2,  // Minutes before onset that alarm crossed threshold
      "peak_classification_prob": 0.87,
      "peak_countdown_regression": 18.3,  // Minutes predicted when peak
      "false_alarms_in_prior_24h": 0,
      "false_alarms_in_prior_1h": 0,
      "confidence_trajectory": "smooth",  // smooth | sudden | oscillating
    }
  ],
  
  "interictal_periods": [
    {
      "duration_hours": 6.0,
      "false_alarms": 0,
      "mean_classification_prob": 0.12,
      "max_classification_prob": 0.34,
    }
  ],
  
  "continuous_stability": {
    "hidden_state_drift_per_hour": 0.002,  // L2 norm change
    "classification_drift_per_hour": 0.001,
    "prediction_oscillation_std": 0.05,
    "any_nan_or_inf": false
  }
}
```

### Population-Level Summary
```
Across all validation patients:
- Average time-to-detection: 7.5 ± 3.2 min (range: 2-15 min)
- Minimum time-to-detection: 2 min (Patient 012, Event 3)
- Maximum time-to-detection: 18 min (Patient 034, Event 1)
- False positive rate during interictal: 0.8 per hour
- Seizure detection rate (≥5 min advance): 85% (17/20 seizures)
- Seizure detection rate (≥10 min advance): 65% (13/20 seizures)
```

---

## Troubleshooting Elevated Alarm Issue

**If alarm elevation is confirmed (uniform high throughout recording):**

1. **Check hidden state saturation:**
   - Print hidden state norms at start, middle, end of recording
   - If norm >> initial norm, hidden state may have diverged

2. **Verify threshold calibration:**
   - Is adaptive threshold (0.57) too low?
   - Should it be patient-specific instead of universal?

3. **Inspect token attention patterns:**
   - Which channels is model attending to?
   - Are they consistently active (normal) or activated inappropriately?

4. **Compare against patient baseline:**
   - Does patient have continuous high heart rate, stress, or other abnormality?
   - Does model's elevated alarm correlate with abnormal ECG/EEG features?

5. **Recheck class weight balance:**
   - Run: `python -c "import json; m=json.load(open('config/default_config.yaml')); print(m['loss']['use_class_weighting'], m['loss']['classification_positive_weight'])"`
   - Should show: `true, 15.0`

---

## Deployment-Readiness Checklist

After longitudinal validation completes:

- [ ] Time-to-detection ≥ 5 minutes for ≥80% of seizures
- [ ] False positive rate < 1 per hour during interictal
- [ ] Continuous inference stable for ≥24 hours
- [ ] No NaN/inf during extended runs
- [ ] Hidden state doesn't diverge over time
- [ ] Alarm elevation issue resolved (or explained)
- [ ] Threshold calibrated to clinical requirements
- [ ] Edge latency < 100 ms per inference step
- [ ] Memory footprint acceptable for target hardware
- [ ] Cold-start strategy defined (how to initialize for new patients)

---

## Commands to Run

### Complete Validation Pipeline
```bash
# Set paths
PATIENT_LIST="validation_patients.json"
MODEL_PATH="models/best_model.pt"
DATASET_ROOT="/mnt/d/Datasets/SeizeIT2"
OUTPUT_DIR="models/longitudinal_validation"

mkdir -p $OUTPUT_DIR

# For each patient: generate continuous inference trace
python scripts/continuous_patient_inference.py \
  --patient-list $PATIENT_LIST \
  --model-path $MODEL_PATH \
  --dataset-root $DATASET_ROOT \
  --output-dir $OUTPUT_DIR \
  --include-hidden-states \
  --visualize-all

# Generate combined report
python scripts/longitudinal_validation_report.py \
  --inference-dir $OUTPUT_DIR \
  --events-file $DATASET_ROOT/events.json \
  --output $OUTPUT_DIR/LONGITUDINAL_VALIDATION_REPORT.md

# Check deployment readiness
python scripts/check_deployment_readiness.py \
  --validation-report $OUTPUT_DIR/LONGITUDINAL_VALIDATION_REPORT.md
```

---

**Next Step:** Implement `continuous_patient_inference.py` and run on 5-10 validation patients.
