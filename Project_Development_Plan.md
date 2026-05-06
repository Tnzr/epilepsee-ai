# Project Development Plan: ECG-Centered Seizure Anticipation with Reactive Countdown

## Executive Summary

This document provides a **comprehensive development roadmap** for implementing an **epileptic seizure anticipation system** using the SeizeIT2 dataset. The system will predict **time-to-seizure** (countdown) using primarily ECG signals with optional multimodal fusion for enhanced accuracy.

Key innovation: **Transition from binary classification (seizure/no-seizure) to continuous-time regression (minutes-to-seizure)**, enabling **early warning capability** with configurable lead time.

---

# 1. Problem Reformulation: Countdown Estimation

## 1.1 Core Objective

Rather than binary classification, the model predicts:
- **Target variable**: Minutes-to-seizure (continuous regression)
- **Output range**: [0, 10] minutes (0 = seizure occurring, ∞ = inter-ictal)
- **Prediction frequency**: Every 1-5 seconds during monitoring
- **Lead-time requirement**: ≥3-5 minutes safe warning window

## 1.2 Advantages Over Binary Classification

| Aspect | Binary Classification | Countdown Regression |
|--------|----------------------|----------------------|
| Output | "Seizure incoming: Yes/No" | "Seizure in 4.3 minutes" |
| Clinical utility | Coarse, binary decision | Precise timing for intervention |
| Early warning | Poor (threshold-dependent) | Excellent (continuous) |
| Model robustness | High false positives | Tunable sensitivity-specificity |
| Patient trust | Limited | Higher (transparent timing) |

---

# 2. Dataset Specification & Structure

## 2.1 SeizeIT2 Dataset Overview

- **Total subjects**: 125 patients with refractory focal epilepsy
- **Total seizures**: 886 recorded seizure episodes
- **Recording duration**: 11,640 hours of continuous wearable data
- **Seizure types**: FA (317), FIA (393), FBTC (55), subclinical (2), unclear (119)
- **Mean seizure duration**: 58 seconds (range: 3s – 16 min)
- **Recording sites**: 5 European EMUs (Leuven, Freiburg, Aachen, Karolinska, Coimbra)

## 2.2 Signal Modalities & Specifications

### Sampling Rates
| Signal | Sampling Rate | Channels | Type |
|--------|---------------|----------|------|
| EEG (behind-ear) | 250 Hz | 1-4 (patient-dependent) | Time-series |
| ECG | 250 Hz | 1 | Time-series |
| EMG (deltoid) | 250 Hz | 1 | Time-series |
| ACC/GYR (movement) | 25 Hz | 6 (3 ACC + 3 GYR) | Time-series |

### Data Organization (BIDS-Formatted)
```
/media/tnzr/HDD1/Datasets/ds005873/
├── sub-001/
│   └── ses-01/
│       ├── ecg/
│       │   ├── sub-001_ses-01_task-szMonitoring_run-01_ecg.edf
│       │   └── sub-001_ses-01_task-szMonitoring_run-01_ecg.json
│       ├── eeg/
│       │   ├── sub-001_ses-01_task-szMonitoring_run-01_eeg.edf
│       │   └── sub-001_ses-01_task-szMonitoring_run-01_eeg.json
│       ├── emg/
│       │   ├── sub-001_ses-01_task-szMonitoring_run-01_emg.edf
│       │   └── sub-001_ses-01_task-szMonitoring_run-01_emg.json
│       └── mov/
│           ├── sub-001_ses-01_task-szMonitoring_run-01_mov.edf
│           └── sub-001_ses-01_task-szMonitoring_run-01_mov.json
├── sub-002/
│   └── ... (continues for 125 subjects)
└── participants.tsv  # Demographic info
```

## 2.3 Event Annotations

**File**: `events.tsv` in each subject directory

**Columns**:
- `onset`: Start time of event (seconds from recording start)
- `duration`: Event duration (seconds)
- `trial_type`: Event label (e.g., "seizure", "non-seizure")
- Additional annotations: seizure type, hemisphere onset, clinical confirmation

**Usage**: Ground truth for generating countdown labels

---

# 3. Data Preprocessing Pipeline

## 3.1 EDF File Reading & Signal Extraction

### Tools & Libraries
- **Primary**: MNE-Python (handles BIDS + EDF natively)
- **Alternative**: PyEDFlib (lightweight, direct EDF access)
- **Supporting**: NumPy, SciPy, Pandas

### Implementation Steps

```python
# Example pseudocode
import mne
from bids import BIDSLayout

# Load BIDS dataset
layout = BIDSLayout('/media/tnzr/HDD1/Datasets/ds005873', validate=False)

# Iterate subjects
for subject in layout.get_subjects():
    for session in layout.get_sessions()[subject]:
        # Load ECG
        ecg_files = layout.get(subject=subject, session=session, 
                             datatype='ecg', extension='edf')
        raw_ecg = mne.io.read_raw_edf(ecg_files[0].path, preload=False)
        
        # Load EEG
        eeg_files = layout.get(subject=subject, session=session, 
                             datatype='eeg', extension='edf')
        raw_eeg = mne.io.read_raw_edf(eeg_files[0].path, preload=False)
        
        # Load movement (for artifact detection)
        mov_files = layout.get(subject=subject, session=session, 
                             datatype='mov', extension='edf')
        raw_mov = mne.io.read_raw_edf(mov_files[0].path, preload=False)
        
        # Load event annotations
        events_file = f"sub-{subject:03d}_ses-01_task-szMonitoring_events.tsv"
        events_df = pd.read_csv(events_file, sep='\t')
```

## 3.2 ECG Signal Preprocessing

### Step 1: Bandpass Filtering
```python
# Remove low-frequency drift and high-frequency noise
# Typical cardiac components: 0.5-40 Hz
raw_ecg_filtered = raw_ecg.filter(l_freq=0.5, h_freq=40.0, 
                                   method='iir', phase='zero')
```

### Step 2: R-Peak Detection
```python
from scipy.signal import find_peaks

# High-pass filter for QRS complex isolation
ecg_detrended = raw_ecg_filtered.copy()
ecg_data = ecg_detrended.get_data()[0]  # 1D signal

# Find peaks (R-peaks)
r_peaks, _ = find_peaks(ecg_data, distance=100, height=np.std(ecg_data)*2)
```

### Step 3: RR Interval Extraction
```python
# Time between successive R-peaks (in seconds)
r_peak_times = r_peaks / sampling_rate  # 250 Hz
rr_intervals = np.diff(r_peak_times)

# Remove ectopic beats / outliers (keep RR between 0.4 - 1.2 seconds)
rr_valid = rr_intervals[(rr_intervals > 0.4) & (rr_intervals < 1.2)]
```

## 3.3 Motion Artifact Detection

```python
# High-pass filter acceleration to detect motion > 0.1g (1 m/s²)
acc_threshold = 1.0  # m/s²
motion_artifact_windows = np.where(np.linalg.norm(acc_data, axis=0) > acc_threshold)[0]

# Mark periods with high motion as "unreliable"
artifact_mask = np.zeros(total_samples)
artifact_mask[motion_artifact_windows] = 1
```

## 3.4 Data Quality Checks

- **Signal completeness**: No missing samples in EDF
- **Artifact burden**: <30% of pre-ictal windows should be flagged as artifacts
- **ECG detectability**: R-peaks detected with >95% sensitivity
- **Synchronization**: Verify ECG, EEG, EMG timestamps align

---

# 4. Feature Extraction

## 4.1 ECG-Derived Features (Primary Features)

### Heart Rate Features
```python
def extract_hr_features(rr_intervals, window_length_s=60, step_s=1):
    """
    Extract heart rate statistics in sliding windows
    """
    hr = 60.0 / rr_intervals  # Convert RR to beats/minute
    
    features = []
    for i in range(0, len(hr) - window_length_s, step_s):
        window = hr[i:i+window_length_s]
        features.append({
            'hr_mean': np.mean(window),
            'hr_std': np.std(window),
            'hr_max': np.max(window),
            'hr_min': np.min(window),
        })
    return pd.DataFrame(features)
```

### Heart Rate Variability (HRV) Metrics

| Feature | Formula | Clinical Meaning |
|---------|---------|------------------|
| **RMSSD** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(RR_i - RR_{i-1})^2}$ | Parasympathetic activity (vagal tone) |
| **SDNN** | $\sqrt{\frac{1}{N}\sum_{i=1}^{N}(RR_i - \overline{RR})^2}$ | Total HRV variability |
| **pNN50** | % of RR intervals differing >50ms | Short-term parasympathetic fluctuations |
| **Entropy** | Shannon/Sample Entropy of RR series | Complexity of cardiac regulation |

### Frequency-Domain Features (LF/HF Ratio)

```python
def extract_hrv_frequency_domain(rr_intervals, method='welch'):
    """
    Compute power spectral density in LF/HF bands
    - LF (0.04-0.15 Hz): sympathetic + parasympathetic activity
    - HF (0.15-0.40 Hz): parasympathetic activity (vagal)
    - LF/HF ratio: Balance between sympathetic and parasympathetic
    
    Pre-ictal elevation of LF/HF → sympathetic dominance
    """
    from scipy.signal import welch
    
    fs = 1.0 / np.mean(rr_intervals)  # Sampling rate
    freq, power = welch(rr_intervals, fs=fs, nperseg=256)
    
    lf_power = np.sum(power[(freq >= 0.04) & (freq < 0.15)])
    hf_power = np.sum(power[(freq >= 0.15) & (freq < 0.40)])
    
    return {
        'lf_power': lf_power,
        'hf_power': hf_power,
        'lf_hf_ratio': lf_power / (hf_power + 1e-6),
    }
```

### Cardiac Instability Indicators

```python
def extract_instability_features(rr_intervals, window_length_s=60):
    """
    Quantify rapid changes in RR intervals (pre-ictal signature)
    """
    rr_diffs = np.diff(rr_intervals)
    
    return {
        'rr_acceleration': np.mean(np.abs(rr_diffs)),
        'rr_jitter': np.std(rr_diffs),
        'rr_detrended_entropy': entropy(rr_intervals - np.convolve(rr_intervals, np.ones(10)/10)),
    }
```

## 4.2 EEG Features (Secondary/Multimodal)

For multimodal models, extract:

```python
def extract_eeg_features(eeg_signal, fs=250, window_s=2):
    """
    Extract spectral and temporal EEG features
    """
    from scipy.signal import welch
    
    freq, power = welch(eeg_signal, fs=fs, nperseg=fs*window_s)
    
    bands = {
        'delta': (0.5, 4),      # 0.5-4 Hz
        'theta': (4, 8),        # 4-8 Hz
        'alpha': (8, 12),       # 8-12 Hz
        'beta': (12, 30),       # 12-30 Hz
    }
    
    features = {}
    for band_name, (low, high) in bands.items():
        band_power = np.sum(power[(freq >= low) & (freq < high)])
        features[f'eeg_{band_name}_power'] = band_power
    
    # Entropy (complexity)
    from entropy import sample_entropy
    features['eeg_entropy'] = sample_entropy(eeg_signal, order=2, metric='chebyshev')
    
    return features
```

## 4.3 Feature Summary Table

| Category | Features | Computation Window | Dimension |
|----------|----------|-------------------|-----------|
| **HR** | Mean, Std, Max, Min | 60s | 4 |
| **HRV Time-domain** | RMSSD, SDNN, pNN50, DFA | 60s | 4 |
| **HRV Frequency** | LF, HF, LF/HF | 256-sample FFT | 3 |
| **Instability** | RR acceleration, jitter, entropy | 60s | 3 |
| **EEG Spectral** | Delta, Theta, Alpha, Beta bands | 2s windows | 4 |
| **EEG Entropy** | Sample entropy, permutation entropy | 60s | 2 |
| **Motion** | Acc magnitude, Gyr magnitude | 1s | 2 |
| **Total ECG-only** | — | — | **14** |
| **Total Multimodal** | — | — | **22+** |

---

# 5. Temporal Labeling Strategy

## 5.1 Label Generation from events.tsv

```python
def generate_countdown_labels(seizure_onset_time, recording_samples, 
                              fs=250, pre_ictal_window=600):
    """
    Convert seizure timestamp to sample-level countdown labels
    
    Args:
        seizure_onset_time: Seizure start in seconds
        recording_samples: Total samples in recording
        fs: Sampling frequency (250 Hz)
        pre_ictal_window: Pre-ictal duration (600 seconds = 10 minutes)
    
    Returns:
        labels: Countdown value for each time point
                - Pre-ictal (0, 600s before): 0 to 600 (minutes)
                - Ictal (during seizure): 0
                - Inter-ictal: -1 (ignore or separate class)
    """
    labels = np.full(recording_samples, -1)  # Init as inter-ictal
    
    seizure_sample = int(seizure_onset_time * fs)
    pre_ictal_start = max(0, seizure_sample - int(pre_ictal_window * fs))
    
    for sample_idx in range(pre_ictal_start, seizure_sample):
        time_to_seizure = (seizure_sample - sample_idx) / fs / 60  # Convert to minutes
        labels[sample_idx] = min(time_to_seizure, 10)  # Cap at 10 minutes
    
    return labels
```

## 5.2 Window Extraction

```python
def create_sliding_windows(features_df, labels, window_length_s=600, 
                          step_s=1, fs=250):
    """
    Create (feature, label) pairs with sliding window
    
    Args:
        features_df: DataFrame with all extracted features
        labels: Countdown labels (minutes-to-seizure)
        window_length_s: Context window (600s = 10 minutes recommended)
        step_s: Stride between windows (1s = 250 samples @ 250Hz)
    
    Returns:
        X: (N_windows, window_samples, N_features)
        y: (N_windows,) - target countdown value
    """
    window_samples = int(window_length_s * fs)
    step_samples = int(step_s * fs)
    
    X, y = [], []
    
    for start_idx in range(0, len(features_df) - window_samples, step_samples):
        end_idx = start_idx + window_samples
        
        window_features = features_df.iloc[start_idx:end_idx].values
        window_label = labels[end_idx - 1]  # Label is the final sample's countdown
        
        # Skip inter-ictal windows (label = -1)
        if window_label >= 0:
            X.append(window_features)
            y.append(window_label)
    
    return np.array(X), np.array(y)
```

## 5.3 Class Importance Weighting

```python
def compute_sample_weights(countdown_labels):
    """
    Weight samples by proximity to seizure (late predictions are more critical)
    
    w(t) = exp(-(T-t)/τ)
    
    where:
    - T: seizure onset time
    - t: current time
    - τ: time constant (e.g., 60 seconds = 1 minute)
    """
    tau = 60.0  # seconds
    weights = np.exp(-(10 - (countdown_labels + 1)) * 60 / tau)
    
    # Normalize
    weights = weights / np.mean(weights)
    return weights
```

---

# 6. Machine Learning Architecture

## 6.1 Baseline: ECG-Only Model (for Smartwatch Feasibility)

### Architecture Option 1: Bidirectional LSTM + Regression Head

```python
import torch
import torch.nn as nn

class ECGCountdownPredictor(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, dropout=0.3):
        super().__init__()
        
        # Bidirectional LSTM (processes temporal context backward & forward)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        
        # Temporal Attention
        self.attention = nn.MultiheadAttention(hidden_dim*2, num_heads=4, 
                                              dropout=dropout, batch_first=True)
        
        # Classification head (is this pre-ictal?)
        self.fc_class = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Probability [0, 1]
        )
        
        # Regression head (minutes-to-seizure)
        self.fc_regress = nn.Sequential(
            nn.Linear(hidden_dim*2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)  # Linear output [0, 10]
        )
    
    def forward(self, x):
        # x shape: (batch, time_steps, features)
        
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time_steps, hidden*2)
        
        # Attention (use final timestep)
        attn_out, _ = self.attention(lstm_out[:, -1:, :], lstm_out, lstm_out)
        pooled = attn_out.mean(dim=1)  # (batch, hidden*2)
        
        # Outputs
        pre_ictal_prob = self.fc_class(pooled)  # (batch, 1)
        countdown = torch.relu(self.fc_regress(pooled))  # (batch, 1)
        
        # Clip countdown to [0, 10]
        countdown = torch.clamp(countdown, 0, 10)
        
        return pre_ictal_prob, countdown
```

### Architecture Option 2: 1D CNN + LSTM Hybrid

```python
class CNNLSTMCountdown(nn.Module):
    def __init__(self, input_dim=14, dropout=0.3):
        super().__init__()
        
        # 1D Convolutions for local feature extraction
        self.conv1 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        
        # LSTM for temporal dependencies
        self.lstm = nn.LSTM(64, 128, num_layers=2, bidirectional=True, 
                           dropout=dropout, batch_first=True)
        
        # Output heads
        self.fc = nn.Linear(256, 64)
        self.fc_class = nn.Linear(64, 1)
        self.fc_regress = nn.Linear(64, 1)
    
    def forward(self, x):
        # x: (batch, time_steps, input_dim)
        x = x.transpose(1, 2)  # (batch, input_dim, time_steps)
        
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)  # (batch, 64, time_steps/4)
        
        x = x.transpose(1, 2)  # (batch, time_steps/4, 64)
        lstm_out, _ = self.lstm(x)
        
        pooled = lstm_out[:, -1, :]  # (batch, 256)
        hidden = torch.relu(self.fc(pooled))
        
        pre_ictal = torch.sigmoid(self.fc_class(hidden))
        countdown = torch.relu(self.fc_regress(hidden))
        countdown = torch.clamp(countdown, 0, 10)
        
        return pre_ictal, countdown
```

## 6.2 Advanced: Multimodal Fusion Model

```python
class MultimodalCountdownPredictor(nn.Module):
    """
    Fuses ECG, EEG, and Motion data streams
    """
    def __init__(self, ecg_dim=14, eeg_dim=6, motion_dim=2, hidden_dim=128):
        super().__init__()
        
        # Separate encoders per modality
        self.ecg_lstm = nn.LSTM(ecg_dim, hidden_dim, num_layers=2, 
                               bidirectional=True, batch_first=True)
        self.eeg_lstm = nn.LSTM(eeg_dim, hidden_dim, num_layers=2, 
                               bidirectional=True, batch_first=True)
        self.motion_dense = nn.Sequential(
            nn.Linear(motion_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, hidden_dim*2)
        )
        
        # Fusion layer (early fusion: concatenate then process)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim*2 * 3, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128)
        )
        
        # Cross-modal attention (optional enhancement)
        self.cross_modal_attention = nn.MultiheadAttention(hidden_dim*2, 4, 
                                                           batch_first=True)
        
        # Output heads
        self.fc_class = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        self.fc_regress = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, ecg_x, eeg_x, motion_x):
        # ECG stream
        ecg_out, _ = self.ecg_lstm(ecg_x)
        ecg_pooled = ecg_out[:, -1, :]  # (batch, hidden*2)
        
        # EEG stream
        eeg_out, _ = self.eeg_lstm(eeg_x)
        eeg_pooled = eeg_out[:, -1, :]
        
        # Motion stream (no LSTM, just dense)
        motion_pooled = self.motion_dense(motion_x[:, -1, :])
        
        # Fusion
        fused = torch.cat([ecg_pooled, eeg_pooled, motion_pooled], dim=-1)
        fused_hidden = self.fusion(fused)
        
        # Outputs
        pre_ictal = torch.sigmoid(self.fc_class(fused_hidden))
        countdown = torch.relu(self.fc_regress(fused_hidden))
        countdown = torch.clamp(countdown, 0, 10)
        
        return pre_ictal, countdown
```

---

# 7. Loss Functions & Training Strategy

## 7.1 Multi-Task Loss (Classification + Regression)

```python
class SeizureCountdownLoss(nn.Module):
    def __init__(self, alpha=0.3, beta=0.7, gamma=0.1):
        super().__init__()
        self.alpha = alpha   # Classification weight
        self.beta = beta     # Regression weight
        self.gamma = gamma   # Ranking weight
        
    def forward(self, pre_ictal_pred, countdown_pred, 
                pre_ictal_true, countdown_true, sample_weights):
        """
        Combined loss function:
        L = α·BCE + β·Weighted_MSE + γ·Ranking_Loss
        """
        
        # 1. Binary classification loss: "Is pre-ictal?" (BCE)
        bce_loss = torch.nn.functional.binary_cross_entropy(
            pre_ictal_pred.squeeze(), pre_ictal_true.float()
        )
        
        # 2. Weighted regression loss: Minutes-to-seizure
        # Weight by proximity (late errors are more critical)
        mse_raw = (countdown_pred.squeeze() - countdown_true) ** 2
        mse_weighted = torch.mean(sample_weights * mse_raw)
        
        # 3. Temporal ranking loss: Countdown should decrease monotonically
        # Penalize if countdown increases over time (model should get "more sure")
        countdown_diffs = countdown_pred[1:] - countdown_pred[:-1]
        ranking_loss = torch.mean(torch.relu(countdown_diffs))  # Penalize increases
        
        total_loss = self.alpha * bce_loss + self.beta * mse_weighted + self.gamma * ranking_loss
        
        return {
            'total': total_loss,
            'bce': bce_loss,
            'mse': mse_weighted,
            'ranking': ranking_loss
        }
```

## 7.2 Training Procedure

```python
def train_end_to_end(model, train_loader, val_loader, device='cuda', 
                     epochs=100, learning_rate=1e-3):
    """
    Full training loop with early stopping
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )
    criterion = SeizureCountdownLoss(alpha=0.3, beta=0.7, gamma=0.1)
    
    best_val_mae = float('inf')
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_mae = 0, 0
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device).float()
            labels = labels.to(device).float()
            
            # Compute sample weights
            sample_weights = compute_temporal_weights(labels).to(device)
            
            # Forward pass
            pre_ictal_pred, countdown_pred = model(features)
            
            # Generate pre-ictal labels (1 if countdown < 10, else 0)
            pre_ictal_labels = (labels > 0).float()
            
            # Loss
            loss_dict = criterion(
                pre_ictal_pred, countdown_pred,
                pre_ictal_labels, labels,
                sample_weights
            )
            
            # Backward
            optimizer.zero_grad()
            loss_dict['total'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss_dict['total'].item()
            train_mae += torch.mean(torch.abs(countdown_pred - labels)).item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_mae = train_mae / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss, val_mae = 0, 0
        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(device).float()
                labels = labels.to(device).float()
                
                pre_ictal_pred, countdown_pred = model(features)
                pre_ictal_labels = (labels > 0).float()
                sample_weights = compute_temporal_weights(labels).to(device)
                
                loss_dict = criterion(
                    pre_ictal_pred, countdown_pred,
                    pre_ictal_labels, labels,
                    sample_weights
                )
                
                val_loss += loss_dict['total'].item()
                val_mae += torch.mean(torch.abs(countdown_pred - labels)).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        
        # Early stopping
        if avg_val_mae < best_val_mae:
            best_val_mae = avg_val_mae
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            patience_counter += 1
        
        # Learning rate scheduling
        scheduler.step(avg_val_mae)
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train: Loss={avg_train_loss:.4f}, MAE={avg_train_mae:.4f}")
        print(f"  Val:   Loss={avg_val_loss:.4f}, MAE={avg_val_mae:.4f}")
        
        if patience_counter >= 15:
            print("Early stopping triggered")
            break
    
    # Load best weights
    model.load_state_dict(torch.load('best_model.pt'))
    return model
```

## 7.3 Learning Rate Schedule

- **Initial LR**: 1e-3
- **Reduction**: ReduceLROnPlateau (factor=0.5, patience=10)
- **Gradient clipping**: max_norm=1.0 (prevent exploding gradients)
- **Batch normalization**: Optional, helps with training stability

---

# 8. Evaluation Metrics & Strategy

## 8.1 Primary Metrics (For Countdown Accuracy)

### Mean Absolute Error (MAE)
$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_{\text{pred},i} - y_{\text{true},i}|$$

**Interpretation**: "On average, predictions are X minutes off"
- **Target**: MAE < 1.5 minutes for lead time ≥ 5 minutes
- **Excellent**: MAE < 0.5 minutes

### Median Absolute Error
More robust to outliers than MAE:
$$\text{Median AE} = \text{median}(|y_{\text{pred}} - y_{\text{true}}|)$$

### Root Mean Square Error (RMSE)
$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_{\text{pred},i} - y_{\text{true},i})^2}$$

**Interpretation**: Penalizes large errors heavily. Use to detect outlier predictions.

## 8.2 Clinical Metrics (For Lead Time Performance)

### Sensitivity @ Lead Times

For different prediction horizons, compute:

```
Sensitivity(lead_time = t) = 
    P(countdown_pred ≥ t | true_countdown ≥ t)
    
i.e., "If there's truly t+ minutes, how often does model predict ≥ t minutes ahead?"
```

**Target thresholds**:
- ≥ 10 minutes ahead: Sensitivity ≥ 60%
- ≥ 5 minutes ahead: Sensitivity ≥ 75%
- ≥ 3 minutes ahead: Sensitivity ≥ 85%

### False Positive Rate (FPR)

```
FPR(threshold) = 
    P(countdown_pred > threshold | inter-ictal epoch)
    
i.e., "How often does model trigger false alarms during normal activity?"
```

**Target**: FPR < 1 false alarm per 8 hours of inter-ictal monitoring

### Prediction Stability

Measure "jitter" in predictions over short windows:

```python
def compute_prediction_stability(countdown_predictions, window=30):
    """
    Countdown predictions should vary smoothly, not jump around
    """
    stabilities = []
    for i in range(len(countdown_predictions) - window):
        window_std = np.std(countdown_predictions[i:i+window])
        stabilities.append(window_std)
    
    return np.median(stabilities)
```

**Target**: Median prediction std < 0.2 minutes (12 seconds) over 30-sample windows

## 8.3 Classification Metrics (Pre-ictal / Inter-ictal)

For the binary classification head:

| Metric | Formula | Target |
|--------|---------|--------|
| **Sensitivity** | TP/(TP+FN) | ≥ 80% |
| **Specificity** | TN/(TN+FP) | ≥ 85% |
| **Precision** | TP/(TP+FP) | ≥ 70% |
| **F1-Score** | 2·(Precision·Recall)/(Precision+Recall) | ≥ 0.75 |
| **AUROC** | Area under ROC curve | ≥ 0.85 |

## 8.4 Cross-Validation Strategy

### Leave-One-Seizure-Out (LOSO)

```python
def leave_one_seizure_out_validation(subjects_data, model_class):
    """
    For each seizure, train on other seizures, test on held-out seizure
    Avoids data leakage from same subject in train/test
    """
    all_seizures = []
    
    # Collect all seizures across all subjects
    for subject_id, subject_data in subjects_data.items():
        for seizure_idx, seizure_info in enumerate(subject_data['seizures']):
            all_seizures.append({
                'subject': subject_id,
                'seizure_idx': seizure_idx,
                'data': seizure_info
            })
    
    scores = {'mae': [], 'sensitivity_5min': [], 'fpr': []}
    
    for test_seizure_idx, test_seizure in enumerate(all_seizures):
        # Create train/test split
        train_seizures = [s for i, s in enumerate(all_seizures) if i != test_seizure_idx]
        test_data = test_seizure['data']
        
        # Train model
        model = model_class().to(device)
        train_on_seizures(model, train_seizures)
        
        # Evaluate on test seizure
        mae, sens_5, fpr = evaluate_on_seizure(model, test_data)
        
        scores['mae'].append(mae)
        scores['sensitivity_5min'].append(sens_5)
        scores['fpr'].append(fpr)
    
    # Report aggregate
    print(f"LOSO-CV Results (N={len(all_seizures)} seizures):")
    print(f"  MAE: {np.mean(scores['mae']):.3f} ± {np.std(scores['mae']):.3f} min")
    print(f"  Sensitivity@5min: {np.mean(scores['sensitivity_5min']):.1%}")
    print(f"  FPR: {np.mean(scores['fpr']):.3f} / 8h")
```

### Leave-One-Subject-Out (LOPO)

*Stricter*: Entire subject held out during training

```python
def leave_one_subject_out_validation(subjects_dict, model_class):
    """
    Each iteration: train on N-1 subjects, test on 1 subject
    Prevents overfitting to subject-specific patterns
    """
    subject_ids = list(subjects_dict.keys())
    scores = []
    
    for test_subject_id in subject_ids:
        # Split
        train_subjects = {s: d for s, d in subjects_dict.items() if s != test_subject_id}
        test_subject_data = subjects_dict[test_subject_id]
        
        # Train
        model = model_class().to(device)
        train_on_subjects(model, train_subjects)
        
        # Evaluate
        metrics = evaluate_on_subject(model, test_subject_data)
        scores.append(metrics)
    
    print(f"LOPO-CV Results (N={len(subject_ids)} subjects):")
    avg_mae = np.mean([s['mae'] for s in scores])
    avg_sens = np.mean([s['sensitivity_5min'] for s in scores])
    print(f"  MAE: {avg_mae:.3f} min")
    print(f"  Sensitivity@5min: {avg_sens:.1%}")
```

---

# 9. Data Preparation & Augmentation

## 9.1 Class & Temporal Balancing

```python
def balance_training_data(X, y, method='weighted_sampler'):
    """
    Addresses two imbalances:
    1. Temporal imbalance: Inter-ictal >> pre-ictal windows
    2. Countdown range imbalance: Early pre-ictal (9-10 min) << late (0-1 min)
    """
    
    if method == 'weighted_sampler':
        # Create weights for each sample based on countdown value
        weights = np.where(
            y >= 0,  # Pre-ictal windows
            np.exp(-(10 - (y + 1)) * 60 / 60),  # Exponential weight (high near seizure)
            0.5  # Inter-ictal windows get lower weight
        )
        
        # Normalize
        weights = weights / np.sum(weights)
        
        # Create WeightedRandomSampler
        sampler = torch.utils.data.WeightedRandomSampler(
            weights, len(X), replacement=True
        )
        return sampler
    
    elif method == 'oversampling':
        # Oversample pre-ictal windows
        preictal_mask = y >= 0
        n_preictal = np.sum(preictal_mask)
        n_inter_ictal = len(y) - n_preictal
        
        # Sample with replacement from pre-ictal to match inter-ictal
        preictal_indices = np.where(preictal_mask)[0]
        oversample_indices = np.random.choice(
            preictal_indices, 
            size=n_inter_ictal, 
            replace=True
        )
        
        balanced_indices = np.concatenate([
            np.where(~preictal_mask)[0],
            oversample_indices
        ])
        
        return X[balanced_indices], y[balanced_indices]
```

## 9.2 Temporal Augmentation

```python
def temporal_augmentation(X, y, augment_prob=0.3):
    """
    Simulate natural variations in pre-ictal progression
    """
    X_aug, y_aug = X.copy(), y.copy()
    
    for i in range(len(X)):
        if np.random.rand() < augment_prob and y[i] >= 0:
            # Random time-stretching: slow down or speed up pre-ictal phase
            stretch_factor = np.random.uniform(0.9, 1.1)
            
            # Resample time axis
            old_length = X[i].shape[0]
            new_length = int(old_length * stretch_factor)
            indices = np.linspace(0, old_length - 1, new_length)
            X_aug[i] = np.interp(indices, np.arange(old_length), X[i, :])
            
            # Adjust countdown accordingly
            y_aug[i] = y[i] / stretch_factor
    
    return X_aug, y_aug

def feature_jittering(X, jitter_std=0.01):
    """
    Add small Gaussian noise to features to prevent overfitting
    """
    noise = np.random.normal(0, jitter_std, X.shape)
    return X + noise
```

---

# 10. Model Deployment & Real-Time Inference

## 10.1 Streaming Inference Pipeline

```python
class RealtimeSeizurePredictor:
    def __init__(self, model_path, window_length_s=600, inference_interval_s=1):
        self.model = torch.load(model_path)
        self.model.eval()
        
        self.window_length_s = window_length_s
        self.inference_interval = int(inference_interval_s * 250)  # 250 Hz fs
        
        self.feature_buffer = deque(maxlen=int(window_length_s * 250))
        self.countdown_history = deque(maxlen=30)  # Last 30 predictions
    
    def process_new_sample(self, ecg_signal, ecg_rr_intervals):
        """
        Called when new ECG sample arrives (every 4 ms @ 250 Hz)
        """
        # Update feature buffer
        features = self._extract_features_online(ecg_signal, ecg_rr_intervals)
        self.feature_buffer.append(features)
        
        # Make prediction periodically
        if len(self.feature_buffer) >= self.window_length_s * 250:
            X = np.array(list(self.feature_buffer))[-self.window_length_s*250:]
            X = torch.from_numpy(X).float().unsqueeze(0).to('cuda')
            
            with torch.no_grad():
                pre_ictal_prob, countdown = self.model(X)
            
            countdown_val = countdown.item()
            self.countdown_history.append(countdown_val)
            
            # Smooth predictions with median filter
            smoothed_countdown = np.median(list(self.countdown_history))
            
            return {
                'countdown_minutes': smoothed_countdown,
                'pre_ictal_probability': pre_ictal_prob.item(),
                'alert_triggered': smoothed_countdown < 5,  # Alert if < 5 min
            }
    
    def _extract_features_online(self, ecg_signal, rr_intervals):
        """
        Extract features from streaming data
        """
        hr = 60.0 / np.mean(rr_intervals) if len(rr_intervals) > 0 else 0.0
        rr_std = np.std(rr_intervals) if len(rr_intervals) > 1 else 0.0
        
        return np.array([hr, rr_std, ...])  # 14-dim feature vector
```

## 10.2 Alert Thresholding

```python
def generate_alert(countdown_val, sensitivity_level='moderate'):
    """
    Determine alert based on countdown with configurable sensitivity
    """
    thresholds = {
        'conservative': {'warning': 8, 'urgent': 3},    # Fewer false alarms
        'moderate': {'warning': 5, 'urgent': 2},        # Balanced
        'aggressive': {'warning': 3, 'urgent': 1},      # More early warnings
    }
    
    thresh = thresholds[sensitivity_level]
    
    if countdown_val < thresh['urgent']:
        return 'URGENT_ALERT'  # Vibration + loud beep
    elif countdown_val < thresh['warning']:
        return 'WARNING'       # Subtle notification
    else:
        return 'NORMAL'        # No alert
```

---

# 11. Implementation Roadmap & Milestones

## Phase 1: Data Preparation (Weeks 1-2)

- [ ] Set up Python environment (MNE, PyTorch, PyEDFlib)
- [ ] Implement BIDS dataset loader
- [ ] Parse 10-15 sample subjects' EDF files
- [ ] Validate data integrity (missing samples, corruption checks)
- [ ] Generate sample countdown labels from events.tsv

**Deliverable**: Working data pipeline with 50 subjects prepared

## Phase 2: Feature Engineering (Weeks 3-4)

- [ ] Implement ECG preprocessing (filtering, R-peak detection)
- [ ] Extract HR, HRV, LF/HF features
- [ ] Extract EEG spectral features
- [ ] Test feature quality on known seizure windows
- [ ] Visualize pre-ictal vs inter-ictal feature distributions

**Deliverable**: 14-dim ECG feature set + 22-dim multimodal features

## Phase 3: Model Development (Weeks 5-6)

- [ ] Implement baseline LSTM-Regression model
- [ ] Implement CNN-LSTM hybrid
- [ ] Implement Multimodal fusion model
- [ ] Set up training loop with PyTorch
- [ ] Configure loss functions (multi-task learning)

**Deliverable**: 3 trained baseline models

## Phase 4: Evaluation & Optimization (Weeks 7-8)

- [ ] Implement LOSO cross-validation
- [ ] Evaluate MAE, Sensitivity@{3,5,10}min, FPR
- [ ] Hyperparameter tuning (learning rate, regularization)
- [ ] Analyze failure cases
- [ ] Generate per-subject performance reports

**Deliverable**: Benchmark results with statistical significance testing

## Phase 5: Advanced Features & Ablation (Weeks 9-10)

- [ ] Add temporal augmentation
- [ ] Implement weighted sampling
- [ ] Run ablation studies (ECG vs multimodal)
- [ ] Test edge cases (rare seizure types, pediatric subjects)
- [ ] Generate publication-quality figures

**Deliverable**: Ablation study results, feature importance analysis

## Phase 6: Real-Time Demo & Documentation (Weeks 11-12)

- [ ] Implement streaming inference pipeline
- [ ] Create alert visualization dashboard
- [ ] Write comprehensive documentation
- [ ] Prepare code for release/publication
- [ ] Generate final report with recommendations

**Deliverable**: Documented codebase, live demo, technical report

---

# 12. Expected Outcomes & Success Criteria

## Minimum Viable Product (MVP)

- **MAE**: < 2 minutes on held-out test set
- **Sensitivity @ 5 min ahead**: ≥ 70%
- **False alarm rate**: < 2 per 8 hours baseline
- **Inference speed**: < 100 ms per prediction

## Excellent Performance

- **MAE**: < 1 minute
- **Sensitivity @ 5 min**: ≥ 85%
- **Sensitivity @ 3 min**: ≥ 75%
- **False alarm rate**: < 1 per 8 hours
- **Prediction stability**: Median jitter < 10 seconds

## Research Impact

- Demonstrate feasibility of wearable seizure anticipation
- Quantify pre-ictal autonomic signatures in seizure-onset epilepsy
- Establish benchmark for future wearable-based systems
- Publication target: IEEE EMBC, NeurIPS ML4Health, or clinical neurology journal

---

# 13. Code Structure & Organization

## Directory Layout

```
Epilepsee-AI/
├── README.md
├── Project_Overview.md
├── Project_Development_Plan.md
├── environment.yml              # Conda environment
├── config/
│   ├── default_config.yaml      # Hyperparameters
│   ├── model_config.py          # Architecture params
│   └── eval_config.py           # Evaluation settings
├── src/
│   ├── __init__.py
│   ├── data_loader.py           # BIDS/EDF reading
│   ├── preprocessing.py         # ECG filtering, R-peak detection
│   ├── feature_extraction.py    # HRV, spectral features
│   ├── labeling.py              # Countdown label generation
│   ├── models.py                # LSTM, CNN-LSTM, Multimodal
│   ├── losses.py                # Multi-task loss function
│   ├── training.py              # Training loop
│   ├── evaluation.py            # Metrics (MAE, sensitivity, FPR)
│   └── inference.py             # Real-time prediction
├── notebooks/
│   ├── 01_exploratory_data_analysis.ipynb
│   ├── 02_feature_analysis.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_evaluation_results.ipynb
│   └── 05_ablation_studies.ipynb
├── scripts/
│   ├── download_seizeit2.sh     # Dataset download (if applicable)
│   ├── preprocess_dataset.py    # Batch preprocessing
│   ├── train_models.py          # Full training pipeline
│   ├── evaluate_models.py       # Cross-validation
│   └── demo_inference.py        # Real-time demo
└── tests/
    ├── test_data_loader.py
    ├── test_preprocessing.py
    ├── test_feature_extraction.py
    └── test_models.py
```

## Key Dependencies

```yaml
# environment.yml
name: seizure-prediction
channels:
  - pytorch
  - conda-forge
dependencies:
  - python=3.10
  - pytorch::pytorch::*[version='>=2.0']
  - pytorch::pytorch-cuda=11.8
  - pytorch::torchvision
  - pytorch::torchaudio
  - pip
  - pip:
    - mne>=1.3.0
    - mne-bids>=0.12.0
    - pyedflib>=0.1.37
    - numpy>=1.24.0
    - scipy>=1.10.0
    - pandas>=2.0.0
    - scikit-learn>=1.2.0
    - matplotlib>=3.7.0
    - seaborn>=0.12.0
    - tensorboard>=2.12.0
    - jupyter>=1.0.0
    - entropy>=0.0.7  # For entropy calculations
```

---

# 14. Technical Best Practices

## 14.1 Data Handling

✅ **DO**:
- Store raw signals uncompressed (lossy compression corrupts ECG)
- Maintain patient anonymization (use subject IDs only)
- Version control preprocessing parameters
- Log all data transformations

❌ **DON'T**:
- Mix sampling rates without explicit resampling
- Use inter-ictal data for pre-ictal labeling
- Normalize across entire dataset (patient-specific normalization better)
- Trust times from EDF files without validation against events.tsv

## 14.2 Model Development

✅ **DO**:
- Use LOSO/LOPO cross-validation (prevent leakage)
- Track all hyperparameters and random seeds
- Save training curves and loss values
- Monitor both train and validation metrics
- Use tensorboard for real-time monitoring

❌ **DON'T**:
- Use random train/test split (temporal structure matters)
- Optimize hyperparameters on final test set
- Train on all data without validation split
- Ignore class imbalance

## 14.3 Evaluation Practices

✅ **DO**:
- Report confidence intervals (via cross-validation)
- Specify which seizures/patients used for testing
- Compare against clinical baselines
- Report failure modes and edge cases
- Use per-subject performance breakdown

❌ **DON'T**:
- Report only best-case results
- Mix evaluation metrics without weighting
- Use single hold-out test set (insufficient for rare event)
- Ignore prediction confidence/uncertainty

---

# 15. References & Related Work

### Key Datasets
- **SeizeIT2**: Bhagubai et al. (2024) - [OpenNeuro](https://openneuro.org/datasets/ds005873)
- **Temple University Hospital (TUH) EEG Corpus**: Obeid & Picone (2016)
- **CHB-MIT Scalp EEG Database**: Shoeb et al. (2010)

### Seizure Prediction Methods
- LSTM-based seizure prediction: (Antoniades et al., 2016)
- Multimodal fusion: (Bandarabadi et al., 2015)
- Temporal point processes: (Halnes et al., 2018)

### Wearable ECG & Autonomic Monitoring
- PPG-based HRV: (Vest et al., 2018)
- Smartwatch seizure detection: (Jeppesen et al., 2015)

---

# 16. Appendix: Troubleshooting & FAQs

### Q: How much training data is needed?

**A**: With ~886 seizures in SeizeIT2:
- MVP: 200 seizures (train: 150, val: 25, test: 25)
- Robust: 600+ seizures with LOSO CV
- Production: All 886 with careful CV strategy

### Q: How long does training take?

**A**: On NVIDIA A100 GPU:
- ECG-only LSTM: ~2-4 hours (100 epochs)
- Multimodal CNN-LSTM: ~6-8 hours
- LOSO cross-validation (886 iterations): ~500 GPU-hours

### Q: Can I use ECG from smartwatch instead of chest ECG?

**A**: Features learned from SeizeIT2 (250 Hz, high-quality) may not directly transfer. You'd need:
- Resampling from wearable sampling rate (50-100 Hz typically)
- Domain adaptation to handle signal quality differences
- Retraining/fine-tuning on wearable-specific data

### Q: How sensitive is the model to hyperparameters?

**A**: Window length and feature extraction are most critical:
- Window < 5 min: Loss of temporal context
- Window > 15 min: Computational overhead
- LR too high (>1e-2): Training diverges
- Dropout < 0.2: Overfitting
- Dropout > 0.5: Underfitting

### Q: What about handling missing data?

**A**: Common in clinical recordings:
- <5 ms gaps: Interpolate
- >1 second gaps: Exclude window as "artifact"
- Full signal loss: Mark run as unusable
- Track artifact percentage for post-hoc filtering

---

# 17. Future Directions

## Short-term (6-12 months)
1. Deploy on smartwatch hardware (Wear OS SDK)
2. Patient-specific model adaptation
3. Multi-seizure-type optimization

## Medium-term (1-2 years)
1. Incorporate brain-computer interface (BCI) signals
2. Causal inference for physiological mechanism
3. Real-world pilot with 5-10 patients

## Long-term (2+ years)
1. FDA medical device validation
2. Integration with closed-loop neuromodulation
3. Generalization across diverse epilepsy phenotypes

---

**Document Version**: 1.0  
**Last Updated**: March 2026  
**Status**: Active Development
