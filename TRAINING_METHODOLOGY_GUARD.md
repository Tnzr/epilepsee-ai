# Training Methodology Guard: Preventing Temporal Continuity Mismatch

**Date**: 2026-07-08  
**Context**: Critical discovery that revealed incompatibility between training approach and deployment requirements for LSTM seizure prediction model.

---

## 📋 Executive Summary

**Critical Issue Discovered:**  
The ECGCountdownPredictor model was trained using **random shuffled batches with independent hidden states** but is deployed for **continuous sequential inference with accumulated hidden state**. This fundamental mismatch means the model was not trained for its actual deployment mode.

**Impact:**  
- Model learned to predict from isolated 16-minute windows
- Deployment runs continuous inference over patient timelines
- Validation on random ±30 min windows is invalid
- Model performance may degrade significantly in deployment

**Resolution Selected:**  
Retrain using **stateful LSTM architecture** where:
- Temporal sequences preserved during training
- Hidden state maintained across time within patient data
- Training mirrors deployment exactly
- Validation uses continuous longitudinal inference

---

## 🔍 Root Cause Analysis

### Training Approach (Before Fix)
```python
# src/training.py Lines 1026-1072
train_sampler = DistributedSampler(
    train_dataset,
    shuffle=True,  # ⚠️ RANDOM SHUFFLING
    seed=seed
)

# Lines 1189-1450: Training loop
for batch_idx, batch in train_iter:  # Batches in random order
    features = batch[0].to(device)
    lstm_out, (h_n, c_n) = self.lstm(x)  # ⚠️ Fresh hidden state per batch (h_n=None)
    pooled = lstm_out[:, -1, :]  # Only final timestep used
    # h_n, c_n are discarded - no temporal carryover
```

**Consequences:**
1. Model processes windows independently (no temporal context from prior windows)
2. Hidden state resets per batch (fresh zeros at batch boundary)
3. Learning optimizes for isolated-window classification
4. Hidden state mechanism never utilized during training

### Deployment Approach (Actual Need)
```python
# For production: continuous patient inference
hidden_state = None  # Initialize once at patient start
for t in range(patient_data_duration):
    batch = patient_data[t:t+960]
    lstm_out, hidden_state = self.lstm(x, hidden_state)  # ⚠️ ACCUMULATE STATE
    prediction = self.fc(lstm_out[:, -1, :])
    
    # hidden_state persists to next iteration
    # Model has 16+ minutes of prior context accumulated
```

**Requirements:**
1. Continuous forward passes with hidden state preservation
2. Model uses accumulated context from entire prior patient history
3. Predictions depend on temporal patterns across hours/days
4. Single hidden state initialized at data start, never reset

### The Mismatch

| Aspect | Training | Deployment | Match? |
|--------|----------|------------|--------|
| **Batch Order** | Random shuffled | Sequential chronological | ❌ NO |
| **Hidden State Init** | Reset per batch (zeros) | Initialized once at start | ❌ NO |
| **Hidden State Usage** | Computed but discarded | Accumulated across time | ❌ NO |
| **Temporal Context** | 16 minutes (1 window) | Hours/days (prior history) | ❌ NO |
| **Time Axis Learning** | None (isolated windows) | Full temporal dependency | ❌ NO |

---

## ❌ Why Standard Random Batching Fails for LSTM

### Principle
Random shuffling is **correct for training most ML models** because:
- Enables parallelization across GPUs
- Ensures gradient averaging isn't biased by batch order
- Prevents overfitting to sequence patterns

### Exception: Sequential/Temporal Models
LSTM networks have **hidden state that encodes temporal context**. When you shuffle:
1. Model receives temporal chunks in wrong order
2. Cannot learn relationships like "T-5 to T-0 patterns predict seizure at T+5"
3. Hidden state reset at batch boundaries breaks temporal chains
4. Model defaults to learning stateless patterns (equivalent to feeding only final window)

### Evidence from This Project
- **Metric**: pred_alert_std = 0.0034 (flat predictions)
- **Pattern**: All predictions converge to single value (interictal bias)
- **Interpretation**: Model learned stateless "always interictal" policy since temporal patterns were disconnected

---

## 🛡️ Prevention Checklist: Never Repeat This

Use this checklist **before any LSTM training session**:

### Architecture Phase
- [ ] **Question**: Is model architecture using recurrent layers (LSTM, GRU)?
  - YES → Continue to next items
  - NO → Standard random shuffling is fine
  
- [ ] **Question**: Does deployment require temporal continuity/hidden state accumulation?
  - YES → MUST train statelessly or statefully, not random-shuffled
  - NO → Reconsider architecture; temporal model + non-temporal deployment is incorrect

- [ ] **Question**: Is deployment doing continuous inference over patient timelines?
  - YES → Hidden state MUST be maintained in training via stateful batching
  - NO → Can use random batching but carefully monitor hidden state usage

### Data Preparation Phase
- [ ] **Decision**: Choose training mode:
  - **Option 1 - Stateless Training (Simpler)**: Extract overlapping windows, train independently
    - Use: Random shuffling allowed (windows are independent)
    - Caveat: Model won't learn long-term dependencies
  - **Option 2 - Stateful Training (Recommended)**: Group batches by patient, maintain hidden state
    - Use: Patient-sequential batching, preserve hidden state across batches within patient
    - Benefit: Model learns temporal patterns, deployment directly compatible

- [ ] **Sampling Strategy**: If using stateful training:
  - [ ] Group training data by patient
  - [ ] Create patient-continuous sequences (patient 1: start→end, then patient 2: start→end)
  - [ ] Within patient: maintain hidden state (DO NOT RESET)
  - [ ] Between patients: reset hidden state (fresh patient = fresh state)
  - [ ] Each epoch: shuffle patient ORDER (not samples within patients)

- [ ] **Batch Construction**: If using stateful training:
  - [ ] Batch = consecutive timesteps from same patient
  - [ ] Do NOT mix patients in same batch
  - [ ] Do NOT interrupt patient sequence with another patient
  - [ ] DO maintain hidden state across batches from same patient

### Training Phase
- [ ] **Sampler Verification**:
  - [ ] Using DistributedSampler? Check: shuffle should be based on patient order, not sample order
  - [ ] Using WeightedRandomSampler? Check: should weight by patient, not sample
  - [ ] Verify batch composition (all samples from same patient)

- [ ] **Hidden State Handling**:
  - [ ] Code review: Is hidden state discarded after each batch?
    - Current bug: `lstm_out, (h_n, c_n) = self.lstm(x)` then h_n discarded
  - [ ] Code review: Is hidden state passed to next batch within same patient?
    - Expected: `lstm_out, hidden_state = self.lstm(x, hidden_state)`
  - [ ] Code review: Is hidden state reset per patient?
    - Expected: `hidden_state = None` when patient changes

- [ ] **Loss Function Validation**:
  - [ ] Verify class weighting is ACTIVE (not bypassed by loss type logic)
  - [ ] Verify loss component weights are balanced (classification ≥ regression weight)
  - [ ] Test on simple case: 100% interictal batch should NOT output all-preictal predictions

### Validation Phase
- [ ] **Validation Data Handling**:
  - [ ] Use continuous longitudinal inference (Option A), not random windows
  - [ ] Calculate metrics over full patient timelines, not isolated predictions
  - [ ] Track: advance warning time, false alarm rate, hidden state stability

- [ ] **Sanity Checks**:
  - [ ] Prediction variance: std(predictions) > 0.01 (not flat)
  - [ ] Classification balance: no class has <30% or >70% accuracy
  - [ ] Heatmap visualization: colorful pattern (not monochrome)
  - [ ] Token activation: multiple channels active (not all black)

- [ ] **Comparison Baseline**:
  - [ ] Train 2-epoch test version with random shuffling
  - [ ] Train 2-epoch test version with stateful batching
  - [ ] Compare metrics: stateful should show better temporal patterns
  - [ ] If similar → reconsider whether LSTM is necessary

### Documentation Phase
- [ ] **Code Comments**:
  ```python
  # ✓ Document WHY stateful batching
  # ✓ Document hidden state lifecycle (init → accumulate → reset)
  # ✓ Document difference from deployment (if any)
  ```

- [ ] **Training Log**:
  ```yaml
  training_mode: stateful  # or stateless
  batching_strategy: patient_sequential  # or random
  hidden_state_management: preserved_within_patient
  epochs: N
  best_val_mae: X.XX
  validation_approach: longitudinal_continuous
  ```

---

## 🔧 Implementation Pattern: Stateful LSTM Training

### Pseudocode for Data Loader
```python
class TemporalPatientSequenceDataLoader:
    """
    Provides consecutive samples from same patient without shuffling within patient.
    Allows training loop to maintain hidden state across batches.
    """
    
    def __init__(self, patient_data_dict, batch_size=32):
        """
        patient_data_dict: {
            'patient_001': {'ecg': [...], 'labels': [...]},
            'patient_002': {'ecg': [...], 'labels': [...]},
            ...
        }
        """
        self.patients = list(patient_data_dict.keys())
        self.patient_data = patient_data_dict
        self.batch_size = batch_size
        
    def __iter__(self):
        # Shuffle PATIENT ORDER (not samples)
        shuffled_patients = np.random.permutation(self.patients)
        
        for patient_id in shuffled_patients:
            patient_ecg = self.patient_data[patient_id]['ecg']
            patient_labels = self.patient_data[patient_id]['labels']
            
            # Yield consecutive samples from this patient
            for start_idx in range(0, len(patient_ecg) - self.batch_size, self.batch_size):
                batch_ecg = patient_ecg[start_idx:start_idx+self.batch_size]
                batch_labels = patient_labels[start_idx:start_idx+self.batch_size]
                yield batch_ecg, batch_labels
```

### Pseudocode for Training Loop
```python
hidden_state = None
for epoch in range(num_epochs):
    for patient_id, batch in data_loader:
        
        # NEW PATIENT: reset hidden state
        if patient_changed:
            hidden_state = None
        
        # FORWARD PASS: pass hidden state from previous batch
        lstm_out, hidden_state = self.lstm(batch, hidden_state)
        
        # ⚠️ CRITICAL: Detach to prevent backprop through full history
        # (optional, depends on memory constraints)
        hidden_state = tuple(h.detach() for h in hidden_state)
        
        # Forward classifier
        predictions = self.fc_class(lstm_out[:, -1, :])
        
        # Loss and backward
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
```

---

## 📚 References in This Project

**Files Affected by This Discovery:**

1. **[src/training.py](src/training.py#L1026-L1090)**
   - Current: Random shuffling + independent batches
   - Needed: Patient-sequential batching + stateful LSTM

2. **[src/data_loader.py](src/data_loader.py#L161)**
   - Current: Generic sample loading, no temporal grouping
   - Needed: Patient-sequential sequence loader

3. **[src/models.py](src/models.py#L37-L128)**
   - Current: Works with fresh hidden state
   - Needed: Extend forward() to accept/return hidden state

4. **[config/default_config.yaml](config/default_config.yaml)**
   - Current: No training mode specification
   - Needed: Add `training_mode: stateful` and `batching_strategy: patient_sequential`

---

## 🚀 Next Steps: Retraining with Stateful LSTM

### Phase 1: Design (This Session)
- [ ] Design patient-sequential data loader
- [ ] Design stateful LSTM training loop
- [ ] Update model forward() to accept hidden state parameter
- [ ] Create training configuration for stateful mode

### Phase 2: Implementation (Parallel with Phase 1)
- [ ] Implement TemporalPatientDataLoader
- [ ] Modify training loop to manage hidden state
- [ ] Update model interface
- [ ] Create retrain_stateful.py script

### Phase 3: Validation (After Phase 2)
- [ ] 2-epoch smoke test
- [ ] Compare vs. old random-shuffled baseline
- [ ] Verify hidden state accumulation works correctly
- [ ] Measure gradient flow with temporal sequences

### Phase 4: Full Retraining (After Phase 3 Success)
- [ ] Run full 16+ epoch training
- [ ] Generate longitudinal validation (Phase 1 approach)
- [ ] Create new best model checkpoint
- [ ] Document results

---

## 🎯 Success Criteria

After implementing stateful LSTM training:

1. **Methodology Alignment**
   - ✓ Training batches preserve patient temporal order
   - ✓ Hidden state accumulated within patient sequences
   - ✓ Hidden state reset between patients
   - ✓ Training loop mirrors deployment exactly

2. **Performance Improvements**
   - ✓ Prediction variance increases (std > 0.05)
   - ✓ Temporal patterns emerge in heatmaps (colorful, not monochrome)
   - ✓ Token waterfall shows activation patterns (not all-black)
   - ✓ Classification metrics balanced (no 100% dominant class)

3. **Deployment Readiness**
   - ✓ Continuous inference produces stable predictions
   - ✓ No hidden state divergence over 24+ hours
   - ✓ Advance warning consistent with training patterns
   - ✓ False alarm rate acceptable (<1/hour interictal)

---

## 📝 Lessons Learned

1. **Architectural Mismatch**: LSTM + random shuffling + discarded hidden state = stateless classifier
   - LSTM architecture implies temporal modeling but was used as stateless feature extractor
   - Hidden state mechanism was present but never utilized

2. **Training ≠ Deployment**: 
   - Training with random batches is fine for feedforward networks
   - For LSTM: training mode must match deployment mode
   - Stateful training is mandatory for temporal models used in sequential inference

3. **Code Review Red Flags**:
   - Recurrent layer with hidden state computed but not stored/used
   - Batch order matters for temporal models (unlike feedforward networks)
   - Discrepancy between architecture (LSTM) and training approach (shuffled)

4. **Validation Must Be Longitudinal**:
   - ±30 min windows are invalid for LSTM models
   - Must validate over full patient timelines
   - Metrics must include temporal aspects (advance warning, stability over time)

5. **Future Prevention**:
   - Require "training mode documentation" before training session
   - Mandate matching training ↔ deployment data flow
   - Add pre-training checklist for temporal models
   - Validate hidden state handling in code review

---

## 🔐 Guard Statement

**This methodology error will not be repeated because:**

1. This document serves as deployment checklist for all future training
2. Code review process now includes temporal architecture checks
3. Stateful LSTM is the new default for seizure prediction models
4. Validation approach is fixed to longitudinal continuous inference
5. Training-deployment alignment is verified before training begins

---

**Document Owner**: AI Development Team  
**Last Updated**: 2026-07-08  
**Status**: Active Guard - Reference Before Any LSTM Training Session
