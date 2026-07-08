# Stateful LSTM Training Architecture Design
**Date:** 2026-07-08  
**Purpose:** Redesign training pipeline to use stateful LSTM with patient-sequential batching  
**Status:** Design Phase - Ready for Implementation

---

## 1. Current Architecture (Random Shuffled - BROKEN FOR DEPLOYMENT)

```
Training Data Flow:
  SeizeIT2 Dataset (100 patients, ~28 recordings each)
    ↓
  BIDSDataLoader → extracts individual 960-sample windows
    ↓
  DistributedSampler(shuffle=True) → RANDOM ORDER
    ↓
  DataLoader → batches of 32 random samples (may mix multiple patients)
    ↓
  Training Loop:
    for batch in random_order_batches:
        hidden_state = None  # ⚠️ RESET per batch (fresh zeros)
        lstm_out, (h_n, c_n) = self.lstm(batch, hidden_state)
        # h_n, c_n computed but DISCARDED - not passed to next batch
        predictions = model(lstm_out[-1, :])
        loss.backward()
        optimizer.step()
```

**Problem:** 
- Random batches prevent temporal learning
- Hidden state reset destroys temporal continuity
- Model learns stateless classifier (equivalent to using final window only)
- Incompatible with continuous deployment

---

## 2. Target Architecture (Stateful LSTM - CORRECT FOR DEPLOYMENT)

```
Training Data Flow:
  SeizeIT2 Dataset (100 patients, ~28 recordings each)
    ↓
  TemporalPatientDataLoader → extracts patient-SEQUENTIAL windows
    ├─ Patient 001: sample 0-32, 32-64, 64-96, ... (SEQUENTIAL ORDER)
    ├─ Patient 002: sample 0-32, 32-64, 64-96, ... (SEQUENTIAL ORDER)  
    └─ Patient 100: sample 0-32, 32-64, 64-96, ... (SEQUENTIAL ORDER)
    ↓
  Epoch shuffles PATIENT ORDER (not sample order)
    ↓
  Training Loop:
    for epoch in epochs:
        patient_list = shuffle(patients)  # Shuffle patient order
        for patient_id in patient_list:
            hidden_state = None  # Initialize at PATIENT START
            for batch in patient_sequential_batches:
                lstm_out, hidden_state = self.lstm(batch, hidden_state)
                # ✓ hidden_state ACCUMULATED across batches
                # ✓ hidden_state CARRIED to next batch in same patient
                predictions = model(lstm_out[-1, :])
                loss.backward()
                # Optionally detach hidden_state to avoid backprop through full history
                hidden_state = (h.detach() for h in hidden_state)
                optimizer.step()
```

**Benefits:**
- Temporal continuity within each patient
- Hidden state accumulates across time
- Matches deployment exactly
- Model learns temporal patterns, not stateless tricks

---

## 3. Component Design

### 3.1 TemporalPatientDataLoader

**Purpose:** Replace BIDSDataLoader for stateful training

**Input:** SeizeIT2 dataset path, patient list  
**Output:** Patient-sequential batches

**Key Features:**
- Groups samples by patient
- Returns consecutive time windows from same patient (NO shuffling within patient)
- Supports multi-epoch iteration (shuffles patient order per epoch)
- Respects batch boundaries (no mixing patients)

**Pseudocode:**
```python
class TemporalPatientSequenceDataLoader:
    def __init__(self, dataset_root, patient_ids, batch_size=32, sequence_overlap=0):
        """
        dataset_root: path to SeizeIT2
        patient_ids: list of patient IDs (e.g., ['sub-001', 'sub-002', ...])
        batch_size: 32 samples per batch
        sequence_overlap: windows may overlap for continuity (default 0=no overlap)
        """
        self.dataset_root = dataset_root
        self.patient_ids = patient_ids
        self.batch_size = batch_size
        self.sequence_overlap = sequence_overlap
        self.patient_data = {}
        
        # Preload all patient data (can be optimized with lazy loading)
        for patient_id in patient_ids:
            ecg, labels = self._load_patient(patient_id)
            self.patient_data[patient_id] = {
                'ecg': ecg,      # shape: (n_samples, 14)
                'labels': labels # shape: (n_samples,)
            }
    
    def __iter__(self):
        """
        Yields: (patient_id, ecg_batch, label_batch)
        - Iterates through patients (shuffled order)
        - For each patient: yields consecutive batches (no shuffling)
        """
        # Shuffle patient order (not sample order)
        shuffled_patients = np.random.permutation(self.patient_ids)
        
        for patient_id in shuffled_patients:
            ecg = self.patient_data[patient_id]['ecg']
            labels = self.patient_data[patient_id]['labels']
            
            # Yield consecutive batches from this patient
            n_samples = len(ecg)
            for start_idx in range(0, n_samples - self.batch_size, self.batch_size):
                end_idx = start_idx + self.batch_size
                yield patient_id, ecg[start_idx:end_idx], labels[start_idx:end_idx]
    
    def _load_patient(self, patient_id):
        # Load ECG and labels from BIDS structure
        # Returns: (ecg, labels) - both shape compatible with training loop
        pass
```

**Implementation Notes:**
- Could lazy-load to reduce memory footprint
- Could use multiprocessing for parallel sample fetching
- For now: preload to keep implementation simple

---

### 3.2 Modified LSTM Forward Pass

**Current (Wrong):**
```python
class ECGCountdownPredictor(nn.Module):
    def forward(self, x):  # x: (batch, time, features)
        lstm_out, (h_n, c_n) = self.lstm(x)  # h_n, c_n: hidden states
        # h_n, c_n DISCARDED - not returned
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out[:, -1, :]
        pre_ictal = self.fc_class(pooled)
        countdown = self.fc_regress(pooled)
        return pre_ictal, countdown
```

**Target (Correct):**
```python
class ECGCountdownPredictor(nn.Module):
    def forward(self, x, hidden_state=None):  # x: (batch, time, features)
        # ✓ Accept hidden_state from previous batch
        lstm_out, hidden_state = self.lstm(x, hidden_state)
        # ✓ Return hidden_state for next batch
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
        pooled = attn_out[:, -1, :]
        pre_ictal = self.fc_class(pooled)
        countdown = self.fc_regress(pooled)
        return pre_ictal, countdown, hidden_state  # ✓ Return hidden state
```

**Type Signature:**
- Input: `x: Tensor(batch, time, features), hidden_state: Tuple[Tensor, Tensor] | None`
- Output: `(pre_ictal: Tensor, countdown: Tensor, hidden_state: Tuple[Tensor, Tensor])`

**Backward Compatibility:**
- Can detect when hidden_state=None (supports old code paths)
- Default behavior: if called without hidden_state, behaves like old version
- Stateful training: explicitly pass hidden_state between calls

---

### 3.3 Modified Training Loop

**Current (Random Shuffled):**
```python
for epoch in range(num_epochs):
    train_sampler.set_epoch(epoch)
    for batch_idx, batch in enumerate(train_loader):  # Random order
        
        features = batch[0].to(device)
        # ⚠️ Fresh hidden state per batch
        lstm_out, _ = self.lstm(features)
        
        pre_ictal_pred, countdown_pred = self.fc_class(lstm_out[-1, :]), ...
        loss = compute_loss(pred, label)
        
        loss.backward()
        optimizer.step()
```

**Target (Stateful Sequential):**
```python
for epoch in range(num_epochs):
    for patient_id, batch in temporal_loader:  # Sequential batches
        
        # Patient changed: reset hidden state
        if patient_id != prev_patient_id:
            hidden_state = None
        
        features = batch[0].to(device)
        
        # ✓ Pass hidden state from previous batch
        lstm_out, hidden_state = self.lstm(features, hidden_state)
        
        # ✓ Optional: Detach to avoid backprop through full history
        # This limits gradient signal to recent N batches (~256 timesteps)
        # vs full patient history (can be 10,000+ samples)
        if epoch % checkpoint_interval == 0:
            hidden_state = tuple(h.detach() for h in hidden_state)
        
        pre_ictal_pred, countdown_pred = self.fc_class(lstm_out[-1, :]), ...
        loss = compute_loss(pred, label)
        
        loss.backward()
        optimizer.step()
        
        prev_patient_id = patient_id
```

**Key Differences:**
1. **Hidden State Management**
   - Initialize: `None` at patient start
   - Accumulate: Same `hidden_state` across batches
   - Reset: When patient changes
   - Detach: Optional, every N batches (prevents backprop explosion)

2. **Loss Computation**
   - Same loss function (no change needed)
   - Gradients flow backward through recent history
   - Detaching controls how far gradients can flow

3. **Metrics Tracking**
   - Per-batch: same as before
   - Per-patient: NEW - track metrics across patient sequence
   - Per-epoch: aggregated across all patient sequences

---

### 3.4 Configuration Updates

**Add to `config/default_config.yaml`:**

```yaml
training:
  # NEW: Training mode specification
  training_mode: "stateful"  # "stateful" or "stateless"
  
  # NEW: Stateful LSTM parameters
  stateful_lstm:
    enabled: true
    hidden_state_detach_interval: 10  # Detach every 10 batches within patient
    max_patient_sequence_length: 10000  # Warn if patient data exceeds this
    reset_hidden_between_epochs: false  # Should hidden state carry between epochs?
  
  # Existing parameters (unchanged)
  batch_size: 32
  num_epochs: 16
  learning_rate: 0.001
  optimizer: "adam"
  scheduler: "reduce_on_plateau"
```

**Rationale:**
- `training_mode`: Allows toggling between stateful and stateless for experiments
- `hidden_state_detach_interval`: Trade-off between gradient flow and stability
  - Small (1-5): Gradients can flow through full patient history → more learning but numerical instability
  - Large (50+): Gradients limited to recent batches → more stable but limited temporal learning
  - Recommended: 10 = ~320 timesteps = ~5 minutes of prior context for gradients
- `max_patient_sequence_length`: Safety check to detect unexpectedly long sequences

---

## 4. Implementation Strategy

### Phase 1: Implement Components (Parallel)

**Task 1.1:** Implement `TemporalPatientDataLoader`
- File: `src/data_loader.py` (new class)
- Dependencies: BIDSDataLoader logic (can reuse sample loading)
- Time: ~2 hours
- Validation: Test that batches are consecutive per patient, shuffled per epoch

**Task 1.2:** Modify `ECGCountdownPredictor.forward()`
- File: `src/models.py` 
- Changes: Add `hidden_state` parameter, return hidden state
- Time: ~30 minutes
- Validation: Verify backward compatibility (hidden_state=None works)

**Task 1.3:** Modify Training Loop
- File: `src/training.py` (modify training function)
- Changes: Integrate temporal loader, manage hidden state
- Time: ~2 hours
- Validation: Test on 2-epoch smoke test

**Task 1.4:** Create Retrain Configuration
- File: `config/stateful_lstm.yaml` (new)
- Based on: `default_config.yaml` + stateful-specific settings
- Time: ~30 minutes

### Phase 2: Integration & Testing

**Task 2.1:** Integration Test
- Combine all components
- Verify data flows correctly through full pipeline
- Check GPU memory usage
- Time: ~1 hour

**Task 2.2:** 2-Epoch Smoke Test
- Run training on 5-10 patients only
- Verify loss decreases (not flat)
- Check hidden state is accumulating (layer norm drift?)
- Time: ~30 minutes (5-10 min per epoch on 2 RTX 3060s)

**Task 2.3:** Compare vs. Old Training
- Run 2 epochs old way (random shuffled)
- Run 2 epochs new way (stateful sequential)
- Compare metrics:
  - Loss trajectory (should both decrease)
  - Prediction variance (stateful should higher?)
  - Heatmap patterns (stateful should show temporal patterns?)
- Time: ~1 hour

### Phase 3: Full Retraining

**Task 3.1:** Full 16+ Epoch Training
- On all patients
- On 2× RTX 3060 via DDP
- Estimated time: ~6.5 minutes (same as before)
- Output: `models/best_model_stateful.pt`

**Task 3.2:** Longitudinal Validation (Phase 1 from previous plan)
- Run continuous inference on 5-10 validation patients
- Generate traces, seizure neighborhoods, metrics
- Time: ~2-4 hours

---

## 5. Risk Mitigation

### 5.1 Memory Concerns
**Risk:** LSTM hidden state grows with batch size and patient length  
**Mitigation:**
- Start with batch_size=32 (same as before)
- Monitor GPU memory during smoke test
- Implement hidden_state_detach to prevent backprop explosion
- Use mixed precision (amp) if needed

### 5.2 Gradient Flow Issues
**Risk:** Gradients don't flow properly through long sequences  
**Mitigation:**
- Detach hidden state every 10 batches (∼5 min context)
- This allows gradients to flow locally but not through full patient sequence
- Comparable to truncated BPTT (backprop through time)

### 5.3 Numerical Stability
**Risk:** Long sequences → accumulated numerical errors  
**Mitigation:**
- Monitor hidden state norm during training
- Use layer normalization in LSTM (already present?)
- Check for NaN/inf values
- Log hidden state statistics per epoch

### 5.4 Training Data Imbalance
**Risk:** Class imbalance (90:10) may reappear  
**Mitigation:**
- Keep class weighting enabled (pos_weight=15.0)
- Monitor classification metrics per batch
- Use stratified patient split (ensure train/val/test have similar seizure density)

---

## 6. Success Criteria

**Smoke Test (2-Epoch):**
- ✓ Training completes without NaN/inf
- ✓ Loss decreases per epoch (0.85 → 0.83)
- ✓ Prediction variance > 0.05
- ✓ No GPU OOM errors

**Full Training (16-Epoch):**
- ✓ Best validation MAE < 10 minutes
- ✓ Classification accuracy > 50% (balanced)
- ✓ Heatmap shows temporal patterns (colorful, not monochrome)
- ✓ Token activation shows multi-channel patterns

**Longitudinal Validation:**
- ✓ Continuous inference runs without errors
- ✓ Advance warning time consistent (e.g., 5-15 min before seizure)
- ✓ False alarm rate acceptable (< 1/hour interictal)
- ✓ Hidden state stable over 24+ hours

---

## 7. Implementation Timeline

| Phase | Tasks | Estimated Time | Blocker |
|-------|-------|-----------------|---------|
| Design | (This document) | — | ✓ Complete |
| 1: Components | 1.1-1.4 | 5-6 hours | None |
| 2: Testing | 2.1-2.3 | 2-3 hours | Phase 1 complete |
| 3: Retraining | 3.1-3.2 | 3-4 hours | Phase 2 successful |
| **Total** | | **10-13 hours** | |

---

## 8. Files to Modify/Create

**New Files:**
- `src/data_loader_stateful.py` - TemporalPatientSequenceDataLoader class
- `config/stateful_lstm.yaml` - Stateful training configuration
- `scripts/retrain_stateful.py` - Script to run stateful training

**Modified Files:**
- `src/models.py` - Add hidden_state parameter to forward()
- `src/training.py` - Integrate temporal loader and hidden state management
- `config/default_config.yaml` - Add training_mode and stateful_lstm settings

**Documentation:**
- `STATEFUL_LSTM_TRAINING_GUIDE.md` - How to use stateful training
- `TRAINING_METHODOLOGY_GUARD.md` - (Already created) Prevention guide

---

## 9. Next Steps

1. ✅ Approve design (THIS DOCUMENT)
2. ⏳ Implement Phase 1 components
3. ⏳ Run Phase 2 smoke test
4. ⏳ Run Phase 3 full retraining
5. ⏳ Run longitudinal validation (Phase 1 approach)

