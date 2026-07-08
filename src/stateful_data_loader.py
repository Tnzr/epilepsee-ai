"""
Stateful LSTM Data Loading: Patient-Sequential Batching

Provides temporal continuity for LSTM training by:
- Grouping samples by patient
- Maintaining temporal order within each patient  
- Shuffling patient order per epoch (not sample order)
- Enabling hidden state accumulation during training

This is a wrapper around SeizureDataset that reorganizes batch construction
for stateful LSTM training (vs. random shuffled batches).
"""

import numpy as np
import logging
from typing import List, Tuple, Optional, Dict, Iterator
from collections import defaultdict

logger = logging.getLogger(__name__)


class TemporalPatientSequenceDataLoader:
    """
    Reorganizes training data for stateful LSTM training.
    
    Instead of random shuffling (which breaks temporal continuity),
    this loader provides consecutive samples from each patient.
    
    Usage in training loop:
        for patient_id, batch in temporal_loader:
            if patient_id != prev_patient:
                hidden_state = None
            lstm_out, hidden_state = model(batch, hidden_state)
            # hidden_state persists to next batch in same patient
    """
    
    def __init__(self, seizure_dataset, batch_size: int = 32, 
                 reset_hidden_between_epochs: bool = False,
                 allow_partial_batches: bool = False):
        """
        Initialize temporal patient sequence loader.
        
        Args:
            seizure_dataset: SeizureDataset instance with features, labels, subject_ids
            batch_size: Samples per batch (default 32)
            reset_hidden_between_epochs: If True, reset hidden state between epochs
            allow_partial_batches: If True, allow last batch per patient to be smaller
        """
        self.seizure_dataset = seizure_dataset
        self.batch_size = batch_size
        self.reset_hidden_between_epochs = reset_hidden_between_epochs
        self.allow_partial_batches = allow_partial_batches
        
        # Extract metadata
        if seizure_dataset.subject_ids is None:
            raise ValueError("SeizureDataset must have subject_ids for stateful training")
        
        self.subject_ids = seizure_dataset.subject_ids
        self.sample_end_times_s = seizure_dataset.sample_end_times_s
        
        # Group samples by patient, maintaining temporal order
        self._build_patient_sequences()
        
        logger.info(f"TemporalPatientSequenceDataLoader: {len(self.patient_sequences)} "
                   f"patients, batch_size={batch_size}")
    
    def _build_patient_sequences(self):
        """
        Group samples by patient ID and sort by time within each patient.
        
        Creates:
            self.patient_sequences: Dict[patient_id] = [(idx, end_time), ...]
                - indices sorted by time
                - only includes complete temporal chains
        """
        self.patient_sequences: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
        
        # Group by patient
        for idx, subject_id in enumerate(self.subject_ids):
            end_time = self.sample_end_times_s[idx] if self.sample_end_times_s is not None else idx
            self.patient_sequences[subject_id].append((idx, end_time))
        
        # Sort each patient's samples by time
        for patient_id in self.patient_sequences:
            sequences = self.patient_sequences[patient_id]
            # Sort by end time (second element of tuple)
            sequences.sort(key=lambda x: x[1])
            logger.debug(f"Patient {patient_id}: {len(sequences)} samples, "
                        f"time range {sequences[0][1]:.1f}s - {sequences[-1][1]:.1f}s")
    
    def __iter__(self) -> Iterator[Tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
        """
        Iterate through patients (shuffled order) and batches (temporal order).
        
        Yields:
            (patient_id, features_batch, labels_batch, weights_batch)
            - Each call returns one batch of temporal consecutive samples from a patient
            - When patient changes, training loop should reset hidden_state
        """
        # Shuffle patient order (not sample order)
        patient_ids = list(self.patient_sequences.keys())
        shuffled_patients = np.random.permutation(patient_ids)
        
        for patient_id in shuffled_patients:
            # Get temporal sequence for this patient
            patient_indices = [idx for idx, _ in self.patient_sequences[patient_id]]
            
            # Yield consecutive batches from this patient (temporal order preserved)
            for batch_start in range(0, len(patient_indices), self.batch_size):
                batch_end = min(batch_start + self.batch_size, len(patient_indices))
                
                # Skip partial batches unless explicitly allowed
                if not self.allow_partial_batches and batch_end - batch_start < self.batch_size:
                    if batch_end < len(patient_indices):
                        # Not last batch of patient, so something is wrong
                        logger.warning(f"Skipping partial batch: patient={patient_id}, "
                                     f"batch_size={batch_end - batch_start}")
                        continue
                
                # Get indices for this batch (consecutive temporal samples)
                batch_indices = patient_indices[batch_start:batch_end]
                
                # Fetch data from underlying dataset
                batch_features = []
                batch_labels = []
                batch_weights = []
                
                for idx in batch_indices:
                    features, label, weight = self.seizure_dataset[idx]
                    batch_features.append(features)
                    batch_labels.append(label)
                    batch_weights.append(weight)
                
                # Stack into tensors
                batch_features = np.stack(batch_features, axis=0)  # (batch, time, features)
                batch_labels = np.array(batch_labels, dtype=np.float32)  # (batch,)
                batch_weights = np.array(batch_weights, dtype=np.float32)  # (batch,)
                
                yield patient_id, batch_features, batch_labels, batch_weights
    
    def get_patient_ids(self) -> List[str]:
        """Return list of unique patient IDs."""
        return list(self.patient_sequences.keys())
    
    def get_patient_sequence_length(self, patient_id: str) -> int:
        """Return number of samples for a patient."""
        return len(self.patient_sequences[patient_id])
    
    def get_stats(self) -> Dict:
        """Return statistics about the loader."""
        total_samples = sum(len(seq) for seq in self.patient_sequences.values())
        patient_lengths = [len(seq) for seq in self.patient_sequences.values()]
        
        return {
            'num_patients': len(self.patient_sequences),
            'total_samples': total_samples,
            'batch_size': self.batch_size,
            'num_batches_per_epoch': sum(
                (len(seq) + self.batch_size - 1) // self.batch_size
                for seq in self.patient_sequences.values()
            ),
            'samples_per_patient_mean': np.mean(patient_lengths),
            'samples_per_patient_min': np.min(patient_lengths),
            'samples_per_patient_max': np.max(patient_lengths),
        }


class HiddenStateManager:
    """
    Manages LSTM hidden state across patient sequences.
    
    Handles:
    - Resetting hidden state when patient changes
    - Detaching hidden state to prevent backprop explosion
    - Tracking hidden state statistics (for debugging)
    """
    
    def __init__(self, device: str = 'cpu', detach_interval: int = 10):
        """
        Initialize hidden state manager.
        
        Args:
            device: 'cpu' or 'cuda'
            detach_interval: Detach hidden state every N batches (0 = never)
        """
        self.device = device
        self.detach_interval = detach_interval
        self.hidden_state = None
        self.current_patient_id = None
        self.batch_count_in_patient = 0
        self.stats = {
            'patient_resets': 0,
            'detaches': 0,
            'total_batches': 0
        }
    
    def update_for_batch(self, patient_id: str, hidden_state=None) -> Tuple:
        """
        Update hidden state manager for new batch.
        
        Args:
            patient_id: Current patient ID
            hidden_state: LSTM hidden state from forward pass
        
        Returns:
            Updated hidden_state (or None to initialize fresh)
        """
        # Reset if patient changed
        if patient_id != self.current_patient_id:
            self.hidden_state = None
            self.current_patient_id = patient_id
            self.batch_count_in_patient = 0
            self.stats['patient_resets'] += 1
            return None
        
        # Store hidden state for next batch
        self.hidden_state = hidden_state
        self.batch_count_in_patient += 1
        self.stats['total_batches'] += 1
        
        # Optionally detach to prevent backprop through full history
        if self.detach_interval > 0 and self.batch_count_in_patient % self.detach_interval == 0:
            if hidden_state is not None:
                if isinstance(hidden_state, tuple):
                    self.hidden_state = tuple(h.detach() for h in hidden_state)
                else:
                    self.hidden_state = hidden_state.detach()
                self.stats['detaches'] += 1
        
        return self.hidden_state
    
    def get_hidden_state(self) -> Optional:
        """Get current hidden state."""
        return self.hidden_state
    
    def reset(self):
        """Explicitly reset hidden state."""
        self.hidden_state = None
        self.current_patient_id = None
        self.batch_count_in_patient = 0
    
    def get_stats(self) -> Dict:
        """Get statistics about hidden state management."""
        return self.stats.copy()


# Example integration into training loop:
"""
# Setup
temporal_loader = TemporalPatientSequenceDataLoader(
    train_dataset,
    batch_size=32,
    allow_partial_batches=False
)
hidden_mgr = HiddenStateManager(device='cuda', detach_interval=10)

# Training
for epoch in range(num_epochs):
    hidden_mgr.reset()  # Optional: reset between epochs
    
    for patient_id, batch_features, batch_labels, batch_weights in temporal_loader:
        # Move to device
        features = batch_features.to(device)
        labels = batch_labels.to(device)
        
        # Get hidden state (None at patient start, accumulated within patient)
        hidden_state = hidden_mgr.get_hidden_state()
        
        # Forward pass with hidden state
        lstm_out, hidden_state_new = model.lstm(features, hidden_state)
        predictions = model.fc(lstm_out[:, -1, :])
        
        # Loss and backward
        loss = criterion(predictions, labels)
        loss.backward()
        
        # Update hidden state manager
        hidden_mgr.update_for_batch(patient_id, hidden_state_new)
        
        optimizer.step()
        optimizer.zero_grad()
"""
