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
from math import ceil
from typing import List, Tuple, Optional, Dict, Iterator, Sequence
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
                 allow_partial_batches: bool = False,
                 patient_subset: Optional[Sequence[str]] = None,
                 shuffle_patients: bool = True):
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
        self.shuffle_patients = shuffle_patients
        
        # Extract metadata
        if seizure_dataset.subject_ids is None:
            raise ValueError("SeizureDataset must have subject_ids for stateful training")
        
        self.subject_ids = seizure_dataset.subject_ids
        self.sample_end_times_s = seizure_dataset.sample_end_times_s
        self.patient_subset = None if patient_subset is None else {str(p) for p in patient_subset}
        
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

        if self.patient_subset is not None:
            self.patient_sequences = {
                patient_id: seq for patient_id, seq in self.patient_sequences.items()
                if str(patient_id) in self.patient_subset
            }

        self._num_batches = sum(
            (len(seq) + self.batch_size - 1) // self.batch_size
            for seq in self.patient_sequences.values()
            if self.allow_partial_batches or len(seq) >= self.batch_size
        )
    
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
        shuffled_patients = np.random.permutation(patient_ids) if self.shuffle_patients else patient_ids
        
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
                # Handle both tensor and numpy returns from underlying dataset
                def _to_numpy(x):
                    import torch
                    return x.numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

                batch_features = np.stack([_to_numpy(f) for f in batch_features], axis=0)  # (batch, time, features)
                batch_labels = np.array([_to_numpy(l).item() if hasattr(_to_numpy(l), 'item') else float(l) for l in batch_labels], dtype=np.float32)
                batch_weights = np.array([_to_numpy(w).item() if hasattr(_to_numpy(w), 'item') else float(w) for w in batch_weights], dtype=np.float32)
                
                yield patient_id, batch_features, batch_labels, batch_weights

    def __len__(self) -> int:
        return int(self._num_batches)
    
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
        
        if len(patient_lengths) == 0:
            return {
                'num_patients': 0,
                'total_samples': 0,
                'batch_size': self.batch_size,
                'num_batches_per_epoch': 0,
                'samples_per_patient_mean': 0.0,
                'samples_per_patient_min': 0,
                'samples_per_patient_max': 0,
            }
        
        return {
            'num_patients': len(self.patient_sequences),
            'total_samples': total_samples,
            'batch_size': self.batch_size,
            'num_batches_per_epoch': sum(
                (len(seq) + self.batch_size - 1) // self.batch_size
                for seq in self.patient_sequences.values()
            ),
            'samples_per_patient_mean': float(np.mean(patient_lengths)),
            'samples_per_patient_min': int(np.min(patient_lengths)),
            'samples_per_patient_max': int(np.max(patient_lengths)),
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


class IndexedTemporalDatasetView:
    """Metadata-preserving index view over a dataset.

    This wraps SeizureDataset/LazyRealDataset-like objects and restricts them to
    a chosen list of indices while preserving patient/timeline metadata needed
    for stateful patient-sequential training.
    """

    def __init__(self, dataset, indices: Sequence[int]):
        self._dataset = dataset
        self._indices = np.asarray(indices, dtype=np.int64)

        for attr in ('subject_ids', 'sample_end_times_s', 'recording_ids', 'labels', 'preictal_labels', 'weights'):
            if hasattr(dataset, attr):
                value = getattr(dataset, attr)
                if value is not None:
                    try:
                        setattr(self, attr, np.asarray(value)[self._indices])
                    except Exception:
                        setattr(self, attr, value)
                else:
                    setattr(self, attr, None)

    def __len__(self):
        return int(len(self._indices))

    def __getitem__(self, idx: int):
        return self._dataset[int(self._indices[idx])]

    @property
    def class_distribution(self) -> Dict[str, int]:
        if hasattr(self, 'preictal_labels') and self.preictal_labels is not None:
            lbl = np.asarray(self.preictal_labels, dtype=np.float32)
            return {
                'preictal': int(np.sum(lbl > 0.5)),
                'interictal': int(len(lbl) - np.sum(lbl > 0.5)),
            }
        if hasattr(self._dataset, 'class_distribution'):
            return self._dataset.class_distribution
        return {'preictal': 0, 'interictal': 0}


def partition_patients_by_batch_count(
    dataset,
    batch_size: int,
    world_size: int,
) -> List[List[str]]:
    """Greedily assign full patients to ranks while balancing batch counts."""
    if getattr(dataset, 'subject_ids', None) is None:
        raise ValueError('Stateful distributed partitioning requires subject_ids')

    subject_ids = np.asarray(dataset.subject_ids).astype(str)
    patient_ids = np.unique(subject_ids)
    patient_rows = []
    for patient_id in patient_ids:
        n_samples = int(np.sum(subject_ids == patient_id))
        n_batches = int(ceil(n_samples / max(batch_size, 1)))
        patient_rows.append((patient_id, n_samples, n_batches))

    patient_rows.sort(key=lambda row: (row[2], row[1]), reverse=True)
    assignments: List[List[str]] = [[] for _ in range(max(world_size, 1))]
    loads = [0 for _ in range(max(world_size, 1))]

    for patient_id, _, n_batches in patient_rows:
        best_rank = min(range(len(assignments)), key=lambda rank: loads[rank])
        assignments[best_rank].append(str(patient_id))
        loads[best_rank] += n_batches

    logger.info(
        'Distributed patient partitioning by batch count: %s',
        ', '.join([f'rank{rank}={loads[rank]} batches/{len(assignments[rank])} patients' for rank in range(len(assignments))])
    )
    return assignments


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
