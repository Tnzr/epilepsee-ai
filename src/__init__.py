"""
Seizure anticipation package: ECG-centered seizure countdown prediction.
"""

__version__ = "1.0.0"
__author__ = "Epilepsee-AI Team"

from .data_loader import BIDSDataLoader, SeizureDataset, SubjectDataCache, WearableDeviceDataLoader, TemporalRingBufferDataset
from .preprocessing import ECGPreprocessor, MotionDetector, SignalProcessor, SignalNormalizer
from .feature_extraction import FeatureExtractor, HRVCalculator, InstabilityExtractor, SpectralFeatureExtractor

# Wrap torch-dependent imports in try/except for optional torch dependency
try:
    from .models import ECGCountdownPredictor, CNNLSTMCountdown, MultimodalCountdownPredictor, ModelFactory
except ImportError:
    import logging
    logging.getLogger(__name__).warning("torch not available - model imports skipped")

try:
    from .losses import SeizureCountdownLoss, SeizureStateLoss, WeightedMSELoss, FocalLoss, LossFactory
except ImportError:
    pass

try:
    from .training import Trainer
except ImportError:
    pass

try:
    from .evaluation import Evaluator
except ImportError:
    pass

__all__ = [
    'BIDSDataLoader',
    'WearableDeviceDataLoader',
    'SeizureDataset',
    'TemporalRingBufferDataset',
    'SubjectDataCache',
    'ECGPreprocessor',
    'MotionDetector',
    'SignalProcessor',
    'SignalNormalizer',
    'FeatureExtractor',
    'HRVCalculator',
    'InstabilityExtractor',
    'SpectralFeatureExtractor',
    'ECGCountdownPredictor',
    'CNNLSTMCountdown',
    'MultimodalCountdownPredictor',
    'ModelFactory',
    'SeizureCountdownLoss',
    'WeightedMSELoss',
    'FocalLoss',
    'LossFactory',
    'Trainer',
    'Evaluator',
]
