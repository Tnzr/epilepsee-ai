"""
Configuration management for seizure anticipation project.
Centralized settings for data, model, training, and evaluation.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional
import yaml
import os


@dataclass
class DataConfig:
    """Data loading and preprocessing configuration."""
    
    # Dataset paths — override via env vars BIDS_DATASET_ROOT / WEARABLE_DATASET_ROOT
    # or by passing --dataset-root on the CLI.
    dataset_root: str = field(default_factory=lambda: os.environ.get('BIDS_DATASET_ROOT', 'data/ds005873'))
    output_dir: str = field(default_factory=lambda: os.environ.get('OUTPUT_DIR', './data'))
    
    # BIDS structure
    bids_validate: bool = False
    
    # Signal specifications
    sampling_rate_high: int = 250  # ECG, EEG, EMG
    sampling_rate_motion: int = 25  # ACC, GYR
    
    # Preprocessing
    ecg_lowcut: float = 0.5  # Hz
    ecg_highcut: float = 40.0  # Hz
    motion_threshold: float = 1.0  # m/s²
    
    # Feature extraction
    feature_window_s: float = 60.0  # 60-second windows
    feature_step_s: float = 1.0  # 1-second stride
    
    # Seizure labeling
    pre_ictal_window_s: float = 600.0  # 10 minutes
    
    # Data splitting
    train_ratio: float = 0.60
    val_ratio: float = 0.15
    test_ratio: float = 0.25
    
    # CV strategy
    cv_strategy: str = "loso"  # "loso" or "lopo"
    random_seed: int = 42

    # Data source tag used by downstream components (e.g., visualization)
    # to adjust modality-specific behavior such as HR/HRV computation.
    # Expected values: "bids" (SeizeIT2 ECG) or "wearable" (OHSU PPG).
    data_source: str = "bids"

    # Safety cap for real-mode wearable datasets to avoid host OOM.
    # When > 0 and using data_source="wearable", the total number of
    # windowed samples across all recordings will be randomly trimmed to
    # at most this value after feature extraction but before splitting.
    # This keeps memory usage bounded even when feature_step_s is small
    # and recordings are long.
    max_wearable_global_samples: int = 50000
    
    # Wearable-specific recording selection controls.
    # When > 0 and using data_source="wearable", limit the number of
    # recordings that contain no seizures to this value. This prevents
    # hundreds of long purely interictal recordings from dominating the
    # windowed dataset and exhausting host RAM without improving preictal
    # coverage.
    max_wearable_nonseizure_recordings: int = 40
    
    # Data augmentation for preictal samples
    augment_preictal: bool = True
    augmentation_factor: int = 15  # Augment each preictal sample N times
    aug_time_warp_rates: List[float] = field(default_factory=lambda: [0.90, 0.95, 1.05, 1.10])
    aug_amplitude_scales: List[float] = field(default_factory=lambda: [0.85, 0.90, 1.10, 1.15])
    aug_noise_levels: List[float] = field(default_factory=lambda: [0.01, 0.02, 0.03])
    aug_time_shifts: List[int] = field(default_factory=lambda: [-5, -3, 3, 5])


@dataclass
class ModelConfig:
    """Neural network architecture configuration."""
    
    # Input dimensions
    ecg_feature_dim: int = 14
    eeg_feature_dim: int = 6
    motion_feature_dim: int = 2
    
    # Architecture type
    # Single-input:  ecg_lstm | cnn_lstm | tcn | eegnet | mobilenet_1d |
    #                inception_1d | temporal_transformer
    # Multimodal:    multimodal | multimodal_transformer
    model_type: str = "ecg_lstm"
    
    # LSTM parameters
    hidden_dim: int = 128
    num_lstm_layers: int = 2
    bidirectional: bool = True
    
    # CNN parameters (for CNN-LSTM)
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    conv_kernel_sizes: List[int] = field(default_factory=lambda: [3, 3, 3])
    pool_size: int = 2
    
    # Dropout & regularization
    dropout: float = 0.3
    use_batch_norm: bool = True
    
    # Attention
    use_attention: bool = True
    num_attention_heads: int = 4
    
    # Multimodal fusion
    fusion_type: str = "early"  # "early", "late", or "hybrid"
    
    # TCN parameters (model_type="tcn")
    # Channels per residual block; dilation doubles each level: 1,2,4,8,...
    tcn_num_channels: List[int] = field(default_factory=lambda: [64, 64, 128, 128])
    tcn_kernel_size: int = 3

    # Transformer parameters (model_type="temporal_transformer" or "multimodal_transformer")
    transformer_d_model: int = 64         # embedding dimension
    transformer_nhead: int = 4            # attention heads (must divide d_model)
    transformer_num_layers: int = 4       # encoder layers
    transformer_dim_feedforward: int = 256  # FFN hidden size

    # Temporal Confidence Head (stage-2 lightweight smoother, ~8k params)
    # Enable to attach a TemporalConfidenceHead on top of any base model.
    temporal_conf_head: bool = False
    temporal_conf_history_k: int = 32     # ring-buffer length (recent predictions)
    temporal_conf_hidden_ch: int = 16     # TCN hidden channels

    # Output
    output_countdown_max: float = 10.0  # minutes
    
    # Initialization
    init_method: str = "xavier"  # "xavier", "kaiming", or "normal"


@dataclass
class LossConfig:
    """Loss function configuration."""
    
    # Multi-task learning weights (normalized during training)
    classification_weight: float = 0.3
    regression_weight: float = 0.7
    ranking_weight: float = 0.0
    
    # Regression loss type
    regression_loss: str = "weighted_mse"  # "mse", "weighted_mse", or "smoothl1"
    
    # Time-based weighting
    weight_tau: float = 60.0  # seconds (exponential time constant)
    
    # Label smoothing
    label_smoothing: float = 0.0
    
    # Classification imbalance handling
    use_class_weighting: bool = True
    classification_positive_weight: Optional[float] = None

    # Classification loss type and focal parameters.
    # "bce"   → standard BCE (optionally class-weighted)
    # "focal" → focal BCE variant for strong imbalance.
    classification_loss_type: str = "bce"  # "bce" or "focal"
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    
    # Decision threshold for binary classification (lower threshold increases sensitivity)
    detection_threshold: float = 0.50  # Base threshold; epoch panels can adapt from validation data


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    
    # Optimization
    optimizer: str = "adam"  # "adam", "sgd", or "adamw"
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    momentum: float = 0.9  # for SGD
    
    # Learning rate schedule
    lr_scheduler: str = "reduce_on_plateau"  # "reduce_on_plateau", "cosine", "step", or "exponential"
    lr_factor: float = 0.5
    lr_patience: int = 10
    lr_min: float = 1e-6
    
    # Training parameters
    num_epochs: int = 100
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    
    # Gradient handling
    gradient_clip_norm: float = 1.0
    gradient_accumulation_steps: int = 1

    # Sampling strategy (non-distributed only). By default we rely on
    # loss-level class weighting and temporal weighting rather than a
    # WeightedRandomSampler, which can otherwise double-correct the
    # imbalance and distort the effective data distribution.
    use_weighted_sampling: bool = False

    # Preserve chronological sample order for day-like continuous training.
    # When enabled, dataloaders avoid randomization so training and validation
    # iterate in timeline order within each split.
    long_sweep_training: bool = False

    # Epoch panel visualization window (minutes). Set to >=180 for longitudinal
    # wearable context in both local files and W&B image logging.
    epoch_panel_window_minutes: float = 180.0
    
    # Early stopping
    early_stopping: bool = True
    early_stopping_patience: int = 15
    early_stopping_metric: str = "val_mae"
    
    # Distributed training (Multi-GPU)
    distributed: bool = True
    num_gpus: int = 2
    backend: str = "nccl"  # "nccl" for GPU, "gloo" for CPU
    
    # Checkpointing
    save_dir: str = field(default_factory=lambda: os.environ.get('OUTPUT_DIR', './models'))
    save_interval: int = 5  # Save every N epochs
    save_best_only: bool = True
    resume_from: Optional[str] = None
    
    # Logging
    log_dir: str = field(default_factory=lambda: os.path.join(os.environ.get('OUTPUT_DIR', '.'), 'logs'))
    log_interval: int = 100  # Log every N batches
    use_tensorboard: bool = True


@dataclass
class EvaluationConfig:
    """Evaluation and metrics configuration."""
    
    # Metrics
    compute_mae: bool = True
    compute_rmse: bool = True
    compute_medae: bool = True
    
    # Lead time thresholds (minutes)
    lead_time_thresholds: List[float] = field(default_factory=lambda: [3.0, 5.0, 10.0])

    # Countdown error buckets (minutes before seizure)
    # Interpreted as contiguous ranges: [b0,b1), [b1,b2), ... with final bucket [b(n-1), bn]
    countdown_metric_buckets: List[float] = field(default_factory=lambda: [0.0, 2.0, 5.0, 10.0])
    
    # FPR evaluation
    fpr_window_hours: float = 8.0
    
    # Stability metrics
    stability_window: int = 30  # samples
    
    # Classification metrics (pre-ictal detection)
    classification_thresholds: List[float] = field(default_factory=lambda: [0.3, 0.5, 0.7])
    
    # Cross-validation
    compute_loso_cv: bool = True
    num_cv_folds: Optional[int] = None  # If None, use all seizures (full LOSO)


@dataclass
class Config:
    """Master configuration class combining all sub-configs."""
    
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Experiment tracking
    experiment_name: str = "seizure_anticipation_baseline"
    description: str = ""
    tags: List[str] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Config':
        """Load configuration from YAML file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'Config':
        """Create config from dictionary."""
        config = cls()
        
        if 'data' in config_dict:
            config.data = DataConfig(**config_dict['data'])
        if 'model' in config_dict:
            config.model = ModelConfig(**config_dict['model'])
        if 'loss' in config_dict:
            config.loss = LossConfig(**config_dict['loss'])
        if 'training' in config_dict:
            config.training = TrainingConfig(**config_dict['training'])
        if 'evaluation' in config_dict:
            config.evaluation = EvaluationConfig(**config_dict['evaluation'])
        
        if 'experiment_name' in config_dict:
            config.experiment_name = config_dict['experiment_name']
        if 'description' in config_dict:
            config.description = config_dict['description']
        if 'tags' in config_dict:
            config.tags = config_dict['tags']
        
        return config
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary."""
        return {
            'data': self.data.__dict__,
            'model': self.model.__dict__,
            'loss': self.loss.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'experiment_name': self.experiment_name,
            'description': self.description,
            'tags': self.tags,
        }
    
    def save_yaml(self, output_path: str) -> None:
        """Save configuration to YAML file."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
    
    def __str__(self) -> str:
        """Pretty print configuration."""
        import json
        return json.dumps(self.to_dict(), indent=2, default=str)


# Default configuration
DEFAULT_CONFIG = Config()
