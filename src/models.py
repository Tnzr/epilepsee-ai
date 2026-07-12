"""
Neural network architectures for seizure countdown prediction.

Models:
- ECGCountdownPredictor:         BiLSTM + Attention (ECG-only)
- CNNLSTMCountdown:              CNN-LSTM hybrid (ECG-only)
- MultimodalCountdownPredictor:  Multimodal early fusion (ECG+EEG+Motion)

TinyML / Edge (sorted smallest → largest):
- EEGNetCountdown:               ~3k params  — ESP32 / microcontrollers
- MobileNetCountdown:            ~60k params — Smartwatch / IoT gateway
- TCNCountdown:                  ~280k params — Capable wearable / edge server
- InceptionTime1DCountdown:      ~430k params — Edge server / gateway

Server / Cloud:
- TemporalTransformerCountdown:       ~500k params — Best accuracy (ECG-only)
- MultimodalTransformerCountdown:     ~1.5M params — Best accuracy (multimodal)

Deployment guide:
    ESP32 (256 KB SRAM):    eegnet   → INT8 quantize to <10 KB
    Smartwatch (CoreML):    mobilenet_1d → INT8 ~200 KB
    Edge server / gateway:  tcn  or  inception_1d
    Cloud server:       temporal_transformer  or  multimodal_transformer
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils.parametrizations import weight_norm
from typing import Tuple, Optional, List

from config.config import ModelConfig


class ECGCountdownPredictor(nn.Module):
    """
    Bidirectional LSTM model for ECG-only seizure countdown prediction.
    
    Architecture:
    - Bidirectional LSTM layers
    - Temporal attention
    - Multi-task heads (classification + regression)
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize ECG countdown predictor.
        
        Args:
            config: ModelConfig with architecture specifications
        """
        super().__init__()
        self.config = config
        
        input_dim = config.ecg_feature_dim
        hidden_dim = config.hidden_dim
        num_layers = config.num_lstm_layers
        dropout = config.dropout
        
        # Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2  # bidirectional
        
        # Temporal attention
        if config.use_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=lstm_output_dim,
                num_heads=config.num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
        else:
            self.attention = None
        
        # Classification head (pre-ictal detection)
        self.fc_class = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # If using coherent heads, regression head takes pooled + class pre-activation
        self.coherent_heads = getattr(config, 'coherent_heads', False)
        regress_in_dim = lstm_output_dim + (1 if self.coherent_heads else 0)
        self.fc_regress = nn.Sequential(
            nn.Linear(regress_in_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        self.last_context_state: Optional[torch.Tensor] = None
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.xavier_uniform_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor, hidden_state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) \
            -> Tuple[torch.Tensor, torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """Forward pass with optional stateful LSTM.
        
        Args:
            x: (batch, time_steps, features)
            hidden_state: Optional (h, c) tuple for stateful LSTM. If None, initialize fresh.
                         Used for maintaining temporal context across batches in training.
        
        Returns:
            Tuple of (pre_ictal_prob, countdown_minutes, hidden_state)
            - For backward compatibility, hidden_state is only returned if requested
        """
        use_cudnn_rnn = bool(getattr(self.config, 'use_cudnn_rnn', True))

        # LSTM with optional hidden state
        if hidden_state is None:
            if use_cudnn_rnn:
                lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, time, hidden*2)
            else:
                # Avoid cuDNN RNN mode-mismatch backward failures in long-running
                # distributed jobs with intermittent eval checkpoints.
                with torch.backends.cudnn.flags(enabled=False):
                    lstm_out, (h_n, c_n) = self.lstm(x)
        else:
            if use_cudnn_rnn:
                lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)  # (batch, time, hidden*2)
            else:
                with torch.backends.cudnn.flags(enabled=False):
                    lstm_out, (h_n, c_n) = self.lstm(x, hidden_state)
        
        # Attention
        if self.attention is not None:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            # Use final timestep from attention output
            pooled = attn_out[:, -1, :]  # (batch, hidden*2)
        else:
            # Pooling: use final timestep
            pooled = lstm_out[:, -1, :]  # (batch, hidden*2)

        self.last_context_state = pooled.detach()
        
        # Classification head
        pre_ictal_prob = self.fc_class[0](pooled)
        pre_ictal_prob = self.fc_class[1](pre_ictal_prob)
        pre_ictal_prob = self.fc_class[2](pre_ictal_prob)
        pre_ictal_logits = self.fc_class[3](pre_ictal_prob)  # (batch, 1) before sigmoid
        pre_ictal_prob = self.fc_class[4](pre_ictal_logits)  # sigmoid

        # Regression head (optionally receives class pre-activation)
        if self.coherent_heads:
            regress_input = torch.cat([pooled, pre_ictal_logits], dim=-1)
        else:
            regress_input = pooled
        countdown = self.fc_regress(regress_input)  # (batch, 1)
        countdown = torch.sigmoid(countdown) * self.config.output_countdown_max

        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1), (h_n, c_n)


class CNNLSTMCountdown(nn.Module):
    """
    CNN-LSTM hybrid model combining local feature extraction with temporal modeling.
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize CNN-LSTM model.
        
        Args:
            config: ModelConfig with architecture specifications
        """
        super().__init__()
        self.config = config
        
        input_dim = config.ecg_feature_dim
        conv_channels = config.conv_channels
        kernel_sizes = config.conv_kernel_sizes
        hidden_dim = config.hidden_dim
        dropout = config.dropout
        
        # 1D Convolutional layers
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.batch_norms_conv = nn.ModuleList()
        
        in_channels = input_dim
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            self.conv_layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            )
            if config.use_batch_norm:
                self.batch_norms_conv.append(nn.BatchNorm1d(out_channels))
            self.pool_layers.append(nn.MaxPool1d(2))
            in_channels = out_channels
        
        # Calculate size after conv/pooling
        # Assuming time dimension halves with each pool
        final_conv_dim = conv_channels[-1]
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=final_conv_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # Classification head
        self.fc_class = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Regression head
        self.fc_regress = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: (batch, time_steps, features)
        
        Returns:
            Tuple of (pre_ictal_prob, countdown_minutes)
        """
        # Transpose for Conv1d: (batch, features, time_steps)
        x = x.transpose(1, 2)
        
        # CNN layers
        for conv, pool, bn in zip(self.conv_layers, self.pool_layers,
                                  self.batch_norms_conv if self.config.use_batch_norm else [None]*len(self.conv_layers)):
            x = conv(x)
            if bn is not None:
                x = bn(x)
            x = F.relu(x)
            x = pool(x)
        
        # Transpose back for LSTM: (batch, time_steps, features)
        x = x.transpose(1, 2)
        
        # LSTM
        lstm_out, _ = self.lstm(x)
        pooled = lstm_out[:, -1, :]  # Final timestep
        
        # Heads
        pre_ictal_prob = self.fc_class(pooled)
        countdown = self.fc_regress(pooled)
        countdown = torch.sigmoid(countdown) * self.config.output_countdown_max
        
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


class MultimodalCountdownPredictor(nn.Module):
    """
    Multimodal fusion model combining ECG, EEG, and motion signals.
    
    Architecture: Early fusion (concatenate features) + LSTM + attention
    """
    
    def __init__(self, config: ModelConfig):
        """Initialize multimodal predictor.
        
        Args:
            config: ModelConfig with architecture specifications
        """
        super().__init__()
        self.config = config
        
        ecg_dim = config.ecg_feature_dim
        eeg_dim = config.eeg_feature_dim
        motion_dim = config.motion_feature_dim
        hidden_dim = config.hidden_dim
        dropout = config.dropout
        
        # Separate LSTM encoders for each modality
        self.ecg_lstm = nn.LSTM(
            input_size=ecg_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        self.eeg_lstm = nn.LSTM(
            input_size=eeg_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            dropout=dropout,
            batch_first=True
        )
        
        # Motion processing (simpler, just dense)
        self.motion_processor = nn.Sequential(
            nn.Linear(motion_dim, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, hidden_dim * 2)
        )
        
        lstm_output_dim = hidden_dim * 2
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(lstm_output_dim * 3, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(256),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Output heads
        self.fc_class = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.fc_regress = nn.Sequential(
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, ecg_x: torch.Tensor, eeg_x: torch.Tensor, 
                motion_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            ecg_x: (batch, time_steps, ecg_dim)
            eeg_x: (batch, time_steps, eeg_dim)
            motion_x: (batch, time_steps, motion_dim)
        
        Returns:
            Tuple of (pre_ictal_prob, countdown_minutes)
        """
        # ECG stream
        ecg_out, _ = self.ecg_lstm(ecg_x)
        ecg_pooled = ecg_out[:, -1, :]  # (batch, hidden*2)
        
        # EEG stream
        eeg_out, _ = self.eeg_lstm(eeg_x)
        eeg_pooled = eeg_out[:, -1, :]
        
        # Motion stream
        motion_final = motion_x[:, -1, :]  # (batch, motion_dim)
        motion_pooled = self.motion_processor(motion_final)  # (batch, hidden*2)
        
        # Fusion
        fused = torch.cat([ecg_pooled, eeg_pooled, motion_pooled], dim=-1)  # (batch, hidden*6)
        fused_hidden = self.fusion(fused)  # (batch, 128)
        
        # Output heads
        pre_ictal_prob = self.fc_class(fused_hidden)
        countdown = self.fc_regress(fused_hidden)
        countdown = torch.sigmoid(countdown) * self.config.output_countdown_max
        
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# TCN — Temporal Convolutional Network
# Target: capable wearable / edge server  (~280k params)
# ---------------------------------------------------------------------------

class _Chomp1d(nn.Module):
    """Remove trailing padding to enforce causality."""
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class _TCNResidualBlock(nn.Module):
    """Dilated causal residual block."""

    def __init__(self, n_inputs: int, n_outputs: int, kernel_size: int,
                 dilation: int, dropout: float = 0.2):
        super().__init__()
        pad = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(n_inputs, n_outputs, kernel_size,
                      padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(n_outputs, n_outputs, kernel_size,
                      padding=pad, dilation=dilation),
            _Chomp1d(pad),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = (nn.Conv1d(n_inputs, n_outputs, 1)
                           if n_inputs != n_outputs else None)
        self.relu = nn.ReLU()
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNCountdown(nn.Module):
    """
    Temporal Convolutional Network for seizure countdown prediction.

    Dilated causal residual convolutions give a large receptive field
    without recurrence → fully parallelisable, fast to train/infer.

    Deployment target: capable wearable (e.g. Wear OS, edge gateway).
    Parameters: ~280 k (default settings).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        in_channels = config.ecg_feature_dim
        num_channels: List[int] = getattr(config, 'tcn_num_channels',
                                          [64, 64, 128, 128])
        kernel_size: int = getattr(config, 'tcn_kernel_size', 3)
        dropout = config.dropout

        # Build TCN layers with exponentially growing dilation
        blocks: List[nn.Module] = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            blocks.append(_TCNResidualBlock(in_channels, out_ch,
                                            kernel_size, dilation, dropout))
            in_channels = out_ch
        self.tcn = nn.Sequential(*blocks)

        final_ch = num_channels[-1]
        self.coherent_heads = getattr(config, 'coherent_heads', False)

        self.fc_class_hidden = nn.Sequential(
            nn.Linear(final_ch, 64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc_class_out = nn.Linear(64, 1)

        regress_in_dim = final_ch + (1 if self.coherent_heads else 0)
        self.fc_regress = nn.Sequential(
            nn.Linear(regress_in_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: x (batch, time, features)"""
        x = x.transpose(1, 2)          # → (batch, features, time)
        x = self.tcn(x)                # → (batch, channels, time)
        pooled = x.mean(dim=-1)        # global average pool → (batch, channels)

        class_hidden = self.fc_class_hidden(pooled)
        pre_ictal_logits = self.fc_class_out(class_hidden)
        pre_ictal_prob = torch.sigmoid(pre_ictal_logits)

        regress_input = torch.cat([pooled, pre_ictal_logits], dim=-1) if self.coherent_heads else pooled
        countdown = self.fc_regress(regress_input)
        countdown = torch.sigmoid(countdown) * self.config.output_countdown_max
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# EEGNet — Ultra-compact depthwise-separable CNN
# Target: ESP32 / microcontroller TinyML  (~3 k params, <10 KB int8)
# ---------------------------------------------------------------------------

class EEGNetCountdown(nn.Module):
    """
    EEGNet-inspired compact architecture for seizure countdown prediction.

    Adapted from Lawhern et al. (2018) for feature-based (not raw signal)
    input.  Uses depthwise & separable 1-D convolutions to minimise
    parameter count while preserving temporal structure.

    Deployment target: ESP32, nRF52840, microcontroller TinyML.
    Parameters: ~3 k  (INT8-quantised ≈ 3 KB model — fits ESP32 SRAM).
    """

    def __init__(self, config: ModelConfig,
                 F1: int = 8, D: int = 2, F2: int = 16,
                 temporal_kernel: int = 32, pool1: int = 8, pool2: int = 8):
        """
        Args:
            F1:              number of temporal filters
            D:               depth multiplier for depthwise conv
            F2:              number of separable filters
            temporal_kernel: length of temporal convolution kernel
            pool1/pool2:     average-pooling strides
        """
        super().__init__()
        self.config = config
        dropout = config.dropout
        in_ch = config.ecg_feature_dim  # treat features as channels

        # Block 1 – temporal conv + depthwise conv
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_ch, F1, temporal_kernel,
                      padding=temporal_kernel // 2, bias=False),
            nn.BatchNorm1d(F1),
        )
        self.depthwise = nn.Sequential(
            nn.Conv1d(F1, F1 * D, 1, groups=F1, bias=False),  # depthwise
            nn.BatchNorm1d(F1 * D),
            nn.ELU(),
            nn.AvgPool1d(pool1),
            nn.Dropout(dropout),
        )

        # Block 2 – separable conv (depthwise + pointwise)
        self.separable = nn.Sequential(
            nn.Conv1d(F1 * D, F2, 3, padding=1, groups=F1 * D, bias=False),   # depthwise
            nn.Conv1d(F2, F2, 1, bias=False),                                  # pointwise
            nn.BatchNorm1d(F2),
            nn.ELU(),
            nn.AvgPool1d(pool2),
            nn.Dropout(dropout),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc_class = nn.Sequential(nn.Linear(F2, 1), nn.Sigmoid())
        self.fc_regress = nn.Linear(F2, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: x (batch, time, features)"""
        x = x.transpose(1, 2)     # → (batch, features, time)
        x = self.temporal_conv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.gap(x).squeeze(-1)  # → (batch, F2)

        pre_ictal_prob = self.fc_class(x)
        countdown = torch.sigmoid(self.fc_regress(x)) * self.config.output_countdown_max
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# MobileNet1D — Depthwise-separable CNN (MobileNet v1 style)
# Target: Smartwatch (Apple Watch, Wear OS via CoreML / TFLite)  (~60 k params)
# ---------------------------------------------------------------------------

class _DWSConvBlock(nn.Module):
    """Depthwise Separable Conv1d block: DW + PW + BN + ReLU6."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1,
                 kernel: int = 3):
        super().__init__()
        self.dw = nn.Conv1d(in_ch, in_ch, kernel, stride=stride,
                            padding=kernel // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv1d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu6(self.bn(self.pw(self.dw(x))))
        return x


class MobileNetCountdown(nn.Module):
    """
    MobileNet v1-style 1-D depthwise-separable CNN.

    Compact yet accurate — well-suited for smartwatch deployment via
    CoreML (Apple Watch) or TFLite (Wear OS).

    Deployment target: Smartwatch / IoT gateway.
    Parameters: ~60 k  (INT8-quantised ≈ 60 KB).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dropout = config.dropout
        in_ch = config.ecg_feature_dim

        # Entry conv (standard, not depthwise)
        self.entry = nn.Sequential(
            nn.Conv1d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU6(),
        )

        # DWS blocks: (in_ch, out_ch, stride)
        dws_cfg = [
            (32, 32, 1), (32, 64, 2),
            (64, 64, 1), (64, 128, 2),
            (128, 128, 1), (128, 128, 2),
            (128, 256, 1),
        ]
        layers: List[nn.Module] = []
        cur = 32
        for (_, out, s) in dws_cfg:
            layers.append(_DWSConvBlock(cur, out, stride=s))
            cur = out
        self.dws = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)

        self.fc_class = nn.Sequential(
            nn.Linear(cur, 1), nn.Sigmoid()
        )
        self.fc_regress = nn.Linear(cur, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: x (batch, time, features)"""
        x = x.transpose(1, 2)        # → (batch, features, time)
        x = self.entry(x)
        x = self.dws(x)
        x = self.gap(x).squeeze(-1)  # global average pool
        x = self.drop(x)

        pre_ictal_prob = self.fc_class(x)
        countdown = torch.sigmoid(self.fc_regress(x)) * self.config.output_countdown_max
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# InceptionTime1D — Multi-scale CNN (InceptionTime adapted for 1-D features)
# Target: Edge server / gateway  (~430 k params)
# ---------------------------------------------------------------------------

class _InceptionModule1D(nn.Module):
    """Multi-scale inception module for 1-D sequences."""

    def __init__(self, in_ch: int, n_filters: int = 32,
                 kernels: Tuple[int, ...] = (9, 19, 39)):
        super().__init__()
        # Bottleneck first
        self.bottleneck = nn.Conv1d(in_ch, n_filters, 1, bias=False)

        self.conv_paths = nn.ModuleList([
            nn.Conv1d(n_filters, n_filters, k, padding=k // 2, bias=False)
            for k in kernels
        ])
        self.maxpool_path = nn.Sequential(
            nn.MaxPool1d(3, stride=1, padding=1),
            nn.Conv1d(in_ch, n_filters, 1, bias=False),
        )
        out_ch = n_filters * len(kernels) + n_filters
        self.bn = nn.BatchNorm1d(out_ch)
        self.out_channels = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        paths = [c(b) for c in self.conv_paths] + [self.maxpool_path(x)]
        return F.relu(self.bn(torch.cat(paths, dim=1)))


class InceptionTime1DCountdown(nn.Module):
    """
    InceptionTime adapted for 1-D feature sequences.

    Multi-scale convolutions at each block capture different temporal
    granularities simultaneously (similar to GoogLeNet/InceptionTime for
    time-series classification – Ismail Fawaz et al., 2020).

    Deployment target: Edge server, capable gateway.
    Parameters: ~430 k.
    """

    def __init__(self, config: ModelConfig,
                 n_filters: int = 32, depth: int = 6):
        super().__init__()
        self.config = config
        dropout = config.dropout
        in_ch = config.ecg_feature_dim

        modules: List[nn.Module] = []
        prev_ch = in_ch
        for i in range(depth):
            mod = _InceptionModule1D(prev_ch, n_filters=n_filters)
            modules.append(mod)
            # Residual shortcut every 3 blocks
            prev_ch = mod.out_channels

        self.inception_blocks = nn.ModuleList(modules)

        # Residual shortcuts every 3 layers
        shortcut_ch = []
        c = in_ch
        for i, m in enumerate(self.inception_blocks):
            if i % 3 == 0 and i > 0:
                shortcut_ch.append((c, m.out_channels))
            c = m.out_channels

        # Build shortcuts
        # We'll handle this inline in forward with a simple identity/project
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.drop = nn.Dropout(dropout)
        final_ch = self.inception_blocks[-1].out_channels  # type: ignore[attr-defined]
        self.coherent_heads = getattr(config, 'coherent_heads', False)

        self.fc_class_hidden = nn.Sequential(
            nn.Linear(final_ch, 64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc_class_out = nn.Linear(64, 1)

        regress_in_dim = final_ch + (1 if self.coherent_heads else 0)
        self.fc_regress = nn.Sequential(
            nn.Linear(regress_in_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: x (batch, time, features)"""
        x = x.transpose(1, 2)      # → (batch, features, time)
        residual = x
        for i, block in enumerate(self.inception_blocks):
            x = block(x)
            # Residual every 3 blocks (project if channels changed)
            if (i + 1) % 3 == 0:
                residual = x   # update residual anchor
        x = self.gap(x).squeeze(-1)
        x = self.drop(x)

        class_hidden = self.fc_class_hidden(x)
        pre_ictal_logits = self.fc_class_out(class_hidden)
        pre_ictal_prob = torch.sigmoid(pre_ictal_logits)

        regress_input = torch.cat([x, pre_ictal_logits], dim=-1) if self.coherent_heads else x
        countdown = torch.sigmoid(self.fc_regress(regress_input)) * self.config.output_countdown_max
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# Temporal Transformer — Transformer encoder (ECG-only, server/cloud)
# Target: Server / cloud  (~500 k params with default d_model=64)
# ---------------------------------------------------------------------------

class _PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class TemporalTransformerCountdown(nn.Module):
    """
    Transformer encoder for seizure countdown prediction (ECG-only).

    Excels at capturing long-range dependencies in physiological time-series.
    Standard sinusoidal positional encoding + N encoder layers.

    Deployment target: Server / cloud inference.
    Parameters: ~500 k (d_model=64, nhead=4, num_layers=4).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dropout = config.dropout
        in_dim = config.ecg_feature_dim

        d_model: int = getattr(config, 'transformer_d_model', 64)
        nhead: int = getattr(config, 'transformer_nhead', 4)
        num_layers: int = getattr(config, 'transformer_num_layers', 4)
        dim_ff: int = getattr(config, 'transformer_dim_feedforward', 256)

        self.proj = nn.Linear(in_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_ff, dropout=dropout,
            batch_first=True, norm_first=True,   # Pre-LN (more stable)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers,
            norm=nn.LayerNorm(d_model)
        )
        self.coherent_heads = getattr(config, 'coherent_heads', False)

        self.fc_class_hidden = nn.Sequential(
            nn.Linear(d_model, 64), nn.ReLU(), nn.Dropout(dropout)
        )
        self.fc_class_out = nn.Linear(64, 1)

        regress_in_dim = d_model + (1 if self.coherent_heads else 0)
        self.fc_regress = nn.Sequential(
            nn.Linear(regress_in_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Args: x (batch, time, features)"""
        x = self.proj(x)           # → (batch, time, d_model)
        x = self.pos_enc(x)
        x = self.transformer(x)    # → (batch, time, d_model)
        pooled = x.mean(dim=1)     # global average pool over time

        class_hidden = self.fc_class_hidden(pooled)
        pre_ictal_logits = self.fc_class_out(class_hidden)
        pre_ictal_prob = torch.sigmoid(pre_ictal_logits)

        regress_input = torch.cat([pooled, pre_ictal_logits], dim=-1) if self.coherent_heads else pooled
        countdown = torch.sigmoid(self.fc_regress(regress_input)) * self.config.output_countdown_max
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# Multimodal Transformer — Cross-attention fusion (server/cloud, multimodal)
# Target: Server / cloud  (~1.5 M params)
# ---------------------------------------------------------------------------

class _ModalityEncoder(nn.Module):
    """Projects one modality into a shared d_model space + Transformer."""

    def __init__(self, in_dim: int, d_model: int, nhead: int, num_layers: int,
                 dim_ff: int, dropout: float):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.pos_enc = _PositionalEncoding(d_model, dropout=dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True, norm_first=True,
        )
        self.enc = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                         norm=nn.LayerNorm(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns time-pooled representation: (batch, d_model)."""
        x = self.pos_enc(self.proj(x))
        x = self.enc(x)
        return x.mean(dim=1)


class MultimodalTransformerCountdown(nn.Module):
    """
    Cross-attention multimodal Transformer for seizure countdown prediction.

    Three parallel Transformer encoders (ECG, EEG, Motion) produce
    modality embeddings that are fused via a learned weighted attention
    gate, then fed into shared prediction heads.

    Deployment target: Server / cloud inference (ECG + EEG + Motion).
    Parameters: ~1.5 M (d_model=128, nhead=4, 2 layers each encoder).
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        dropout = config.dropout

        d_model: int = getattr(config, 'transformer_d_model', 128)
        nhead: int = getattr(config, 'transformer_nhead', 4)
        num_layers: int = getattr(config, 'transformer_num_layers', 2)
        dim_ff: int = getattr(config, 'transformer_dim_feedforward', 512)

        self.ecg_enc = _ModalityEncoder(config.ecg_feature_dim,   d_model,
                                        nhead, num_layers, dim_ff, dropout)
        self.eeg_enc = _ModalityEncoder(config.eeg_feature_dim,   d_model,
                                        nhead, num_layers, dim_ff, dropout)
        self.mot_enc = _ModalityEncoder(config.motion_feature_dim, d_model,
                                        nhead, num_layers, dim_ff, dropout)

        # Learned gate: softmax over three modality contributions
        self.gate = nn.Linear(d_model * 3, 3)

        self.fusion = nn.Sequential(
            nn.Linear(d_model * 3, 256), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128), nn.GELU(),
            nn.Dropout(dropout),
        )

        self.fc_class = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1), nn.Sigmoid()
        )
        self.fc_regress = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(32, 1)
        )

    def forward(self, ecg_x: torch.Tensor, eeg_x: torch.Tensor,
                motion_x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            ecg_x:    (batch, time, ecg_dim)
            eeg_x:    (batch, time, eeg_dim)
            motion_x: (batch, time, motion_dim)
        """
        e_ecg = self.ecg_enc(ecg_x)   # (batch, d_model)
        e_eeg = self.eeg_enc(eeg_x)
        e_mot = self.mot_enc(motion_x)

        concat = torch.cat([e_ecg, e_eeg, e_mot], dim=-1)  # (batch, d_model*3)

        # Soft attention gate over modalities
        gates = torch.softmax(self.gate(concat), dim=-1)   # (batch, 3)
        d = e_ecg.shape[-1]
        gated = (gates[:, 0:1] * e_ecg
                 + gates[:, 1:2] * e_eeg
                 + gates[:, 2:3] * e_mot)                  # (batch, d_model)
        # Append gated summary to full concat for richer fusion
        fused = self.fusion(concat + gated.repeat(1, 3))   # broadcast add

        pre_ictal_prob = self.fc_class(fused)
        countdown = torch.sigmoid(self.fc_regress(fused)) * self.config.output_countdown_max
        return pre_ictal_prob.squeeze(-1), countdown.squeeze(-1)


# ---------------------------------------------------------------------------
# TemporalConfidenceHead  — lightweight stage-2 temporal smoother
# ---------------------------------------------------------------------------

class _CausalResBlock(nn.Module):
    """Causal dilated residual block used by TemporalConfidenceHead."""

    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        pad = (kernel_size - 1) * dilation  # causal (future) padding
        self.conv1 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=pad)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size,
                               dilation=dilation, padding=pad)
        self.drop = nn.Dropout(dropout)
        self.relu = nn.ReLU()

    @staticmethod
    def _trim(x: torch.Tensor, target_len: int) -> torch.Tensor:
        """Remove the extra future-padding to keep the tensor causal."""
        return x[..., :target_len] if x.shape[-1] > target_len else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        L = x.shape[-1]
        residual = x
        out = self.relu(self._trim(self.conv1(x), L))
        out = self.drop(self.relu(self._trim(self.conv2(out), L)))
        return self.relu(out + residual)


class TemporalConfidenceHead(nn.Module):
    """Lightweight causal TCN operating on a ring-buffer of K recent alarms.

    Stage-2 temporal smoother: takes the last K raw alarm probabilities
    (from *any* base model) and outputs a refined, temporally-aware
    confidence score.

    **Why this helps:**
    Base models classify each 2-min window in isolation — no memory of
    previous outputs.  Even TCN/Transformer only have context *within* the
    window.  High-frequency noise in the raw output stream hides the true
    escalation pattern.  This head learns:

    * Sustained low-level activity vs random spikes.
    * Monotonically rising patterns (~10 → 0 min countdown).
    * The difference between a burst of two positive windows and a true
      building alarm over 8+ consecutive windows.

    Architecture: 1-D input projection → 3 causal dilated residual blocks
                  (d=1,2,4) → global-average pool → Linear(32,1) → Sigmoid.

    Parameters: ~8 k (K=32, hidden_ch=16) — runs on any wearable after INT8.

    Deployment pattern::

        # During live inference (e.g. smartwatch):
        buffer = collections.deque(maxlen=K)   # populated with 0.0 initially
        for window in ecg_stream:
            raw_prob, _ = base_model(window)
            buffer.append(raw_prob)
            alarm_conf = tch(torch.tensor(list(buffer)).unsqueeze(0))

    Args:
        history_k:  Number of recent predictions in the buffer (default 32).
        hidden_ch:  Internal channel width (default 16).
        kernel_size: Causal conv kernel (default 3).
        num_blocks: Number of dilated residual blocks (default 3).
        dropout:    Dropout rate (default 0.10).
    """

    def __init__(
        self,
        history_k: int = 32,
        hidden_ch: int = 16,
        kernel_size: int = 3,
        num_blocks: int = 3,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.history_k = history_k

        # Scalar prob → hidden_ch feature maps
        self.input_proj = nn.Conv1d(1, hidden_ch, kernel_size=1)

        # Causal backbone with exponentially growing dilation
        blocks: List[nn.Module] = []
        for i in range(num_blocks):
            blocks.append(_CausalResBlock(hidden_ch, kernel_size, 2 ** i, dropout))
        self.backbone = nn.Sequential(*blocks)

        # Alarm head
        self.head = nn.Sequential(
            nn.Linear(hidden_ch, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, history: torch.Tensor) -> torch.Tensor:
        """Refine alarm confidence from recent prediction history.

        Args:
            history: (batch, K) — last K raw alarm probabilities, oldest first.

        Returns:
            refined_alarm: (batch,) — smoothed alarm probability in [0, 1].
        """
        x = history.unsqueeze(1)         # (B, 1, K)
        x = self.input_proj(x)           # (B, hidden_ch, K)
        x = self.backbone(x)             # (B, hidden_ch, K) causal
        pooled = x.mean(dim=-1)          # (B, hidden_ch)
        return self.head(pooled).squeeze(-1)   # (B,)


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

class ModelFactory:
    """Factory for creating models based on configuration."""

    #: Maps model_type string → constructor
    _REGISTRY = {
        # Original models
        "ecg_lstm":   ECGCountdownPredictor,
        "cnn_lstm":   CNNLSTMCountdown,
        "multimodal": MultimodalCountdownPredictor,
        # TinyML / Edge
        "eegnet":       EEGNetCountdown,
        "mobilenet_1d": MobileNetCountdown,
        "tcn":          TCNCountdown,
        "inception_1d": InceptionTime1DCountdown,
        # Server / Cloud
        "temporal_transformer":    TemporalTransformerCountdown,
        "multimodal_transformer":  MultimodalTransformerCountdown,
    }

    #: Model types that consume separate (ecg, eeg, motion) tensors
    MULTIMODAL_TYPES = frozenset({"multimodal", "multimodal_transformer"})

    @staticmethod
    def create_model(config: ModelConfig) -> nn.Module:
        """Instantiate a model from *config.model_type*.

        Args:
            config: ModelConfig dataclass.

        Returns:
            Instantiated nn.Module.

        Raises:
            ValueError: If model_type is not registered.
        """
        model_type = config.model_type.lower()
        cls = ModelFactory._REGISTRY.get(model_type)
        if cls is None:
            valid = sorted(ModelFactory._REGISTRY.keys())
            raise ValueError(
                f"Unknown model type: '{model_type}'. "
                f"Valid options: {valid}"
            )
        return cls(config)

    @staticmethod
    def is_multimodal(model_type: str) -> bool:
        """Return True if the model expects separate modality tensors."""
        return model_type.lower() in ModelFactory.MULTIMODAL_TYPES


if __name__ == "__main__":
    from config.config import DEFAULT_CONFIG
    import sys

    cfg = DEFAULT_CONFIG.model
    B, T = 4, 600

    single_input_models = [
        ("ecg_lstm",           "ECG BiLSTM+Attention"),
        ("cnn_lstm",           "CNN-LSTM Hybrid"),
        ("tcn",                "TCN (Temporal Conv Net)"),
        ("eegnet",             "EEGNet (TinyML/ESP32)"),
        ("mobilenet_1d",       "MobileNet-1D (Smartwatch)"),
        ("inception_1d",       "InceptionTime-1D (Edge Server)"),
        ("temporal_transformer","Temporal Transformer (Server)"),
    ]

    multimodal_models = [
        ("multimodal",             "Multimodal LSTM Fusion"),
        ("multimodal_transformer", "Multimodal Transformer (Server)"),
    ]

    x = torch.randn(B, T, cfg.ecg_feature_dim)

    for model_type, label in single_input_models:
        cfg.model_type = model_type
        model = ModelFactory.create_model(cfg)
        model.eval()
        with torch.no_grad():
            prob, cd = model(x)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{label:42s}  prob={prob.shape}  countdown={cd.shape}  "
              f"params={n_params:,}")

    x_ecg    = torch.randn(B, T, cfg.ecg_feature_dim)
    x_eeg    = torch.randn(B, T, cfg.eeg_feature_dim)
    x_motion = torch.randn(B, T, cfg.motion_feature_dim)

    for model_type, label in multimodal_models:
        cfg.model_type = model_type
        model = ModelFactory.create_model(cfg)
        model.eval()
        with torch.no_grad():
            prob, cd = model(x_ecg, x_eeg, x_motion)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"{label:42s}  prob={prob.shape}  countdown={cd.shape}  "
              f"params={n_params:,}")
