# ECGCountdownPredictor Architecture Report

## Overview
The `ECGCountdownPredictor` is a neural network designed for seizure countdown prediction using ECG signals. It is built with a focus on multi-task learning, supporting both pre-ictal classification and onset (countdown) regression. The architecture is configurable and supports a UNet-style connection between the classification and regression heads for improved coherence.

## Architecture Diagram

```
Input (ECG features)
      │
  [BiLSTM layers]
      │
[Temporal Attention] (optional)
      │
   [Pooling]
      │
 ┌───────────────┬─────────────────────┐
 │               │                     │
 │        [Classification Head]        │
 │   (Linear → ReLU → Dropout → Linear │
 │    → Sigmoid)                       │
 │               │                     │
 │        (Pre-activation output)      │
 │               │                     │
 │        [Regression Head]            │
 │   (Linear → ReLU → Dropout → Linear │
 │    → Sigmoid × max countdown)       │
 │               │                     │
 └───────────────┴─────────────────────┘
```

- **UNet-style connection:** If `coherent_heads` is enabled in the config, the regression head receives both the pooled encoder output and the pre-activation output of the classification head as input (concatenated).

## Key Components
- **BiLSTM Encoder:** Captures temporal dependencies in ECG features.
- **Temporal Attention:** (Optional) Focuses on important time steps.
- **Classification Head:** Predicts pre-ictal probability (sigmoid output).
- **Regression Head:** Predicts countdown to onset (sigmoid output scaled by max countdown).
- **Coherent Heads:** When enabled, regression head input = [pooled encoder output, classification pre-activation].

## Configurable Parameters
- `hidden_dim`, `num_lstm_layers`, `dropout`, `use_attention`, `num_attention_heads`
- `coherent_heads`: Enables/disables UNet-style connection
- Loss weights: `classification_weight`, `regression_weight`

## Usage
- Set `model.coherent_heads: true` in the config to enable the UNet-style connection.
- Adjust loss weights to prioritize onset prediction as needed.

## Benefits
- **Multi-task learning:** Jointly optimizes for both pre-ictal detection and onset countdown.
- **Configurable coherence:** UNet-style connection can improve temporal consistency and prediction accuracy.
- **Flexible:** Easily toggled via config for ablation studies.

---
_Last updated: May 18, 2026_
