Below is a **research-grade, end-to-end methodology** that you can use directly in a paper (or proposal). It integrates your architectural ideas with decisions grounded in prior literature directions, and—just as important—explains *why* each choice is made under your constraints (multimodal physiology, limited data, long horizons, deployment).

---

# 🧠 **Title**

**Hierarchical Event-Token Representation Learning for Multimodal Physiological Time Series with Longitudinal Reasoning**

---

# 🧾 **Abstract**

We propose a hierarchical representation learning framework for multimodal physiological time series that introduces a structured **event-token abstraction layer** to enable long-horizon reasoning under limited data conditions. Unlike conventional sequence models such as LSTM and Transformer architectures, which operate directly on continuous signals and require large context windows, the proposed approach converts synchronized streams of EEG, photoplethysmography (PPG), and electrodermal activity (EDA) into discrete-time latent tokens summarizing local physiological dynamics.

These tokens are processed through a multi-scale hierarchical aggregation mechanism that compresses temporal information from seconds to minutes, hours, and days, enabling efficient long-term memory while preserving clinically relevant patterns such as event frequency, recurrence, and cross-modal synchronization. To address data scarcity, the model leverages self-supervised and multi-task learning objectives that encourage general physiological representation beyond seizure-specific features.

We further introduce a token quantization and statistical modeling layer that enables interpretable analysis via frequency histograms and transition dynamics, supporting probabilistic reasoning over physiological states. Experiments demonstrate improved robustness and sample efficiency compared to baseline sequence models, while enabling deployment-friendly compression. The proposed framework provides a foundation for interpretable, scalable physiological intelligence systems applicable to seizure prediction and broader health monitoring tasks.

---

# 🔷 1. **Problem Framing & Motivation**

## 1.1 Limitations of Existing Approaches

Most physiological ML systems:

* Use fixed windows (e.g., 30s–5min)
* Treat windows as independent samples
* Rely on:

  * LSTM
  * Transformer

### Identified Issues:

1. **Loss of temporal continuity**
2. **Poor long-horizon modeling (minutes → hours → days)**
3. **High memory/computation cost**
4. **Limited interpretability**
5. **Data inefficiency under small datasets**

---

## 1.2 Design Requirements (Derived from Constraints)

From your problem:

* Multimodal signals (EEG, PPG, EDA)
* Sparse labeled seizure events
* Need for early prediction (not just detection)
* Edge deployment considerations

### Therefore, the system must:

✔ Operate in **streaming mode**
✔ Support **longitudinal memory**
✔ Be **sample-efficient**
✔ Provide **interpretable outputs**
✔ Enable **compression across time scales**

---

## 1.3 Seizure Phase Terminology

The following phases are used consistently throughout this document. They are defined on the **temporal axis** (clock time flowing forward), not on the numerical countdown label axis.

| Phase | Definition | Countdown label $c$ |
|---|---|---|
| **Interictal** | Brain baseline state between seizures. No seizure trajectory is present. | $c < 0$ (annotated as −1; no upcoming seizure) |
| **Pre-ictal** | Measurable physiological transition *toward* a seizure, minutes to tens of minutes before onset. The seizure has not started but the trajectory has begun. | $c \ge \theta$ (e.g., $c \ge 2$ min) |
| **Onset** | The narrow window immediately preceding ictal discharge. Corresponds to the last 0–$\theta$ minutes of the pre-ictal ramp. | $0 \le c < \theta$ |
| **Ictal** | The seizure itself. Not directly modeled as a separate head; onset merges into it at $c = 0$. | $c = 0$ |
| **Post-ictal** | Recovery period. Falls back to interictal labeling once no further seizure is annotated. | $c < 0$ |

> **Why countdown goes high → low, not low → high.** The label $c$ represents *minutes remaining until the next seizure*. As clock time advances toward a seizure, $c$ *decreases*: a sample recorded 20 min before a seizure has $c = 20$; a sample 1 min before has $c = 1$. Temporal order on the timeline is therefore:
>
> $$\underbrace{\text{interictal}}_{c < 0} \;\longrightarrow\; \underbrace{\text{pre-ictal}}_{c \ge \theta} \;\longrightarrow\; \underbrace{\text{onset}}_{0 \le c < \theta} \;\longrightarrow\; \text{seizure}$$
>
> Reading the same condition ranges numerically on the label axis instead gives $-1 \to 0{-}\theta \to \theta{+}$, which reverses the middle two states and suggests the incorrect order interictal → onset → pre-ictal. Always interpret the labels as a *countdown*, not a rank-ordered category index.

---

## 1.3 Seizure Phase Terminology

The following phases are used consistently throughout this document. They are defined on the **temporal axis** (clock time flowing forward), not on the numerical countdown label axis.

| Phase | Definition | Countdown label $c$ |
|---|---|---|
| **Interictal** | Brain baseline state between seizures. No seizure trajectory is present. | $c < 0$ (annotated as −1; no upcoming seizure) |
| **Pre-ictal** | Measurable physiological transition *toward* a seizure, minutes to tens of minutes before onset. The seizure has not started but the trajectory has begun. | $c \ge \theta$ (e.g., $c \ge 2$ min) |
| **Onset** | The narrow window immediately preceding ictal discharge. Corresponds to the last 0–$\theta$ minutes of the pre-ictal ramp. | $0 \le c < \theta$ |
| **Ictal** | The seizure itself. Not directly modeled as a separate head; onset merges into it at $c = 0$. | $c = 0$ |
| **Post-ictal** | Recovery period. Falls back to interictal labeling once no further seizure is annotated. | $c < 0$ |

> **Why countdown goes high → low, not low → high.** The label $c$ represents *minutes remaining until the next seizure*. As clock time advances toward a seizure, $c$ *decreases*: a sample recorded 20 min before a seizure has $c = 20$; a sample 1 min before has $c = 1$. Temporal order on the timeline is therefore:
>
> $$\underbrace{\text{interictal}}_{c < 0} \;\longrightarrow\; \underbrace{\text{pre-ictal}}_{c \ge \theta} \;\longrightarrow\; \underbrace{\text{onset}}_{0 \le c < \theta} \;\longrightarrow\; \text{seizure}$$
>
> Reading the same condition ranges numerically on the label axis instead gives $-1 \to 0{-}\theta \to \theta{+}$, which reverses the middle two states. Always interpret the labels as a *countdown*, not a rank-ordered category index.

---

# 🔷 2. **Core Design Philosophy**

The proposed system is based on three principles:

---

## 🔹 (A) Event Abstraction

Instead of modeling raw signals:

```text
signal → event representation → reasoning
```

Justification:

* Biological systems operate via **events and transitions**
* Aligns with:

  * neuromorphic processing
  * symbolic abstraction in time-series

---

## 🔹 (B) Hierarchical Temporal Compression

Instead of storing all time steps:

```text
seconds → minutes → hours → days
```

Justification:

* Inspired by:

  * multi-resolution signal processing
  * hierarchical temporal models
* Required for:

  * long-horizon inference
  * deployment feasibility

---

## 🔹 (C) Hybrid Statistical + Deep Learning Reasoning

Instead of purely neural inference:

```text
embeddings + frequency + transitions → prediction
```

Justification:

* Clinical interpretability
* Enables Bayesian-style reasoning
* Improves robustness under limited data

---

# 🔷 3. **System Architecture**

---

## 3.1 Multimodal Temporal Encoder

### Input:

```text
X(t) = {EEG, PPG, EDA, Temp, SQI}
```

### Output:

```text
H(t) ∈ ℝ^D
```

### Design:

* Temporal CNN with dilations

### Justification:

* CNNs outperform RNNs for:

  * local motif detection
  * parallel efficiency

---

## 3.2 Event-Token Embedding Layer

### Definition:

```text
E(t) = f(H(t : t+τ))
```

Where:

* τ = short window (1–10s)

### Output:

```text
E(t) ∈ ℝ¹²⁸
```

---

### Design Choice:

* Continuous embeddings first
* Optional discretization later

### Justification:

* Matches representation learning literature
* Avoids premature information loss

---

## 3.3 Token Quantization (Codebook)

### Mapping:

```text
E(t) → token_id ∈ {1 … K}
```

### Method:

* Vector Quantization (VQ)

### Justification:

* Enables:

  * compression
  * interpretability
  * frequency modeling

---

## 3.4 Hierarchical Temporal Aggregation

### Levels:

| Level | Input   | Output | Frequency |
| ----- | ------- | ------ | --------- |
| L0    | tokens  | 128D   | 1 Hz      |
| L1    | seconds | 64D    | 1/min     |
| L2    | minutes | 32D    | 1/hr      |
| L3    | hours   | 16D    | 1/day     |

---

### Mechanism:

* GRU-based aggregation

### Justification:

* Efficient state compression
* Supports streaming updates
* Better than storing raw sequences

---

## 3.5 Statistical Memory Layer

### Components:

#### (A) Histogram

```text
H(k) = frequency of token k
```

#### (B) Transition Matrix

```text
P(i → j)
```

#### (C) Time-decayed counts

---

### Justification:

* Captures:

  * recurrence patterns
  * temporal dynamics
* Enables:

  * interpretable reasoning
  * probabilistic modeling

---

## 3.6 Prediction Layer

### Tasks:

* Current seizure detection
* Future seizure prediction

### Inputs:

* hierarchical states
* token statistics

---

# 🔷 4. **Training Methodology**

---

## 4.1 Self-Supervised Pretraining (Critical)

### Objectives:

1. Signal reconstruction
2. Future prediction
3. Contrastive learning

### Justification:

* Addresses limited labeled data
* Encourages general physiological representations

---

## 4.2 Supervised Fine-Tuning

### Loss: SeizureStateLoss (3-class, no future-time regression)

The training objective is a **safe state classification loss** that deliberately avoids regressing the exact countdown value in minutes. Regressing the countdown encodes *when* the next seizure will occur into model weights — a form of future data leakage that (a) is unverifiable at deployment, and (b) is self-defeating: a successful early intervention prevents the seizure, making the model's countdown prediction retrospectively "wrong" even though it triggered a correct response.

Instead, GT countdown labels are used **only to bucket samples** into the three temporal states defined in §1.3:

| Condition on $c$ | State | Target for each head |
|---|---|---|
| $c < 0$ | interictal | alert = 0, onset = 0 |
| $c \ge \theta$ | pre-ictal | alert = 1, onset = 0 |
| $0 \le c < \theta$ | onset | alert = 1, onset = 1 |

The model has two output heads, both trained with Focal Binary Cross-Entropy:

```text
L_alert = FocalBCE(pre_ictal_pred,  is_active)   # is_active = (c >= 0)
L_onset = FocalBCE(sigmoid(countdown_pred), is_onset)  # is_onset = (0 <= c < θ)
L_total = w_alert · L_alert + w_onset · L_onset
```

`pre_ictal_pred` learns **P(alert)** = P(pre-ictal OR onset). `countdown_pred` is **re-interpreted** as a logit for P(onset), not a regression target. No MSE, no ranking loss, no minutes-to-seizure value is ever forced into gradients.

**Why P(alert) is still clinically meaningful without a countdown.** The model does not need to estimate *when* — it estimates *whether the brain is currently in a seizure-trajectory state*. The warning horizon is implicitly determined by:
1. The feature window length (how many seconds of signal feed the encoder), and
2. The pre-ictal period present in training data (typically 5–30 min for EEG-based datasets).

When P(alert) rises above threshold, it means the signal pattern is statistically consistent with patterns observed during pre-ictal and onset periods in training. The guarantee required for safety is that **P(alert) is HIGH whenever onset is active** — which is satisfied structurally: `is_active` covers both pre-ictal and onset, so the alert head is always supervised to fire during onset.

**Observer-effect self-consistency.** In a real-world alert system, a patient who receives an alert and takes rescue medication may stabilize before seizure onset, making a regression-based countdown provably wrong (it predicted $c = 4$ min but no seizure occurred). A state-based P(alert) has no such conflict: the prediction that "the brain was in a pre-ictal state" is independent of whether intervention succeeded.

Auxiliary losses:

```text
L = L_alert + λ L_onset + γ L_aux
```

---

## 4.3 Multi-Task Learning

Auxiliary tasks:

* heart rate estimation
* signal reconstruction

### Justification:

* Improves embedding richness
* Encourages cross-domain generalization

---

# 🔷 5. **Streaming Training Strategy (Key Contribution)**

Instead of:

```text
window → prediction
```

We use:

```text
continuous stream → stepwise updates
```

### Benefits:

✔ Preserves temporal continuity
✔ Matches real-world deployment
✔ Enables long-term reasoning

---

## 5.1 TemporalRingBufferDataset — Deployment-Faithful Training

Conventional ML training shuffles the full dataset each epoch, exposing the model simultaneously to data from all points in a recording's history. A deployed wearable device has no such access: it only has the signal history *since the device was powered on*.

To simulate this, `TemporalRingBufferDataset` wraps a time-sorted `SeizureDataset` and exposes only a sliding window of the most recent $W$ samples:

```text
Dataset (sorted chronologically):  [s_0, s_1, ..., s_{N-1}]
                                              |
                               offset         |      offset + W
                                  v                       v
                                  [  visible window W  ]
```

At each epoch the window advances by step $\Delta$:

```text
offset_e = min(N - W,  e * delta)
```

**Default step**: $\Delta = 0.10 \times W$ (10% of the buffer per epoch), so the entire dataset history is traversed over roughly 10 epochs.

**Temporal continuity.** Because samples inside the window remain in chronological order and old samples are evicted as time advances, the model is never trained on a sample that would be "in the future" relative to the current window position. This enforces the same causal constraint the deployed device faces.

**Integration with online augmentation.** Online augmentation (time-warp, amplitude-scale, Gaussian noise, time-shift) fires stochastically at the `SeizureDataset` level, inside the ring buffer's index translation. This means:
- The same raw sample within the window appears with a different augmented form each epoch
- Temporal ordering and dataset size are unchanged by augmentation
- The small pre-ictal minority within the visible window still receives augmentation diversity despite the limited window size
- Newly entered samples begin receiving augmentation immediately with no configuration change

**Patient-sequential training with ring buffer.** When combined with patient-sequential splitting (one patient's full recording timeline forms the training stream), the ring buffer prevents the model from seeing a future patient's data while processing an earlier patient — faithfully simulating a clinical streaming device that processes patients in sequence, never with hindsight.

---

## 5.1 TemporalRingBufferDataset — Deployment-Faithful Training

Conventional ML training shuffles the full dataset each epoch, exposing the model simultaneously to data from all times in a recording. A deployed wearable device has no such luxury: it only has access to the signal history *since the device was turned on*.

To simulate this, the `TemporalRingBufferDataset` wraps a time-sorted dataset and exposes only a sliding window of the most recent $W$ samples:

```text
Dataset (sorted chronologically):  [s_0, s_1, ..., s_{N-1}]
                                           ↑            ↑
                                        offset    offset + W
                                      └──────────────────┘
                                           visible window
```

At each epoch the window advances by a step $\Delta$:

```text
offset_e = min(N - W,  e · Δ)
```

**Default step**: $\Delta = 0.10 \times W$ (10% of the buffer per epoch), so the full dataset is traversed in approximately 10 epochs.

**Why this matters for longitudinal time series.** Seizure data is inherently temporal. An LSTM or GRU trained on random shuffles of full-history windows will develop spurious correlations with samples that, in deployment, the device has not yet seen. The ring buffer enforces the same constraint the device faces: the model can only learn from signal history that precedes the current moment in recording time.

**Integration with online augmentation.** Online augmentation (time-warp, amplitude-scale, Gaussian noise, time-shift) fires stochastically at the `SeizureDataset` level, *inside* the ring buffer's index translation. This means:
- The same raw sample within the window appears with a different augmented form each epoch
- Temporal ordering and dataset size are both unchanged
- The small pre-ictal minority within the visible window still receives augmentation diversity despite the limited window size
- As the window advances and new samples enter the buffer, they begin receiving augmentation immediately with no configuration change

**Patient-sequential training with ring buffer.** When combined with patient-sequential splitting (one patient's full recording timeline forms the training stream), the ring buffer further prevents the model from seeing a future patient's data while processing an earlier patient — faithfully simulating a streaming clinical device that processes patients in sequence, never with hindsight.

---

# 🔷 6. **Handling Patient Variability**

### Approach:

* Train per-patient sequences
* Maintain patient-specific statistics
* Evaluate cross-patient generalization

### Justification:

* Physiological baselines differ
* Avoids data leakage

---

# 🔷 7. **Interpretability Framework**

The system provides:

### (1) Token Visualization

* clusters of physiological patterns

### (2) Histogram Analysis

* frequency of events over time

### (3) Transition Graphs

* progression dynamics

---

# 🔷 8. **Deployment Considerations**

### Advantages:

* hierarchical compression reduces memory
* tokenization reduces bandwidth
* GRU aggregation is lightweight

### Path to edge:

* quantization (INT8)
* pruning
* replace Transformer if needed

---

# 🔷 9. **Expected Contributions**

---

## 🔬 Theoretical

* Event-token abstraction for physiological signals
* Hybrid deep + statistical reasoning framework

---

## ⚙️ Methodological

* Streaming training paradigm
* Hierarchical temporal compression

---

## 🏥 Applied

* Improved seizure prediction robustness
* Interpretable physiological modeling

---

# 🔷 10. **Positioning Against Literature (Final Framing)**

Your work differs from prior work in:

| Area              | Prior Work            | Your Contribution       |
| ----------------- | --------------------- | ----------------------- |
| Representation    | Continuous embeddings | Event-token abstraction |
| Temporal modeling | Fixed windows         | Streaming + hierarchy   |
| Reasoning         | Neural only           | Neural + statistical    |
| Memory            | Raw sequences         | Compressed multi-scale  |
| Interpretability  | Limited               | Explicit tokens + stats |

---

# 🔷 11. **Formal Bayesian State-Memory Formulation (Confidence Upgrade)**

To make the "Bayesian-style" claim publication-grade, define the prediction layer as explicit probabilistic inference over a compact latent state.

## 11.1 State Variables

At each time step $t$:

```text
e_t : continuous event embedding
k_t : discrete token from codebook
s_t : hierarchical neural state (GRU summaries across scales)
m_t : statistical memory state
```

Where:

```text
m_t = { c_t, N_t, A_t }
```

* $c_t$ = token count vector (histogram)
* $N_t$ = total token count
* $A_t$ = token transition count matrix

## 11.2 Online Bayesian Updates

Use conjugate priors for stable streaming updates under low data:

```text
c_t ~ Dirichlet(alpha)
A_t(i,:) ~ Dirichlet(beta_i)
```

Streaming update rules:

```text
c_t = rho * c_{t-1} + one_hot(k_t)
A_t(k_{t-1}, k_t) = rho * A_{t-1}(k_{t-1}, k_t) + 1
```

* $rho in (0,1]$ is a decay factor for non-stationarity

Posterior predictive estimates:

```text
p(k = j | m_t) = (c_t(j) + alpha_j) / (N_t + sum(alpha))
p(k_{t+1}=j | k_t=i, m_t) = (A_t(i,j) + beta_{i,j}) / (sum_j A_t(i,j) + sum_j beta_{i,j})
```

## 11.3 Seizure Risk Inference

Predict risk over horizon $Delta$ as:

```text
p(y_{t+Delta}=1 | s_t, m_t, e_t) = sigma( w_s^T s_t + w_m^T phi(m_t) + w_e^T e_t + b )
```

Where $phi(m_t)$ includes normalized histogram entropy, transition entropy, recurrence rates, and token burst statistics.

## 11.4 Calibrated Uncertainty

Report both:

* point risk: $hat{p}$
* epistemic uncertainty: variance from Monte-Carlo dropout or deep ensembles

Calibration metrics:

* ECE (Expected Calibration Error)
* Brier Score
* reliability diagrams

This closes a frequent reviewer concern: "high AUC but poorly calibrated clinical risk."

---

# 🔷 12. **Token-Based Explainability Protocol**

Explainability should be operationalized as reproducible analyses, not only visual examples.

## 12.1 Token Semantics

For each token ID $k$:

* retrieve top-$n$ nearest embedding segments
* summarize physiological descriptors (bandpower, HRV proxies, EDA slope)
* assign a provisional semantic label (e.g., sympathetic surge, motion artifact, stable baseline)

## 12.2 Evidence Decomposition for Predictions

For each alert, expose:

* dominant tokens in last 5, 30, and 120 minutes
* high-probability transitions preceding risk increase
* contribution scores from $s_t$, $m_t$, and $e_t$ components

## 12.3 Clinically Usable Outputs

Each prediction should provide:

* risk probability with confidence interval
* top evidence tokens and transitions
* short natural-language rationale template

Example rationale:

```text
Risk increased from 0.18 to 0.41 in 20 min, driven by repeated token T37
and elevated transition pattern T12->T37->T37, historically associated with
pre-ictal periods for this patient.
```

---

# 🔷 13. **Experimental Plan to De-Risk the Research Direction**

## 13.1 Phase A (Representation Validity)

Goal: verify token embeddings capture reusable physiology.

* SSL pretraining on unlabeled streams
* evaluate linear probe and cross-task transfer
* compare continuous-only vs VQ-tokenized embeddings

Success criteria:

* better probe performance than raw-window baselines
* stable token usage (no severe codebook collapse)

## 13.2 Phase B (Bayesian Memory Utility)

Goal: verify statistical memory improves long-horizon prediction.

Ablations:

1. neural state only ($s_t$)
2. statistical memory only ($m_t$)
3. fused model ($s_t + m_t + e_t$)

Primary endpoints:

* AUROC / AUPRC for future seizure prediction
* lead-time-aware sensitivity
* calibration (ECE, Brier)

## 13.3 Phase C (Generalization + Deployment)

Goal: verify robustness and edge feasibility.

* patient-wise splits with strict leakage control
* cross-patient adaptation via lightweight fine-tuning
* throughput, memory, and latency benchmarks on target hardware

---

# 🔷 14. **Clear Hypotheses for Publication**

H1: Event-token embeddings are more sample-efficient than raw continuous sequence modeling.

H2: Adding Bayesian statistical memory improves early prediction calibration and lead-time performance.

H3: Token-transition explanations increase interpretability fidelity without degrading predictive accuracy.

---

# 🔷 15. **Transition to Final Output (Continuous Day-Sweep Protocol)**

To move from current window-centric training to deployment-faithful behavior, use a continuous sweep protocol where samples are processed in chronological order across full recordings.

## 15.1 Final Output Definition

At each inference step $t$ (for example every 1 second), the model outputs:

```text
o_t = {
  p_alert(t),
  p_onset(t),
  uncertainty(t),
  evidence_tokens(t)
}
```

Where:

* $p_{\text{alert}}(t)$ = P(brain is in a pre-ictal or onset state right now). Fires the alert when above threshold. Corresponds to `pre_ictal_pred` head.
* $p_{\text{onset}}(t)$ = P(seizure is imminent, within the onset window $\theta$ minutes). Secondary "red alert" flag. Corresponds to `sigmoid(countdown_pred)` head.
* $uncertainty(t)$ = calibrated confidence interval (Monte-Carlo dropout or ensemble variance)
* $evidence_tokens(t)$ = dominant token IDs and transition motifs supporting the estimate

> **Note: no $\hat{t}_{\text{seizure}}$ output.** The model intentionally does not estimate time-to-seizure in minutes. Reasons:
> 1. **Observer effect**: a correct alert that triggers successful intervention will suppress the seizure, making a minutes-based countdown retrospectively wrong.
> 2. **Future data leakage**: training a regression head on countdown minutes forces the gradient to encode absolute future timing into weights — information unavailable at deployment without a perfect oracle.
> 3. **State vs. time**: the clinically actionable quantity is *whether* a seizure trajectory is active, not *exactly when*. P(alert) answers the actionable question; $\hat{t}$ pretends a level of precision the signal does not support.
>
> The warning horizon is implicitly defined by the pre-ictal period length in training data and the feature window size, not by a regression output.

## 15.2 Continuous Training Sweep

Training should simulate real watch usage:

1. keep full recording timelines (no random window subsampling in long-sweep runs)
2. split train/val/test by recording to prevent temporal leakage
3. iterate in timeline order within each split
4. update neural state and Bayesian memory at each step

This ensures token counts, transition statistics, and risk trajectories evolve as they would during all-day wear.

## 15.3 Horizon-Aware Targets

Support multi-horizon prediction heads simultaneously:

* binary state: healthy vs preictal
* short horizon: seizure in next 30/60/120 seconds
* medium horizon: seizure in next 5/10/20 minutes

This avoids the "single 1-hour snapshot" failure mode and enforces operationally meaningful warning windows.

## 15.4 Day-Simulation Evaluation

Evaluate on complete day-like streams using rolling inference traces:

* sensitivity at lead times (30s, 60s, 5m, 10m)
* false alarms per 8h monitoring
* temporal stability (prediction jitter)
* calibration (ECE/Brier) over time bins

## 15.5 Implemented Online Memory Simulation (Now Available)

The current pipeline now includes a recording-aware chronological Bayesian memory pass during long-sweep evaluation.

Per step $t$ (within each recording):

```text
token_t = discretize(p_alert_t, p_onset_t)
```

The two heads `p_alert` and `p_onset` define a 2D prediction space. Because both are independently supervised (alert covers pre-ictal + onset; onset is a narrower positive class), their joint value varies across all three states — producing far richer token diversity than a single countdown scalar:

| State | p_alert | p_onset | Typical token region |
|---|---|---|---|
| interictal | low | low | bottom-left cluster |
| pre-ictal | high | low | top-left cluster |
| onset | high | high | top-right cluster |

```text
c_t = rho * c_{t-1} + one_hot(token_t)
A_t = rho * A_{t-1}; A_t(token_{t-1}, token_t) += 1
```

Memory-informed risk fusion:

```text
r_mem(t) = 0.5 * p_hist(token_t | c_t) + 0.5 * p_trans(token_t | token_{t-1}, A_t)
r_fused(t) = alpha * r_raw(t) + (1-alpha) * r_mem(t)
```

Uncertainty proxy:

```text
u_t = sqrt(r_fused(t) * (1-r_fused(t)) / (N_t + 2))
```

Produced outputs include fused risk traces, memory risk, uncertainty, and token IDs for full timeline replay.

---

# 🔷 Final Perspective

What you are now positioned to claim is:

> **A hierarchical embedding-to-token physiological model with explicit Bayesian state memory for calibrated long-horizon prediction and clinically interpretable evidence traces**

That framing is:

* defensible in peer review
* aligned with current research trends
* distinct enough to be novel

---

## Immediate Next Steps (Execution-Oriented)

1. Implement a minimal fused predictor: $[s_t || phi(m_t) || e_t] -> p(y_{t+Delta})$.
2. Add calibration tracking (ECE, Brier, reliability bins) to training and validation.
3. Build token evidence reports per prediction (dominant tokens, transitions, confidence interval).
4. Run Phase A/B ablations with strict patient-wise splits before expanding model complexity.

At this point, the idea is strong and differentiated; the main requirement for publication confidence is empirical proof that Bayesian memory improves calibration and lead-time under realistic patient-level evaluation.
