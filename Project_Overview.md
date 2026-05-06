Here is a **concise project summary for a Functional Proof of Concept (PoC)** using the **SeizeIT2 dataset**. This format is suitable for a **research proposal, grant, thesis, or engineering project description**.

---

# Functional Proof of Concept: ECG-Centered Epileptic Seizure Anticipation Using the SeizeIT2 Dataset

## Project Overview

This project aims to develop a **functional proof of concept (PoC)** for an **epileptic seizure anticipation system** based primarily on **cardiac signals (ECG)** and supported by **wearable EEG and motion data**. The goal is to investigate whether **pre-ictal autonomic changes detectable in ECG signals** can be used to provide **early warnings of focal seizures**, with the long-term objective of enabling **wrist-wearable monitoring devices such as smartwatches**.

The PoC will be developed using the **SeizeIT2 dataset**, a large multimodal epilepsy monitoring dataset publicly available through OpenNeuro.

---

# Dataset

The project uses the **SeizeIT2 Dataset**, a BIDS-formatted multimodal wearable dataset derived from a multicenter clinical study (ClinicalTrials.gov: NCT04284072).

Dataset characteristics:

* **125 patients** with refractory focal epilepsy
* **5 European epilepsy monitoring units**
* **11,640 hours of wearable recordings**
* **886 recorded seizures**
* Multimodal wearable sensing

### Available Signal Modalities

| Modality             | Description             | Sampling Rate |
| -------------------- | ----------------------- | ------------- |
| EEG (behind-the-ear) | Wearable EEG channels   | 250 Hz        |
| ECG                  | Chest ECG               | 250 Hz        |
| EMG                  | Deltoid muscle activity | 250 Hz        |
| ACC / GYR            | Motion sensors          | 25 Hz         |

Each subject contains multiple monitoring runs organized in **BIDS format**, including:

* `.edf` physiological signal recordings
* `.json` metadata files
* `.tsv` event annotations containing seizure timing and labels.

Seizure annotations are synchronized with **clinical video-EEG monitoring**, providing reliable seizure onset ground truth.

---

# Research Objective

The main objective is to determine whether **pre-seizure autonomic signatures observable in ECG signals** can predict seizure onset before clinical manifestation.

Specific goals:

1. Identify **pre-ictal cardiac signatures** preceding focal seizures.
2. Train machine learning models to distinguish:

   * **Inter-ictal (normal)** periods
   * **Pre-ictal (seizure approaching)** periods
3. Evaluate whether **ECG-only models** can anticipate seizures.
4. Compare performance with **multimodal ECG + EEG models**.

The PoC will assess whether ECG-based monitoring can serve as a **foundation for future wearable seizure-warning systems**.

---

# System Architecture

The proof-of-concept system will include the following pipeline.

## 1 Data Ingestion

Load physiological recordings from BIDS-structured EDF files.

Signals used:

* ECG (primary signal)
* EEG (secondary validation signal)
* Motion (artifact detection)

Tools:

* Python
* MNE-Python
* NumPy / SciPy
* PyTorch or TensorFlow

---

## 2 Signal Preprocessing

### ECG

* Bandpass filtering
* R-peak detection
* RR interval extraction

Derived features:

* Heart rate
* Heart rate variability
* Autonomic metrics

### EEG

Used to confirm seizure onset timing and detect pre-ictal neurological changes.

### Motion

Used to filter segments containing excessive motion artifacts.

---

## 3 Feature Extraction

### ECG Features

Autonomic nervous system indicators:

* HR (Heart Rate)
* HRV metrics

  * RMSSD
  * SDNN
  * LF/HF ratio
* Entropy metrics
* RR interval variability
* Cardiac instability indicators

### EEG Features

Optional model enhancement:

* spectral band power
* entropy
* seizure precursor patterns

---

## 4 Temporal Labeling

Using the provided **events.tsv** annotations:

Define time windows:

| Window Type | Description                 |
| ----------- | --------------------------- |
| Inter-ictal | Normal background activity  |
| Pre-ictal   | 1–10 minutes before seizure |
| Ictal       | Seizure period              |

The model will be trained to detect **pre-ictal windows**.

---

## 5 Machine Learning Model

Two architectures will be evaluated:

### ECG-only model

Goal: assess smartwatch feasibility.

Possible architectures:

* Random Forest
* Gradient Boosted Trees
* LSTM time-series model

### Multimodal model

Fusion of:

* ECG
* EEG
* Motion

Possible architecture:

```
ECG features → LSTM
EEG features → CNN
Motion features → dense layer
        ↓
    Fusion layer
        ↓
Seizure risk prediction
```

Output:
**Probability of impending seizure**

---

## 6 Evaluation Metrics

The PoC will evaluate:

* Seizure prediction accuracy
* Sensitivity
* False alarm rate
* Prediction horizon (minutes before seizure)

Important evaluation metric:

**Time-to-seizure prediction capability**

---

# Expected Outcome

The project aims to demonstrate that:

1. **ECG contains detectable pre-ictal signatures**
2. Multimodal signals improve prediction reliability
3. Cardiac-based seizure anticipation is technically feasible

The final PoC will produce:

* A working seizure-risk prediction pipeline
* Performance evaluation on the SeizeIT2 dataset
* A baseline model for future wearable implementations

---

# Relevance to Wearable Devices

Although SeizeIT2 uses **chest ECG and behind-ear EEG**, the physiological patterns learned from the dataset can be translated to **wrist-based wearable devices** that measure:

* ECG (single lead)
* PPG-derived heart rate
* motion data

Thus the project provides a **foundation for smartwatch-based seizure warning systems**.

---

# Deliverables

1. Data preprocessing pipeline
2. Feature extraction modules
3. Seizure anticipation model
4. Evaluation framework
5. Demonstration of real-time seizure risk estimation

---

If you want, I can also help you build a **much stronger version suitable for a PhD proposal or paper**, including:

* a **clear novelty statement**
* **baseline comparisons**
* **model architecture diagrams**
* **expected performance benchmarks**
* **publication targets** (IEEE EMBC, NeurIPS ML4Health, etc.).
