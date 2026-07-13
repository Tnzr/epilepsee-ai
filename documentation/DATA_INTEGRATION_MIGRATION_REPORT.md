# Data Integration / Migration Summary Report

**Project:** Epilepsee-AI  
**Date:** 2026-03-16  
**Scope:** Integrate wearable wrist-based Oregon dataset into existing seizure anticipation pipeline currently centered on BIDS dataset `ds005873`.

---

## 1) Executive Summary

The integration is in place and functional for both data sources:

- **Existing pipeline remains compatible** with BIDS (`ds005873`) loaders and workflows.
- **New wearable loader path is implemented** for the Oregon wrist-device dataset (CSV streams + Excel seizure annotations).
- **Core migration objective is achieved:** the project can now ingest and label both dataset styles in a consistent training-ready format.

Although seizure-positive wearable samples are currently limited, this integration establishes the required ingestion, labeling, and export scaffolding so the team can scale quickly once more wearable seizure data is acquired.

---

## 2) What Was Integrated

### Implemented capabilities

- Wearable dataset ingestion from:
  - `WearablwDevice-Oregon/.../CSV/*.csv` signal streams
  - Master seizure annotation workbook (`Wearable Seizre Detection Master Database.xlsx`)
- Seizure-time parsing from absolute timestamps (date + time)
- Windowed feature extraction and countdown labeling (preictal vs interictal)
- Compatibility with existing `SeizureDataset` object for downstream model training
- Debug/test artifacts and workflow notebook support for reproducible runs

### Migration design outcome

The system now supports **dual-source ingestion**:

1. **BIDS/EDF event-based workflow** (existing)
2. **Wearable/CSV absolute-time workflow** (new)

Both converge into a common model-facing tensor/label interface.

---

## 3) Dataset Comparison (Differences, Pros/Cons, Best Use)

## A) `ds005873` (BIDS, EDF-based clinical-style data)

### Characteristics
- Structured BIDS hierarchy (`sub-*/ses-*`)
- Signal files typically in EDF format
- Seizure events via `events.tsv` and run/session metadata
 - 125 patients with 886 focal seizures in the full cohort

### Pros
- Strong standardized structure (BIDS conventions)
- Rich neurophysiology context and cleaner event linkage
- Better for reproducible cross-subject benchmarking in research settings

### Cons
- Not wrist-wearable native
- Domain mismatch for smartwatch deployment scenarios
- Potentially less representative of ambulatory real-world wrist signal noise

### Best for
- Baseline model development
- Method benchmarking and controlled evaluation
- Architecture/loss function experimentation with clearer annotation structure

---

## B) WearablwDevice-Oregon (wrist wearable, CSV stream-based)

### Characteristics
- Wrist-device streams (e.g., PPG, motion/ADXL, EDA, temperature)
- Recording metadata distributed across folder structure and stream files
- Seizure labels maintained in a master Excel annotation source
 - Current curated integration covers **7 seizure‑positive patients**, with only a small number of seizures across them

### Pros
- Directly aligned with target deployment modality (wrist-based sensing)
- Captures practical sensor behavior/noise expected in real-world wearable usage
- Enables pipeline maturation for eventual field-grade seizure anticipation

### Cons
- Less standardized than BIDS; more custom parsing logic required
- Annotation linkage is less explicit at file level (requires cross-reference to master DB)
- Current seizure-positive volume is **very limited**:
  - Only 7 seizure‑positive patients in the present integration.
  - Only a handful of seizures overall, versus hundreds of mostly seizure‑free recordings in the raw corpus.
  - When converted into 60‑s, 1‑s‑stride windows, this yields many tens of thousands of windows, but only a tiny fraction are genuinely preictal.

### Best for
- Domain adaptation toward wearable deployment
- Validation of robustness to real-world sensor variability
- Building the production-facing ingestion and preprocessing pipeline

In other words, `ds005873` provides broad coverage (hundreds of seizures across 125 patients) and is well suited for benchmarking architectures and losses, while the Oregon wearable path currently functions as a **scarce, 7‑patient pilot cohort** that is primarily valuable for exercising the wrist‑specific ingestion, labeling, and windowing stack and for early, highly exploratory modeling.

---

## 4) Current Data Sufficiency Reality

The available wearable seizure-positive data is limited for broad generalization and stable model selection on its own. This affects:

- Reliable threshold calibration
- Robust patient-independent performance estimates
- Confidence in early-warning performance under distribution shift

That said, this phase is still high value because it:

- De-risks future onboarding of new wearable cohorts
- Validates ingestion + labeling + feature windowing assumptions now
- Reduces future turnaround time once additional seizure data arrives

---

## 5) Recommended Practical Use Right Now

### Near-term strategy

1. Use **`ds005873`** for primary model prototyping and objective benchmarking.
2. Use **Wearable-Oregon** to validate end-to-end wrist-specific preprocessing and labeling behavior.
3. Treat current wearable modeling as **pipeline readiness + exploratory analysis**, not final clinical performance claims.

### Model development posture

- Prioritize robust preprocessing invariants and failure handling.
- Maintain strict separation of exploratory wearable metrics from production claims.
- Track class imbalance and event sparsity explicitly in each experiment summary.

---

## 6) Why This Integration Matters Before More Data Requests

This integration establishes the operational foundation needed to scale:

- The codebase now supports both reference clinical-like and target wearable modalities.
- New wearable data can be dropped in with substantially less engineering overhead.
- Future data collection requests can be made with precise schema/annotation requirements based on concrete pipeline experience.

In short, even with limited current seizure events, this migration is the correct preparatory step and materially improves readiness for the next data acquisition phase.

---

## 7) Suggested Next Step (Data Request Framing)

When requesting additional wearable data, prioritize:

- More seizure-positive recording hours per subject
- Explicit per-event onset/offset timestamps in machine-readable format
- Consistent device/session metadata mapping
- Signal quality markers and missing-data indicators

This will directly improve model training stability, validation quality, and translational confidence.
