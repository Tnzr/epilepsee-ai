SHELL := /bin/bash

ifneq ($(strip $(CONDA_PREFIX)),)
PYTHON ?= $(CONDA_PREFIX)/bin/python
TORCHRUN ?= $(CONDA_PREFIX)/bin/torchrun
else
PYTHON ?= python3
TORCHRUN ?= torchrun
endif

# Prefer the dedicated project env if it exists (more reliable than base conda)
ifneq ($(wildcard /home/tnzr/.local/share/mamba/envs/epilepsee-ai/bin/python),)
override PYTHON := /home/tnzr/.local/share/mamba/envs/epilepsee-ai/bin/python
override TORCHRUN := /home/tnzr/.local/share/mamba/envs/epilepsee-ai/bin/torchrun
endif
NPROC ?= 2
GPU ?= 0,1
MODEL ?= ecg_lstm
EPOCHS ?= 80
BATCH ?= 32
LR ?= 0.001
CONFIG ?=
DATA_MODE ?= real
DATA_SOURCE ?= bids
WANDB_MODE ?= online
WANDB_PROJECT ?= epilepsee-ai
DATASET_ROOT ?=
MAX_RECORDINGS ?= 120
MAX_SAMPLES_PER_RECORDING ?= 120
REAL_STRIDE_SECONDS ?=
Q_MODE ?= dynamic_int8
Q_OUTPUT_DIR ?= models/quantized
Q_SEQUENCE_LENGTH ?= 600
Q_BATCH ?= 1
Q_ITERS ?= 100
Q_CHECKPOINT ?= models/best_model.pt
Q_CONFIG ?= models/config.yaml
ARTIFACTS_DIR ?= models/artifacts
THRESH_OBJECTIVE ?= balanced_accuracy
THRESH_MIN_SENS ?= 0.75
THRESH_MAX_FPR ?= 0.35
SEED ?= 42
STRIDE_SWEEP ?= 0.2 0.5 1.0 2.0 5.0
SWEEP_OUTPUT_DIR ?= models/sampling_sweeps
LONG_SWEEP ?= 0
ONLINE_AUG_PROB ?= 0.7
STATE_LOSS ?= 0
ONSET_THRESHOLD_MIN ?= 2.0
RING_BUFFER_SIZE ?= 0
OPTIMIZER_STEP_SCOPE ?= batch
PATIENTS_PER_STEP ?= 1

TRAIN_ARGS = --model-type $(MODEL) --epochs $(EPOCHS) --batch-size $(BATCH) --learning-rate $(LR) --data-mode $(DATA_MODE) --data-source $(DATA_SOURCE) --max-recordings $(MAX_RECORDINGS) --max-samples-per-recording $(MAX_SAMPLES_PER_RECORDING)
TRAIN_ARGS += --wandb-mode $(WANDB_MODE) --wandb-project $(WANDB_PROJECT)
ifneq ($(strip $(DATASET_ROOT)),)
TRAIN_ARGS += --dataset-root $(DATASET_ROOT)
endif
ifneq ($(strip $(REAL_STRIDE_SECONDS)),)
TRAIN_ARGS += --real-stride-seconds $(REAL_STRIDE_SECONDS)
endif
ifneq ($(strip $(CONFIG)),)
TRAIN_ARGS += --config $(CONFIG)
endif
ifneq ($(strip $(ONLINE_AUG_PROB)),)
TRAIN_ARGS += --online-aug-prob $(ONLINE_AUG_PROB)
endif
ifeq ($(strip $(LONG_SWEEP)),1)
TRAIN_ARGS += --long-sweep-training
endif
ifeq ($(strip $(PATIENT_SEQUENTIAL)),1)
TRAIN_ARGS += --patient-sequential
endif
ifeq ($(strip $(STATE_LOSS)),1)
TRAIN_ARGS += --state-loss --onset-threshold-min $(ONSET_THRESHOLD_MIN)
endif
ifneq ($(strip $(RING_BUFFER_SIZE)),0)
TRAIN_ARGS += --ring-buffer-size $(RING_BUFFER_SIZE)
endif
TRAIN_ARGS += --optimizer-step-scope $(OPTIMIZER_STEP_SCOPE) --patients-per-step $(PATIENTS_PER_STEP)

.PHONY: help test test-training train train-ddp train-dummy train-ddp-dummy \
	train-temporal train-temporal-ddp train-multimodal-ddp \
	train-eegnet train-eegnet-ddp train-mobilenet train-mobilenet-ddp \
	train-tcn train-tcn-ddp train-inception train-inception-ddp \
	train-ecg-lstm-ddp train-cnn-lstm-ddp \
	train-watch-eegnet train-watch-mobilenet train-watch-tcn \
	train-watch-eegnet-ddp train-watch-mobilenet-ddp train-watch-tcn-ddp \
	train-wearable-recommended train-wearable-ddp train-wearable-config \
	train-wearable-config-mobilenet \
	train-patient-sequential train-patient-sequential-ddp \
	train-safe-state-loss train-safe-state-loss-ring \
	train-ecg-progression train-ecg-progression-ddp \
	quantize quantize-eegnet quantize-mobilenet quantize-tcn \
	quantize-temporal-fp16 quantize-multimodal-transformer-fp16 \
	quantize-ecg-progression quantize-watch \
	snapshot-model snapshot-eegnet snapshot-mobilenet snapshot-tcn \
	snapshot-temporal snapshot-multimodal-transformer \
	train-snapshot-eegnet train-snapshot-mobilenet train-snapshot-tcn \
	train-snapshot-temporal train-snapshot-multimodal-transformer \
	quantize-eegnet-artifact quantize-mobilenet-artifact quantize-tcn-artifact \
	train-reliable-ecg train-reliable-ecg-ddp \
	train-auto-threshold train-auto-threshold-ddp \
	train-auto-threshold-sampling-sweep train-auto-threshold-ddp-sampling-sweep \
	summarize-sampling-sweep \
	visualize-wearable visualize-wearable-multi visualize-wearable-all \
	visualize-wearable-all-signals visualize-wearable-all-signals-extended \
	visualize-explainability \
	dataset-check env-check

help: ## Show all available commands
	@echo "Epilepsee-AI Make targets"
	@echo ""
	@grep -E '^[a-zA-Z0-9_.-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  %-24s %s\n", $$1, $$2}'
	@echo ""
	@echo "Example:"
	@echo "  make train-ddp MODEL=temporal_transformer EPOCHS=100 DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make train-eegnet-ddp EPOCHS=25 DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make train-eegnet GPU=0  # force single-GPU"
	@echo "  make train-watch-eegnet DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make train-ecg-progression DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make quantize MODEL=eegnet Q_MODE=dynamic_int8"
	@echo "  make quantize MODEL=eegnet Q_CHECKPOINT=models/best_model.pt Q_CONFIG=models/config.yaml"
	@echo "  make train-snapshot-eegnet DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make quantize-eegnet-artifact"
	@echo "  make train-reliable-ecg-ddp DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make train-auto-threshold MODEL=tcn DATASET_ROOT=/path/to/SeizeIT2"
	@echo "  make train-wearable-recommended"
	@echo "  make quantize-watch"

env-check: ## Print Python/Torch environment diagnostics
	@echo "Python: $$($(PYTHON) -V 2>&1)"
	@echo "Torchrun: $$(command -v $(TORCHRUN) || echo 'not found')"
	@$(PYTHON) -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count())"

dataset-check: ## Verify DATASET_ROOT exists and looks like a BIDS dataset
	@if [ -z "$(DATASET_ROOT)" ]; then \
		echo "ERROR: DATASET_ROOT is empty."; \
		echo "Usage: make dataset-check DATASET_ROOT=/path/to/SeizeIT2"; \
		exit 2; \
	fi
	@if [ ! -d "$(DATASET_ROOT)" ]; then \
		echo "ERROR: DATASET_ROOT does not exist: $(DATASET_ROOT)"; \
		echo "Hint: ensure the drive is mounted in THIS shell session."; \
		exit 2; \
	fi
	@if [ ! -f "$(DATASET_ROOT)/participants.tsv" ]; then \
		echo "WARNING: participants.tsv not found at $(DATASET_ROOT)/participants.tsv"; \
		echo "Path exists but may not be dataset root."; \
	else \
		echo "OK: found participants.tsv"; \
	fi
	@echo "DATASET_ROOT is accessible: $(DATASET_ROOT)"

test: test-training ## Run verification tests

test-training: ## Run training pipeline verification script
	$(PYTHON) scripts/test_training.py

train: ## Single-GPU training (override MODEL/EPOCHS/etc)
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/train.py $(TRAIN_ARGS)

train-ddp: ## Multi-GPU DDP training via torchrun
	$(TORCHRUN) --nproc_per_node=$(NPROC) scripts/train.py $(TRAIN_ARGS)

train-dummy: ## Single-GPU dummy-data smoke run
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/train.py $(TRAIN_ARGS) --data-mode dummy

train-ddp-dummy: ## Multi-GPU dummy-data smoke run
	$(TORCHRUN) --nproc_per_node=$(NPROC) scripts/train.py $(TRAIN_ARGS) --data-mode dummy

train-temporal: ## Single-GPU temporal transformer run
	$(MAKE) train MODEL=temporal_transformer

train-temporal-ddp: ## Multi-GPU temporal transformer run
	$(MAKE) train-ddp MODEL=temporal_transformer

train-multimodal-ddp: ## Multi-GPU multimodal transformer run
	$(MAKE) train-ddp MODEL=multimodal_transformer

train-eegnet: ## TinyML-friendly EEGNet baseline
	$(MAKE) train MODEL=eegnet

train-eegnet-ddp: ## Multi-GPU EEGNet run
	$(MAKE) train-ddp MODEL=eegnet

train-mobilenet: ## Smartwatch-friendly MobileNet1D baseline
	$(MAKE) train MODEL=mobilenet_1d

train-mobilenet-ddp: ## Multi-GPU MobileNet1D run
	$(MAKE) train-ddp MODEL=mobilenet_1d

train-tcn: ## TCN baseline for edge gateways
	$(MAKE) train MODEL=tcn

train-tcn-ddp: ## Multi-GPU TCN run
	$(MAKE) train-ddp MODEL=tcn

train-inception: ## InceptionTime-1D baseline
	$(MAKE) train MODEL=inception_1d

train-inception-ddp: ## Multi-GPU InceptionTime-1D run
	$(MAKE) train-ddp MODEL=inception_1d

train-ecg-lstm-ddp: ## Multi-GPU ECG LSTM baseline run
	$(MAKE) train-ddp MODEL=ecg_lstm

train-cnn-lstm-ddp: ## Multi-GPU CNN-LSTM baseline run
	$(MAKE) train-ddp MODEL=cnn_lstm

train-watch-eegnet: ## Watch preset (ECG-first): EEGNet, small footprint
	$(MAKE) train MODEL=eegnet BATCH=64 EPOCHS=25 MAX_RECORDINGS=0

train-watch-mobilenet: ## Watch preset (ECG-first): MobileNet1D
	$(MAKE) train MODEL=mobilenet_1d BATCH=64 EPOCHS=40 MAX_RECORDINGS=0

train-watch-tcn: ## Watch/edge preset (ECG-first): TCN
	$(MAKE) train MODEL=tcn BATCH=64 EPOCHS=50 MAX_RECORDINGS=0

train-watch-eegnet-ddp: ## Multi-GPU watch preset: EEGNet
	$(MAKE) train-ddp MODEL=eegnet BATCH=64 EPOCHS=25 MAX_RECORDINGS=0

train-watch-mobilenet-ddp: ## Multi-GPU watch preset: MobileNet1D
	$(MAKE) train-ddp MODEL=mobilenet_1d BATCH=64 EPOCHS=40 MAX_RECORDINGS=0

train-watch-tcn-ddp: ## Multi-GPU watch preset: TCN
	$(MAKE) train-ddp MODEL=tcn BATCH=64 EPOCHS=50 MAX_RECORDINGS=0

train-wearable-recommended: ## Recommended wearable countdown run: single-GPU TCN + auto-threshold
	$(MAKE) train-auto-threshold \
		MODEL=tcn \
		DATA_SOURCE=wearable \
		DATASET_ROOT=/media/tnzr/HDD11/Datasets/WearablwDevice-Oregon \
		BATCH=4 \
		EPOCHS=20 \
		LR=0.001 \
		MAX_RECORDINGS=0 \
		MAX_SAMPLES_PER_RECORDING=0

train-wearable-config: ## Wearable TCN + focal loss config (single-GPU auto-threshold, low-FPR priority)
	$(MAKE) train-auto-threshold \
		MODEL=tcn \
		DATA_SOURCE=wearable \
		DATASET_ROOT=/media/tnzr/HDD11/Datasets/WearablwDevice-Oregon \
		BATCH=16 \
		EPOCHS=3 \
		LR=0.001 \
		MAX_RECORDINGS=0 \
		MAX_SAMPLES_PER_RECORDING=0 \
		THRESH_OBJECTIVE=f1 \
		THRESH_MIN_SENS=0.50 \
		THRESH_MAX_FPR=0.10 \
		CONFIG=config/wearable_tcn.yaml

train-wearable-config-mobilenet: ## Wearable MobileNet1D + focal loss config (mirrors train-wearable-config but with MODEL=mobilenet_1d)
	$(MAKE) train-auto-threshold \
		MODEL=mobilenet_1d \
		DATA_SOURCE=wearable \
		DATASET_ROOT=/media/tnzr/HDD11/Datasets/WearablwDevice-Oregon \
		BATCH=4 \
		EPOCHS=2 \
		LR=0.001 \
		MAX_RECORDINGS=0 \
		MAX_SAMPLES_PER_RECORDING=0 \
		THRESH_OBJECTIVE=f1 \
		THRESH_MIN_SENS=0.50 \
		THRESH_MAX_FPR=0.10 \
		CONFIG=config/wearable_tcn.yaml

train-wearable-ddp: ## Recommended wearable countdown run: 2-GPU TCN (DDP) + auto-threshold
	$(MAKE) train-auto-threshold-ddp \
		MODEL=tcn \
		DATA_SOURCE=wearable \
		DATASET_ROOT=/media/tnzr/HDD11/Datasets/WearablwDevice-Oregon \
		BATCH=4 \
		EPOCHS=3 \
		LR=0.001 \
		MAX_RECORDINGS=0 \
		MAX_SAMPLES_PER_RECORDING=0

train-patient-sequential: ## Train with patient-by-patient sequential processing (BIDS dataset)
	$(MAKE) train-auto-threshold \
		MODEL=eegnet \
		DATA_SOURCE=bids \
		DATASET_ROOT=/media/tnzr/HDD11/Datasets/ds005873 \
		ONLINE_AUG_PROB=0.95 \
		PATIENT_SEQUENTIAL=1

train-patient-sequential-ddp: ## Train DDP with patient-by-patient sequential processing (BIDS dataset)
	PATIENT_SEQUENTIAL=1 $(TORCHRUN) --nproc_per_node=$(NPROC) /home/tnzr/Documents/FIU/Research/Epilepsee-AI/scripts/train.py $(TRAIN_ARGS) \
		--data-source bids \
		--dataset-root /media/tnzr/HDD11/Datasets/ds005873 \
		--strict-real-data \
		--auto-threshold \
		--threshold-objective $(THRESH_OBJECTIVE) \
		--threshold-min-sensitivity $(THRESH_MIN_SENS) \
		--threshold-max-fpr $(THRESH_MAX_FPR) \
		--write-threshold-to-config \
		--seed $(SEED) \
		--patient-sequential \
		--online-aug-prob 0.95

train-safe-state-loss: ## Patient-sequential DDP with safe 3-class state loss (no GT countdown regression)
	$(MAKE) train-patient-sequential-ddp \
		STATE_LOSS=1 \
		ONSET_THRESHOLD_MIN=$(ONSET_THRESHOLD_MIN)

train-safe-state-loss-ring: ## State-loss DDP + ring-buffer temporal streaming (most deployment-realistic)
	$(MAKE) train-safe-state-loss \
		RING_BUFFER_SIZE=$(RING_BUFFER_SIZE)

quantize: ## Quantize models/best_model.pt (override MODEL/Q_MODE)
	$(PYTHON) scripts/quantize.py \
		--config $(Q_CONFIG) \
		--checkpoint $(Q_CHECKPOINT) \
		--model-type $(MODEL) \
		--mode $(Q_MODE) \
		--output-dir $(Q_OUTPUT_DIR) \
		--sequence-length $(Q_SEQUENCE_LENGTH) \
		--batch-size $(Q_BATCH) \
		--iters $(Q_ITERS)

quantize-eegnet: ## Quantize EEGNet checkpoint
	$(MAKE) quantize MODEL=eegnet

quantize-mobilenet: ## Quantize MobileNet1D checkpoint
	$(MAKE) quantize MODEL=mobilenet_1d

quantize-tcn: ## Quantize TCN checkpoint
	$(MAKE) quantize MODEL=tcn

quantize-temporal-fp16: ## Quantize temporal transformer with FP16
	$(MAKE) quantize MODEL=temporal_transformer Q_MODE=float16

quantize-multimodal-transformer-fp16: ## Quantize multimodal transformer with FP16
	$(MAKE) quantize MODEL=multimodal_transformer Q_MODE=float16

quantize-ecg-progression: ## Quantize ECG progression models
	$(MAKE) quantize-eegnet
	$(MAKE) quantize-mobilenet
	$(MAKE) quantize-tcn

quantize-watch: ## Watch-oriented quantization bundle (INT8)
	$(MAKE) quantize-ecg-progression Q_MODE=dynamic_int8

snapshot-model: ## Snapshot current models/best_model.pt + config.yaml into models/artifacts
	@mkdir -p $(ARTIFACTS_DIR)
	cp -f models/best_model.pt $(ARTIFACTS_DIR)/$(MODEL)_best_model.pt
	cp -f models/config.yaml $(ARTIFACTS_DIR)/$(MODEL)_config.yaml
	@echo "Saved: $(ARTIFACTS_DIR)/$(MODEL)_best_model.pt"
	@echo "Saved: $(ARTIFACTS_DIR)/$(MODEL)_config.yaml"

snapshot-eegnet: ## Snapshot EEGNet artifacts
	$(MAKE) snapshot-model MODEL=eegnet

snapshot-mobilenet: ## Snapshot MobileNet1D artifacts
	$(MAKE) snapshot-model MODEL=mobilenet_1d

snapshot-tcn: ## Snapshot TCN artifacts
	$(MAKE) snapshot-model MODEL=tcn

snapshot-temporal: ## Snapshot temporal transformer artifacts
	$(MAKE) snapshot-model MODEL=temporal_transformer

snapshot-multimodal-transformer: ## Snapshot multimodal transformer artifacts
	$(MAKE) snapshot-model MODEL=multimodal_transformer

train-snapshot-eegnet: ## Train EEGNet then snapshot checkpoint/config
	$(MAKE) train-watch-eegnet
	$(MAKE) snapshot-eegnet

train-snapshot-mobilenet: ## Train MobileNet1D then snapshot checkpoint/config
	$(MAKE) train-watch-mobilenet
	$(MAKE) snapshot-mobilenet

train-snapshot-tcn: ## Train TCN then snapshot checkpoint/config
	$(MAKE) train-watch-tcn
	$(MAKE) snapshot-tcn

train-snapshot-temporal: ## Train temporal transformer then snapshot checkpoint/config
	$(MAKE) train-temporal
	$(MAKE) snapshot-temporal

train-snapshot-multimodal-transformer: ## Train multimodal transformer then snapshot checkpoint/config
	$(MAKE) train-multimodal-ddp
	$(MAKE) snapshot-multimodal-transformer

quantize-eegnet-artifact: ## Quantize from saved EEGNet artifact snapshot
	$(MAKE) quantize MODEL=eegnet Q_CHECKPOINT=$(ARTIFACTS_DIR)/eegnet_best_model.pt Q_CONFIG=$(ARTIFACTS_DIR)/eegnet_config.yaml

quantize-mobilenet-artifact: ## Quantize from saved MobileNet1D artifact snapshot
	$(MAKE) quantize MODEL=mobilenet_1d Q_CHECKPOINT=$(ARTIFACTS_DIR)/mobilenet_1d_best_model.pt Q_CONFIG=$(ARTIFACTS_DIR)/mobilenet_1d_config.yaml

quantize-tcn-artifact: ## Quantize from saved TCN artifact snapshot
	$(MAKE) quantize MODEL=tcn Q_CHECKPOINT=$(ARTIFACTS_DIR)/tcn_best_model.pt Q_CONFIG=$(ARTIFACTS_DIR)/tcn_config.yaml

train-reliable-ecg: ## Reliable ECG run (single GPU): TCN with stronger training budget
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/train.py $(TRAIN_ARGS) \
		--model-type tcn --epochs 120 --batch-size 64 --learning-rate 0.0005 \
		--max-recordings 0 --max-samples-per-recording 0 --strict-real-data

train-reliable-ecg-ddp: ## Reliable ECG run (DDP): TCN with stronger training budget
	$(TORCHRUN) --nproc_per_node=$(NPROC) scripts/train.py $(TRAIN_ARGS) \
		--model-type tcn --epochs 120 --batch-size 64 --learning-rate 0.0005 \
		--max-recordings 0 --max-samples-per-recording 0 --strict-real-data

train-auto-threshold: ## Train single-GPU and auto-select detection threshold
	CUDA_VISIBLE_DEVICES=$(GPU) $(PYTHON) scripts/train.py $(TRAIN_ARGS) \
		--strict-real-data \
		--auto-threshold \
		--threshold-objective $(THRESH_OBJECTIVE) \
		--threshold-min-sensitivity $(THRESH_MIN_SENS) \
		--threshold-max-fpr $(THRESH_MAX_FPR) \
		--write-threshold-to-config

train-auto-threshold-ddp: ## Train DDP and auto-select detection threshold
	$(TORCHRUN) --nproc_per_node=$(NPROC) scripts/train.py $(TRAIN_ARGS) \
		--strict-real-data \
		--auto-threshold \
		--threshold-objective $(THRESH_OBJECTIVE) \
		--threshold-min-sensitivity $(THRESH_MIN_SENS) \
		--threshold-max-fpr $(THRESH_MAX_FPR) \
		--write-threshold-to-config \
		--seed $(SEED)

train-overnight: ## Reliable overnight DDP training with W&B online
	@if [ -z "$(DATASET_ROOT)" ]; then \
		echo "DATASET_ROOT is required. Example: make train-overnight MODEL=tcn DATASET_ROOT=/mnt/d/Datasets/SeizeIT2 EPOCHS=7"; \
		exit 1; \
	fi
	PYTHON=$(PYTHON) TORCHRUN=$(TORCHRUN) $(SHELL) scripts/run_reliable_training.sh --ddp --gpu $(GPU) --nproc $(NPROC) --dataset-root $(DATASET_ROOT) --wandb-mode online --wandb-project $(WANDB_PROJECT) --run-name overnight_$(MODEL) --epochs $(EPOCHS) --batch-size $(BATCH) --data-source $(DATA_SOURCE) -- --model-type $(MODEL)

train-auto-threshold-sampling-sweep: ## Single-GPU sweep over REAL_STRIDE_SECONDS values
	@set -e; \
	mkdir -p $(SWEEP_OUTPUT_DIR); \
	for stride in $(STRIDE_SWEEP); do \
		run_stamp=$$(date +%Y%m%d_%H%M%S); \
		stride_tag=$$(echo $$stride | tr '.' 'p'); \
		out_dir="$(SWEEP_OUTPUT_DIR)/stride_$${stride_tag}_$${run_stamp}"; \
		echo ""; \
		echo "=== Sampling sweep: REAL_STRIDE_SECONDS=$$stride (single-GPU) ==="; \
		$(MAKE) train-auto-threshold REAL_STRIDE_SECONDS=$$stride; \
		mkdir -p "$$out_dir"; \
		cp -f models/results.json "$$out_dir/"; \
		test -f models/threshold_selection.json && cp -f models/threshold_selection.json "$$out_dir/" || true; \
		test -f models/config.yaml && cp -f models/config.yaml "$$out_dir/" || true; \
		test -f models/gt_vs_inference_panel.png && cp -f models/gt_vs_inference_panel.png "$$out_dir/" || true; \
		echo "{\"stride_seconds\": $$stride, \"seed\": $(SEED), \"mode\": \"single\", \"timestamp\": \"$$run_stamp\"}" > "$$out_dir/run_metadata.json"; \
		echo "Saved run artifacts -> $$out_dir"; \
	done

train-auto-threshold-ddp-sampling-sweep: ## DDP sweep over REAL_STRIDE_SECONDS values
	@set -e; \
	mkdir -p $(SWEEP_OUTPUT_DIR); \
	for stride in $(STRIDE_SWEEP); do \
		run_stamp=$$(date +%Y%m%d_%H%M%S); \
		stride_tag=$$(echo $$stride | tr '.' 'p'); \
		out_dir="$(SWEEP_OUTPUT_DIR)/stride_$${stride_tag}_$${run_stamp}"; \
		echo ""; \
		echo "=== Sampling sweep: REAL_STRIDE_SECONDS=$$stride (DDP) ==="; \
		$(MAKE) train-auto-threshold-ddp REAL_STRIDE_SECONDS=$$stride; \
		mkdir -p "$$out_dir"; \
		cp -f models/results.json "$$out_dir/"; \
		test -f models/threshold_selection.json && cp -f models/threshold_selection.json "$$out_dir/" || true; \
		test -f models/config.yaml && cp -f models/config.yaml "$$out_dir/" || true; \
		test -f models/gt_vs_inference_panel.png && cp -f models/gt_vs_inference_panel.png "$$out_dir/" || true; \
		echo "{\"stride_seconds\": $$stride, \"seed\": $(SEED), \"mode\": \"ddp\", \"timestamp\": \"$$run_stamp\"}" > "$$out_dir/run_metadata.json"; \
		echo "Saved run artifacts -> $$out_dir"; \
	done

summarize-sampling-sweep: ## Build stride-vs-metrics summary table from sweep outputs
	$(PYTHON) scripts/summarize_sampling_sweep.py --sweep-dir $(SWEEP_OUTPUT_DIR)

# ============================================================================
# Wearable Device Visualization Targets
# ============================================================================

visualize-wearable: ## Visualize patient #2 wearable signals around first seizure (3min before, 1.5min after)
	$(PYTHON) scripts/visualize_wearable_sample_v2.py --patient 2 --before-min 3.0 --after-min 1.5

visualize-wearable-multi: ## Visualize patient #2 with extended time windows (5min before, 2min after)
	$(PYTHON) scripts/visualize_wearable_sample_v2.py --patient 2 --before-min 5.0 --after-min 2.0 --all-signals

visualize-wearable-all: ## Visualize all seizures for patient #2
	$(PYTHON) scripts/visualize_wearable_sample_v2.py --patient 2 --all-seizures --before-min 3.0 --after-min 1.5 --all-signals

visualize-wearable-all-signals: ## Visualize patient #2 with comprehensive all-signals figure (PPG, ADPD, ADXL, EDA, etc.)
	$(PYTHON) scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals

visualize-wearable-all-signals-extended: ## All signals for patient #2 with extended windows (5min before, 2min after)
	$(PYTHON) scripts/visualize_wearable_sample_v2.py --patient 2 --all-signals --before-min 10.0 --after-min 2.0

visualize-explainability: ## Build token histogram + rolling token/embedding waterfall explainability figure
	OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 nice -n 15 \
		$(PYTHON) scripts/visualize_token_explainability.py --predictions-npz models/test_predictions.npz --output-dir visualizations/explainability
