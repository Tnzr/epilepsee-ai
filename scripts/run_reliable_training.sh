#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PYTHON="${PYTHON:-python3}"
TORCHRUN="${TORCHRUN:-torchrun}"

DDP=0
SUBSET=0
GPU="0,1"
NPROC=2
LOG_DIR="${PROJECT_ROOT}/logs"
RUN_NAME="train_$(date +%Y%m%d_%H%M%S)"
WANDB_MODE="online"
WANDB_PROJECT="epilepsee-ai"
WANDB_RUN_NAME=""
DATA_MODE="real"
DATA_SOURCE="bids"
DATASET_ROOT=""
EPOCHS="7"
BATCH_SIZE="32"
MAX_RECORDINGS=""
MAX_SAMPLES_PER_RECORDING=""
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $0 [options] [-- extra train args]

Options:
  --ddp                       Launch with torchrun for multi-GPU training
  --subset                    Run a quick subset test before full training
  --gpu GPU_IDS               CUDA_VISIBLE_DEVICES (default: 0,1)
  --nproc N                   torchrun --nproc_per_node (default: 2)
  --run-name NAME             Custom run name for log files
  --log-dir DIR               Directory for logs (default: ./logs)
  --wandb-mode MODE           online|offline|disabled (default: online)
  --wandb-project NAME        W&B project name (default: epilepsee-ai)
  --wandb-run-name NAME       Explicit W&B run name
  --data-mode MODE            real|dummy (default: real)
  --data-source SOURCE        bids|wearable (default: bids)
  --dataset-root PATH         Dataset root path
  --epochs N                  Number of epochs (default: 7)
  --batch-size N              Batch size (default: 32)
  --max-recordings N          Real-mode recordings cap for subset mode
  --max-samples-per-recording N  Real-mode per-recording sample cap for subset mode
  --help                      Show this help message

Any remaining arguments after -- are passed through to scripts/train.py.
EOF
}

if [[ $# -eq 0 ]]; then
  usage
  exit 0
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ddp)
      DDP=1
      shift
      ;;
    --subset)
      SUBSET=1
      shift
      ;;
    --gpu)
      GPU="$2"
      shift 2
      ;;
    --nproc)
      NPROC="$2"
      shift 2
      ;;
    --run-name)
      RUN_NAME="$2"
      shift 2
      ;;
    --log-dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --wandb-mode)
      WANDB_MODE="$2"
      shift 2
      ;;
    --wandb-project)
      WANDB_PROJECT="$2"
      shift 2
      ;;
    --wandb-run-name)
      WANDB_RUN_NAME="$2"
      shift 2
      ;;
    --data-mode)
      DATA_MODE="$2"
      shift 2
      ;;
    --data-source)
      DATA_SOURCE="$2"
      shift 2
      ;;
    --dataset-root)
      DATASET_ROOT="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    --batch-size)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --max-recordings)
      MAX_RECORDINGS="$2"
      shift 2
      ;;
    --max-samples-per-recording)
      MAX_SAMPLES_PER_RECORDING="$2"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ "$SUBSET" -eq 1 ]]; then
  RUN_NAME="${RUN_NAME}-subset"
  EPOCHS="1"
  BATCH_SIZE="16"
  if [[ -z "$MAX_RECORDINGS" ]]; then
    MAX_RECORDINGS="8"
  fi
  if [[ -z "$MAX_SAMPLES_PER_RECORDING" ]]; then
    MAX_SAMPLES_PER_RECORDING="12"
  fi
  EXTRA_ARGS+=(--wandb-run-name "${RUN_NAME}")
  EXTRA_ARGS+=(--training-mode stateful)
  EXTRA_ARGS+=(--batching-strategy patient_sequential)
  EXTRA_ARGS+=(--patient-sequential)
fi

if [[ "$DATA_MODE" != "dummy" && -z "$DATASET_ROOT" ]]; then
  echo "ERROR: --dataset-root must be provided for real mode training."
  echo "Run with --dataset-root /path/to/SeizeIT2"
  exit 2
fi

CMD=()
if [[ "$DDP" -eq 1 ]]; then
  CMD+=("$TORCHRUN" "--nproc_per_node=${NPROC}" "scripts/train.py")
else
  CMD+=("$PYTHON" "scripts/train.py")
fi

CMD+=(--data-mode "$DATA_MODE"
  --data-source "$DATA_SOURCE"
  --dataset-root "$DATASET_ROOT"
  --wandb-mode "$WANDB_MODE"
  --wandb-project "$WANDB_PROJECT"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
)

if [[ -n "$MAX_RECORDINGS" ]]; then
  CMD+=(--max-recordings "$MAX_RECORDINGS")
fi

if [[ -n "$MAX_SAMPLES_PER_RECORDING" ]]; then
  CMD+=(--max-samples-per-recording "$MAX_SAMPLES_PER_RECORDING")
fi

if [[ -n "$WANDB_RUN_NAME" ]]; then
  CMD+=(--wandb-run-name "$WANDB_RUN_NAME")
fi

if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  CMD+=("${EXTRA_ARGS[@]}")
fi

mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$RUN_NAME.log"
PID_FILE="$LOG_DIR/$RUN_NAME.pid"
CMD_FILE="$LOG_DIR/$RUN_NAME.cmd"

printf '%s\n' "${CMD[@]}" > "$CMD_FILE"

export WANDB_MODE="$WANDB_MODE"
export WANDB_PROJECT="$WANDB_PROJECT"
if [[ "$DDP" -eq 0 ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU"
fi

echo "Starting reliable training run"
echo "  run name: $RUN_NAME"
echo "  log file: $LOG_FILE"
echo "  dataset root: $DATASET_ROOT"
echo "  wandb mode: $WANDB_MODE"
echo "  wandb project: $WANDB_PROJECT"
echo "  command saved to: $CMD_FILE"

nohup stdbuf -oL -eL "${CMD[@]}" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!
echo "$TRAIN_PID" > "$PID_FILE"

sleep 2

if ps -p "$TRAIN_PID" >/dev/null 2>&1; then
  echo "Started background training PID $TRAIN_PID"
  echo "Log tail:"
  tail -n 10 "$LOG_FILE"
  echo "Use 'tail -f $LOG_FILE' to follow progress."
  echo "PID file: $PID_FILE"
else
  echo "Training process terminated quickly; check the log for details: $LOG_FILE"
  tail -n 30 "$LOG_FILE" || true
  exit 1
fi
