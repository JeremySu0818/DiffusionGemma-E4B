#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

STUDENT_INIT="${DG_STUDENT_INIT:-transplant}"
STUDENT_MODEL="${DG_STUDENT_MODEL:-${DG_MODEL:-google/gemma-4-E4B-it}}"
MODEL_INPUT="${DG_TRANSPLANT_DIR:-artifacts/transplanted}"
TOKENIZER_SOURCE="${DG_TOKENIZER_SOURCE:-$STUDENT_MODEL}"

if [[ "$STUDENT_INIT" != "transplant" ]]; then
  echo "Unsupported DG_STUDENT_INIT=$STUDENT_INIT; production requires the E4B transplant path." >&2
  exit 2
fi

echo "Stage 1: building the E4B-sized diffusion student from $STUDENT_MODEL..."
student_args=(
  --base-model "$STUDENT_MODEL"
  --output-dir "$MODEL_INPUT"
  --canvas-length "${DG_CANVAS_LENGTH:-256}"
  --dtype "${DG_DTYPE:-bfloat16}"
  --device-map "${DG_DEVICE_MAP:-auto}"
)
python -m diffusiongemma_e4b.student "${student_args[@]}"

echo "Stage 2: training the E4B Diffusion Transformer (${DG_TRAIN_MODE:-qlora})..."
train_args=(
  --model-dir "$MODEL_INPUT"
  --tokenizer-source "$TOKENIZER_SOURCE"
  --data-dir "${DG_CORRUPTION_DIR:-data/corruption}"
  --output-dir "${DG_TRAIN_OUTPUT_DIR:-artifacts/conversion_training}"
  --train-mode "${DG_TRAIN_MODE:-qlora}"
  --device-map "${DG_DEVICE_MAP:-auto}"
  --dtype "${DG_DTYPE:-bfloat16}"
  --batch-size "${DG_BATCH_SIZE:-1}"
  --gradient-accumulation-steps "${DG_GRAD_ACCUM:-32}"
  --learning-rate "${DG_LEARNING_RATE:-${DG_LR:-1e-5}}"
  --weight-decay "${DG_WEIGHT_DECAY:-0.1}"
  --max-grad-norm "${DG_MAX_GRAD_NORM:-1.0}"
  --max-optimizer-steps "${DG_MAX_OPTIMIZER_STEPS:-125000}"
  --warmup-steps "${DG_WARMUP_STEPS:-3750}"
  --save-interval "${DG_SAVE_INTERVAL:-1000}"
  --val-interval "${DG_VAL_INTERVAL:-500}"
  --val-batches "${DG_VAL_BATCHES:-64}"
  --min-relative-val-improvement "${DG_MIN_RELATIVE_VAL_IMPROVEMENT:-0.001}"
  --checkpoint-retention "${DG_CHECKPOINT_RETENTION:-3}"
  --self-conditioning-prob "${DG_SELF_CONDITIONING_PROB:-0.5}"
  --clean-token-loss-weight "${DG_CLEAN_TOKEN_LOSS_WEIGHT:-0.0}"
  --noise-min "${DG_NOISE_MIN:-0.05}"
  --noise-max "${DG_NOISE_MAX:-0.95}"
  --lora-r "${DG_LORA_R:-64}"
  --lora-alpha "${DG_LORA_ALPHA:-128}"
  --lora-dropout "${DG_LORA_DROPOUT:-0.05}"
  --lora-target-modules "${DG_LORA_TARGET_MODULES:-auto}"
  --val-fraction "${DG_VAL_FRACTION:-0.02}"
  --num-workers "${DG_NUM_WORKERS:-0}"
  --seed "${DG_SEED:-1337}"
)

if [[ "${DG_OFFLINE_CORRUPTION:-0}" == "1" ]]; then
  train_args+=(--offline-corruption)
fi
if [[ "${DG_GRADIENT_CHECKPOINTING:-0}" == "1" ]]; then
  train_args+=(--gradient-checkpointing)
fi
if [[ "${DG_RESUME:-1}" == "1" ]]; then
  train_args+=(--resume)
fi

python -m diffusiongemma_e4b.train "${train_args[@]}"
