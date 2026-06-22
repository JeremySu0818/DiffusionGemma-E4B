#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

echo "Stage 1: Weight Transplantation..."
python -m diffusiongemma_e4b.student \
  --base-model "${DG_MODEL:-google/gemma-4-E4B-it}" \
  --output-dir "${DG_TRANSPLANT_DIR:-artifacts/transplanted}" \
  --canvas-length "${DG_CANVAS_LENGTH:-256}" \
  --dtype "${DG_DTYPE:-bfloat16}" \
  --device-map "${DG_DEVICE_MAP:-auto}"

echo "Stage 2: Model Training (QLoRA)..."
python -m diffusiongemma_e4b.train \
  --model-dir "${DG_TRANSPLANT_DIR:-artifacts/transplanted}" \
  --data-dir "${DG_CORRUPTION_DIR:-data/corruption}" \
  --output-dir "${DG_TRAIN_OUTPUT_DIR:-artifacts/conversion_training}" \
  --train-mode "${DG_TRAIN_MODE:-qlora}" \
  --batch-size "${DG_BATCH_SIZE:-1}" \
  --gradient-accumulation-steps "${DG_GRAD_ACCUM:-32}" \
  --learning-rate "${DG_LR:-2e-4}" \
  --max-steps "${DG_MAX_STEPS:-200000}" \
  --save-interval "${DG_SAVE_INTERVAL:-1000}" \
  --val-interval "${DG_VAL_INTERVAL:-500}" \
  --val-batches "${DG_VAL_BATCHES:-16}" \
  --self-conditioning-prob "${DG_SELF_CONDITIONING_PROB:-0.5}" \
  --seed "${DG_SEED:-1337}" \
  --gradient-checkpointing \
  --resume
