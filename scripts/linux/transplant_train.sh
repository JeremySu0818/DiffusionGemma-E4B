#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

echo "Stage 1: Weight Transplantation..."
python -m diffusiongemma_e4b.student \
  --base-model google/gemma-4-E4B-it \
  --output-dir artifacts/transplanted \
  --canvas-length 256 \
  --dtype bfloat16 \
  --device-map auto

echo "Stage 2: Model Training (QLoRA)..."
python -m diffusiongemma_e4b.train \
  --model-dir artifacts/transplanted \
  --data-dir data/corruption \
  --output-dir artifacts/conversion_training \
  --train-mode qlora \
  --batch-size 1 \
  --gradient-accumulation-steps 32 \
  --learning-rate 2e-4 \
  --max-steps 200000 \
  --save-interval 1000 \
  --val-interval 500 \
  --val-batches 16 \
  --self-conditioning-prob 0.5 \
  --gradient-checkpointing \
  --resume
