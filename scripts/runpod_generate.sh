#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
python -m diffusiongemma_e4b.teacher \
  --runtime openai-compatible \
  --model google/gemma-4-E4B-it \
  --base-url "${DG_TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --output data/raw_self_continuation/self_continuation.jsonl \
  --progress data/raw_self_continuation/progress.json \
  --target-estimated-tokens 50000000 \
  --max-tokens-per-sample 4096 \
  --temperature 0.95 \
  --top-p 0.98 \
  --timeout-s 900
