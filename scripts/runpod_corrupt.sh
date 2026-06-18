#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DiffusionGemma-E4B
source .venv/bin/activate
python -m diffusiongemma_e4b.corruption \
  --raw-jsonl data/raw_self_continuation/self_continuation.jsonl \
  --output-dir data/corruption \
  --tokenizer google/gemma-4-E4B-it \
  --target-blocks 200000 \
  --canvas-length 256 \
  --prefix-length 512 \
  --shard-blocks 4096
