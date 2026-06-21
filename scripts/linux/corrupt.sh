#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
python -m diffusiongemma_e4b.corruption \
  --raw-jsonl data/raw_self_continuation/self_continuation.jsonl \
  --output-dir data/corruption \
  --tokenizer google/gemma-4-E4B-it \
  --target-blocks 200000 \
  --canvas-length 256 \
  --prefix-length 512 \
  --shard-blocks 4096
