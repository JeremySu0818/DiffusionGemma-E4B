#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
python -m diffusiongemma_e4b.corruption \
  --raw-jsonl data/teacher_supervised/teacher_outputs.jsonl \
  --output-dir data/corruption \
  --tokenizer "${DG_MODEL:-google/gemma-4-E4B-it}" \
  --target-blocks "${DG_TARGET_BLOCKS:-200000}" \
  --canvas-length "${DG_CANVAS_LENGTH:-256}" \
  --prefix-length "${DG_PREFIX_LENGTH:-512}" \
  --shard-blocks "${DG_SHARD_BLOCKS:-4096}"
