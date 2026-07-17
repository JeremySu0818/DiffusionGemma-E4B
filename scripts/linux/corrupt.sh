#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
python -m diffusiongemma_e4b.corruption \
  --raw-jsonl "${DG_TEACHER_OUTPUT:-data/teacher_supervised/teacher_outputs.jsonl}" \
  --output-dir "${DG_CORRUPTION_DIR:-data/corruption}" \
  --tokenizer "${DG_STUDENT_MODEL:-google/gemma-4-E4B-it}" \
  --target-blocks "${DG_TARGET_BLOCKS:-200000}" \
  --canvas-length "${DG_CANVAS_LENGTH:-256}" \
  --prefix-length "${DG_PREFIX_LENGTH:-512}" \
  --shard-blocks "${DG_SHARD_BLOCKS:-4096}" \
  --seed "${DG_SEED:-1337}" \
  --record-order "${DG_RECORD_ORDER:-shuffled}" \
  --max-multimodal-bytes "${DG_MAX_MULTIMODAL_SHARD_BYTES:-68719476736}" \
  --max-shard-uncompressed-bytes "${DG_MAX_SHARD_UNCOMPRESSED_BYTES:-536870912}" \
  --dataset-config "${DG_DATASET_CONFIG:-configs/dataset_sources.json}" \
  --sources "${DG_DATASET_SOURCES:-}"
