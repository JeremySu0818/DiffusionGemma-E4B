#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
# The outer worker pool owns CPU parallelism; avoid one native pool per worker.
export OMP_NUM_THREADS="${DG_CORRUPTION_NATIVE_THREADS:-1}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export TOKENIZERS_PARALLELISM=false
python -m diffusiongemma_e4b.corruption \
  --raw-jsonl "${DG_TEACHER_OUTPUT:-data/teacher_supervised/teacher_outputs.jsonl}" \
  --output-dir "${DG_CORRUPTION_DIR:-data/corruption}" \
  --tokenizer "${DG_STUDENT_MODEL:-google/gemma-4-E4B-it}" \
  --target-blocks "${DG_TARGET_BLOCKS:-0}" \
  --canvas-length "${DG_CANVAS_LENGTH:-256}" \
  --prefix-length "${DG_PREFIX_LENGTH:-512}" \
  --shard-blocks "${DG_SHARD_BLOCKS:-4096}" \
  --seed "${DG_SEED:-1337}" \
  --record-order "${DG_RECORD_ORDER:-shuffled}" \
  --num-workers "${DG_CORRUPTION_WORKERS:-0}" \
  --prefetch-records "${DG_CORRUPTION_PREFETCH:-0}" \
  --max-multimodal-bytes "${DG_MAX_MULTIMODAL_SHARD_BYTES:-68719476736}" \
  --max-shard-uncompressed-bytes "${DG_MAX_SHARD_UNCOMPRESSED_BYTES:-536870912}" \
  --min-free-disk-bytes "${DG_MIN_FREE_DISK_BYTES:-2147483648}" \
  --dataset-config "${DG_DATASET_CONFIG:-configs/dataset_sources.json}" \
  --sources "${DG_DATASET_SOURCES:-}"
