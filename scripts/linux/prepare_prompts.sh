#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
ARGS=(
  --config "${DG_DATASET_CONFIG:-configs/dataset_sources.json}"
  --output "${DG_PROMPT_JSONL:-data/prompt_context/prompts.jsonl}"
  --max-chars "${DG_MAX_PROMPT_CHARS:-12000}"
  --media-dir "${DG_MEDIA_CACHE_DIR:-data/media_cache}"
  --sources "${DG_DATASET_SOURCES:-}"
  --max-records-per-source "${DG_MAX_RECORDS_PER_SOURCE:-0}"
  --max-total-records "${DG_MAX_TOTAL_PROMPT_RECORDS:-0}"
)
python -m diffusiongemma_e4b.data_sources "${ARGS[@]}"
