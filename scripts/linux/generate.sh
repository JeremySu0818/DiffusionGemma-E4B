#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
if ! python - <<'PY'
import os
import requests

base_url = os.environ.get("DG_TEACHER_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
for url in (base_url + "/models", base_url.replace("/v1", "") + "/health"):
    try:
        response = requests.get(url, timeout=3)
        if response.status_code < 500:
            raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
then
  echo "Teacher server is not reachable at ${DG_TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}." >&2
  echo "Use bash scripts/linux/run_pipeline.sh to auto-start it, or start it manually with bash scripts/linux/serve_teacher.sh." >&2
  exit 1
fi
python -m diffusiongemma_e4b.teacher \
  --runtime openai-compatible \
  --model "${DG_MODEL:-google/gemma-4-E4B-it}" \
  --base-url "${DG_TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --source-config "${DG_DATASET_CONFIG:-configs/dataset_sources.json}" \
  --media-dir "${DG_MEDIA_CACHE_DIR:-data/media_cache}" \
  --max-prompt-chars "${DG_MAX_PROMPT_CHARS:-12000}" \
  --sources "${DG_DATASET_SOURCES:-}" \
  --output data/teacher_supervised/teacher_outputs.jsonl \
  --progress data/teacher_supervised/progress.json \
  --target-estimated-tokens "${DG_TARGET_ESTIMATED_TOKENS:-50000000}" \
  --max-tokens-per-sample "${DG_MAX_TOKENS_PER_SAMPLE:-4096}" \
  --temperature "${DG_TEACHER_TEMPERATURE:-0.95}" \
  --top-p "${DG_TEACHER_TOP_P:-0.98}" \
  --timeout-s "${DG_TEACHER_TIMEOUT_S:-900}"
