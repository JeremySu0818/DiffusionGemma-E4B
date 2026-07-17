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
api_key = os.environ.get("DG_TEACHER_API_KEY")
expected_model = os.environ.get(
    "DG_TEACHER_SERVED_MODEL_NAME",
    os.environ.get("DG_MODEL", "google/gemma-4-E4B-it"),
)
headers = {"Authorization": f"Bearer {api_key}"} if api_key else None
try:
    response = requests.get(base_url + "/models", headers=headers, timeout=5)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    served = {
        str(row.get("id")) for row in (rows or [])
        if isinstance(row, dict) and row.get("id") is not None
    }
    if expected_model in served:
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
  --model "${DG_TEACHER_SERVED_MODEL_NAME:-${DG_MODEL:-google/gemma-4-E4B-it}}" \
  --base-url "${DG_TEACHER_BASE_URL:-http://127.0.0.1:8000/v1}" \
  --source-config "${DG_DATASET_CONFIG:-configs/dataset_sources.json}" \
  --media-dir "${DG_MEDIA_CACHE_DIR:-data/media_cache}" \
  --max-prompt-chars "${DG_MAX_PROMPT_CHARS:-11000}" \
  --sources "${DG_DATASET_SOURCES:-}" \
  --max-records-per-source "${DG_MAX_RECORDS_PER_SOURCE:-0}" \
  --max-total-records "${DG_MAX_TOTAL_PROMPT_RECORDS:-0}" \
  --output "${DG_TEACHER_OUTPUT:-data/teacher_supervised/teacher_outputs.jsonl}" \
  --progress "${DG_TEACHER_PROGRESS:-data/teacher_supervised/progress.json}" \
  --target-estimated-tokens "${DG_TARGET_ESTIMATED_TOKENS:-0}" \
  --max-tokens-per-sample "${DG_MAX_TOKENS_PER_SAMPLE:-4096}" \
  --temperature "${DG_TEACHER_TEMPERATURE:-0.2}" \
  --top-p "${DG_TEACHER_TOP_P:-0.95}" \
  --timeout-s "${DG_TEACHER_TIMEOUT_S:-900}" \
  --max-retries "${DG_TEACHER_MAX_RETRIES:-5}" \
  --retry-base-s "${DG_TEACHER_RETRY_BASE_S:-2}" \
  --min-estimated-tokens "${DG_MIN_TEACHER_ESTIMATED_TOKENS:-8}" \
  --tokenizer "${DG_STUDENT_MODEL:-google/gemma-4-E4B-it}" \
  --max-consecutive-failures "${DG_TEACHER_MAX_CONSECUTIVE_FAILURES:-20}" \
  --concurrency "${DG_TEACHER_CONCURRENCY:-8}" \
  --student-prefix-length "${DG_PREFIX_LENGTH:-2048}"
