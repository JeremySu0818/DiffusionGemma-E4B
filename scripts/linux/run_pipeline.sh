#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
PRESET="gpu"
echo "Using pipeline preset: $PRESET"
eval "$(python -m diffusiongemma_e4b.pipeline_presets --preset "$PRESET" --shell bash)"

export DG_TEACHER_BASE_URL="${DG_TEACHER_BASE_URL:-http://127.0.0.1:${DG_TEACHER_PORT:-8000}/v1}"
TEACHER_PID=""

teacher_ready() {
  python - <<'PY'
import os
import requests

base = os.environ["DG_TEACHER_BASE_URL"].rstrip("/")
urls = [base + "/models", base.replace("/v1", "") + "/health"]
for url in urls:
    try:
        response = requests.get(url, timeout=5)
        if response.status_code < 500:
            raise SystemExit(0)
    except Exception:
        pass
raise SystemExit(1)
PY
}

cleanup_teacher() {
  if [[ -n "$TEACHER_PID" ]] && kill -0 "$TEACHER_PID" 2>/dev/null; then
    kill "$TEACHER_PID" 2>/dev/null || true
    wait "$TEACHER_PID" 2>/dev/null || true
  fi
}
trap cleanup_teacher EXIT

if ! teacher_ready; then
  mkdir -p outputs/logs
  echo "Starting teacher server in background..."
  "$SCRIPT_DIR/serve_teacher.sh" > outputs/logs/teacher_server.log 2>&1 &
  TEACHER_PID="$!"
  for _ in $(seq 1 180); do
    if teacher_ready; then
      break
    fi
    if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
      echo "Teacher server exited early. See outputs/logs/teacher_server.log" >&2
      exit 1
    fi
    sleep 5
  done
  if ! teacher_ready; then
    echo "Teacher server did not become ready. See outputs/logs/teacher_server.log" >&2
    exit 1
  fi
fi

"$SCRIPT_DIR/generate.sh"
"$SCRIPT_DIR/corrupt.sh"
"$SCRIPT_DIR/transplant_train.sh"
