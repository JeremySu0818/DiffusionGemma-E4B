#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run: bash scripts/linux/setup.sh" >&2
  exit 2
fi
source .venv/bin/activate

PRESET="${DG_PRESET:-gpu}"
eval "$(python -m diffusiongemma_e4b.pipeline_presets --preset "$PRESET" --shell bash)"
export DG_PRESET="$PRESET"
export DG_TEACHER_HOST="${DG_TEACHER_HOST:-127.0.0.1}"
export DG_TEACHER_BASE_URL="${DG_TEACHER_BASE_URL:-http://127.0.0.1:${DG_TEACHER_PORT:-8000}/v1}"

LOG_DIR="${DG_LOG_DIR:-outputs/logs}"
mkdir -p "$LOG_DIR"
TEACHER_PID=""
CURRENT_STAGE="startup"

teacher_ready() {
  python - <<'PY'
import os
from diffusiongemma_e4b.preflight import probe_teacher
result = probe_teacher(
    os.environ["DG_TEACHER_BASE_URL"],
    api_key=os.environ.get("DG_TEACHER_API_KEY"),
    expected_model=os.environ.get(
        "DG_TEACHER_SERVED_MODEL_NAME",
        os.environ.get("DG_MODEL", "google/gemma-4-E4B-it"),
    ),
)
raise SystemExit(0 if result["ready"] else 1)
PY
}

stop_auto_teacher() {
  if [[ -n "$TEACHER_PID" ]] && kill -0 "$TEACHER_PID" 2>/dev/null; then
    echo "Stopping auto-started teacher (PID $TEACHER_PID)..."
    kill "$TEACHER_PID" 2>/dev/null || true
    local deadline=$((SECONDS + ${DG_TEACHER_STOP_TIMEOUT_S:-30}))
    while kill -0 "$TEACHER_PID" 2>/dev/null && (( SECONDS < deadline )); do
      sleep 1
    done
    if kill -0 "$TEACHER_PID" 2>/dev/null; then
      echo "Teacher did not stop after SIGTERM; sending SIGKILL." >&2
      kill -KILL "$TEACHER_PID" 2>/dev/null || true
    fi
    wait "$TEACHER_PID" 2>/dev/null || true
  fi
  TEACHER_PID=""
}

on_exit() {
  local status=$?
  trap - EXIT INT TERM
  stop_auto_teacher
  if (( status != 0 )); then
    echo "Pipeline failed in stage '$CURRENT_STAGE' (exit $status). See $LOG_DIR/${CURRENT_STAGE}.log when present." >&2
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 130' INT TERM

run_stage() {
  local stage="$1"
  shift
  CURRENT_STAGE="$stage"
  echo
  echo "[$stage] $*"
  "$@" 2>&1 | tee "$LOG_DIR/${stage}.log"
}

echo "Using pipeline preset: $PRESET"
run_stage preflight python -m diffusiongemma_e4b.preflight \
  --preset "$PRESET" \
  --dataset-config "${DG_DATASET_CONFIG:-configs/dataset_sources.json}" \
  --output "$LOG_DIR/preflight.json"

CURRENT_STAGE="teacher_server"
if teacher_ready; then
  echo "Reusing healthy teacher endpoint at $DG_TEACHER_BASE_URL (only HTTP 2xx is accepted)."
else
  echo "Starting teacher server in background..."
  : > "$LOG_DIR/teacher_server.log"
  "$SCRIPT_DIR/serve_teacher.sh" >> "$LOG_DIR/teacher_server.log" 2>&1 &
  TEACHER_PID="$!"
  deadline=$((SECONDS + ${DG_TEACHER_START_TIMEOUT_S:-1800}))
  while (( SECONDS < deadline )); do
    if teacher_ready; then
      break
    fi
    if ! kill -0 "$TEACHER_PID" 2>/dev/null; then
      echo "Teacher server exited early. See $LOG_DIR/teacher_server.log" >&2
      exit 1
    fi
    sleep 5
  done
  if ! teacher_ready; then
    echo "Teacher server did not return HTTP 2xx before timeout. See $LOG_DIR/teacher_server.log" >&2
    exit 1
  fi
fi

run_stage generate "$SCRIPT_DIR/generate.sh"
# The teacher reserves most of the accelerator. Release it before corruption
# and training; never stop a server that the user supplied independently.
stop_auto_teacher

run_stage corrupt "$SCRIPT_DIR/corrupt.sh"
run_stage train "$SCRIPT_DIR/transplant_train.sh"
run_stage validate_export "$SCRIPT_DIR/validate_export.sh"
CURRENT_STAGE="complete"
echo
printf 'Pipeline complete. Final artifact: %s\n' "${DG_TRAIN_OUTPUT_DIR:-artifacts/conversion_training}/final"
