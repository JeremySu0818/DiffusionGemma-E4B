#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ ! -f .venv/bin/activate ]]; then
  echo "Missing .venv. Run: bash scripts/linux/setup.sh" >&2
  exit 1
fi
source .venv/bin/activate

# Make CUDA libraries installed in the virtual environment discoverable.
for d in "${REPO_DIR}"/.venv/lib/python3.*/site-packages/nvidia/*/lib; do
  if [[ -d "$d" ]]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

MODEL="${DG_MODEL:-google/gemma-4-E4B-it}"
SERVED_MODEL_NAME="${DG_TEACHER_SERVED_MODEL_NAME:-$MODEL}"
HOST="${DG_TEACHER_HOST:-127.0.0.1}"
PORT="${DG_TEACHER_PORT:-8000}"
GPU_MEMORY_UTILIZATION="${DG_TEACHER_GPU_MEMORY_UTILIZATION:-0.80}"
TENSOR_PARALLEL_SIZE="${DG_TEACHER_TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${DG_TEACHER_MAX_MODEL_LEN:-16384}"
DTYPE="${DG_DTYPE:-bfloat16}"

if ! python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  echo "vLLM is required for the local teacher server but is not installed." >&2
  echo "Re-run setup, or set DG_TEACHER_BASE_URL to an external OpenAI-compatible Gemma 4 E4B server." >&2
  exit 1
fi

extra_args=()
if [[ "${DG_TEACHER_ENFORCE_EAGER:-0}" == "1" ]]; then
  extra_args+=(--enforce-eager)
fi
if [[ -n "${DG_TEACHER_API_KEY:-}" ]]; then
  extra_args+=(--api-key "$DG_TEACHER_API_KEY")
fi

exec python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_MODEL_NAME" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
  --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
  --max-model-len "$MAX_MODEL_LEN" \
  --trust-remote-code \
  "${extra_args[@]}"
