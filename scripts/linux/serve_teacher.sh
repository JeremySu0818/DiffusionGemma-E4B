#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate

# Dynamically add virtual environment NVIDIA CUDA libraries to LD_LIBRARY_PATH
for d in "${REPO_DIR}"/.venv/lib/python3.*/site-packages/nvidia/*/lib; do
  if [ -d "$d" ]; then
    export LD_LIBRARY_PATH="$d:${LD_LIBRARY_PATH:-}"
  fi
done

MODEL="${DG_MODEL:-google/gemma-4-E4B-it}"
HOST="${DG_TEACHER_HOST:-0.0.0.0}"
PORT="${DG_TEACHER_PORT:-8000}"

if python - <<'PY'
import importlib.util
raise SystemExit(0 if importlib.util.find_spec("vllm") else 1)
PY
then
  exec python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host "$HOST" \
    --port "$PORT" \
    --gpu-memory-utilization 0.75 \
    --trust-remote-code
fi

echo "vLLM is not installed in this environment." >&2
echo "Install it with: uv pip install vllm" >&2
echo "Or start any OpenAI-compatible server and set DG_TEACHER_BASE_URL." >&2
exit 1
