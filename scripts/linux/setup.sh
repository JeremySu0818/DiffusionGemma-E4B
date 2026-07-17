#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

if [[ "${EUID}" -eq 0 && "${DG_SKIP_APT:-0}" != "1" ]]; then
  apt-get update
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential git git-lfs curl ca-certificates tmux htop nvtop aria2 rsync \
    python3 python3-venv python3-dev
  git lfs install
else
  echo "Skipping apt packages; expecting Python, build tools, git-lfs, and CUDA drivers to be present."
fi

if ! command -v uv >/dev/null 2>&1; then
  if ! command -v curl >/dev/null 2>&1; then
    echo "uv is missing and curl is unavailable. Install uv first: https://docs.astral.sh/uv/" >&2
    exit 1
  fi
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

export PATH="$HOME/.local/bin:$PATH"
PYTHON_BIN="${DG_PYTHON:-python3}"
if [[ ! -x .venv/bin/python ]]; then
  uv venv .venv --python "$PYTHON_BIN"
fi
source .venv/bin/activate

# vLLM pins a compatible CUDA-enabled torch build. Let it establish that stack
# when this machine will host the local teacher; otherwise install PyTorch from
# the configurable CUDA wheel index.
TORCH_BACKEND="${DG_TORCH_BACKEND:-auto}"
if [[ "${DG_SKIP_LOCAL_TEACHER:-0}" != "1" ]]; then
  # Resolve vLLM and this project in one transaction so a later install cannot
  # silently replace vLLM's compiled-against torch/transformers stack.
  uv pip install --torch-backend="$TORCH_BACKEND" "vllm==0.23.0" -e '.[train,dev]'
else
  if [[ -n "${DG_TORCH_INDEX_URL:-}" ]]; then
    uv pip install --index-url "$DG_TORCH_INDEX_URL" \
      "torch==2.11.*" "torchvision==0.26.*" "torchaudio==2.11.*" -e '.[train,dev]'
  else
    uv pip install --torch-backend="$TORCH_BACKEND" \
      "torchaudio==2.11.*" -e '.[train,dev]'
  fi
fi
uv pip check

if [[ "${DG_INSTALL_FLASH_ATTN:-0}" == "1" ]]; then
  uv pip install flash-attn --no-build-isolation
fi

python - <<'PY'
import sys
import torch
import transformers

required = (3, 11)
if sys.version_info < required:
    raise SystemExit(f"Python {required[0]}.{required[1]}+ is required")
if not hasattr(transformers, "DiffusionGemmaForBlockDiffusion"):
    raise SystemExit(
        "This Transformers build lacks DiffusionGemmaForBlockDiffusion. "
        "Install the version constrained by pyproject.toml."
    )
if not hasattr(transformers, "AutoModelForMultimodalLM"):
    raise SystemExit("This Transformers build lacks AutoModelForMultimodalLM.")
print(
    {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "cuda_available": torch.cuda.is_available(),
        "bf16_supported": bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
    }
)
PY

echo "Environment ready. Run: bash scripts/linux/run_pipeline.sh"
