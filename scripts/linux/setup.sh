#!/usr/bin/env bash
set -euo pipefail

# Make sure apt-get commands are run as root if needed
if [ "$EUID" -ne 0 ]; then
  echo "Please run as root (using sudo) to install system packages, or ensure you have the required packages pre-installed."
else
  apt-get update
  apt-get install -y git git-lfs curl wget tmux htop nvtop aria2 rsync python3.11 python3.11-venv python3.11-dev build-essential
  git lfs install
fi

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

# Automatically install uv if not present
if ! command -v uv &> /dev/null; then
  echo "uv is not installed. Installing uv..."
  if command -v curl &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  elif command -v wget &> /dev/null; then
    wget -qO- https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
  else
    python3.11 -m pip install --user uv || python3 -m pip install --user uv || pip install --user uv
    export PATH="$HOME/.local/bin:$PATH"
  fi
fi

# Ensure uv is in PATH
export PATH="$HOME/.local/bin:$PATH"

echo "Setting up virtual environment in $REPO_DIR..."
uv venv .venv --python python3.11
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
uv pip install -e .[train,dev]
uv pip install vllm || true
uv pip install flash-attn --no-build-isolation || true
python -m diffusiongemma_e4b.config --base-model google/gemma-4-E4B-it --output-dir configs/diffusiongemma-e4b
