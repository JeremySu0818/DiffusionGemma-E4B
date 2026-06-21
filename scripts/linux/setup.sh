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

echo "Setting up virtual environment in $REPO_DIR..."
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -e .[train,dev]
pip install flash-attn --no-build-isolation || true
python -m diffusiongemma_e4b.config --base-model google/gemma-4-E4B-it --output-dir configs/diffusiongemma-e4b
