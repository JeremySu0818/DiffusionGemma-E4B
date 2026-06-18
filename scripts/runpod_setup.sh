#!/usr/bin/env bash
set -euo pipefail
apt-get update
apt-get install -y git git-lfs curl wget tmux htop nvtop aria2 rsync python3.11 python3.11-venv python3.11-dev build-essential
git lfs install
cd /workspace
if [ ! -d DiffusionGemma-E4B ]; then
  git clone "$DG_REPO_URL" DiffusionGemma-E4B
fi
cd /workspace/DiffusionGemma-E4B
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip wheel setuptools
pip install --index-url https://download.pytorch.org/whl/cu128 torch torchvision torchaudio
pip install -e .[train,dev]
pip install flash-attn --no-build-isolation || true
python -m diffusiongemma_e4b.config --base-model google/gemma-4-E4B-it --output-dir configs/diffusiongemma-e4b
