#!/usr/bin/env bash
set -euo pipefail

# Locate repository directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate
python -m diffusiongemma_e4b.validate \
  --model-dir artifacts/conversion_training/final \
  --data-dir data/corruption \
  --output outputs/validation/validation_report.json

python -m diffusiongemma_e4b.infer \
  --model-dir artifacts/conversion_training/final \
  --prompt "" \
  --output outputs/validation/strict_diffusion_inference.json \
  --max-new-tokens 256 \
  --denoise-steps 32 \
  --entropy-bound 0.1

python -m diffusiongemma_e4b.export \
  --output artifacts/diffusiongemma-e4b-repro-bundle.tar.gz
