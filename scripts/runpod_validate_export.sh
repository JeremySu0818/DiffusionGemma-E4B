#!/usr/bin/env bash
set -euo pipefail
cd /workspace/DiffusionGemma-E4B
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
