$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m diffusiongemma_e4b.validate `
  --model-dir artifacts/conversion_training/final `
  --data-dir data/corruption `
  --output outputs/validation/validation_report.json
python -m diffusiongemma_e4b.infer `
  --model-dir artifacts/conversion_training/final `
  --prompt "" `
  --output outputs/validation/strict_diffusion_inference.json `
  --max-new-tokens 256
python -m diffusiongemma_e4b.export `
  --output artifacts/diffusiongemma-e4b-repro-bundle.tar.gz
