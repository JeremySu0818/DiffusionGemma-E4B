$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m diffusiongemma_e4b.train `
  --model-dir artifacts/transplanted `
  --data-dir data/corruption `
  --output-dir artifacts/conversion_training `
  --train-mode lora `
  --batch-size 1 `
  --gradient-accumulation-steps 32 `
  --learning-rate 2e-4 `
  --max-steps 200000 `
  --save-interval 1000 `
  --val-interval 500 `
  --self-conditioning-prob 0.5 `
  --gradient-checkpointing `
  --resume
