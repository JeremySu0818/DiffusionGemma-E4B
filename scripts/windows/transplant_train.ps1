$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Stage 1: Weight Transplantation..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.student `
  --base-model google/gemma-4-E4B-it `
  --output-dir artifacts/transplanted `
  --canvas-length 256 `
  --dtype bfloat16 `
  --device-map auto

Write-Host "Stage 2: Model Training (LoRA)..." -ForegroundColor Cyan
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
