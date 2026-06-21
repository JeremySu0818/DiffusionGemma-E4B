$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Running validation..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.validate `
  --model-dir artifacts/conversion_training/final `
  --data-dir data/corruption `
  --output outputs/validation/validation_report.json

Write-Host "Testing strict diffusion inference..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.infer `
  --model-dir artifacts/conversion_training/final `
  --prompt "" `
  --output outputs/validation/strict_diffusion_inference.json `
  --max-new-tokens 256

Write-Host "Exporting reproduction bundle..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.export `
  --output artifacts/diffusiongemma-e4b-repro-bundle.tar.gz
