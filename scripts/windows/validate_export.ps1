$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Running validation..." -ForegroundColor Cyan
$trainOutputDir = if ($env:DG_TRAIN_OUTPUT_DIR) { $env:DG_TRAIN_OUTPUT_DIR } else { "artifacts/conversion_training" }
$finalModelDir = Join-Path $trainOutputDir "final"
$corruptionDir = if ($env:DG_CORRUPTION_DIR) { $env:DG_CORRUPTION_DIR } else { "data/corruption" }
$validationDir = if ($env:DG_VALIDATION_DIR) { $env:DG_VALIDATION_DIR } else { "outputs/validation" }
$transplantDir = if ($env:DG_TRANSPLANT_DIR) { $env:DG_TRANSPLANT_DIR } else { "artifacts/transplanted" }
$exportOutput = if ($env:DG_EXPORT_OUTPUT) { $env:DG_EXPORT_OUTPUT } else { "artifacts/diffusiongemma-e4b-repro-bundle.tar.gz" }
python -m diffusiongemma_e4b.validate `
  --model-dir $finalModelDir `
  --base-model $transplantDir `
  --data-dir $corruptionDir `
  --output (Join-Path $validationDir "validation_report.json")

Write-Host "Testing strict diffusion inference..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.infer `
  --model-dir $finalModelDir `
  --base-model $transplantDir `
  --prompt "Explain why the sky appears blue in three concise sentences." `
  --output (Join-Path $validationDir "strict_diffusion_inference.json") `
  --max-new-tokens 256

Write-Host "Exporting reproduction bundle..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.export `
  --output $exportOutput `
  --model-dir $finalModelDir `
  --validation-dir $validationDir `
  --base-model-dir $transplantDir
