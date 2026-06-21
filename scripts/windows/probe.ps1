$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Running local hardware and memory probe..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.local_probe `
  --base-model google/gemma-4-E4B-it `
  --output outputs/local_probe/local_feasibility_report.json `
  --allocation-gb 7.0
