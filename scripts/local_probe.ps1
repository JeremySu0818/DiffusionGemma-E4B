$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m diffusiongemma_e4b.local_probe `
  --base-model google/gemma-4-E4B-it `
  --output outputs/local_probe/local_feasibility_report.json `
  --allocation-gb 7.0
