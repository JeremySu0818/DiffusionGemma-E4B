$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
python -m diffusiongemma_e4b.student `
  --base-model google/gemma-4-E4B-it `
  --output-dir artifacts/transplanted `
  --canvas-length 256 `
  --dtype bfloat16 `
  --device-map auto
