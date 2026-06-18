$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)

$model = if ($env:DG_MODEL) { $env:DG_MODEL } else { "google/gemma-4-e4b" }
lms server start | Out-Host
lms load $model | Out-Host

python -m diffusiongemma_e4b.teacher `
  --runtime lmstudio `
  --model $model `
  --base-url http://127.0.0.1:1234/v1 `
  --output data/raw_self_continuation/self_continuation.jsonl `
  --progress data/raw_self_continuation/progress.json `
  --target-estimated-tokens 50000000 `
  --max-tokens-per-sample 4096 `
  --temperature 0.95 `
  --top-p 0.98
