$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Running block corruption data preparation..." -ForegroundColor Cyan
python -m diffusiongemma_e4b.corruption `
  --raw-jsonl data/raw_self_continuation/self_continuation.jsonl `
  --output-dir data/corruption `
  --tokenizer google/gemma-4-E4B-it `
  --target-blocks 200000 `
  --canvas-length 256 `
  --prefix-length 512 `
  --shard-blocks 4096
