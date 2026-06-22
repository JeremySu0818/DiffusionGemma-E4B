$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Running block corruption data preparation..." -ForegroundColor Cyan
$model = if ($env:DG_MODEL) { $env:DG_MODEL } else { "google/gemma-4-E4B-it" }
$targetBlocks = if ($env:DG_TARGET_BLOCKS) { $env:DG_TARGET_BLOCKS } else { "200000" }
$canvasLength = if ($env:DG_CANVAS_LENGTH) { $env:DG_CANVAS_LENGTH } else { "256" }
$prefixLength = if ($env:DG_PREFIX_LENGTH) { $env:DG_PREFIX_LENGTH } else { "512" }
$shardBlocks = if ($env:DG_SHARD_BLOCKS) { $env:DG_SHARD_BLOCKS } else { "4096" }
$seed = if ($env:DG_SEED) { $env:DG_SEED } else { "1337" }
$recordOrder = if ($env:DG_RECORD_ORDER) { $env:DG_RECORD_ORDER } else { "shuffled" }
python -m diffusiongemma_e4b.corruption `
  --raw-jsonl data/teacher_supervised/teacher_outputs.jsonl `
  --output-dir data/corruption `
  --tokenizer $model `
  --target-blocks $targetBlocks `
  --canvas-length $canvasLength `
  --prefix-length $prefixLength `
  --shard-blocks $shardBlocks `
  --seed $seed `
  --record-order $recordOrder
