$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Running block corruption data preparation..." -ForegroundColor Cyan
$model = if ($env:DG_MODEL) { $env:DG_MODEL } else { "google/gemma-4-E4B-it" }
$targetBlocks = if ($env:DG_TARGET_BLOCKS) { $env:DG_TARGET_BLOCKS } else { "0" }
$canvasLength = if ($env:DG_CANVAS_LENGTH) { $env:DG_CANVAS_LENGTH } else { "256" }
$prefixLength = if ($env:DG_PREFIX_LENGTH) { $env:DG_PREFIX_LENGTH } else { "512" }
$shardBlocks = if ($env:DG_SHARD_BLOCKS) { $env:DG_SHARD_BLOCKS } else { "4096" }
$seed = if ($env:DG_SEED) { $env:DG_SEED } else { "1337" }
$recordOrder = if ($env:DG_RECORD_ORDER) { $env:DG_RECORD_ORDER } else { "shuffled" }
$workers = if ($env:DG_CORRUPTION_WORKERS) { $env:DG_CORRUPTION_WORKERS } else { "0" }
$prefetch = if ($env:DG_CORRUPTION_PREFETCH) { $env:DG_CORRUPTION_PREFETCH } else { "0" }
$diskReserve = if ($env:DG_MIN_FREE_DISK_BYTES) { $env:DG_MIN_FREE_DISK_BYTES } else { "2147483648" }
$nativeThreads = if ($env:DG_CORRUPTION_NATIVE_THREADS) { $env:DG_CORRUPTION_NATIVE_THREADS } else { "1" }
$env:OMP_NUM_THREADS = $nativeThreads
$env:MKL_NUM_THREADS = $nativeThreads
$env:OPENBLAS_NUM_THREADS = $nativeThreads
$env:TOKENIZERS_PARALLELISM = "false"
$teacherOutput = if ($env:DG_TEACHER_OUTPUT) { $env:DG_TEACHER_OUTPUT } else { "data/teacher_supervised/teacher_outputs.jsonl" }
$corruptionDir = if ($env:DG_CORRUPTION_DIR) { $env:DG_CORRUPTION_DIR } else { "data/corruption" }
$datasetConfig = if ($env:DG_DATASET_CONFIG) { $env:DG_DATASET_CONFIG } else { "configs/dataset_sources.json" }
python -m diffusiongemma_e4b.corruption `
  --raw-jsonl $teacherOutput `
  --output-dir $corruptionDir `
  --tokenizer $model `
  --target-blocks $targetBlocks `
  --canvas-length $canvasLength `
  --prefix-length $prefixLength `
  --shard-blocks $shardBlocks `
  --seed $seed `
  --record-order $recordOrder `
  --num-workers $workers `
  --prefetch-records $prefetch `
  --min-free-disk-bytes $diskReserve `
  --dataset-config $datasetConfig
