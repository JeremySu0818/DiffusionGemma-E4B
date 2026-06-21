$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

$model = if ($env:DG_MODEL) { $env:DG_MODEL } else { "google/gemma-4-e4b" }
Write-Host "Starting LM-Studio server and loading model: $model..." -ForegroundColor Cyan
lms server start | Out-Host
lms load $model | Out-Host

Write-Host "Running teacher generation pipeline..." -ForegroundColor Cyan
$sourceConfig = if ($env:DG_DATASET_CONFIG) { $env:DG_DATASET_CONFIG } else { "configs/dataset_sources.json" }
$mediaDir = if ($env:DG_MEDIA_CACHE_DIR) { $env:DG_MEDIA_CACHE_DIR } else { "data/media_cache" }
$maxPromptChars = if ($env:DG_MAX_PROMPT_CHARS) { $env:DG_MAX_PROMPT_CHARS } else { "12000" }
$sources = if ($env:DG_DATASET_SOURCES) { $env:DG_DATASET_SOURCES } else { "" }
$targetTokens = if ($env:DG_TARGET_ESTIMATED_TOKENS) { $env:DG_TARGET_ESTIMATED_TOKENS } else { "50000000" }
$maxTokens = if ($env:DG_MAX_TOKENS_PER_SAMPLE) { $env:DG_MAX_TOKENS_PER_SAMPLE } else { "4096" }
$temperature = if ($env:DG_TEACHER_TEMPERATURE) { $env:DG_TEACHER_TEMPERATURE } else { "0.95" }
$topP = if ($env:DG_TEACHER_TOP_P) { $env:DG_TEACHER_TOP_P } else { "0.98" }
python -m diffusiongemma_e4b.teacher `
  --runtime lmstudio `
  --model $model `
  --base-url http://127.0.0.1:1234/v1 `
  --source-config $sourceConfig `
  --media-dir $mediaDir `
  --max-prompt-chars $maxPromptChars `
  --sources $sources `
  --output data/teacher_supervised/teacher_outputs.jsonl `
  --progress data/teacher_supervised/progress.json `
  --target-estimated-tokens $targetTokens `
  --max-tokens-per-sample $maxTokens `
  --temperature $temperature `
  --top-p $topP
