$ErrorActionPreference = "Stop"

$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

$config = if ($env:DG_DATASET_CONFIG) { $env:DG_DATASET_CONFIG } else { "configs/dataset_sources.json" }
$output = if ($env:DG_PROMPT_JSONL) { $env:DG_PROMPT_JSONL } else { "data/prompt_context/prompts.jsonl" }
$maxChars = if ($env:DG_MAX_PROMPT_CHARS) { $env:DG_MAX_PROMPT_CHARS } else { "12000" }
$mediaDir = if ($env:DG_MEDIA_CACHE_DIR) { $env:DG_MEDIA_CACHE_DIR } else { "data/media_cache" }
$sources = if ($env:DG_DATASET_SOURCES) { $env:DG_DATASET_SOURCES } else { "" }
$maxPerSource = if ($env:DG_MAX_RECORDS_PER_SOURCE) { $env:DG_MAX_RECORDS_PER_SOURCE } else { "0" }
$maxTotal = if ($env:DG_MAX_TOTAL_PROMPT_RECORDS) { $env:DG_MAX_TOTAL_PROMPT_RECORDS } else { "0" }

$argsList = @(
  "-m", "diffusiongemma_e4b.data_sources",
  "--config", $config,
  "--output", $output,
  "--max-chars", $maxChars,
  "--media-dir", $mediaDir,
  "--sources", $sources,
  "--max-records-per-source", $maxPerSource,
  "--max-total-records", $maxTotal
)

Write-Host "Preparing full GPU prompt/context bank..." -ForegroundColor Cyan
python @argsList
