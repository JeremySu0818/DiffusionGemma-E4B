$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

$runtime = if ($env:DG_TEACHER_RUNTIME) { $env:DG_TEACHER_RUNTIME } else { "openai-compatible" }
$model = if ($env:DG_TEACHER_SERVED_MODEL_NAME) { $env:DG_TEACHER_SERVED_MODEL_NAME } elseif ($env:DG_MODEL) { $env:DG_MODEL } else { "google/gemma-4-E4B-it" }
$baseUrl = if ($env:DG_TEACHER_BASE_URL) { $env:DG_TEACHER_BASE_URL } else { "http://127.0.0.1:1234/v1" }

if (-not $env:DG_SKIP_LOCAL_TEACHER -and ($baseUrl -like "*127.0.0.1*" -or $baseUrl -like "*localhost*")) {
    if (Get-Command lms -ErrorAction SilentlyContinue) {
        Write-Host "Starting LM-Studio server and loading model: $model..." -ForegroundColor Cyan
        lms server start | Out-Host
        lms load $model | Out-Host
    }
}

Write-Host "Running teacher generation pipeline..." -ForegroundColor Cyan
$sourceConfig = if ($env:DG_DATASET_CONFIG) { $env:DG_DATASET_CONFIG } else { "configs/dataset_sources.json" }
$mediaDir = if ($env:DG_MEDIA_CACHE_DIR) { $env:DG_MEDIA_CACHE_DIR } else { "data/media_cache" }
$maxPromptChars = if ($env:DG_MAX_PROMPT_CHARS) { $env:DG_MAX_PROMPT_CHARS } else { "11000" }
$sources = if ($env:DG_DATASET_SOURCES) { $env:DG_DATASET_SOURCES } else { "" }
$maxPerSource = if ($env:DG_MAX_RECORDS_PER_SOURCE) { $env:DG_MAX_RECORDS_PER_SOURCE } else { "0" }
$maxTotal = if ($env:DG_MAX_TOTAL_PROMPT_RECORDS) { $env:DG_MAX_TOTAL_PROMPT_RECORDS } else { "0" }
$targetTokens = if ($env:DG_TARGET_ESTIMATED_TOKENS) { $env:DG_TARGET_ESTIMATED_TOKENS } else { "0" }
$temperature = if ($env:DG_TEACHER_TEMPERATURE) { $env:DG_TEACHER_TEMPERATURE } else { "0.2" }
$topP = if ($env:DG_TEACHER_TOP_P) { $env:DG_TEACHER_TOP_P } else { "0.95" }
$timeoutS = if ($env:DG_TEACHER_TIMEOUT_S) { $env:DG_TEACHER_TIMEOUT_S } else { "900" }
$maxRetries = if ($env:DG_TEACHER_MAX_RETRIES) { $env:DG_TEACHER_MAX_RETRIES } else { "5" }
$retryBaseS = if ($env:DG_TEACHER_RETRY_BASE_S) { $env:DG_TEACHER_RETRY_BASE_S } else { "2" }
$minEstimatedTokens = if ($env:DG_MIN_TEACHER_ESTIMATED_TOKENS) { $env:DG_MIN_TEACHER_ESTIMATED_TOKENS } else { "8" }
$tokenizer = if ($env:DG_STUDENT_MODEL) { $env:DG_STUDENT_MODEL } else { "artifacts/tokenizer_processor_gemma4_e4b" }
$maxConsecutiveFailures = if ($env:DG_TEACHER_MAX_CONSECUTIVE_FAILURES) { $env:DG_TEACHER_MAX_CONSECUTIVE_FAILURES } else { "20" }
$teacherOutput = if ($env:DG_TEACHER_OUTPUT) { $env:DG_TEACHER_OUTPUT } else { "data/teacher_supervised/teacher_outputs.jsonl" }
$teacherProgress = if ($env:DG_TEACHER_PROGRESS) { $env:DG_TEACHER_PROGRESS } else { "data/teacher_supervised/progress.json" }
$concurrency = if ($env:DG_TEACHER_CONCURRENCY) { $env:DG_TEACHER_CONCURRENCY } else { "10" }
$prefetchRecords = if ($env:DG_TEACHER_PREFETCH_RECORDS) { $env:DG_TEACHER_PREFETCH_RECORDS } else { "16384" }
$prefetchDir = if ($env:DG_TEACHER_PREFETCH_DIR) { $env:DG_TEACHER_PREFETCH_DIR } else { "data/teacher_supervised/prompt_spool" }
$prefixLength = if ($env:DG_PREFIX_LENGTH) { $env:DG_PREFIX_LENGTH } else { "2048" }
$resumeBaseUrl = if ($env:DG_TEACHER_RESUME_BASE_URL) { $env:DG_TEACHER_RESUME_BASE_URL } elseif ($env:DG_TEACHER_ORIGINAL_BASE_URL) { $env:DG_TEACHER_ORIGINAL_BASE_URL } else { "" }
$allowResumeFingerprint = if ($env:DG_ALLOW_RESUME_FINGERPRINT) { $env:DG_ALLOW_RESUME_FINGERPRINT } else { "" }

python -m diffusiongemma_e4b.teacher `
  --runtime $runtime `
  --model $model `
  --base-url $baseUrl `
  --source-config $sourceConfig `
  --media-dir $mediaDir `
  --max-prompt-chars $maxPromptChars `
  --sources $sources `
  --max-records-per-source $maxPerSource `
  --max-total-records $maxTotal `
  --output $teacherOutput `
  --progress $teacherProgress `
  --target-estimated-tokens $targetTokens `
  --temperature $temperature `
  --top-p $topP `
  --timeout-s $timeoutS `
  --max-retries $maxRetries `
  --retry-base-s $retryBaseS `
  --min-estimated-tokens $minEstimatedTokens `
  --tokenizer $tokenizer `
  --max-consecutive-failures $maxConsecutiveFailures `
  --concurrency $concurrency `
  --prefetch-records $prefetchRecords `
  --prefetch-dir $prefetchDir `
  --student-prefix-length $prefixLength `
  --resume-base-url $resumeBaseUrl `
  --allow-resume-fingerprint $allowResumeFingerprint
