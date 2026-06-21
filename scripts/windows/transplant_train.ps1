$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Stage 1: Weight Transplantation..." -ForegroundColor Cyan
$model = if ($env:DG_MODEL) { $env:DG_MODEL } else { "google/gemma-4-E4B-it" }
$transplantDir = if ($env:DG_TRANSPLANT_DIR) { $env:DG_TRANSPLANT_DIR } else { "artifacts/transplanted" }
$canvasLength = if ($env:DG_CANVAS_LENGTH) { $env:DG_CANVAS_LENGTH } else { "256" }
$dtype = if ($env:DG_DTYPE) { $env:DG_DTYPE } else { "bfloat16" }
$deviceMap = if ($env:DG_DEVICE_MAP) { $env:DG_DEVICE_MAP } else { "auto" }
python -m diffusiongemma_e4b.student `
  --base-model $model `
  --output-dir $transplantDir `
  --canvas-length $canvasLength `
  --dtype $dtype `
  --device-map $deviceMap

Write-Host "Stage 2: Model Training (LoRA)..." -ForegroundColor Cyan
$dataDir = if ($env:DG_CORRUPTION_DIR) { $env:DG_CORRUPTION_DIR } else { "data/corruption" }
$trainOutputDir = if ($env:DG_TRAIN_OUTPUT_DIR) { $env:DG_TRAIN_OUTPUT_DIR } else { "artifacts/conversion_training" }
$trainMode = if ($env:DG_TRAIN_MODE) { $env:DG_TRAIN_MODE } else { "lora" }
$batchSize = if ($env:DG_BATCH_SIZE) { $env:DG_BATCH_SIZE } else { "1" }
$gradAccum = if ($env:DG_GRAD_ACCUM) { $env:DG_GRAD_ACCUM } else { "32" }
$lr = if ($env:DG_LR) { $env:DG_LR } else { "2e-4" }
$maxSteps = if ($env:DG_MAX_STEPS) { $env:DG_MAX_STEPS } else { "200000" }
$saveInterval = if ($env:DG_SAVE_INTERVAL) { $env:DG_SAVE_INTERVAL } else { "1000" }
$valInterval = if ($env:DG_VAL_INTERVAL) { $env:DG_VAL_INTERVAL } else { "500" }
$selfConditioningProb = if ($env:DG_SELF_CONDITIONING_PROB) { $env:DG_SELF_CONDITIONING_PROB } else { "0.5" }
python -m diffusiongemma_e4b.train `
  --model-dir $transplantDir `
  --data-dir $dataDir `
  --output-dir $trainOutputDir `
  --train-mode $trainMode `
  --batch-size $batchSize `
  --gradient-accumulation-steps $gradAccum `
  --learning-rate $lr `
  --max-steps $maxSteps `
  --save-interval $saveInterval `
  --val-interval $valInterval `
  --self-conditioning-prob $selfConditioningProb `
  --gradient-checkpointing `
  --resume
