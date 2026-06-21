$ErrorActionPreference = "Stop"

$preset = "gpu"
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

if (Test-Path ".venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

Write-Host "Using pipeline preset: $preset" -ForegroundColor Cyan
$presetScript = python -m diffusiongemma_e4b.pipeline_presets --preset $preset --shell powershell
Invoke-Expression ($presetScript -join "`n")

& "$PSScriptRoot\generate.ps1"
& "$PSScriptRoot\corrupt.ps1"
& "$PSScriptRoot\transplant_train.ps1"
