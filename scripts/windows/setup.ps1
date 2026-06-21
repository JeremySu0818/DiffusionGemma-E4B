$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

Write-Host "Setting up python virtual environment in $RepoRoot..." -ForegroundColor Cyan
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -e .[dev]
.\.venv\Scripts\python.exe -m pip install peft

Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
