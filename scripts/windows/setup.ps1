$ErrorActionPreference = "Stop"

# Get the repo root directory
$RepoRoot = (Get-Item $PSScriptRoot).Parent.Parent.FullName
Set-Location $RepoRoot

# Automatically install uv if not present
if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv is not installed. Installing uv..." -ForegroundColor Yellow
    try {
        Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
        $env:Path += ";$env:USERPROFILE\.local\bin"
    } catch {
        Write-Host "Failed to install uv via installer, trying pip..." -ForegroundColor Yellow
        python -m pip install --user uv
    }
}

# Automatically install jq if winget is present and jq is not installed
if (!(Get-Command jq -ErrorAction SilentlyContinue) -and (Get-Command winget -ErrorAction SilentlyContinue)) {
    try {
        Write-Host "Installing jq via winget..." -ForegroundColor Yellow
        winget install --id jqlang.jq --exact --silent --accept-source-agreements --accept-package-agreements | Out-Null
    } catch {
        Write-Host "Optional jq installation skipped." -ForegroundColor DarkGray
    }
}

Write-Host "Setting up python virtual environment in $RepoRoot..." -ForegroundColor Cyan
uv venv .venv
. .\.venv\Scripts\Activate.ps1

uv pip install -e .[train,dev]
uv pip install peft

Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
