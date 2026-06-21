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
        $user_site = python -c "import site; print(site.USER_BASE)"
        $env:Path += ";$user_site\Scripts"
    }
}

Write-Host "Setting up python virtual environment in $RepoRoot..." -ForegroundColor Cyan
uv venv .venv
. .\.venv\Scripts\Activate.ps1

uv pip install -e .[train,dev]
uv pip install peft

Write-Host "Bootstrap complete. Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
