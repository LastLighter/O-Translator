$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
if (Test-Path ".venv\Scripts\Activate.ps1") {
    & ".venv\Scripts\Activate.ps1"
}
Write-Host "Starting O-Translator API on http://127.0.0.1:8765" -ForegroundColor Cyan
python -m backend.main
