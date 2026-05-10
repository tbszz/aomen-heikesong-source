$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "[run] Missing .venv, preparing environment first..."
    & "$PSScriptRoot\setup_env.ps1"
}

Write-Host "[run] Launching recorder. Hold SPACE to record, release to save temp.wav."
.\.venv\Scripts\python .\physical_trigger_recorder.py
