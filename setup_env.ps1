$ErrorActionPreference = "Stop"

Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv")) {
    Write-Host "[setup] Creating .venv with Python 3.12..."
    py -3.12 -m venv .venv
}

Write-Host "[setup] Upgrading pip..."
.\.venv\Scripts\python -m pip install --upgrade pip

Write-Host "[setup] Installing dependencies..."
.\.venv\Scripts\python -m pip install keyboard pyaudio

Write-Host "[setup] Done."
