$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "[setup] Missing .venv, creating with Python 3.12..."
    py -3.12 -m venv .venv
}

Write-Host "[setup] Installing required packages..."
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install keyboard pyaudio requests edge-tts

if (-not $env:SILICONFLOW_API_KEY) {
    Write-Host "[warn] SILICONFLOW_API_KEY is not set in this shell."
    Write-Host "[warn] Please set it before running:"
    Write-Host "       `$env:SILICONFLOW_API_KEY='your_key_here'"
}

Write-Host "[run] Starting VoiceBridge Step2 loop..."
.\.venv\Scripts\python .\voicebridge_step2_loop.py
