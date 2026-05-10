$ErrorActionPreference = "Stop"
Set-Location -LiteralPath $PSScriptRoot

if (-not (Test-Path -LiteralPath ".\.venv\Scripts\python.exe")) {
    Write-Host "[setup] Missing .venv, creating environment..."
    py -3.12 -m venv .venv
}

Write-Host "[setup] Installing Python dependencies..."
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install fastapi uvicorn python-multipart requests edge-tts numpy scipy librosa soundfile faiss-cpu torch transformers

if (-not (Test-Path -LiteralPath ".\node_modules")) {
    Write-Host "[setup] Installing Node dependencies..."
    npm install
}

Write-Host "[build] Building TypeScript frontend..."
npm run build:frontend

Write-Host "[run] Opening browser at http://127.0.0.1:8765"
Start-Process "http://127.0.0.1:8765"

Write-Host "[run] Starting backend..."
.\.venv\Scripts\python -m uvicorn web_ui_app:app --host 127.0.0.1 --port 8765
