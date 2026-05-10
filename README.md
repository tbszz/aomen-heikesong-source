# VoiceBridge (Open Source)

VoiceBridge is a phrase-level speech bridge for dysarthric / throat-mic input.
It captures short audio, runs phrase matching, and returns a normalized phrase with optional TTS.

## Scope of this public repo
This repository publishes **source code only**.
It intentionally excludes:

- personal/local datasets
- recorded audio samples
- model artifacts and index caches
- runtime logs and OMX local state

## Tech stack

- Backend: Python (FastAPI style app in `web_ui_app.py`)
- Frontend: TypeScript (`webui/src/main.ts`) + static page (`webui/static/index.html`)
- Utility scripts: PowerShell + Python scripts in root and `tools/`

## Quick start (Windows PowerShell)

1. Create and activate venv

```powershell
./setup_env.ps1
```

2. Optional: set cloud ASR key (fallback path only)

```powershell
$env:SILICONFLOW_API_KEY="your_key_here"
```

3. Start web UI

```powershell
./launch_web_ui.ps1
```

4. If frontend changes were made, rebuild static JS

```powershell
npm install
npm run build:frontend
```

## Privacy and data policy

- Do not commit user audio, training data, logs, or runtime state.
- Keep secrets in environment variables only.
- `.gitignore` is configured for safe public-source publishing by default.

## License

MIT (see `LICENSE`).
