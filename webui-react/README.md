# VoiceBridge Browser MVP (TF.js Speech Commands)

Browser-only MVP for phrase recognition using transfer learning.

## Run

```powershell
bun install
bun run dev
```

Open `http://127.0.0.1:5173`.

## Flow

1. Record phrase samples + background noise
2. Train transfer recognizer
3. Run real-time recognition with threshold rejection and browser TTS

Trained model is saved in IndexedDB and reused on next load.
