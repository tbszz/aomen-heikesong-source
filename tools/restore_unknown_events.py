from __future__ import annotations

import hashlib
import json
import sys
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_ui_app import (  # noqa: E402
    V2_UNKNOWN_DIR,
    load_active_v2_engine,
    load_v2_manifest,
    now_iso,
    save_v2_manifest,
    v2_match_audio,
    v2_unknown_event,
)


def duration_sec(path: Path) -> float:
    with wave.open(str(path), "rb") as wf:
        rate = wf.getframerate()
        return round(wf.getnframes() / float(rate), 4) if rate else 0.0


def row_for_unknown_wav(path: Path) -> dict:
    data = path.read_bytes()
    return {
        "sample_id": path.stem.split("_")[-1],
        "split": "unknown",
        "source": "unknown_capture_restored",
        "file_rel": path.relative_to(ROOT).as_posix(),
        "duration_sec": duration_sec(path),
        "quality_flags": [],
        "sha256": hashlib.sha256(data).hexdigest(),
        "created_at": now_iso(),
    }


def main() -> int:
    manifest = load_v2_manifest()
    existing_rels = {
        row.get("file_rel")
        for row in manifest.get("unknown_events", [])
        if isinstance(row, dict) and isinstance(row.get("file_rel"), str)
    }
    restored = []
    engine = str(load_active_v2_engine().get("engine_id") or "mfcc_dtw_safe_v1")
    for path in sorted(V2_UNKNOWN_DIR.glob("*.wav")):
        row = row_for_unknown_wav(path)
        if row["file_rel"] in existing_rels:
            continue
        matched, debug = v2_match_audio(path, engine=engine)
        restored.append(v2_unknown_event(row, matched, debug))
    if restored:
        manifest.setdefault("unknown_events", []).extend(restored)
        save_v2_manifest(manifest)
    print(json.dumps({"ok": True, "restored": len(restored), "engine": engine}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
