from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402
from web_ui_app import app  # noqa: E402


def main() -> int:
    client = TestClient(app)

    engines = client.get("/api/v2/engines")
    if engines.status_code != 200:
        raise AssertionError(f"/api/v2/engines failed: {engines.text}")
    payload = engines.json()
    active_engine = payload.get("active_engine", {}).get("engine_id")
    if active_engine not in {"engine_v3_personalized_top1", "mfcc_dtw_safe_v1"}:
        raise AssertionError(f"unexpected active engine: {payload}")

    unknown_export = client.get("/api/v2/unknown/export")
    if unknown_export.status_code != 200:
        raise AssertionError(f"/api/v2/unknown/export failed: {unknown_export.text}")
    unknown_payload = unknown_export.json()
    if "summary" not in unknown_payload:
        raise AssertionError("unknown export missing summary")
    if unknown_payload["summary"].get("false_accept_rate") is None:
        raise AssertionError("unknown summary missing false_accept_rate")

    print(json.dumps({
        "ok": True,
        "active_engine": payload.get("active_engine"),
        "unknown_export_summary": unknown_payload.get("summary"),
        "note": "smoke is read-only; it does not reset user negative-test records",
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
