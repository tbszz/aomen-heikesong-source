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

    report_resp = client.post("/api/v2/eval/run", json={"engines": ["engine_v3_personalized_top1"]})
    if report_resp.status_code != 200:
        raise AssertionError(f"/api/v2/eval/run failed: {report_resp.text}")
    payload = report_resp.json()
    if not payload.get("ok"):
        raise AssertionError(f"/api/v2/eval/run not ok: {payload}")

    reports = payload.get("reports_by_engine", {})
    safe_report = reports.get("engine_v3_personalized_top1")
    if not isinstance(safe_report, dict):
        raise AssertionError("missing v3 engine report")

    summary = safe_report.get("summary", {}) if isinstance(safe_report.get("summary"), dict) else {}
    if float(summary.get("reject_rate", -1.0)) != 0.0:
        raise AssertionError(f"reject_rate should be 0.0 in no-reject mode, got: {summary.get('reject_rate')}")

    reasons = summary.get("failure_reasons", {})
    if reasons not in ({}, {"none": summary.get("total", 0)}):
        if isinstance(reasons, dict):
            bad = [k for k, v in reasons.items() if k != "none" and int(v or 0) > 0]
            if bad:
                raise AssertionError(f"unexpected reject reasons in no-reject mode: {bad}")

    print(
        json.dumps(
            {
                "ok": True,
                "engine": "engine_v3_personalized_top1",
                "total": summary.get("total"),
                "top1_rate": summary.get("top1_rate"),
                "top2_rate": summary.get("top2_rate"),
                "reject_rate": summary.get("reject_rate"),
                "failure_reasons": reasons,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
