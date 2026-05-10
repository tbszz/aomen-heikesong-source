from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_ui_app import (  # noqa: E402
    compare_v2_engines,
    get_v2_engine_registry,
    load_active_v2_engine,
)


def main() -> int:
    registry = get_v2_engine_registry()
    required = {
        "engine_v3_personalized_top1",
        "mfcc_dtw_baseline",
        "mfcc_dtw_safe_v1",
        "mfcc_dtw_lenient_v1",
        "mfcc_classifier_v1",
        "ssl_proto_v1",
        "ssl_fusion_v1",
        "ssl_classifier_v1",
    }
    found = {row.get("engine_id") for row in registry}
    missing = sorted(required - found)
    if missing:
        raise AssertionError(f"missing engines: {missing}")

    comparison = compare_v2_engines(["mfcc_dtw_baseline"])
    if not comparison.get("ok"):
        raise AssertionError(f"comparison failed: {comparison}")
    if not comparison.get("report_file"):
        raise AssertionError("comparison did not write report_file")
    if "mfcc_dtw_baseline" not in comparison.get("reports_by_engine", {}):
        raise AssertionError("comparison missing baseline report")
    baseline = comparison["reports_by_engine"]["mfcc_dtw_baseline"]
    if "error_buckets" not in baseline:
        raise AssertionError("baseline report missing error_buckets")

    active = load_active_v2_engine()
    if active.get("engine_id") not in required:
        raise AssertionError(f"active engine is invalid: {active}")

    print(json.dumps({"ok": True, "report_file": comparison["report_file"], "active": active}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
