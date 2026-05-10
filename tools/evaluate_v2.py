from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from web_ui_app import compare_v2_engines, compare_v3_engines, load_v2_index_meta, load_v3_index_meta, v2_build_index, v3_build_index  # noqa: E402


def main() -> int:
    args = [x for x in sys.argv[1:] if x]
    dataset = "v2"
    engines: list[str] = []
    for arg in args:
        token = arg.strip().lower()
        if token in {"--v3", "v3"}:
            dataset = "v3"
            continue
        if token in {"--v2", "v2"}:
            dataset = "v2"
            continue
        engines.append(arg)
    if not engines:
        engines = ["mfcc_dtw_baseline", "engine_v3_personalized_top1"] if dataset == "v3" else ["engine_v3_personalized_top1"]

    if dataset == "v3":
        rebuild = v3_build_index()
        meta = load_v3_index_meta()
        comparison = compare_v3_engines(engines)
    else:
        rebuild = v2_build_index()
        meta = load_v2_index_meta()
        comparison = compare_v2_engines(engines)
    calibration = meta.get("calibration", {}) if isinstance(meta, dict) else {}
    baseline = comparison.get("reports_by_engine", {}).get("mfcc_dtw_baseline", {})
    compact = {
        "ok": True,
        "dataset": dataset,
        "rebuild_ok": bool(rebuild.get("ok")),
        "template_count": rebuild.get("template_count", 0),
        "report_file": comparison.get("report_file"),
        "selection_hint": comparison.get("selection_hint"),
        "failures": comparison.get("failures"),
        "engine": baseline.get("engine"),
        "train_count": baseline.get("train_count"),
        "eval_count": baseline.get("eval_count"),
        "train_leave_one_out_summary": calibration.get("summary"),
        "thresholds_by_phrase": meta.get("thresholds_by_phrase", {}),
        "reports_by_engine": {
            engine_id: report.get("summary")
            for engine_id, report in comparison.get("reports_by_engine", {}).items()
            if isinstance(report, dict)
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
