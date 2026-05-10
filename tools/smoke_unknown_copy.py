from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INDEX_HTML = ROOT / "webui" / "static" / "index.html"
MAIN_TS = ROOT / "webui" / "src" / "main.ts"


def require(text: str, needle: str, source: Path) -> None:
    if needle not in text:
        raise AssertionError(f"{source.relative_to(ROOT)} missing required copy: {needle}")


def forbid(text: str, needle: str, source: Path) -> None:
    if needle in text:
        raise AssertionError(f"{source.relative_to(ROOT)} still contains misleading copy: {needle}")


def main() -> int:
    html = INDEX_HTML.read_text(encoding="utf-8")
    ts = MAIN_TS.read_text(encoding="utf-8")

    require(html, "负样本测试（不要说四句）", INDEX_HTML)
    require(html, "如果你说“你好 / 我想喝水 / 我想吃饭 / 请帮我”，这条不算负样本", INDEX_HTML)
    require(html, "v22_negative_copy_20260502", INDEX_HTML)
    require(ts, "负样本测试中：只录四句之外的声音", MAIN_TS)
    require(ts, "已拒识：这是负样本模式", MAIN_TS)
    require(ts, "第一候选只供排错", MAIN_TS)

    forbid(html, "未知/乱说测试", INDEX_HTML)
    forbid(ts, "未知/乱说测试", MAIN_TS)

    print("unknown negative-test copy smoke passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
