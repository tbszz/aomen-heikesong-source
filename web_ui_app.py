# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import argparse
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import wave
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set
from uuid import uuid4

import edge_tts
import librosa
import numpy as np
import requests
from fastapi import Body, FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    import faiss  # type: ignore[import-not-found]
except Exception:  # noqa: BLE001
    faiss = None

try:
    import torch
    from transformers import AutoFeatureExtractor, AutoModel, HubertModel
except Exception:  # noqa: BLE001
    torch = None
    AutoFeatureExtractor = None
    AutoModel = None
    HubertModel = None


ROOT_DIR = Path(__file__).resolve().parent
WEB_DIR = ROOT_DIR / "webui"
STATIC_DIR = WEB_DIR / "static"
AUDIO_DIR = WEB_DIR / "audio"
DATA_DIR = ROOT_DIR / "data"
DATA_V2_DIR = ROOT_DIR / "data_v2"
DATA_V3_DIR = ROOT_DIR / "data_v3"
BATCH_DIR = DATA_DIR / "batch_eval"
PHRASE_DATA_DIR = DATA_DIR / "phrases"
INDEX_DIR = DATA_DIR / "index"
MANIFEST_PATH = DATA_DIR / "manifest.json"
V2_MANIFEST_PATH = DATA_V2_DIR / "manifest.json"
V2_INDEX_DIR = DATA_V2_DIR / "index"
V2_INDEX_META_PATH = V2_INDEX_DIR / "meta.json"
V2_REPORTS_DIR = DATA_V2_DIR / "reports"
V2_AUDIT_REPORT_PATH = DATA_V2_DIR / "audit_report.json"
V2_MODELS_DIR = DATA_V2_DIR / "models"
V2_ACTIVE_ENGINE_PATH = V2_MODELS_DIR / "active_engine.json"
V2_FEATURES_DIR = DATA_V2_DIR / "features"
V2_UNKNOWN_DIR = DATA_V2_DIR / "unknown"
V3_MANIFEST_PATH = DATA_V3_DIR / "manifest.json"
V3_INDEX_DIR = DATA_V3_DIR / "index"
V3_INDEX_META_PATH = V3_INDEX_DIR / "meta.json"
V3_REPORTS_DIR = DATA_V3_DIR / "reports"
V3_AUDIT_REPORT_PATH = DATA_V3_DIR / "audit_report_v3.json"
V3_MODELS_DIR = DATA_V3_DIR / "models"
V3_ACTIVE_ENGINE_PATH = V3_MODELS_DIR / "active_engine.json"
V3_FEATURES_DIR = DATA_V3_DIR / "features"
V3_UNKNOWN_DIR = DATA_V3_DIR / "unknown"
SIGNATURE_DB_PATH = ROOT_DIR / "signature_db.json"
BATCH_EVENTS_PATH = BATCH_DIR / "events.jsonl"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
INDEX_META_PATH = INDEX_DIR / "meta.json"
MIN_GAP_PROFILE_PATH = INDEX_DIR / "min_gap_profile.json"
PROTOTYPE_CACHE_PATH = INDEX_DIR / "prototype_cache.npz"

SILICONFLOW_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
ASR_PRIMARY_MODEL = "FunAudioLLM/SenseVoiceSmall"
ASR_FALLBACK_MODEL = "TeleAI/TeleSpeechASR"
TTS_VOICE = "zh-CN-YunxiNeural"

TARGET_SAMPLE_RATE = 16000
MIN_SAMPLE_DURATION_SEC = 0.18
MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE = 0.8
TRAIN_TARGET_PER_PHRASE = 15
EVAL_TARGET_PER_PHRASE = 15
TRIM_TOP_DB = 35
EMBED_N_MFCC = 13
INDEX_TOP_K = 24
MAX_CANDIDATE_PHRASES = 3
DTW_TOPK_PER_PHRASE = 3
DEFAULT_DISTANCE_THRESHOLD = 0.02
DEFAULT_MIN_GAP = 0.0012
DEFAULT_RATIO_MIN = 1.08
PROTOTYPE_TOP_K = 2
PROTO_DTW_TOPK_PER_PHRASE = 4
FUSION_WEIGHT_PROTO = 0.65
FUSION_WEIGHT_DTW = 0.35
HUBERT_MODEL_NAME = "facebook/hubert-base-ls960"
WAVLM_MODEL_NAME = "microsoft/wavlm-base-plus"
SSL_PROTO_ENGINE_ID = "ssl_proto_v1"
SSL_FUSION_ENGINE_ID = "ssl_fusion_v1"
SSL_CLASSIFIER_ENGINE_ID = "ssl_classifier_v1"
MFCC_CLASSIFIER_ENGINE_ID = "mfcc_classifier_v1"
MFCC_LENIENT_ENGINE_ID = "mfcc_dtw_lenient_v1"
MFCC_SAFE_ENGINE_ID = "mfcc_dtw_safe_v1"
ENGINE_V3_PERSONALIZED_TOP1 = "engine_v3_personalized_top1"
V2_ENGINE_IDS = [
    ENGINE_V3_PERSONALIZED_TOP1,
    "mfcc_dtw_baseline",
    MFCC_SAFE_ENGINE_ID,
    MFCC_LENIENT_ENGINE_ID,
    MFCC_CLASSIFIER_ENGINE_ID,
    SSL_PROTO_ENGINE_ID,
    SSL_FUSION_ENGINE_ID,
    SSL_CLASSIFIER_ENGINE_ID,
]
V2_DEFAULT_COMPARE_ENGINES = [ENGINE_V3_PERSONALIZED_TOP1, "mfcc_dtw_baseline", MFCC_SAFE_ENGINE_ID, MFCC_CLASSIFIER_ENGINE_ID]
MIN_GAP_MULTIPLIER_MIN = 0.4
MIN_GAP_MULTIPLIER_MAX = 1.6

ALNUM_CJK_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")

PHRASE_PACK: List[Dict[str, str]] = [
    {"id": "p01_ni_hao", "text": "你好"},
    {"id": "p02_he_shui", "text": "我想喝水"},
    {"id": "p03_chi_fan", "text": "我想吃饭"},
    {"id": "p04_qing_bang_wo", "text": "请帮我"},
    {"id": "p05_bu_shu_fu", "text": "我不舒服"},
    {"id": "p06_shang_ce_suo", "text": "我想上厕所"},
    {"id": "p07_man_yi_dian", "text": "请慢一点"},
    {"id": "p08_xie_xie_ni", "text": "谢谢你"},
]
PHRASE_ID_TO_TEXT = {item["id"]: item["text"] for item in PHRASE_PACK}
PHRASE_TEXT_TO_ID = {item["text"]: item["id"] for item in PHRASE_PACK}
ACTIVE_PHRASE_IDS_DEFAULT = [item["id"] for item in PHRASE_PACK]


def env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw.strip())
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw.strip())
    except Exception:
        return default


# Runtime switch: default to reject-enabled (no force-top1) unless explicitly overridden.
NO_REJECT_MODE = env_bool("VB_NO_REJECT_MODE", False)
VB_ENABLE_CLOUD_FALLBACK = env_bool("VB_ENABLE_CLOUD_FALLBACK", True)
VB_FALLBACK_POLICY = (os.environ.get("VB_FALLBACK_POLICY", "low_conf_only") or "low_conf_only").strip().lower()
VB_REBUILD_POLICY = (os.environ.get("VB_REBUILD_POLICY", "deferred") or "deferred").strip().lower()
VB_BATCH_REBUILD_EVERY = max(1, env_int("VB_BATCH_REBUILD_EVERY", 10))
VB_FALLBACK_MARGIN_THRESHOLD = env_float("VB_FALLBACK_MARGIN_THRESHOLD", 0.0025)
VB_FALLBACK_SCORE_THRESHOLD = env_float("VB_FALLBACK_SCORE_THRESHOLD", 0.04)
VB_MIN_ACTIVE_PER_PHRASE = max(1, env_int("VB_MIN_ACTIVE_PER_PHRASE", 8))
VB_PURIFY_DISABLE_THRESHOLD = min(1.0, max(0.0, env_float("VB_PURIFY_DISABLE_THRESHOLD", 0.75)))
VB_PURIFY_RECENT_ERROR_WINDOW = max(20, env_int("VB_PURIFY_RECENT_ERROR_WINDOW", 240))
VB_PURIFY_PER_PHRASE_MAX_DISABLE = max(1, env_int("VB_PURIFY_PER_PHRASE_MAX_DISABLE", 6))


def detect_torch_backend() -> str:
    if torch is None:
        return "none"
    try:
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if hasattr(torch, "xpu"):
        try:
            if torch.xpu.is_available():
                return "xpu"
        except Exception:
            pass
    return "cpu"


SSL_BACKEND = detect_torch_backend()
HUBERT_DEVICE = "cuda" if SSL_BACKEND == "cuda" else ("xpu" if SSL_BACKEND == "xpu" else "cpu")
V3_SSL_ENABLED = bool(torch is not None and AutoFeatureExtractor is not None and AutoModel is not None)

_HUBERT_FEATURE_EXTRACTOR = None
_HUBERT_MODEL = None
_SSL_FEATURE_EXTRACTOR = None
_SSL_MODEL = None
_SSL_MODEL_NAME = None
_SSL_MODEL_ERROR: Optional[str] = None


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    return "".join(ALNUM_CJK_PATTERN.findall(raw_text))


def normalize_rebuild_policy(policy: Optional[str]) -> str:
    p = (policy or "").strip().lower()
    if p in {"immediate", "deferred"}:
        return p
    return VB_REBUILD_POLICY if VB_REBUILD_POLICY in {"immediate", "deferred"} else "deferred"


def runtime_config_snapshot() -> Dict[str, Any]:
    return {
        "enable_cloud_fallback": bool(VB_ENABLE_CLOUD_FALLBACK),
        "fallback_policy": VB_FALLBACK_POLICY,
        "rebuild_policy_default": VB_REBUILD_POLICY,
        "batch_rebuild_every": VB_BATCH_REBUILD_EVERY,
        "fallback_margin_threshold": VB_FALLBACK_MARGIN_THRESHOLD,
        "fallback_score_threshold": VB_FALLBACK_SCORE_THRESHOLD,
        "no_reject_mode": bool(NO_REJECT_MODE),
        "selection_mode": "force_top1_no_reject" if NO_REJECT_MODE else "reject_then_correct",
        "active_phrase_count": len(PHRASE_PACK),
    }


def normalize_to_phrase_id(raw_text: str) -> Optional[str]:
    cleaned = clean_text(raw_text)
    if not cleaned:
        return None
    for phrase in PHRASE_PACK:
        p_text = phrase["text"]
        p_clean = clean_text(p_text)
        if cleaned == p_clean or p_clean in cleaned or cleaned in p_clean:
            return phrase["id"]
    return None


def is_low_confidence(debug: Dict[str, Any]) -> bool:
    reject_reason = str(debug.get("reject_reason") or "none")
    if reject_reason != "none":
        return True
    margin = debug.get("margin", debug.get("gap"))
    if isinstance(margin, (int, float)) and float(margin) < VB_FALLBACK_MARGIN_THRESHOLD:
        return True
    score = debug.get("top1_score", debug.get("score", debug.get("best_dist")))
    if isinstance(score, (int, float)) and float(score) > VB_FALLBACK_SCORE_THRESHOLD:
        return True
    return False


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def ensure_storage() -> None:
    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    BATCH_DIR.mkdir(parents=True, exist_ok=True)
    PHRASE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    for phrase in PHRASE_PACK:
        (PHRASE_DATA_DIR / phrase["id"]).mkdir(parents=True, exist_ok=True)

    if not MANIFEST_PATH.exists():
        manifest = {
            "version": 1,
            "created_at": now_iso(),
            "phrases": {phrase["id"]: {"text": phrase["text"], "samples": []} for phrase in PHRASE_PACK},
        }
        save_json(MANIFEST_PATH, manifest)

    if not SIGNATURE_DB_PATH.exists():
        signature = {
            "version": 2,
            "updated_at": now_iso(),
            "phrases": {phrase["text"]: {"phrase_id": phrase["id"], "signature_counts": {}} for phrase in PHRASE_PACK},
        }
        save_json(SIGNATURE_DB_PATH, signature)


def ensure_v2_storage() -> None:
    DATA_V2_DIR.mkdir(parents=True, exist_ok=True)
    V2_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    V2_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    V2_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    V2_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    V2_UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval", "corrections", "rejected"):
        for phrase in PHRASE_PACK:
            (DATA_V2_DIR / split / phrase["id"]).mkdir(parents=True, exist_ok=True)
    if not V2_MANIFEST_PATH.exists():
        manifest = {
            "version": 2,
            "created_at": now_iso(),
            "phrase_pack": PHRASE_PACK,
            "samples": [],
            "events": [],
            "corrections": [],
            "rejected": [],
            "unknown_events": [],
        }
        save_json(V2_MANIFEST_PATH, manifest)


def ensure_v3_storage() -> None:
    DATA_V3_DIR.mkdir(parents=True, exist_ok=True)
    V3_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    V3_REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    V3_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    V3_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    V3_UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "eval", "corrections", "rejected"):
        for phrase in PHRASE_PACK:
            (DATA_V3_DIR / split / phrase["id"]).mkdir(parents=True, exist_ok=True)
    if not V3_MANIFEST_PATH.exists():
        manifest = {
            "version": 3,
            "created_at": now_iso(),
            "phrase_pack": PHRASE_PACK,
            "samples": [],
            "events": [],
            "corrections": [],
            "rejected": [],
            "unknown_events": [],
        }
        save_json(V3_MANIFEST_PATH, manifest)


def load_v2_manifest() -> Dict[str, Any]:
    ensure_v2_storage()
    try:
        manifest = json.loads(V2_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    manifest.setdefault("version", 2)
    manifest.setdefault("created_at", now_iso())
    manifest["phrase_pack"] = PHRASE_PACK
    manifest.setdefault("samples", [])
    manifest.setdefault("events", [])
    manifest.setdefault("corrections", [])
    manifest.setdefault("rejected", [])
    manifest.setdefault("unknown_events", [])
    for row in manifest.get("samples", []):
        if not isinstance(row, dict):
            continue
        row.setdefault("status", "active")
        row.setdefault("disabled_reason", None)
        row.setdefault("disabled_at", None)
        row.setdefault("quality_score", None)
        row.setdefault("suspicion_score", None)
        row.setdefault("purify_signals", [])
    return manifest


def save_v2_manifest(manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    save_json(V2_MANIFEST_PATH, manifest)


def load_v3_manifest() -> Dict[str, Any]:
    ensure_v3_storage()
    try:
        manifest = json.loads(V3_MANIFEST_PATH.read_text(encoding="utf-8"))
    except Exception:
        manifest = {}
    manifest.setdefault("version", 3)
    manifest.setdefault("created_at", now_iso())
    manifest["phrase_pack"] = PHRASE_PACK
    manifest.setdefault("samples", [])
    manifest.setdefault("events", [])
    manifest.setdefault("corrections", [])
    manifest.setdefault("rejected", [])
    manifest.setdefault("unknown_events", [])
    for row in manifest.get("samples", []):
        if not isinstance(row, dict):
            continue
        row.setdefault("status", "active")
        row.setdefault("disabled_reason", None)
        row.setdefault("disabled_at", None)
        row.setdefault("quality_score", None)
        row.setdefault("suspicion_score", None)
        row.setdefault("purify_signals", [])
    return manifest


def save_v3_manifest(manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    save_json(V3_MANIFEST_PATH, manifest)


def v2_path_for_rel(file_rel: str) -> Path:
    candidate = (ROOT_DIR / file_rel).resolve()
    allowed_roots = [DATA_V2_DIR.resolve(), DATA_V3_DIR.resolve(), (ROOT_DIR / "data_v2").resolve(), (ROOT_DIR / "data_v3").resolve()]
    for root in allowed_roots:
        if candidate == root or root in candidate.parents:
            return candidate
    raise ValueError("path outside managed data roots")


def v3_path_for_rel(file_rel: str) -> Path:
    candidate = (ROOT_DIR / file_rel).resolve()
    allowed_roots = [DATA_V3_DIR.resolve(), (ROOT_DIR / "data_v3").resolve()]
    for root in allowed_roots:
        if candidate == root or root in candidate.parents:
            return candidate
    raise ValueError("path outside data_v3")


def load_manifest() -> Dict[str, Any]:
    ensure_storage()
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest.setdefault("phrases", {})
    for phrase in PHRASE_PACK:
        manifest["phrases"].setdefault(phrase["id"], {"text": phrase["text"], "samples": []})
        manifest["phrases"][phrase["id"]]["text"] = phrase["text"]
        manifest["phrases"][phrase["id"]].setdefault("samples", [])
    return manifest


def save_manifest(manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = now_iso()
    save_json(MANIFEST_PATH, manifest)


def load_signature_db() -> Dict[str, Any]:
    ensure_storage()
    db = json.loads(SIGNATURE_DB_PATH.read_text(encoding="utf-8"))
    db.setdefault("phrases", {})
    return db


def load_index_meta() -> Dict[str, Any]:
    if not INDEX_META_PATH.exists():
        return {}
    try:
        payload = json.loads(INDEX_META_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def default_min_gap_profile() -> Dict[str, Any]:
    return {
        "version": 1,
        "updated_at": now_iso(),
        "global_multiplier": 1.0,
        "per_phrase_multiplier": {phrase["id"]: 1.0 for phrase in PHRASE_PACK},
    }


def normalize_multiplier_value(value: Any) -> float:
    try:
        f = float(value)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("invalid multiplier type") from exc
    if not np.isfinite(f):
        raise ValueError("invalid multiplier value")
    if f < MIN_GAP_MULTIPLIER_MIN or f > MIN_GAP_MULTIPLIER_MAX:
        raise ValueError(f"multiplier out of range [{MIN_GAP_MULTIPLIER_MIN}, {MIN_GAP_MULTIPLIER_MAX}]")
    return round(f, 6)


def normalize_min_gap_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    base = default_min_gap_profile()
    global_mul = payload.get("global_multiplier", base["global_multiplier"])
    per_phrase = payload.get("per_phrase_multiplier", {})
    if not isinstance(per_phrase, dict):
        raise ValueError("per_phrase_multiplier must be an object")

    normalized_per_phrase: Dict[str, float] = {}
    for phrase in PHRASE_PACK:
        pid = phrase["id"]
        normalized_per_phrase[pid] = normalize_multiplier_value(per_phrase.get(pid, 1.0))

    return {
        "version": 1,
        "updated_at": now_iso(),
        "global_multiplier": normalize_multiplier_value(global_mul),
        "per_phrase_multiplier": normalized_per_phrase,
    }


def load_min_gap_profile() -> Dict[str, Any]:
    ensure_storage()
    if not MIN_GAP_PROFILE_PATH.exists():
        profile = default_min_gap_profile()
        save_json(MIN_GAP_PROFILE_PATH, profile)
        return profile

    try:
        payload = json.loads(MIN_GAP_PROFILE_PATH.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("profile is not object")
        profile = normalize_min_gap_profile(payload)
        save_json(MIN_GAP_PROFILE_PATH, profile)
        return profile
    except Exception:
        profile = default_min_gap_profile()
        save_json(MIN_GAP_PROFILE_PATH, profile)
        return profile


def save_min_gap_profile(profile: Dict[str, Any]) -> None:
    save_json(MIN_GAP_PROFILE_PATH, profile)


def append_batch_event(payload: Dict[str, Any]) -> None:
    ensure_storage()
    row = dict(payload)
    row["server_received_at"] = now_iso()
    with BATCH_EVENTS_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_batch_events(limit: int = 2000) -> List[Dict[str, Any]]:
    if not BATCH_EVENTS_PATH.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with BATCH_EVENTS_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    if limit > 0:
        rows = rows[-limit:]
    return rows


def count_batch_events_for_phrase(phrase_id: str) -> int:
    return sum(1 for row in read_batch_events(limit=0) if row.get("truth_phrase_id") == phrase_id)


def get_phrase_stats(manifest: Dict[str, Any], signature_db: Dict[str, Any], index_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    signature_count_by_pid: Dict[str, int] = {}
    phrases = signature_db.get("phrases", {})
    if isinstance(phrases, dict):
        for phrase_text, detail in phrases.items():
            if not isinstance(detail, dict):
                continue
            pid = detail.get("phrase_id") or PHRASE_TEXT_TO_ID.get(phrase_text)
            counts = detail.get("signature_counts", {})
            if isinstance(pid, str) and isinstance(counts, dict):
                signature_count_by_pid[pid] = len(counts)

    calibration = index_meta.get("calibration", {}) if isinstance(index_meta, dict) else {}
    rows: List[Dict[str, Any]] = []
    for phrase in PHRASE_PACK:
        node = manifest.get("phrases", {}).get(phrase["id"], {})
        samples = node.get("samples", []) if isinstance(node, dict) else []
        cal = calibration.get(phrase["id"], {}) if isinstance(calibration, dict) else {}
        rows.append(
            {
                "phrase_id": phrase["id"],
                "text": phrase["text"],
                "sample_count": len(samples) if isinstance(samples, list) else 0,
                "target_count": TRAIN_TARGET_PER_PHRASE,
                "signature_count": signature_count_by_pid.get(phrase["id"], 0),
                "threshold_ready": bool(cal.get("threshold_ready", False)),
                "last_calibrated_at": cal.get("last_calibrated_at"),
                "phrase_threshold": cal.get("distance_threshold"),
                "min_gap": cal.get("min_gap"),
            }
        )
    return rows


def validate_audio_sample(data: bytes) -> Tuple[bool, str, float]:
    if not data:
        return False, "empty audio", 0.0
    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            nframes = wf.getnframes()
            rate = wf.getframerate()
            duration = nframes / float(rate) if rate > 0 else 0.0
    except wave.Error:
        return False, "invalid wav", 0.0
    if duration < MIN_SAMPLE_DURATION_SEC:
        return False, f"audio too short ({duration:.3f}s)", duration
    return True, "ok", duration


def analyze_wav_bytes(data: bytes, existing_hashes: Optional[set[str]] = None) -> Dict[str, Any]:
    sha = hashlib.sha256(data).hexdigest()
    report: Dict[str, Any] = {
        "ok": False,
        "sha256": sha,
        "duration_sec": 0.0,
        "sample_rate": None,
        "channels": None,
        "sample_width": None,
        "rms": 0.0,
        "peak": 0.0,
        "silence_ratio": 1.0,
        "quality_flags": [],
        "warning_flags": [],
    }
    flags: List[str] = []
    warnings: List[str] = []
    if not data:
        flags.append("empty_audio")
        report["quality_flags"] = flags
        return report

    try:
        with wave.open(io.BytesIO(data), "rb") as wf:
            nframes = wf.getnframes()
            rate = wf.getframerate()
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            raw = wf.readframes(nframes)
    except wave.Error:
        flags.append("invalid_wav")
        report["quality_flags"] = flags
        return report

    duration = nframes / float(rate) if rate else 0.0
    report.update(
        {
            "duration_sec": round(duration, 4),
            "sample_rate": rate,
            "channels": channels,
            "sample_width": sample_width,
        }
    )

    if rate != TARGET_SAMPLE_RATE:
        flags.append("unexpected_sample_rate")
    if channels != 1:
        flags.append("not_mono")
    if sample_width != 2:
        flags.append("not_16bit")
    if duration < MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:
        flags.append("too_short")
    if duration > 6.0:
        flags.append("too_long")
    if existing_hashes and sha in existing_hashes:
        flags.append("duplicate_audio")

    samples = np.array([], dtype=np.float32)
    if sample_width == 2 and raw:
        samples = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        if channels > 1 and samples.size >= channels:
            samples = samples.reshape(-1, channels).mean(axis=1)
    if samples.size:
        abs_samples = np.abs(samples)
        peak = float(np.max(abs_samples))
        rms = float(np.sqrt(np.mean(samples * samples)))
        silence_ratio = float(np.mean(abs_samples < 0.01))
        report["peak"] = round(peak, 6)
        report["rms"] = round(rms, 6)
        report["silence_ratio"] = round(silence_ratio, 6)
        if rms < 0.003:
            flags.append("too_quiet")
        if peak > 0.98:
            warnings.append("clipping_risk")
        if silence_ratio > 0.92:
            flags.append("mostly_silence")
    else:
        flags.append("no_pcm_samples")

    report["quality_flags"] = flags
    report["warning_flags"] = warnings
    report["ok"] = len(flags) == 0
    return report


def existing_v2_hashes(manifest: Optional[Dict[str, Any]] = None) -> set[str]:
    manifest = manifest or load_v2_manifest()
    hashes: set[str] = set()
    for bucket in ("samples", "events", "corrections", "unknown_events"):
        rows = manifest.get(bucket, [])
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and isinstance(row.get("sha256"), str):
                    hashes.add(row["sha256"])
    return hashes


def clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return float(value)


def v2_all_train_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v2_manifest()
    rows = manifest.get("samples", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("split") == "train"]


def v2_sample_quality_score(row: Dict[str, Any]) -> float:
    score = 1.0
    duration = row.get("duration_sec")
    if isinstance(duration, (int, float)):
        d = float(duration)
        if d < 0.9:
            score -= 0.22
        elif d < 1.2:
            score -= 0.08
        elif d > 3.5:
            score -= 0.08
    flags = [str(x) for x in list(row.get("quality_flags") or [])]
    warning_flags = [str(x) for x in list(row.get("warning_flags") or [])]
    severe = {"too_short", "mostly_silence", "too_quiet", "not_mono", "invalid_wav", "no_pcm_samples"}
    for flag in flags:
        if flag in severe:
            score -= 0.32
        elif flag in {"unexpected_sample_rate", "not_16bit", "too_long"}:
            score -= 0.18
        elif flag == "duplicate_audio":
            score -= 0.08
    if "clipping_risk" in warning_flags:
        score -= 0.08
    return round(clamp01(score), 4)


def v2_train_templates_from_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    templates: List[Dict[str, Any]] = []
    for row in rows:
        rel = row.get("file_rel")
        pid = row.get("phrase_id")
        if not isinstance(rel, str) or not isinstance(pid, str):
            continue
        try:
            path = v2_path_for_rel(rel)
        except ValueError:
            continue
        if not path.exists():
            continue
        templates.append(
            {
                "sample_id": row.get("sample_id"),
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "file_rel": rel,
                "duration_sec": row.get("duration_sec"),
            }
        )
    return templates


def v2_recent_confusion_map(
    events: List[Dict[str, Any]],
    window: int = VB_PURIFY_RECENT_ERROR_WINDOW,
) -> Dict[str, Dict[str, int]]:
    if window <= 0:
        return {}
    picked = events[-window:] if len(events) > window else events
    out: Dict[str, Dict[str, int]] = {}
    for event in picked:
        truth = event.get("truth_phrase_id")
        pred = event.get("predicted_phrase_id") or event.get("best_phrase_id")
        if not isinstance(truth, str) or truth not in PHRASE_ID_TO_TEXT:
            continue
        if not isinstance(pred, str) or pred not in PHRASE_ID_TO_TEXT:
            continue
        reason = str(event.get("reject_reason") or "unknown")
        if pred == truth and reason == "none":
            continue
        out.setdefault(truth, {})
        out[truth][pred] = out[truth].get(pred, 0) + 1
    return out


def v2_auto_purify_train_samples(
    manifest: Optional[Dict[str, Any]] = None,
    trigger_reason: str = "corrections_confirm",
) -> Dict[str, Any]:
    manifest = manifest or load_v2_manifest()
    now = now_iso()
    all_train = v2_all_train_records(manifest)
    active_rows = [row for row in all_train if str(row.get("status") or "active") == "active"]
    events = v2_eval_records(manifest)
    confusion_map = v2_recent_confusion_map(events)
    meta = load_v2_index_meta()
    per_phrase_thresholds = meta.get("thresholds_by_phrase", {}) if isinstance(meta, dict) else {}
    templates = v2_train_templates_from_rows(active_rows)

    active_by_phrase_before: Dict[str, int] = {}
    for row in active_rows:
        pid = row.get("phrase_id")
        if isinstance(pid, str):
            active_by_phrase_before[pid] = active_by_phrase_before.get(pid, 0) + 1

    active_by_phrase_after = dict(active_by_phrase_before)
    disabled_per_phrase: Dict[str, int] = {pid: 0 for pid in PHRASE_ID_TO_TEXT}
    protected_by_min_active = 0
    candidates = 0
    disabled_rows: List[Dict[str, Any]] = []
    touched_rows = 0

    for row in active_rows:
        pid = row.get("phrase_id")
        sid = row.get("sample_id")
        rel = row.get("file_rel")
        if not isinstance(pid, str) or pid not in PHRASE_ID_TO_TEXT:
            continue
        if not isinstance(sid, str) or not isinstance(rel, str):
            continue

        touched_rows += 1
        signals: List[str] = []
        suspicion = 0.0
        quality_score = v2_sample_quality_score(row)
        row["quality_score"] = quality_score

        quality_flags = [str(x) for x in list(row.get("quality_flags") or [])]
        severe_flags = [f for f in quality_flags if f in {"too_short", "mostly_silence", "too_quiet", "not_mono", "invalid_wav", "no_pcm_samples"}]
        if severe_flags:
            suspicion += 0.52
            signals.extend([f"quality_severe:{f}" for f in severe_flags[:2]])
        if "clipping_risk" in [str(x) for x in list(row.get("warning_flags") or [])]:
            suspicion += 0.10
            signals.append("quality_warning:clipping_risk")

        try:
            query_path = v2_path_for_rel(rel)
        except ValueError:
            query_path = None
        loo_debug: Dict[str, Any] = {}
        if query_path is not None and query_path.exists() and len(templates) >= 2:
            _, loo_debug = v2_score_against_templates(query_path, templates, exclude_sample_id=sid)
            best_pid = loo_debug.get("best_phrase_id")
            best_dist = loo_debug.get("best_dist")
            if isinstance(best_pid, str) and best_pid != pid:
                suspicion += 0.43
                signals.append("loo_misclassified")
            th_row = per_phrase_thresholds.get(pid, {}) if isinstance(per_phrase_thresholds, dict) else {}
            th = th_row.get("distance_threshold") if isinstance(th_row, dict) else None
            if isinstance(th, (int, float)) and isinstance(best_dist, (int, float)):
                if float(best_dist) > float(th) * 1.15:
                    suspicion += 0.23
                    signals.append("loo_distance_too_high")
            confuse = confusion_map.get(pid, {}) if isinstance(confusion_map, dict) else {}
            if isinstance(confuse, dict):
                hot_preds = [p for p, cnt in confuse.items() if isinstance(p, str) and isinstance(cnt, int) and cnt >= 2 and p != pid]
                if isinstance(best_pid, str) and best_pid in hot_preds:
                    suspicion += 0.18
                    signals.append("overlap_recent_confusion")

        suspicion += (1.0 - quality_score) * 0.35
        suspicion = round(clamp01(suspicion), 4)
        row["suspicion_score"] = suspicion
        row["purify_signals"] = signals

        if suspicion < VB_PURIFY_DISABLE_THRESHOLD:
            continue
        candidates += 1
        if disabled_per_phrase.get(pid, 0) >= VB_PURIFY_PER_PHRASE_MAX_DISABLE:
            continue
        active_left = active_by_phrase_after.get(pid, 0)
        if active_left - 1 < VB_MIN_ACTIVE_PER_PHRASE:
            protected_by_min_active += 1
            continue

        row["status"] = "disabled"
        row["disabled_reason"] = f"auto_purify:{trigger_reason}"
        row["disabled_at"] = now
        active_by_phrase_after[pid] = active_left - 1
        disabled_per_phrase[pid] = disabled_per_phrase.get(pid, 0) + 1
        disabled_rows.append(
            {
                "sample_id": sid,
                "phrase_id": pid,
                "file_rel": rel,
                "suspicion_score": suspicion,
                "quality_score": quality_score,
                "disabled_reason": row["disabled_reason"],
                "purify_signals": signals,
            }
        )

    save_v2_manifest(manifest)
    by_phrase: List[Dict[str, Any]] = []
    for phrase in PHRASE_PACK:
        pid = phrase["id"]
        total = sum(1 for row in all_train if row.get("phrase_id") == pid)
        active_before = active_by_phrase_before.get(pid, 0)
        active_after = active_by_phrase_after.get(pid, 0)
        by_phrase.append(
            {
                "phrase_id": pid,
                "phrase_text": phrase["text"],
                "total_train": total,
                "active_before": active_before,
                "active_after": active_after,
                "disabled_this_round": max(0, active_before - active_after),
            }
        )

    return {
        "ok": True,
        "trigger_reason": trigger_reason,
        "threshold": VB_PURIFY_DISABLE_THRESHOLD,
        "min_active_per_phrase": VB_MIN_ACTIVE_PER_PHRASE,
        "max_disable_per_phrase": VB_PURIFY_PER_PHRASE_MAX_DISABLE,
        "touched_rows": touched_rows,
        "disable_candidates": candidates,
        "disabled_count": len(disabled_rows),
        "protected_by_min_active": protected_by_min_active,
        "disabled_samples": disabled_rows,
        "by_phrase": by_phrase,
    }


def v2_train_health_report(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    manifest = manifest or load_v2_manifest()
    all_train = v2_all_train_records(manifest)
    phrase_rows: List[Dict[str, Any]] = []
    suspicious_samples: List[Dict[str, Any]] = []
    for phrase in PHRASE_PACK:
        pid = phrase["id"]
        rows = [row for row in all_train if row.get("phrase_id") == pid]
        active_rows = [row for row in rows if str(row.get("status") or "active") == "active"]
        disabled_rows = [row for row in rows if str(row.get("status") or "active") == "disabled"]
        suspect = [
            row
            for row in active_rows
            if isinstance(row.get("suspicion_score"), (int, float))
            and float(row["suspicion_score"]) >= max(0.6, VB_PURIFY_DISABLE_THRESHOLD - 0.12)
        ]
        for row in suspect[:8]:
            suspicious_samples.append(
                {
                    "sample_id": row.get("sample_id"),
                    "phrase_id": pid,
                    "phrase_text": phrase["text"],
                    "file_rel": row.get("file_rel"),
                    "suspicion_score": row.get("suspicion_score"),
                    "quality_score": row.get("quality_score"),
                    "purify_signals": row.get("purify_signals") or [],
                }
            )
        if len(active_rows) < VB_MIN_ACTIVE_PER_PHRASE:
            suggestion = f"active<{VB_MIN_ACTIVE_PER_PHRASE}: prioritize truth re-record"
        elif suspect:
            suggestion = "review high-suspicion active samples"
        elif disabled_rows:
            suggestion = "can manually re-enable disabled after spot-check"
        else:
            suggestion = "healthy"
        phrase_rows.append(
            {
                "phrase_id": pid,
                "phrase_text": phrase["text"],
                "total": len(rows),
                "active": len(active_rows),
                "disabled": len(disabled_rows),
                "suspect_active": len(suspect),
                "suggested_action": suggestion,
            }
        )

    return {
        "ok": True,
        "dataset": "data_v3" if DATA_V2_DIR.resolve() == DATA_V3_DIR.resolve() else "data_v2",
        "min_active_per_phrase": VB_MIN_ACTIVE_PER_PHRASE,
        "disable_threshold": VB_PURIFY_DISABLE_THRESHOLD,
        "items": phrase_rows,
        "suspicious_samples": sorted(
            suspicious_samples,
            key=lambda row: float(row.get("suspicion_score") or 0.0),
            reverse=True,
        )[:64],
    }


def v2_update_samples_status(
    sample_ids: List[str],
    status: str,
    reason: str,
) -> Dict[str, Any]:
    target_status = status.strip().lower()
    if target_status not in {"active", "disabled"}:
        return {"ok": False, "error": "status must be active|disabled"}
    if not sample_ids:
        return {"ok": False, "error": "sample_ids is empty"}

    manifest = load_v2_manifest()
    touched = 0
    updated = 0
    missing: List[str] = []
    id_set: Set[str] = set([sid for sid in sample_ids if isinstance(sid, str) and sid.strip()])
    rows = manifest.get("samples", [])
    if not isinstance(rows, list):
        return {"ok": False, "error": "manifest samples invalid"}
    idx: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if isinstance(row, dict) and isinstance(row.get("sample_id"), str):
            idx[str(row["sample_id"])] = row
    for sid in id_set:
        row = idx.get(sid)
        if not isinstance(row, dict):
            missing.append(sid)
            continue
        if row.get("split") != "train":
            missing.append(sid)
            continue
        touched += 1
        prev = str(row.get("status") or "active")
        if target_status == prev:
            continue
        row["status"] = target_status
        if target_status == "disabled":
            row["disabled_reason"] = reason or "manual_disable"
            row["disabled_at"] = now_iso()
        else:
            row["disabled_reason"] = None
            row["disabled_at"] = None
        if row.get("quality_score") is None:
            row["quality_score"] = v2_sample_quality_score(row)
        updated += 1
    save_v2_manifest(manifest)
    return {"ok": True, "touched": touched, "updated": updated, "missing": sorted(missing)}


def save_media_file(prefix: str, data: bytes, ext: str) -> Path:
    name = f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.{ext}"
    path = AUDIO_DIR / name
    path.write_bytes(data)
    return path


def transcribe_siliconflow(audio_path: Path, model_name: str, api_key: str) -> str:
    with audio_path.open("rb") as f:
        files = {"file": (audio_path.name, f, "audio/wav")}
        data = {"model": model_name, "language": "zh", "temperature": "0.0"}
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(SILICONFLOW_URL, headers=headers, data=data, files=files, timeout=30)
    response.raise_for_status()
    payload = response.json()

    text = payload.get("text")
    if isinstance(text, str):
        return text.strip()
    choices = payload.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            candidate = first.get("text")
            if isinstance(candidate, str):
                return candidate.strip()
            msg = first.get("message")
            if isinstance(msg, dict):
                content = msg.get("content")
                if isinstance(content, str):
                    return content.strip()
    return ""


def transcribe_with_fallback(audio_path: Path, api_key: str) -> Tuple[str, List[Dict[str, Any]]]:
    steps: List[Dict[str, Any]] = []
    try:
        raw = transcribe_siliconflow(audio_path, ASR_PRIMARY_MODEL, api_key)
        steps.append({"stage": "asr", "status": "ok", "model": ASR_PRIMARY_MODEL, "message": f"primary result: {raw or '<empty>'}"})
        return raw, steps
    except Exception as exc:
        steps.append({"stage": "asr", "status": "warn", "model": ASR_PRIMARY_MODEL, "message": f"primary failed: {exc}"})

    try:
        raw = transcribe_siliconflow(audio_path, ASR_FALLBACK_MODEL, api_key)
        steps.append({"stage": "asr", "status": "ok", "model": ASR_FALLBACK_MODEL, "message": f"fallback result: {raw or '<empty>'}"})
        return raw, steps
    except Exception as exc:
        steps.append({"stage": "asr", "status": "error", "model": ASR_FALLBACK_MODEL, "message": f"fallback failed: {exc}"})
        return "", steps


async def synthesize_tts_async(text: str, output_path: Path) -> None:
    communicator = edge_tts.Communicate(text=text, voice=TTS_VOICE)
    await communicator.save(str(output_path))


def run_tts_sync(text: str, output_path: Path) -> None:
    try:
        asyncio.get_running_loop()
        in_running_loop = True
    except RuntimeError:
        in_running_loop = False

    if not in_running_loop:
        asyncio.run(synthesize_tts_async(text, output_path))
        return

    errors: List[BaseException] = []

    def _worker() -> None:
        try:
            asyncio.run(synthesize_tts_async(text, output_path))
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()
    if errors:
        raise errors[0]


def preprocess_waveform(y: np.ndarray) -> np.ndarray:
    if y.size == 0:
        return y
    trimmed, _ = librosa.effects.trim(y, top_db=TRIM_TOP_DB)
    if trimmed.size >= 256:
        y = trimmed
    peak = float(np.max(np.abs(y))) if y.size > 0 else 0.0
    if peak > 1e-6:
        y = y / peak
    return y.astype(np.float32)


def load_wave(path: Path) -> Optional[np.ndarray]:
    try:
        y, _ = librosa.load(str(path), sr=TARGET_SAMPLE_RATE, mono=True)
    except Exception:
        return None
    if y.size == 0:
        return None
    y = preprocess_waveform(y)
    if y.size < 256:
        return None
    return y


def l2_normalize(vec: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(vec))
    if denom < 1e-12:
        return vec.astype(np.float32)
    return (vec / denom).astype(np.float32)


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    an = l2_normalize(a)
    bn = l2_normalize(b)
    sim = float(np.dot(an, bn))
    return float(1.0 - sim)


def get_hubert() -> Tuple[Any, Any]:
    global _HUBERT_FEATURE_EXTRACTOR, _HUBERT_MODEL
    if AutoFeatureExtractor is None or HubertModel is None:
        raise RuntimeError("transformers/torch not available")
    if _HUBERT_FEATURE_EXTRACTOR is None:
        _HUBERT_FEATURE_EXTRACTOR = AutoFeatureExtractor.from_pretrained(HUBERT_MODEL_NAME)
    if _HUBERT_MODEL is None:
        _HUBERT_MODEL = HubertModel.from_pretrained(HUBERT_MODEL_NAME)
        _HUBERT_MODEL.eval()
        _HUBERT_MODEL.to(HUBERT_DEVICE)
    return _HUBERT_FEATURE_EXTRACTOR, _HUBERT_MODEL


def get_ssl_model() -> Tuple[Any, Any, str]:
    global _SSL_FEATURE_EXTRACTOR, _SSL_MODEL, _SSL_MODEL_NAME, _SSL_MODEL_ERROR
    if AutoFeatureExtractor is None or AutoModel is None or torch is None:
        _SSL_MODEL_ERROR = "transformers/torch not available"
        raise RuntimeError(_SSL_MODEL_ERROR)
    if _SSL_FEATURE_EXTRACTOR is not None and _SSL_MODEL is not None and _SSL_MODEL_NAME:
        return _SSL_FEATURE_EXTRACTOR, _SSL_MODEL, str(_SSL_MODEL_NAME)

    errors: List[str] = []
    for model_name in (WAVLM_MODEL_NAME, HUBERT_MODEL_NAME):
        try:
            extractor = AutoFeatureExtractor.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name)
            model.eval()
            model.to(HUBERT_DEVICE)
            _SSL_FEATURE_EXTRACTOR = extractor
            _SSL_MODEL = model
            _SSL_MODEL_NAME = model_name
            _SSL_MODEL_ERROR = None
            return extractor, model, model_name
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{model_name}: {exc}")
            continue
    _SSL_MODEL_ERROR = " | ".join(errors) if errors else "failed to load SSL model"
    raise RuntimeError(_SSL_MODEL_ERROR)


def extract_hubert_embedding(y: np.ndarray) -> Optional[np.ndarray]:
    if torch is None:
        return None
    if y.size < 256:
        return None
    try:
        extractor, model = get_hubert()
        with torch.inference_mode():
            inputs = extractor(
                y,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(HUBERT_DEVICE) for k, v in inputs.items()}
            out = model(**inputs)
            hs = out.last_hidden_state
            emb = hs.mean(dim=1).squeeze(0).detach().cpu().numpy().astype(np.float32)
        return l2_normalize(emb)
    except Exception:
        return None


def _ssl_feature_paths(file_rel: str) -> Tuple[Path, Path]:
    key = hashlib.sha256(file_rel.replace("\\", "/").encode("utf-8")).hexdigest()
    if file_rel.replace("\\", "/").startswith("data_v3/"):
        feature_root = V3_FEATURES_DIR
    else:
        feature_root = V2_FEATURES_DIR
    base = feature_root / "ssl_v1" / key
    return base.with_suffix(".emb.npy"), base.with_suffix(".seq.npy")


def extract_ssl_features_from_wave(y: np.ndarray) -> Optional[Dict[str, Any]]:
    if torch is None or y.size < 256:
        return None
    try:
        extractor, model, model_name = get_ssl_model()
        with torch.inference_mode():
            inputs = extractor(
                y,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
            inputs = {k: v.to(HUBERT_DEVICE) for k, v in inputs.items()}
            out = model(**inputs)
            seq = out.last_hidden_state.squeeze(0).detach().cpu().numpy().astype(np.float32)
            if seq.ndim != 2 or seq.shape[0] < 1:
                return None
            emb = l2_normalize(np.mean(seq, axis=0).astype(np.float32))
            return {"embedding": emb, "sequence": seq.astype(np.float32), "model_name": model_name}
    except Exception:
        return None


def load_or_extract_ssl_features(path: Path, file_rel: Optional[str] = None) -> Optional[Dict[str, Any]]:
    rel = file_rel or path.relative_to(ROOT_DIR).as_posix()
    emb_path, seq_path = _ssl_feature_paths(rel)
    if emb_path.exists() and seq_path.exists():
        try:
            emb = np.load(emb_path).astype(np.float32)
            seq = np.load(seq_path).astype(np.float32)
            if emb.size > 0 and seq.ndim == 2 and seq.shape[0] > 0:
                return {"embedding": l2_normalize(emb), "sequence": seq, "model_name": _SSL_MODEL_NAME or "cached_ssl"}
        except Exception:
            pass

    y = load_wave(path)
    if y is None:
        return None
    features = extract_ssl_features_from_wave(y)
    if features is None:
        return None
    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, features["embedding"])
    np.save(seq_path, features["sequence"])
    return features


def ssl_sequence_dtw_distance(query_seq: np.ndarray, sample_seq: np.ndarray) -> float:
    if query_seq.ndim != 2 or sample_seq.ndim != 2 or query_seq.shape[0] < 1 or sample_seq.shape[0] < 1:
        return float("inf")
    D, _ = librosa.sequence.dtw(X=query_seq.T, Y=sample_seq.T, metric="cosine")
    return float(D[-1, -1]) / max(float(query_seq.shape[0] + sample_seq.shape[0]), 1.0)


def extract_embedding(y: np.ndarray) -> Optional[np.ndarray]:
    try:
        mfcc = librosa.feature.mfcc(y=y, sr=TARGET_SAMPLE_RATE, n_mfcc=EMBED_N_MFCC)
    except Exception:
        return None
    if mfcc.ndim != 2 or mfcc.shape[1] < 2:
        return None
    mean = np.mean(mfcc, axis=1)
    std = np.std(mfcc, axis=1)
    vec = np.concatenate([mean, std]).astype(np.float32)
    return vec


def mfcc_dtw_distance_from_waves(query_y: np.ndarray, sample_y: np.ndarray) -> float:
    m1 = librosa.feature.mfcc(y=query_y, sr=TARGET_SAMPLE_RATE, n_mfcc=EMBED_N_MFCC)
    m2 = librosa.feature.mfcc(y=sample_y, sr=TARGET_SAMPLE_RATE, n_mfcc=EMBED_N_MFCC)
    if m1.shape[1] < 2 or m2.shape[1] < 2:
        return float("inf")
    D, _ = librosa.sequence.dtw(X=m1, Y=m2, metric="cosine")
    path_cost = float(D[-1, -1])
    norm = float(m1.shape[1] + m2.shape[1])
    return path_cost / max(norm, 1.0)


def v2_train_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v2_manifest()
    rows = manifest.get("samples", [])
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("split") == "train" and str(row.get("status") or "active") == "active"
    ]


def v3_train_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v3_manifest()
    rows = manifest.get("samples", [])
    if not isinstance(rows, list):
        return []
    return [
        row
        for row in rows
        if isinstance(row, dict) and row.get("split") == "train" and str(row.get("status") or "active") == "active"
    ]


def v2_eval_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v2_manifest()
    rows = manifest.get("events", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("split") == "eval"]


def v3_eval_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v3_manifest()
    rows = manifest.get("events", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("split") == "eval"]


def v2_unknown_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v2_manifest()
    rows = manifest.get("unknown_events", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("split") == "unknown"]


def v3_unknown_records(manifest: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    manifest = manifest or load_v3_manifest()
    rows = manifest.get("unknown_events", [])
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict) and row.get("split") == "unknown"]


def v2_phrase_counts(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, int]]:
    manifest = manifest or load_v2_manifest()
    counts = {
        phrase["id"]: {"train": 0, "eval": 0, "corrections": 0, "rejected": 0}
        for phrase in PHRASE_PACK
    }
    for row in manifest.get("samples", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("phrase_id")
        split = row.get("split")
        if isinstance(pid, str) and pid in counts and split in counts[pid]:
            counts[pid][split] += 1
    for row in manifest.get("events", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("truth_phrase_id")
        if isinstance(pid, str) and pid in counts:
            counts[pid]["eval"] += 1
    for row in manifest.get("corrections", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("truth_phrase_id")
        if isinstance(pid, str) and pid in counts:
            counts[pid]["corrections"] += 1
    for row in manifest.get("rejected", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("phrase_id") or row.get("truth_phrase_id")
        if isinstance(pid, str) and pid in counts:
            counts[pid]["rejected"] += 1
    return counts


def v3_phrase_counts(manifest: Optional[Dict[str, Any]] = None) -> Dict[str, Dict[str, int]]:
    manifest = manifest or load_v3_manifest()
    counts = {
        phrase["id"]: {"train": 0, "eval": 0, "corrections": 0, "rejected": 0}
        for phrase in PHRASE_PACK
    }
    for row in manifest.get("samples", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("phrase_id")
        split = row.get("split")
        if isinstance(pid, str) and pid in counts and split in counts[pid]:
            counts[pid][split] += 1
    for row in manifest.get("events", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("truth_phrase_id")
        if isinstance(pid, str) and pid in counts:
            counts[pid]["eval"] += 1
    for row in manifest.get("corrections", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("truth_phrase_id")
        if isinstance(pid, str) and pid in counts:
            counts[pid]["corrections"] += 1
    for row in manifest.get("rejected", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("phrase_id") or row.get("truth_phrase_id")
        if isinstance(pid, str) and pid in counts:
            counts[pid]["rejected"] += 1
    return counts


def v2_build_index() -> Dict[str, Any]:
    manifest = load_v2_manifest()
    train_rows = v2_train_records(manifest)
    templates: List[Dict[str, Any]] = []
    phrase_durations: Dict[str, List[float]] = {}

    for row in train_rows:
        rel = row.get("file_rel")
        pid = row.get("phrase_id")
        if not isinstance(rel, str) or not isinstance(pid, str):
            continue
        try:
            path = v2_path_for_rel(rel)
        except ValueError:
            continue
        if not path.exists():
            continue
        y = load_wave(path)
        if y is None:
            continue
        templates.append(
            {
                "sample_id": row.get("sample_id"),
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "file_rel": rel,
                "duration_sec": row.get("duration_sec"),
            }
        )
        if isinstance(row.get("duration_sec"), (int, float)):
            phrase_durations.setdefault(pid, []).append(float(row["duration_sec"]))

    templates_by_phrase: Dict[str, int] = {}
    for tpl in templates:
        pid = tpl["phrase_id"]
        templates_by_phrase[pid] = templates_by_phrase.get(pid, 0) + 1

    meta = {
        "version": 2,
        "updated_at": now_iso(),
        "engine": ENGINE_V3_PERSONALIZED_TOP1,
        "feature": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "n_mfcc": EMBED_N_MFCC,
            "trim_top_db": TRIM_TOP_DB,
            "dtw_topk_per_phrase": DTW_TOPK_PER_PHRASE,
        },
        "templates": templates,
        "templates_by_phrase": templates_by_phrase,
        "phrase_stats": {
            pid: {
                "template_count": templates_by_phrase.get(pid, 0),
                "duration_median": round(float(np.median(vals)), 4) if vals else None,
            }
            for pid, vals in phrase_durations.items()
        },
        "thresholds": {
            "distance_threshold": 0.035,
            "min_gap": 0.003,
            "ratio_min": 1.06,
        },
    }
    calibration = v2_calibrate_thresholds_from_templates(templates)
    if calibration.get("thresholds_by_phrase"):
        meta["thresholds_by_phrase"] = calibration["thresholds_by_phrase"]
        meta["calibration"] = calibration
    save_json(V2_INDEX_META_PATH, meta)
    return {"ok": bool(templates), "meta": meta, "template_count": len(templates)}


@contextmanager
def v3_runtime_context():
    global DATA_V2_DIR, V2_MANIFEST_PATH, V2_INDEX_DIR, V2_INDEX_META_PATH
    global V2_REPORTS_DIR, V2_AUDIT_REPORT_PATH, V2_MODELS_DIR, V2_ACTIVE_ENGINE_PATH
    global V2_FEATURES_DIR, V2_UNKNOWN_DIR
    old = (
        DATA_V2_DIR,
        V2_MANIFEST_PATH,
        V2_INDEX_DIR,
        V2_INDEX_META_PATH,
        V2_REPORTS_DIR,
        V2_AUDIT_REPORT_PATH,
        V2_MODELS_DIR,
        V2_ACTIVE_ENGINE_PATH,
        V2_FEATURES_DIR,
        V2_UNKNOWN_DIR,
    )
    DATA_V2_DIR = DATA_V3_DIR
    V2_MANIFEST_PATH = V3_MANIFEST_PATH
    V2_INDEX_DIR = V3_INDEX_DIR
    V2_INDEX_META_PATH = V3_INDEX_META_PATH
    V2_REPORTS_DIR = V3_REPORTS_DIR
    V2_AUDIT_REPORT_PATH = V3_AUDIT_REPORT_PATH
    V2_MODELS_DIR = V3_MODELS_DIR
    V2_ACTIVE_ENGINE_PATH = V3_ACTIVE_ENGINE_PATH
    V2_FEATURES_DIR = V3_FEATURES_DIR
    V2_UNKNOWN_DIR = V3_UNKNOWN_DIR
    try:
        yield
    finally:
        (
            DATA_V2_DIR,
            V2_MANIFEST_PATH,
            V2_INDEX_DIR,
            V2_INDEX_META_PATH,
            V2_REPORTS_DIR,
            V2_AUDIT_REPORT_PATH,
            V2_MODELS_DIR,
            V2_ACTIVE_ENGINE_PATH,
            V2_FEATURES_DIR,
            V2_UNKNOWN_DIR,
        ) = old


def v2_score_against_templates(query_path: Path, templates: List[Dict[str, Any]], exclude_sample_id: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "engine": "mfcc_dtw_baseline",
        "reject_reason": "none",
        "best_phrase": None,
        "best_phrase_id": None,
        "second_phrase": None,
        "second_phrase_id": None,
        "best_dist": None,
        "second_dist": None,
        "gap": None,
        "ratio": None,
        "candidates": [],
    }
    query_wave = load_wave(query_path)
    if query_wave is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug
    by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for tpl in templates:
        if exclude_sample_id and tpl.get("sample_id") == exclude_sample_id:
            continue
        pid = tpl.get("phrase_id")
        rel = tpl.get("file_rel")
        if isinstance(pid, str) and isinstance(rel, str):
            by_phrase.setdefault(pid, []).append(tpl)
    phrase_scores: List[Dict[str, Any]] = []
    for pid, rows in by_phrase.items():
        dtw_vals: List[float] = []
        used: List[str] = []
        for tpl in rows:
            rel = tpl.get("file_rel")
            if not isinstance(rel, str):
                continue
            try:
                sample_path = v2_path_for_rel(rel)
            except ValueError:
                continue
            sample_wave = load_wave(sample_path)
            if sample_wave is None:
                continue
            dist = mfcc_dtw_distance_from_waves(query_wave, sample_wave)
            if np.isfinite(dist):
                dtw_vals.append(float(dist))
                used.append(rel)
        if not dtw_vals:
            continue
        dtw_vals.sort()
        top_vals = dtw_vals[: min(DTW_TOPK_PER_PHRASE, len(dtw_vals))]
        phrase_scores.append(
            {
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "score": round(float(np.mean(top_vals)), 6),
                "nearest": [round(v, 6) for v in top_vals],
                "template_count": len(rows),
                "used_samples": used[:DTW_TOPK_PER_PHRASE],
            }
        )
    if not phrase_scores:
        debug["reject_reason"] = "no_templates"
        return None, debug
    phrase_scores.sort(key=lambda row: float(row["score"]))
    best = phrase_scores[0]
    second = phrase_scores[1] if len(phrase_scores) > 1 else None
    best_dist = float(best["score"])
    second_dist = float(second["score"]) if second else float("inf")
    gap = second_dist - best_dist if np.isfinite(second_dist) else float("inf")
    ratio = second_dist / best_dist if best_dist > 1e-9 and np.isfinite(second_dist) else float("inf")
    debug.update(
        {
            "best_phrase": best["phrase_text"],
            "best_phrase_id": best["phrase_id"],
            "second_phrase": second["phrase_text"] if second else None,
            "second_phrase_id": second["phrase_id"] if second else None,
            "best_dist": round(best_dist, 6),
            "second_dist": safe_float(round(second_dist, 6) if np.isfinite(second_dist) else second_dist),
            "gap": safe_float(round(gap, 6) if np.isfinite(gap) else gap),
            "ratio": safe_float(round(ratio, 6) if np.isfinite(ratio) else ratio),
            "candidates": phrase_scores,
        }
    )
    return str(best["phrase_text"]), debug


def v2_calibrate_thresholds_from_templates(templates: List[Dict[str, Any]]) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    by_truth: Dict[str, List[Dict[str, Any]]] = {}
    for tpl in templates:
        rel = tpl.get("file_rel")
        pid = tpl.get("phrase_id")
        sample_id = tpl.get("sample_id")
        if not isinstance(rel, str) or not isinstance(pid, str):
            continue
        try:
            path = v2_path_for_rel(rel)
        except ValueError:
            continue
        _, debug = v2_score_against_templates(path, templates, exclude_sample_id=str(sample_id) if sample_id else None)
        row = {
            "sample_id": sample_id,
            "truth_phrase_id": pid,
            "best_phrase_id": debug.get("best_phrase_id"),
            "second_phrase_id": debug.get("second_phrase_id"),
            "best_dist": debug.get("best_dist"),
            "second_dist": debug.get("second_dist"),
            "gap": debug.get("gap"),
            "ratio": debug.get("ratio"),
            "reject_reason": debug.get("reject_reason"),
            "file_rel": rel,
        }
        rows.append(row)
        by_truth.setdefault(pid, []).append(row)

    thresholds_by_phrase: Dict[str, Dict[str, Any]] = {}
    for pid, phrase_rows in by_truth.items():
        correct = [
            row
            for row in phrase_rows
            if row.get("best_phrase_id") == pid and isinstance(row.get("best_dist"), (int, float))
        ]
        positive_dists = [float(row["best_dist"]) for row in correct]
        correct_gaps = [float(row["gap"]) for row in correct if isinstance(row.get("gap"), (int, float))]
        wrong_best = [
            float(row["best_dist"])
            for row in phrase_rows
            if row.get("best_phrase_id") != pid and isinstance(row.get("best_dist"), (int, float))
        ]
        if positive_dists:
            p95 = float(np.percentile(np.array(positive_dists, dtype=np.float32), 95))
            p90 = float(np.percentile(np.array(positive_dists, dtype=np.float32), 90))
            distance_threshold = max(p95 + 0.002, p90 + 0.003, 0.02)
        else:
            distance_threshold = 0.035
        if wrong_best:
            distance_threshold = min(distance_threshold, max(0.02, float(np.percentile(np.array(wrong_best, dtype=np.float32), 30)) - 0.001))
        if correct_gaps:
            gap_p10 = float(np.percentile(np.array(correct_gaps, dtype=np.float32), 10))
            min_gap = max(0.001, min(0.006, gap_p10 * 0.65))
        else:
            min_gap = 0.003
        thresholds_by_phrase[pid] = {
            "distance_threshold": round(float(distance_threshold), 6),
            "min_gap": round(float(min_gap), 6),
            "ratio_min": 1.04,
            "train_loo_total": len(phrase_rows),
            "train_loo_correct": len(correct),
            "train_loo_accuracy": round(len(correct) / len(phrase_rows), 4) if phrase_rows else 0.0,
            "positive_dist_p95": round(float(np.percentile(np.array(positive_dists, dtype=np.float32), 95)), 6) if positive_dists else None,
            "wrong_best_count": len(wrong_best),
        }
    summary = v2_eval_summary(
        [
            {
                "truth_phrase_id": row.get("truth_phrase_id"),
                "predicted_phrase_id": row.get("best_phrase_id"),
                "best_phrase_id": row.get("best_phrase_id"),
                "second_phrase_id": row.get("second_phrase_id"),
                "reject_reason": "none",
            }
            for row in rows
        ]
    )
    return {
        "created_at": now_iso(),
        "method": "train_leave_one_out",
        "summary": summary,
        "thresholds_by_phrase": thresholds_by_phrase,
        "rows": rows,
    }


def load_v2_index_meta() -> Dict[str, Any]:
    if not V2_INDEX_META_PATH.exists():
        return {}
    try:
        payload = json.loads(V2_INDEX_META_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def load_v3_index_meta() -> Dict[str, Any]:
    if not V3_INDEX_META_PATH.exists():
        return {}
    try:
        payload = json.loads(V3_INDEX_META_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def v2_engine_debug(engine: str) -> Dict[str, Any]:
    return {
        "engine": engine,
        "engine_id": engine,
        "reject_reason": "none",
        "best_phrase": None,
        "best_phrase_id": None,
        "second_phrase": None,
        "second_phrase_id": None,
        "best_dist": None,
        "second_dist": None,
        "score": None,
        "margin": None,
        "gap": None,
        "ratio": None,
        "distance_threshold": None,
        "min_gap": None,
        "ratio_min": None,
        "candidates": [],
    }


def apply_v2_decision_gates(
    debug: Dict[str, Any],
    meta: Dict[str, Any],
    best: Dict[str, Any],
    second: Optional[Dict[str, Any]],
    default_threshold: float = 0.035,
    default_min_gap: float = 0.003,
    default_ratio_min: float = 1.06,
    threshold_key: str = "distance_threshold",
    min_gap_key: str = "min_gap",
    ratio_key: str = "ratio_min",
) -> Tuple[Optional[str], Dict[str, Any]]:
    best_dist = float(best["score"])
    second_dist = float(second["score"]) if second else float("inf")
    gap = second_dist - best_dist if np.isfinite(second_dist) else float("inf")
    ratio = second_dist / best_dist if best_dist > 1e-9 and np.isfinite(second_dist) else float("inf")
    thresholds = meta.get("thresholds", {}) if isinstance(meta, dict) else {}
    distance_threshold = float(thresholds.get(threshold_key, thresholds.get("distance_threshold", default_threshold)))
    min_gap = float(thresholds.get(min_gap_key, thresholds.get("min_gap", default_min_gap)))
    ratio_min = float(thresholds.get(ratio_key, thresholds.get("ratio_min", default_ratio_min)))
    thresholds_by_phrase = meta.get("thresholds_by_phrase", {}) if isinstance(meta, dict) else {}
    phrase_thresholds = thresholds_by_phrase.get(best["phrase_id"], {}) if isinstance(thresholds_by_phrase, dict) else {}
    if isinstance(phrase_thresholds, dict):
        distance_threshold = float(
            phrase_thresholds.get(threshold_key, phrase_thresholds.get("distance_threshold", distance_threshold))
        )
        min_gap = float(phrase_thresholds.get(min_gap_key, phrase_thresholds.get("min_gap", min_gap)))
        ratio_min = float(phrase_thresholds.get(ratio_key, phrase_thresholds.get("ratio_min", ratio_min)))

    debug.update(
        {
            "best_phrase": best["phrase_text"],
            "best_phrase_id": best["phrase_id"],
            "second_phrase": second["phrase_text"] if second else None,
            "second_phrase_id": second["phrase_id"] if second else None,
            "best_dist": round(best_dist, 6),
            "second_dist": safe_float(round(second_dist, 6) if np.isfinite(second_dist) else second_dist),
            "score": round(best_dist, 6),
            "margin": safe_float(round(gap, 6) if np.isfinite(gap) else gap),
            "gap": safe_float(round(gap, 6) if np.isfinite(gap) else gap),
            "ratio": safe_float(round(ratio, 6) if np.isfinite(ratio) else ratio),
            "distance_threshold": round(distance_threshold, 6),
            "phrase_threshold": round(distance_threshold, 6),
            "min_gap": round(min_gap, 6),
            "ratio_min": round(ratio_min, 6),
        }
    )
    if NO_REJECT_MODE:
        debug["reject_reason"] = "none"
        return str(best["phrase_text"]), debug
    if best_dist > distance_threshold:
        debug["reject_reason"] = "distance_too_high"
        return None, debug
    if second is not None and (gap < min_gap or ratio < ratio_min):
        debug["reject_reason"] = "separation_too_low"
        return None, debug
    debug["reject_reason"] = "none"
    return str(best["phrase_text"]), debug


def v2_match_audio_mfcc(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "engine": "mfcc_dtw_baseline",
        "engine_id": "mfcc_dtw_baseline",
        "reject_reason": "none",
        "best_phrase": None,
        "best_phrase_id": None,
        "second_phrase": None,
        "second_phrase_id": None,
        "best_dist": None,
        "second_dist": None,
        "gap": None,
        "ratio": None,
        "candidates": [],
    }
    templates = meta.get("templates", []) if isinstance(meta, dict) else []
    if not isinstance(templates, list) or not templates:
        debug["reject_reason"] = "index_not_ready"
        return None, debug

    query_wave = load_wave(audio_path)
    if query_wave is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug

    by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for tpl in templates:
        if not isinstance(tpl, dict):
            continue
        pid = tpl.get("phrase_id")
        rel = tpl.get("file_rel")
        if isinstance(pid, str) and isinstance(rel, str):
            by_phrase.setdefault(pid, []).append(tpl)

    phrase_scores: List[Dict[str, Any]] = []
    for pid, rows in by_phrase.items():
        dtw_vals: List[float] = []
        used: List[str] = []
        for tpl in rows:
            rel = tpl.get("file_rel")
            if not isinstance(rel, str):
                continue
            try:
                sample_path = v2_path_for_rel(rel)
            except ValueError:
                continue
            sample_wave = load_wave(sample_path)
            if sample_wave is None:
                continue
            dist = mfcc_dtw_distance_from_waves(query_wave, sample_wave)
            if np.isfinite(dist):
                dtw_vals.append(float(dist))
                used.append(rel)
        if not dtw_vals:
            continue
        dtw_vals.sort()
        top_vals = dtw_vals[: min(DTW_TOPK_PER_PHRASE, len(dtw_vals))]
        phrase_scores.append(
            {
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "score": round(float(np.mean(top_vals)), 6),
                "nearest": [round(v, 6) for v in top_vals],
                "template_count": len(rows),
                "used_samples": used[:DTW_TOPK_PER_PHRASE],
            }
        )

    if not phrase_scores:
        debug["reject_reason"] = "no_templates"
        return None, debug
    phrase_scores.sort(key=lambda row: float(row["score"]))
    debug["candidates"] = phrase_scores
    best = phrase_scores[0]
    second = phrase_scores[1] if len(phrase_scores) > 1 else None
    return apply_v2_decision_gates(debug, meta, best, second)


def v2_match_audio_mfcc_lenient(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    matched, debug = v2_match_audio_mfcc(audio_path, meta)
    debug["engine"] = MFCC_LENIENT_ENGINE_ID
    debug["engine_id"] = MFCC_LENIENT_ENGINE_ID
    if matched or debug.get("reject_reason") not in {"distance_too_high", "separation_too_low"}:
        return matched, debug
    best_id = debug.get("best_phrase_id")
    best_dist = debug.get("best_dist")
    gap = debug.get("gap")
    ratio = debug.get("ratio")
    if not isinstance(best_id, str) or not isinstance(best_dist, (int, float)):
        return None, debug
    # In the current stage pack, p03 is already stable under the strict gate.
    # Letting a rejected p03 win leniently mostly turns p02 clips into false accepts.
    if best_id == "p03_chi_fan":
        debug["lenient_blocked_reason"] = "p03_requires_strict_gate"
        return None, debug

    base_threshold = float(debug.get("distance_threshold") or 0.035)
    base_gap = float(debug.get("min_gap") or 0.003)
    relaxed_threshold = base_threshold + max(0.004, base_threshold * 0.22)
    relaxed_gap = max(0.0008, base_gap * 0.45)
    relaxed_ratio = 1.015
    gap_ok = not isinstance(gap, (int, float)) or float(gap) >= relaxed_gap
    ratio_ok = not isinstance(ratio, (int, float)) or float(ratio) >= relaxed_ratio
    dist_ok = float(best_dist) <= relaxed_threshold
    debug["lenient_threshold"] = round(relaxed_threshold, 6)
    debug["lenient_min_gap"] = round(relaxed_gap, 6)
    debug["lenient_ratio_min"] = relaxed_ratio
    if dist_ok and gap_ok and ratio_ok:
        debug["reject_reason"] = "none"
        debug["distance_threshold"] = round(relaxed_threshold, 6)
        debug["phrase_threshold"] = round(relaxed_threshold, 6)
        debug["min_gap"] = round(relaxed_gap, 6)
        debug["ratio_min"] = relaxed_ratio
        return str(debug.get("best_phrase") or PHRASE_ID_TO_TEXT.get(best_id, best_id)), debug
    return None, debug


def v2_match_audio_mfcc_safe(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    matched, debug = v2_match_audio_mfcc(audio_path, meta)
    debug["engine"] = MFCC_SAFE_ENGINE_ID
    debug["engine_id"] = MFCC_SAFE_ENGINE_ID
    if NO_REJECT_MODE and debug.get("best_phrase"):
        debug["reject_reason"] = "none"
        return str(debug.get("best_phrase")), debug
    if debug.get("reject_reason") != "none" or not matched:
        return None, debug

    best_dist = debug.get("best_dist")
    gap = debug.get("gap")
    ratio = debug.get("ratio")
    threshold = debug.get("distance_threshold", debug.get("phrase_threshold"))
    min_gap = debug.get("min_gap")
    ratio_min = debug.get("ratio_min")
    if not isinstance(best_dist, (int, float)):
        debug["reject_reason"] = "distance_too_high"
        return None, debug

    safe_threshold = min(float(threshold or 0.035), 0.0235)
    safe_min_gap = max(float(min_gap or 0.003), 0.0035)
    safe_ratio_min = max(float(ratio_min or 1.06), 1.12)
    debug["safe_threshold"] = round(safe_threshold, 6)
    debug["safe_min_gap"] = round(safe_min_gap, 6)
    debug["safe_ratio_min"] = round(safe_ratio_min, 6)
    debug["distance_threshold"] = round(safe_threshold, 6)
    debug["phrase_threshold"] = round(safe_threshold, 6)
    debug["min_gap"] = round(safe_min_gap, 6)
    debug["ratio_min"] = round(safe_ratio_min, 6)

    if float(best_dist) > safe_threshold:
        debug["reject_reason"] = "distance_too_high"
        debug["safe_reject_reason"] = "safe_distance_too_high"
        return None, debug
    if isinstance(gap, (int, float)) and float(gap) < safe_min_gap:
        debug["reject_reason"] = "separation_too_low"
        debug["safe_reject_reason"] = "safe_gap_too_low"
        return None, debug
    if isinstance(ratio, (int, float)) and float(ratio) < safe_ratio_min:
        debug["reject_reason"] = "separation_too_low"
        debug["safe_reject_reason"] = "safe_ratio_too_low"
        return None, debug
    debug["reject_reason"] = "none"
    return matched, debug


def build_ssl_phrase_models(meta: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[str]]:
    templates = meta.get("templates", []) if isinstance(meta, dict) else []
    if not isinstance(templates, list) or not templates:
        return {}, "index_not_ready"
    by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for tpl in templates:
        pid = tpl.get("phrase_id") if isinstance(tpl, dict) else None
        rel = tpl.get("file_rel") if isinstance(tpl, dict) else None
        if isinstance(pid, str) and isinstance(rel, str):
            try:
                path = v2_path_for_rel(rel)
            except ValueError:
                continue
            features = load_or_extract_ssl_features(path, rel)
            if features is None:
                continue
            by_phrase.setdefault(pid, []).append(
                {
                    **tpl,
                    "embedding": features["embedding"],
                    "sequence": features["sequence"],
                }
            )
    models: Dict[str, Any] = {}
    for pid, rows in by_phrase.items():
        embeddings = [row["embedding"] for row in rows if isinstance(row.get("embedding"), np.ndarray)]
        if not embeddings:
            continue
        stack = np.vstack(embeddings).astype(np.float32)
        mean_proto = l2_normalize(np.mean(stack, axis=0))
        center_d = [cosine_distance(v, mean_proto) for v in embeddings]
        keep_idx = np.argsort(np.array(center_d, dtype=np.float32))[: max(1, int(np.ceil(0.8 * len(embeddings))))]
        robust_proto = l2_normalize(np.mean(stack[keep_idx], axis=0))
        models[pid] = {
            "phrase_id": pid,
            "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
            "rows": rows,
            "mean_proto": mean_proto,
            "robust_proto": robust_proto,
            "proto_var": float(np.mean(center_d)) if center_d else 0.0,
            "count": len(rows),
        }
    if not models:
        return {}, _SSL_MODEL_ERROR or "ssl_features_unavailable"
    return models, None


def ssl_calibration_from_models(models: Dict[str, Any], engine: str) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    for pid, model in models.items():
        pos: List[float] = []
        neg: List[float] = []
        for row in model.get("rows", []):
            emb = row.get("embedding")
            if not isinstance(emb, np.ndarray):
                continue
            own_score = min(cosine_distance(emb, model["mean_proto"]), cosine_distance(emb, model["robust_proto"]))
            pos.append(float(own_score))
            for other_pid, other_model in models.items():
                if other_pid == pid:
                    continue
                other_score = min(cosine_distance(emb, other_model["mean_proto"]), cosine_distance(emb, other_model["robust_proto"]))
                neg.append(float(other_score))
        pos_stats = percentile_stats(pos)
        neg_stats = percentile_stats(neg)
        pos_p95 = pos_stats.get("p95") or pos_stats.get("max") or 0.25
        neg_p10 = neg_stats.get("p10")
        threshold = float(pos_p95) + 0.015
        if neg_p10 is not None:
            threshold = min(threshold, max(float(pos_p95) + 0.005, float(neg_p10) - 0.005))
        thresholds[pid] = {
            "distance_threshold": round(float(max(threshold, 0.005)), 6),
            "min_gap": round(float(max(0.002, 0.08 * max(threshold, 0.005))), 6),
            "ratio_min": 1.03 if engine == SSL_CLASSIFIER_ENGINE_ID else 1.04,
        }
    return thresholds


def v2_match_audio_ssl_proto(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug = v2_engine_debug(SSL_PROTO_ENGINE_ID)
    query = load_or_extract_ssl_features(audio_path)
    if query is None:
        debug["reject_reason"] = "ssl_unavailable"
        debug["error"] = _SSL_MODEL_ERROR or "failed to extract query SSL features"
        return None, debug
    models, error = build_ssl_phrase_models(meta)
    if error:
        debug["reject_reason"] = "index_not_ready" if error == "index_not_ready" else "ssl_unavailable"
        debug["error"] = error
        return None, debug
    candidates: List[Dict[str, Any]] = []
    q_emb = query["embedding"]
    for pid, model in models.items():
        score = min(cosine_distance(q_emb, model["mean_proto"]), cosine_distance(q_emb, model["robust_proto"]))
        candidates.append(
            {
                "phrase_id": pid,
                "phrase_text": model["phrase_text"],
                "score": round(float(score), 6),
                "proto_dist": round(float(score), 6),
                "template_count": model["count"],
            }
        )
    candidates.sort(key=lambda row: float(row["score"]))
    if not candidates:
        debug["reject_reason"] = "no_templates"
        return None, debug
    debug["candidates"] = candidates
    ssl_thresholds = ssl_calibration_from_models(models, SSL_PROTO_ENGINE_ID)
    ssl_meta = {**meta, "thresholds_by_phrase": ssl_thresholds, "thresholds": {"distance_threshold": 0.35, "min_gap": 0.01, "ratio_min": 1.04}}
    return apply_v2_decision_gates(debug, ssl_meta, candidates[0], candidates[1] if len(candidates) > 1 else None, default_threshold=0.35, default_min_gap=0.01, default_ratio_min=1.04)


def v2_match_audio_ssl_fusion(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug = v2_engine_debug(SSL_FUSION_ENGINE_ID)
    query = load_or_extract_ssl_features(audio_path)
    query_wave = load_wave(audio_path)
    if query is None or query_wave is None:
        debug["reject_reason"] = "ssl_unavailable" if query is None else "audio_too_short"
        debug["error"] = _SSL_MODEL_ERROR or "failed to extract query SSL features"
        return None, debug
    models, error = build_ssl_phrase_models(meta)
    if error:
        debug["reject_reason"] = "index_not_ready" if error == "index_not_ready" else "ssl_unavailable"
        debug["error"] = error
        return None, debug
    proto_rank: List[Tuple[str, float]] = []
    q_emb = query["embedding"]
    q_seq = query["sequence"]
    for pid, model in models.items():
        proto_rank.append((pid, min(cosine_distance(q_emb, model["mean_proto"]), cosine_distance(q_emb, model["robust_proto"]))))
    proto_rank.sort(key=lambda item: item[1])
    candidates: List[Dict[str, Any]] = []
    for pid, proto_dist in proto_rank[: min(3, len(proto_rank))]:
        model = models[pid]
        ssl_dtw_vals: List[float] = []
        mfcc_dtw_vals: List[float] = []
        used: List[str] = []
        ranked_rows = sorted(
            model.get("rows", []),
            key=lambda row: cosine_distance(q_emb, row["embedding"]) if isinstance(row.get("embedding"), np.ndarray) else float("inf"),
        )
        for row in ranked_rows[: min(PROTO_DTW_TOPK_PER_PHRASE, len(ranked_rows))]:
            rel = row.get("file_rel")
            seq = row.get("sequence")
            if not isinstance(rel, str) or not isinstance(seq, np.ndarray):
                continue
            try:
                sample_wave = load_wave(v2_path_for_rel(rel))
            except ValueError:
                sample_wave = None
            sd = ssl_sequence_dtw_distance(q_seq, seq)
            md = mfcc_dtw_distance_from_waves(query_wave, sample_wave) if sample_wave is not None else float("inf")
            if np.isfinite(sd):
                ssl_dtw_vals.append(float(sd))
            if np.isfinite(md):
                mfcc_dtw_vals.append(float(md))
            used.append(rel)
        ssl_dtw_vals.sort()
        mfcc_dtw_vals.sort()
        ssl_dtw = float(np.mean(ssl_dtw_vals[: min(PROTO_DTW_TOPK_PER_PHRASE, len(ssl_dtw_vals))])) if ssl_dtw_vals else float(proto_dist)
        mfcc_dtw = float(np.mean(mfcc_dtw_vals[: min(DTW_TOPK_PER_PHRASE, len(mfcc_dtw_vals))])) if mfcc_dtw_vals else 0.035
        final_score = 0.55 * float(proto_dist) + 0.30 * ssl_dtw + 0.15 * mfcc_dtw
        candidates.append(
            {
                "phrase_id": pid,
                "phrase_text": model["phrase_text"],
                "score": round(float(final_score), 6),
                "proto_dist": round(float(proto_dist), 6),
                "ssl_dtw": round(float(ssl_dtw), 6),
                "mfcc_dtw": round(float(mfcc_dtw), 6),
                "used_samples": used,
                "template_count": model["count"],
            }
        )
    candidates.sort(key=lambda row: float(row["score"]))
    if not candidates:
        debug["reject_reason"] = "no_templates"
        return None, debug
    debug["candidates"] = candidates
    fusion_thresholds = ssl_calibration_from_models(models, SSL_FUSION_ENGINE_ID)
    fusion_meta = {**meta, "thresholds_by_phrase": fusion_thresholds, "thresholds": {"distance_threshold": 0.35, "min_gap": 0.01, "ratio_min": 1.04}}
    return apply_v2_decision_gates(debug, fusion_meta, candidates[0], candidates[1] if len(candidates) > 1 else None, default_threshold=0.35, default_min_gap=0.01, default_ratio_min=1.04)


def v2_match_audio_ssl_classifier(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug = v2_engine_debug(SSL_CLASSIFIER_ENGINE_ID)
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # noqa: BLE001
        debug["reject_reason"] = "classifier_unavailable"
        debug["error"] = str(exc)
        return None, debug
    query = load_or_extract_ssl_features(audio_path)
    if query is None:
        debug["reject_reason"] = "ssl_unavailable"
        debug["error"] = _SSL_MODEL_ERROR or "failed to extract query SSL features"
        return None, debug
    templates = meta.get("templates", []) if isinstance(meta, dict) else []
    x_rows: List[np.ndarray] = []
    y_rows: List[str] = []
    for tpl in templates if isinstance(templates, list) else []:
        rel = tpl.get("file_rel") if isinstance(tpl, dict) else None
        pid = tpl.get("phrase_id") if isinstance(tpl, dict) else None
        if not isinstance(rel, str) or not isinstance(pid, str):
            continue
        try:
            features = load_or_extract_ssl_features(v2_path_for_rel(rel), rel)
        except ValueError:
            continue
        if features is None:
            continue
        x_rows.append(features["embedding"])
        y_rows.append(pid)
    if len(set(y_rows)) < 2 or len(x_rows) < 8:
        debug["reject_reason"] = "no_templates"
        return None, debug
    x = np.vstack(x_rows).astype(np.float32)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    clf.fit(x, np.array(y_rows))
    probs = clf.predict_proba(query["embedding"].reshape(1, -1))[0]
    rows = sorted(
        [
            {
                "phrase_id": str(pid),
                "phrase_text": PHRASE_ID_TO_TEXT.get(str(pid), str(pid)),
                "score": round(float(1.0 - prob), 6),
                "probability": round(float(prob), 6),
            }
            for pid, prob in zip(clf.classes_, probs)
        ],
        key=lambda row: float(row["score"]),
    )
    debug["candidates"] = rows
    if not rows:
        debug["reject_reason"] = "no_templates"
        return None, debug
    cls_meta = {
        **meta,
        "thresholds_by_phrase": {
            pid: {"distance_threshold": 0.68, "min_gap": 0.02, "ratio_min": 1.02}
            for pid in set(y_rows)
        },
        "thresholds": {"distance_threshold": 0.68, "min_gap": 0.02, "ratio_min": 1.02},
    }
    return apply_v2_decision_gates(debug, cls_meta, rows[0], rows[1] if len(rows) > 1 else None, default_threshold=0.68, default_min_gap=0.02, default_ratio_min=1.02)


def v2_match_audio_mfcc_classifier(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug = v2_engine_debug(MFCC_CLASSIFIER_ENGINE_ID)
    try:
        from sklearn.linear_model import LogisticRegression
    except Exception as exc:  # noqa: BLE001
        debug["reject_reason"] = "classifier_unavailable"
        debug["error"] = str(exc)
        return None, debug
    query_wave = load_wave(audio_path)
    if query_wave is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug
    q_emb = extract_embedding(query_wave)
    if q_emb is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug
    templates = meta.get("templates", []) if isinstance(meta, dict) else []
    x_rows: List[np.ndarray] = []
    y_rows: List[str] = []
    for tpl in templates if isinstance(templates, list) else []:
        rel = tpl.get("file_rel") if isinstance(tpl, dict) else None
        pid = tpl.get("phrase_id") if isinstance(tpl, dict) else None
        if not isinstance(rel, str) or not isinstance(pid, str):
            continue
        try:
            wave_y = load_wave(v2_path_for_rel(rel))
        except ValueError:
            continue
        if wave_y is None:
            continue
        emb = extract_embedding(wave_y)
        if emb is None:
            continue
        x_rows.append(emb)
        y_rows.append(pid)
    if len(set(y_rows)) < 2 or len(x_rows) < 8:
        debug["reject_reason"] = "no_templates"
        return None, debug
    x = np.vstack(x_rows).astype(np.float32)
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0) + 1e-6
    x_norm = (x - mean) / std
    q_norm = ((q_emb.astype(np.float32) - mean) / std).reshape(1, -1)
    clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=7)
    clf.fit(x_norm, np.array(y_rows))
    probs = clf.predict_proba(q_norm)[0]
    rows = sorted(
        [
            {
                "phrase_id": str(pid),
                "phrase_text": PHRASE_ID_TO_TEXT.get(str(pid), str(pid)),
                "score": round(float(1.0 - prob), 6),
                "probability": round(float(prob), 6),
            }
            for pid, prob in zip(clf.classes_, probs)
        ],
        key=lambda row: float(row["score"]),
    )
    debug["candidates"] = rows
    if not rows:
        debug["reject_reason"] = "no_templates"
        return None, debug
    cls_meta = {
        **meta,
        "thresholds_by_phrase": {
            pid: {"distance_threshold": 0.74, "min_gap": 0.015, "ratio_min": 1.02}
            for pid in set(y_rows)
        },
        "thresholds": {"distance_threshold": 0.74, "min_gap": 0.015, "ratio_min": 1.02},
    }
    return apply_v2_decision_gates(debug, cls_meta, rows[0], rows[1] if len(rows) > 1 else None, default_threshold=0.74, default_min_gap=0.015, default_ratio_min=1.02)


def load_active_v2_engine() -> Dict[str, Any]:
    ensure_v2_storage()
    if not V2_ACTIVE_ENGINE_PATH.exists():
        payload = {"engine_id": ENGINE_V3_PERSONALIZED_TOP1, "updated_at": now_iso(), "reason": "default_v3_personalized_top1"}
        save_json(V2_ACTIVE_ENGINE_PATH, payload)
        return payload
    try:
        payload = json.loads(V2_ACTIVE_ENGINE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("engine_id") in V2_ENGINE_IDS:
            return payload
    except Exception:
        pass
    payload = {"engine_id": ENGINE_V3_PERSONALIZED_TOP1, "updated_at": now_iso(), "reason": "invalid_active_engine_reset"}
    save_json(V2_ACTIVE_ENGINE_PATH, payload)
    return payload


def load_active_v3_engine() -> Dict[str, Any]:
    ensure_v3_storage()
    if not V3_ACTIVE_ENGINE_PATH.exists():
        payload = {"engine_id": ENGINE_V3_PERSONALIZED_TOP1, "updated_at": now_iso(), "reason": "default_v3_personalized_top1"}
        save_json(V3_ACTIVE_ENGINE_PATH, payload)
        return payload
    try:
        payload = json.loads(V3_ACTIVE_ENGINE_PATH.read_text(encoding="utf-8"))
        if isinstance(payload, dict) and payload.get("engine_id") in V2_ENGINE_IDS:
            return payload
    except Exception:
        pass
    payload = {"engine_id": ENGINE_V3_PERSONALIZED_TOP1, "updated_at": now_iso(), "reason": "invalid_active_engine_reset"}
    save_json(V3_ACTIVE_ENGINE_PATH, payload)
    return payload


def v2_match_audio_engine_v3(audio_path: Path, meta: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    # Prefer SSL fusion (best separability for personalized small-sample phrases),
    # then degrade gracefully to SSL proto and MFCC-safe fallback.
    debug: Dict[str, Any] = {}
    debug2: Dict[str, Any] = {}
    if V3_SSL_ENABLED:
        matched, debug = v2_match_audio_ssl_fusion(audio_path, meta)
        if matched and debug.get("best_phrase"):
            debug["engine"] = ENGINE_V3_PERSONALIZED_TOP1
            debug["engine_id"] = ENGINE_V3_PERSONALIZED_TOP1
            debug["top1_score"] = debug.get("score", debug.get("best_dist"))
            return matched, debug

        matched2, debug2 = v2_match_audio_ssl_proto(audio_path, meta)
        if matched2 and debug2.get("best_phrase"):
            debug2["engine"] = ENGINE_V3_PERSONALIZED_TOP1
            debug2["engine_id"] = ENGINE_V3_PERSONALIZED_TOP1
            debug2["top1_score"] = debug2.get("score", debug2.get("best_dist"))
            debug2["fallback_from"] = "ssl_fusion_v1"
            return matched2, debug2

    matched3, debug3 = v2_match_audio_mfcc_safe(audio_path, meta)
    if matched3 and debug3.get("best_phrase"):
        debug3["engine"] = ENGINE_V3_PERSONALIZED_TOP1
        debug3["engine_id"] = ENGINE_V3_PERSONALIZED_TOP1
        debug3["top1_score"] = debug3.get("score", debug3.get("best_dist"))
        debug3["fallback_from"] = "ssl_proto_v1"
        return matched3, debug3

    # Last-resort force from best candidate if debug contains a phrase.
    for dbg in (debug, debug2, debug3):
        best_phrase = dbg.get("best_phrase")
        if isinstance(best_phrase, str) and best_phrase:
            dbg["engine"] = ENGINE_V3_PERSONALIZED_TOP1
            dbg["engine_id"] = ENGINE_V3_PERSONALIZED_TOP1
            dbg["reject_reason"] = "none"
            dbg["top1_score"] = dbg.get("score", dbg.get("best_dist"))
            if not V3_SSL_ENABLED:
                dbg["fallback_from"] = "ssl_disabled_cpu_mode"
            return best_phrase, dbg
    fail_debug = v2_engine_debug(ENGINE_V3_PERSONALIZED_TOP1)
    fail_debug["reject_reason"] = "no_templates"
    fail_debug["error"] = "no candidate phrase available"
    return None, fail_debug


def v2_match_audio(audio_path: Path, engine: Optional[str] = None) -> Tuple[Optional[str], Dict[str, Any]]:
    meta = load_v2_index_meta()
    engine_id = engine or str(load_active_v2_engine().get("engine_id") or "mfcc_dtw_baseline")
    if engine_id == ENGINE_V3_PERSONALIZED_TOP1:
        return v2_match_audio_engine_v3(audio_path, meta)
    if engine_id == "mfcc_dtw_baseline":
        return v2_match_audio_mfcc(audio_path, meta)
    if engine_id == MFCC_SAFE_ENGINE_ID:
        return v2_match_audio_mfcc_safe(audio_path, meta)
    if engine_id == MFCC_LENIENT_ENGINE_ID:
        return v2_match_audio_mfcc_lenient(audio_path, meta)
    if engine_id == MFCC_CLASSIFIER_ENGINE_ID:
        return v2_match_audio_mfcc_classifier(audio_path, meta)
    if engine_id == SSL_PROTO_ENGINE_ID:
        return v2_match_audio_ssl_proto(audio_path, meta)
    if engine_id == SSL_FUSION_ENGINE_ID:
        return v2_match_audio_ssl_fusion(audio_path, meta)
    if engine_id == SSL_CLASSIFIER_ENGINE_ID:
        return v2_match_audio_ssl_classifier(audio_path, meta)
    debug = v2_engine_debug(engine_id)
    debug["reject_reason"] = "engine_unavailable"
    debug["error"] = f"unknown engine: {engine_id}"
    return None, debug


def v2_store_rejected(
    data: bytes,
    reason: str,
    phrase_id: Optional[str] = None,
    truth_phrase_id: Optional[str] = None,
    source: str = "unknown",
    quality: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    ensure_v2_storage()
    target_pid = phrase_id or truth_phrase_id or "unknown"
    safe_pid = target_pid if target_pid in PHRASE_ID_TO_TEXT else "unknown"
    target_dir = DATA_V2_DIR / "rejected" / safe_pid
    target_dir.mkdir(parents=True, exist_ok=True)
    rejected_id = uuid4().hex
    path = target_dir / f"rejected_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{rejected_id[:8]}.wav"
    path.write_bytes(data)
    rel = path.relative_to(ROOT_DIR).as_posix()
    row = {
        "sample_id": rejected_id,
        "phrase_id": phrase_id,
        "truth_phrase_id": truth_phrase_id,
        "split": "rejected",
        "source": source,
        "file_rel": rel,
        "reason": reason,
        "quality": quality or analyze_wav_bytes(data),
        "created_at": now_iso(),
    }
    manifest = load_v2_manifest()
    manifest["rejected"].append(row)
    save_v2_manifest(manifest)
    return row


def v2_store_audio_sample(
    data: bytes,
    phrase_id: str,
    split: str,
    source: str,
    allow_duplicate: bool = False,
    speaker_id: str = "speaker_default",
) -> Tuple[bool, Dict[str, Any]]:
    if phrase_id not in PHRASE_ID_TO_TEXT:
        return False, {"ok": False, "error": "invalid phrase_id"}
    if split not in {"train", "eval", "corrections"}:
        return False, {"ok": False, "error": "invalid split"}

    manifest = load_v2_manifest()
    quality = analyze_wav_bytes(data, None if allow_duplicate else existing_v2_hashes(manifest))
    if not quality.get("ok"):
        rejected = v2_store_rejected(
            data,
            reason="quality_gate_failed",
            phrase_id=phrase_id if split == "train" else None,
            truth_phrase_id=phrase_id if split in {"eval", "corrections"} else None,
            source=source,
            quality=quality,
        )
        return False, {"ok": False, "error": "quality_gate_failed", "quality": quality, "rejected": rejected}

    sample_id = uuid4().hex
    target_dir = DATA_V2_DIR / split / phrase_id
    target_dir.mkdir(parents=True, exist_ok=True)
    prefix = "sample" if split == "train" else ("eval" if split == "eval" else "correction")
    path = target_dir / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample_id[:8]}.wav"
    path.write_bytes(data)
    rel = path.relative_to(ROOT_DIR).as_posix()
    row = {
        "sample_id": sample_id,
        "speaker_id": speaker_id or "speaker_default",
        "phrase_id": phrase_id,
        "truth_phrase_id": phrase_id if split in {"eval", "corrections"} else None,
        "phrase_text": PHRASE_ID_TO_TEXT[phrase_id],
        "split": split,
        "source": source,
        "file_rel": rel,
        "duration_sec": quality["duration_sec"],
        "quality_flags": quality["quality_flags"],
        "warning_flags": quality.get("warning_flags", []),
        "sha256": quality["sha256"],
        "status": "active",
        "disabled_reason": None,
        "disabled_at": None,
        "quality_score": None,
        "suspicion_score": None,
        "purify_signals": [],
        "created_at": now_iso(),
    }
    if split == "train":
        row["quality_score"] = v2_sample_quality_score(row)
    manifest = load_v2_manifest()
    if split == "train":
        manifest["samples"].append(row)
    elif split == "eval":
        # The caller appends the full eval event after prediction.
        pass
    else:
        manifest["corrections"].append(row)
    save_v2_manifest(manifest)
    return True, {"ok": True, "sample": row, "quality": quality}


def v2_process_demo_audio(data: bytes) -> Dict[str, Any]:
    quality = analyze_wav_bytes(data)
    if not quality.get("ok"):
        raw_path = save_media_file("raw", data, "wav")
        return {
            "ok": False,
            "error": "quality_gate_failed",
            "quality": quality,
            "raw_audio_url": f"/audio/{raw_path.name}",
            "matched_phrase": None,
            "tts_audio_url": None,
            "audio_match_debug": {"reject_reason": "audio_too_short" if "too_short" in quality.get("quality_flags", []) else "quality_gate_failed"},
        }
    raw_path = save_media_file("raw", data, "wav")
    started = time.perf_counter()
    matched_phrase, debug = v2_match_audio(raw_path)
    result: Dict[str, Any] = {
        "ok": False,
        "quality": quality,
        "raw_audio_url": f"/audio/{raw_path.name}",
        "matched_phrase": matched_phrase,
        "tts_audio_url": None,
        "audio_match_debug": debug,
        "engine_id": debug.get("engine_id", debug.get("engine")),
        "top1_score": debug.get("score", debug.get("best_dist")),
        "score": debug.get("score", debug.get("best_dist")),
        "margin": debug.get("margin", debug.get("gap")),
        "ratio": debug.get("ratio"),
        "model_report_ref": load_active_v2_engine().get("selected_from_report"),
        "latency_ms": {},
    }
    if not matched_phrase:
        result["latency_ms"]["release_to_verdict"] = round((time.perf_counter() - started) * 1000.0, 2)
        return result
    tts_path = save_media_file("tts", b"", "mp3")
    tts_start = time.perf_counter()
    try:
        run_tts_sync(matched_phrase, tts_path)
    except Exception as exc:
        tts_path.unlink(missing_ok=True)
        result["error"] = f"tts failed: {exc}"
        result["latency_ms"]["release_to_verdict"] = round((tts_start - started) * 1000.0, 2)
        return result
    result["ok"] = True
    result["tts_audio_url"] = f"/audio/{tts_path.name}"
    result["latency_ms"]["release_to_verdict"] = round((tts_start - started) * 1000.0, 2)
    result["latency_ms"]["release_to_tts"] = round((time.perf_counter() - started) * 1000.0, 2)
    return result


def v2_append_eval_event(sample_row: Dict[str, Any], matched_phrase: Optional[str], debug: Dict[str, Any]) -> Dict[str, Any]:
    manifest = load_v2_manifest()
    truth_phrase_id = str(sample_row["truth_phrase_id"])
    predicted_phrase_id = debug.get("best_phrase_id")
    if not isinstance(predicted_phrase_id, str):
        predicted_phrase_id = None
    predicted_text = matched_phrase if isinstance(matched_phrase, str) and matched_phrase else None
    if predicted_text is None and predicted_phrase_id:
        predicted_text = PHRASE_ID_TO_TEXT.get(predicted_phrase_id, predicted_phrase_id)
    event = {
        "event_id": uuid4().hex,
        "split": "eval",
        "source": sample_row.get("source", "eval_upload"),
        "truth_phrase_id": truth_phrase_id,
        "truth_text": PHRASE_ID_TO_TEXT.get(truth_phrase_id, truth_phrase_id),
        "predicted_phrase_id": predicted_phrase_id,
        "best_phrase_id": debug.get("best_phrase_id"),
        "predicted_text": predicted_text,
        "best_text": debug.get("best_phrase"),
        "second_phrase_id": debug.get("second_phrase_id"),
        "second_text": debug.get("second_phrase"),
        "reject_reason": debug.get("reject_reason"),
        "best_dist": debug.get("best_dist"),
        "top1_score": debug.get("score", debug.get("best_dist")),
        "score": debug.get("score", debug.get("best_dist")),
        "second_dist": debug.get("second_dist"),
        "margin": debug.get("margin", debug.get("gap")),
        "gap": debug.get("gap"),
        "ratio": debug.get("ratio"),
        "engine": debug.get("engine"),
        "engine_id": debug.get("engine_id", debug.get("engine")),
        "file_rel": sample_row.get("file_rel"),
        "raw_audio_url": sample_row.get("raw_audio_url"),
        "duration_sec": sample_row.get("duration_sec"),
        "quality_flags": sample_row.get("quality_flags", []),
        "sha256": sample_row.get("sha256"),
        "created_at": now_iso(),
    }
    manifest["events"].append(event)
    save_v2_manifest(manifest)
    return event


def v2_eval_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    accepted = [e for e in events if e.get("reject_reason") == "none"]
    top1 = [e for e in accepted if e.get("predicted_phrase_id") == e.get("truth_phrase_id")]
    top2 = [
        e
        for e in events
        if e.get("truth_phrase_id") in {e.get("best_phrase_id"), e.get("second_phrase_id")}
    ]
    reasons: Dict[str, int] = {}
    per_phrase: Dict[str, Dict[str, int]] = {}
    matrix: Dict[str, Dict[str, int]] = {}
    for event in events:
        truth = str(event.get("truth_phrase_id") or "unknown")
        pred = str(event.get("predicted_phrase_id") or "MISS")
        reason = str(event.get("reject_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        per_phrase.setdefault(truth, {"total": 0, "top1": 0, "top2": 0, "miss": 0})
        per_phrase[truth]["total"] += 1
        if pred == truth and reason == "none":
            per_phrase[truth]["top1"] += 1
        if truth in {event.get("best_phrase_id"), event.get("second_phrase_id")}:
            per_phrase[truth]["top2"] += 1
        if reason != "none":
            per_phrase[truth]["miss"] += 1
        matrix.setdefault(truth, {})
        matrix[truth][pred] = matrix[truth].get(pred, 0) + 1

    return {
        "total": total,
        "accepted": len(accepted),
        "top1": len(top1),
        "top2": len(top2),
        "top1_rate": round(len(top1) / total, 4) if total else 0.0,
        "top2_rate": round(len(top2) / total, 4) if total else 0.0,
        "reject_rate": round((total - len(accepted)) / total, 4) if total else 0.0,
        "failure_reasons": reasons,
        "per_phrase": per_phrase,
        "confusion_matrix": matrix,
    }


def v2_unknown_summary(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(events)
    accepted = [e for e in events if e.get("predicted_phrase_id")]
    reasons: Dict[str, int] = {}
    accepted_by_phrase: Dict[str, int] = {}
    examples: List[Dict[str, Any]] = []
    for event in events:
        reason = str(event.get("reject_reason") or "unknown")
        reasons[reason] = reasons.get(reason, 0) + 1
        pred = event.get("predicted_phrase_id")
        if isinstance(pred, str):
            accepted_by_phrase[pred] = accepted_by_phrase.get(pred, 0) + 1
            if len(examples) < 12:
                examples.append(
                    {
                        "predicted_phrase_id": pred,
                        "predicted_text": event.get("predicted_text"),
                        "best_phrase_id": event.get("best_phrase_id"),
                        "second_phrase_id": event.get("second_phrase_id"),
                        "score": event.get("score", event.get("best_dist")),
                        "margin": event.get("margin", event.get("gap")),
                        "ratio": event.get("ratio"),
                        "file_rel": event.get("file_rel"),
                    }
                )
    return {
        "total": total,
        "false_accepts": len(accepted),
        "correct_rejects": total - len(accepted),
        "false_accept_rate": round(len(accepted) / total, 4) if total else 0.0,
        "failure_reasons": reasons,
        "accepted_by_phrase": accepted_by_phrase,
        "false_accept_examples": examples,
    }


def v2_top1_score_stats(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    all_scores: List[float] = []
    per_phrase: Dict[str, List[float]] = {}
    for event in events:
        score = event.get("top1_score", event.get("score", event.get("best_dist")))
        pid = event.get("best_phrase_id")
        if isinstance(score, (int, float)) and np.isfinite(float(score)):
            sval = float(score)
            all_scores.append(sval)
            if isinstance(pid, str):
                per_phrase.setdefault(pid, []).append(sval)
    out: Dict[str, Any] = {"all": percentile_stats(all_scores), "per_phrase": {}}
    for pid, vals in per_phrase.items():
        out["per_phrase"][pid] = {
            "phrase_id": pid,
            "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
            "count": len(vals),
            "stats": percentile_stats(vals),
        }
    return out


def v2_confusion_pairs(events: List[Dict[str, Any]], top_n: int = 8) -> List[Dict[str, Any]]:
    pairs: Dict[str, int] = {}
    for event in events:
        truth = event.get("truth_phrase_id")
        pred = event.get("predicted_phrase_id")
        if not isinstance(truth, str) or not isinstance(pred, str):
            continue
        if truth == pred:
            continue
        key = f"{truth}->{pred}"
        pairs[key] = pairs.get(key, 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    rows: List[Dict[str, Any]] = []
    for key, count in ranked:
        truth, pred = key.split("->", 1)
        rows.append(
            {
                "truth_phrase_id": truth,
                "truth_text": PHRASE_ID_TO_TEXT.get(truth, truth),
                "pred_phrase_id": pred,
                "pred_text": PHRASE_ID_TO_TEXT.get(pred, pred),
                "count": count,
            }
        )
    return rows


def eval_error_diagnosis(event: Dict[str, Any]) -> Dict[str, str]:
    truth = event.get("truth_phrase_id")
    pred = event.get("predicted_phrase_id")
    best = event.get("best_phrase_id")
    second = event.get("second_phrase_id")
    reason = str(event.get("reject_reason") or "unknown")
    score = event.get("score", event.get("best_dist"))
    margin = event.get("margin", event.get("gap"))
    ratio = event.get("ratio")
    threshold = event.get("phrase_threshold", event.get("distance_threshold"))

    if reason == "none" and isinstance(pred, str) and pred == truth:
        return {
            "diagnosis_code": "accepted_correct",
            "diagnosis_text": "Prediction matches truth and was accepted.",
            "suggested_fix": "No action needed.",
        }
    if reason == "none" and isinstance(pred, str) and pred != truth:
        return {
            "diagnosis_code": "accepted_wrong",
            "diagnosis_text": "Wrong phrase passed acceptance gate.",
            "suggested_fix": "Add 3 truth clips and 2 contrast clips against predicted phrase, then confirm corrections.",
        }
    if reason in {"distance_too_high", "audio_too_short"}:
        detail = f"score={safe_float(float(score))}" if isinstance(score, (int, float)) else "score unavailable"
        if isinstance(threshold, (int, float)):
            detail += f", threshold={safe_float(float(threshold))}"
        return {
            "diagnosis_code": "distance_too_high",
            "diagnosis_text": f"Best candidate is too far from known templates ({detail}).",
            "suggested_fix": "Re-record clearer truth audio (3 clips) and add 1-2 hard negatives.",
        }
    if reason == "separation_too_low":
        detail = []
        if isinstance(margin, (int, float)):
            detail.append(f"margin={safe_float(float(margin))}")
        if isinstance(ratio, (int, float)):
            detail.append(f"ratio={safe_float(float(ratio))}")
        detail_text = ", ".join(detail) if detail else "low separation"
        if isinstance(second, str) and second == truth:
            return {
                "diagnosis_code": "truth_second",
                "diagnosis_text": f"Truth is second candidate but separation is too small ({detail_text}).",
                "suggested_fix": "Add 3 truth clips + 2 contrast clips against top-1 confused phrase.",
            }
        return {
            "diagnosis_code": "separation_too_low",
            "diagnosis_text": f"Top candidates are too close to separate reliably ({detail_text}).",
            "suggested_fix": "Collect directional contrast clips for this confusion pair and rebuild index.",
        }
    if reason in {"index_not_ready", "no_templates"}:
        return {
            "diagnosis_code": "index_not_ready",
            "diagnosis_text": "Index/template set is not ready for reliable matching.",
            "suggested_fix": "Collect train samples then rebuild index before evaluation.",
        }
    if reason != "none":
        return {
            "diagnosis_code": "open_set_reject",
            "diagnosis_text": f"Sample rejected by gate ({reason}).",
            "suggested_fix": "Review audio quality and add truth-specific samples before retraining.",
        }
    if isinstance(best, str) and best != truth:
        return {
            "diagnosis_code": "confusion_wrong_best",
            "diagnosis_text": "Best candidate is consistently a wrong phrase.",
            "suggested_fix": "Add contrast set between truth and wrong-best phrase, then rerun fresh eval.",
        }
    return {
        "diagnosis_code": "unknown",
        "diagnosis_text": "Unable to classify failure mode from current event fields.",
        "suggested_fix": "Inspect audio and debug fields manually, then add targeted corrections.",
    }


def build_v3_eval_error_payload(events: List[Dict[str, Any]], low_margin_threshold: Optional[float] = None) -> Dict[str, Any]:
    threshold = VB_FALLBACK_MARGIN_THRESHOLD if low_margin_threshold is None else float(low_margin_threshold)
    items: List[Dict[str, Any]] = []
    confusion_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    per_phrase: Dict[str, Dict[str, int]] = {}
    low_margin_cases: List[Dict[str, Any]] = []
    accepted_wrong_cases: List[Dict[str, Any]] = []

    for ev in events:
        if not isinstance(ev, dict):
            continue
        truth_id = str(ev.get("truth_phrase_id") or "")
        pred_id = str(ev.get("predicted_phrase_id") or "") if isinstance(ev.get("predicted_phrase_id"), str) else None
        best_id = str(ev.get("best_phrase_id") or "") if isinstance(ev.get("best_phrase_id"), str) else None
        second_id = str(ev.get("second_phrase_id") or "") if isinstance(ev.get("second_phrase_id"), str) else None
        reason = str(ev.get("reject_reason") or "unknown")
        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        per_phrase.setdefault(truth_id or "unknown", {"total": 0, "errors": 0})
        per_phrase[truth_id or "unknown"]["total"] += 1

        text_truth = ev.get("truth_text") if isinstance(ev.get("truth_text"), str) else None
        text_pred = ev.get("predicted_text") if isinstance(ev.get("predicted_text"), str) else None
        text_mismatch = isinstance(text_truth, str) and isinstance(text_pred, str) and text_truth != text_pred
        is_error = reason != "none" or (pred_id is not None and pred_id != truth_id) or text_mismatch
        if is_error:
            per_phrase[truth_id or "unknown"]["errors"] += 1

        if truth_id and pred_id and truth_id != pred_id:
            key = f"{truth_id}->{pred_id}"
            confusion_counts[key] = confusion_counts.get(key, 0) + 1

        margin = ev.get("margin", ev.get("gap"))
        ratio = ev.get("ratio")
        if isinstance(margin, (int, float)) and float(margin) < threshold:
            low_margin_cases.append(
                {
                    "event_id": ev.get("event_id"),
                    "truth_phrase_id": truth_id,
                    "predicted_phrase_id": pred_id,
                    "margin": safe_float(float(margin)),
                    "ratio": safe_float(float(ratio)) if isinstance(ratio, (int, float)) else None,
                    "reject_reason": reason,
                }
            )

        if reason == "none" and pred_id and pred_id != truth_id:
            accepted_wrong_cases.append(
                {
                    "event_id": ev.get("event_id"),
                    "truth_phrase_id": truth_id,
                    "predicted_phrase_id": pred_id,
                    "best_phrase_id": best_id,
                    "second_phrase_id": second_id,
                    "score": ev.get("score", ev.get("best_dist")),
                    "margin": margin,
                    "ratio": ratio,
                }
            )

        diagnosis = eval_error_diagnosis(ev)
        file_rel = ev.get("file_rel")
        audio_url = ev.get("raw_audio_url")
        if not isinstance(audio_url, str) or not audio_url:
            audio_url = f"/api/v3/audio?file_rel={file_rel}" if isinstance(file_rel, str) and file_rel else None
        elif audio_url.startswith("/api/v2/audio"):
            audio_url = audio_url.replace("/api/v2/audio", "/api/v3/audio")
        item = {
            "event_id": ev.get("event_id"),
            "truth_phrase_id": truth_id or None,
            "truth_text": ev.get("truth_text") or PHRASE_ID_TO_TEXT.get(truth_id, truth_id),
            "predicted_phrase_id": pred_id,
            "predicted_text": ev.get("predicted_text"),
            "best_phrase_id": best_id,
            "best_text": ev.get("best_text"),
            "second_phrase_id": second_id,
            "second_text": ev.get("second_text"),
            "score": ev.get("score", ev.get("best_dist")),
            "margin": margin,
            "gap": ev.get("gap"),
            "ratio": ratio,
            "reject_reason": reason,
            "diagnosis_code": diagnosis["diagnosis_code"],
            "diagnosis_text": diagnosis["diagnosis_text"],
            "suggested_fix": diagnosis["suggested_fix"],
            "is_error": bool(is_error),
            "file_rel": file_rel,
            "audio_url": audio_url,
            "engine_id": ev.get("engine_id", ev.get("engine")),
            "created_at": ev.get("created_at"),
        }
        items.append(item)

    confusion_pairs: List[Dict[str, Any]] = []
    for pair, count in sorted(confusion_counts.items(), key=lambda kv: kv[1], reverse=True):
        truth_id, pred_id = pair.split("->", 1)
        confusion_pairs.append(
            {
                "truth_phrase_id": truth_id,
                "truth_text": PHRASE_ID_TO_TEXT.get(truth_id, truth_id),
                "pred_phrase_id": pred_id,
                "pred_text": PHRASE_ID_TO_TEXT.get(pred_id, pred_id),
                "count": count,
            }
        )

    per_phrase_error_rate: List[Dict[str, Any]] = []
    for pid, stat in sorted(per_phrase.items(), key=lambda kv: kv[1]["errors"], reverse=True):
        total = int(stat.get("total") or 0)
        errors = int(stat.get("errors") or 0)
        per_phrase_error_rate.append(
            {
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "total": total,
                "errors": errors,
                "error_rate": safe_div(errors, total),
            }
        )

    analysis = {
        "confusion_pairs": confusion_pairs,
        "reason_counts": reason_counts,
        "error_count": sum(1 for row in items if bool(row.get("is_error"))),
        "per_phrase_error_rate": per_phrase_error_rate,
        "low_margin_cases": sorted(
            low_margin_cases,
            key=lambda row: float(row.get("margin")) if isinstance(row.get("margin"), (int, float)) else 999.0,
        ),
        "accepted_wrong_cases": accepted_wrong_cases,
    }
    return {"items": items, "analysis": analysis}


def v2_error_buckets(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    buckets: Dict[str, int] = {
        "accepted_correct": 0,
        "best_correct_rejected": 0,
        "accepted_wrong": 0,
        "truth_second": 0,
        "truth_not_top2": 0,
    }
    examples: Dict[str, List[Dict[str, Any]]] = {key: [] for key in buckets}
    for event in events:
        truth = event.get("truth_phrase_id")
        best = event.get("best_phrase_id")
        second = event.get("second_phrase_id")
        predicted = event.get("predicted_phrase_id")
        reason = event.get("reject_reason")
        if reason == "none" and predicted == truth:
            bucket = "accepted_correct"
        elif reason == "none" and predicted != truth:
            bucket = "accepted_wrong"
        elif best == truth:
            bucket = "best_correct_rejected"
        elif second == truth:
            bucket = "truth_second"
        else:
            bucket = "truth_not_top2"
        buckets[bucket] += 1
        if len(examples[bucket]) < 12:
            examples[bucket].append(
                {
                    "truth_phrase_id": truth,
                    "best_phrase_id": best,
                    "second_phrase_id": second,
                    "reject_reason": reason,
                    "score": event.get("score", event.get("best_dist")),
                    "margin": event.get("margin", event.get("gap")),
                    "ratio": event.get("ratio"),
                    "file_rel": event.get("file_rel"),
                }
            )
    return {"counts": buckets, "examples": examples}


def v2_store_unknown_audio(data: bytes, source: str = "unknown_capture") -> Tuple[bool, Dict[str, Any]]:
    manifest = load_v2_manifest()
    quality = analyze_wav_bytes(data, existing_v2_hashes(manifest))
    if not quality.get("ok"):
        rejected = v2_store_rejected(
            data,
            reason="unknown_quality_gate_failed",
            phrase_id=None,
            truth_phrase_id=None,
            source=source,
            quality=quality,
        )
        return False, {"ok": False, "error": "quality_gate_failed", "quality": quality, "rejected": rejected}
    unknown_id = uuid4().hex
    V2_UNKNOWN_DIR.mkdir(parents=True, exist_ok=True)
    path = V2_UNKNOWN_DIR / f"unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unknown_id[:8]}.wav"
    path.write_bytes(data)
    rel = path.relative_to(ROOT_DIR).as_posix()
    row = {
        "sample_id": unknown_id,
        "split": "unknown",
        "source": source,
        "file_rel": rel,
        "duration_sec": quality["duration_sec"],
        "quality_flags": quality["quality_flags"],
        "sha256": quality["sha256"],
        "created_at": now_iso(),
    }
    return True, {"ok": True, "sample": row, "quality": quality}


def v2_unknown_event(sample_row: Dict[str, Any], matched_phrase: Optional[str], debug: Dict[str, Any]) -> Dict[str, Any]:
    predicted_phrase_id = debug.get("best_phrase_id")
    if not isinstance(predicted_phrase_id, str):
        predicted_phrase_id = None
    predicted_text = matched_phrase if isinstance(matched_phrase, str) and matched_phrase else None
    if predicted_text is None and predicted_phrase_id:
        predicted_text = PHRASE_ID_TO_TEXT.get(predicted_phrase_id, predicted_phrase_id)
    return {
        "event_id": uuid4().hex,
        "split": "unknown",
        "source": sample_row.get("source", "unknown_capture"),
        "predicted_phrase_id": predicted_phrase_id,
        "best_phrase_id": debug.get("best_phrase_id"),
        "predicted_text": predicted_text,
        "best_text": debug.get("best_phrase"),
        "second_phrase_id": debug.get("second_phrase_id"),
        "second_text": debug.get("second_phrase"),
        "reject_reason": debug.get("reject_reason"),
        "best_dist": debug.get("best_dist"),
        "top1_score": debug.get("score", debug.get("best_dist")),
        "score": debug.get("score", debug.get("best_dist")),
        "second_dist": debug.get("second_dist"),
        "margin": debug.get("margin", debug.get("gap")),
        "gap": debug.get("gap"),
        "ratio": debug.get("ratio"),
        "engine": debug.get("engine"),
        "engine_id": debug.get("engine_id", debug.get("engine")),
        "file_rel": sample_row.get("file_rel"),
        "duration_sec": sample_row.get("duration_sec"),
        "quality_flags": sample_row.get("quality_flags", []),
        "sha256": sample_row.get("sha256"),
        "created_at": now_iso(),
    }


def v2_append_unknown_event(sample_row: Dict[str, Any], matched_phrase: Optional[str], debug: Dict[str, Any]) -> Dict[str, Any]:
    manifest = load_v2_manifest()
    event = v2_unknown_event(sample_row, matched_phrase, debug)
    manifest["unknown_events"].append(event)
    save_v2_manifest(manifest)
    return event


def v2_run_unknown_eval(engine: str) -> Dict[str, Any]:
    manifest = load_v2_manifest()
    rerun_events: List[Dict[str, Any]] = []
    for event in v2_unknown_records(manifest):
        rel = event.get("file_rel")
        if not isinstance(rel, str):
            continue
        try:
            path = v2_path_for_rel(rel)
        except ValueError:
            continue
        if not path.exists():
            continue
        matched, debug = v2_match_audio(path, engine=engine)
        rerun_events.append({**event, **v2_unknown_event(event, matched, debug)})
    return v2_unknown_summary(rerun_events)


def v2_run_eval(engine: str = "mfcc_dtw_baseline") -> Dict[str, Any]:
    manifest = load_v2_manifest()
    source_events = v2_eval_records(manifest)
    rerun_events: List[Dict[str, Any]] = []
    for event in source_events:
        rel = event.get("file_rel")
        truth = event.get("truth_phrase_id")
        if not isinstance(rel, str) or not isinstance(truth, str):
            continue
        try:
            path = v2_path_for_rel(rel)
        except ValueError:
            continue
        if not path.exists():
            continue
        started = time.perf_counter()
        matched, debug = v2_match_audio(path, engine=engine)
        latency_ms = round((time.perf_counter() - started) * 1000.0, 2)
        predicted_id = debug.get("best_phrase_id")
        if not isinstance(predicted_id, str):
            predicted_id = None
        predicted_text = matched if isinstance(matched, str) and matched else None
        if predicted_text is None and predicted_id:
            predicted_text = PHRASE_ID_TO_TEXT.get(predicted_id, predicted_id)
        rerun_events.append(
            {
                **event,
                "predicted_phrase_id": predicted_id,
                "predicted_text": predicted_text,
                "best_phrase_id": debug.get("best_phrase_id"),
                "best_text": debug.get("best_phrase"),
                "second_phrase_id": debug.get("second_phrase_id"),
                "second_text": debug.get("second_phrase"),
                "reject_reason": debug.get("reject_reason"),
                "best_dist": debug.get("best_dist"),
                "top1_score": debug.get("score", debug.get("best_dist")),
                "score": debug.get("score", debug.get("best_dist")),
                "second_dist": debug.get("second_dist"),
                "margin": debug.get("margin", debug.get("gap")),
                "gap": debug.get("gap"),
                "ratio": debug.get("ratio"),
                "engine": debug.get("engine", engine),
                "engine_id": debug.get("engine_id", debug.get("engine", engine)),
                "latency_ms": latency_ms,
            }
        )
    latencies = [float(e["latency_ms"]) for e in rerun_events if isinstance(e.get("latency_ms"), (int, float))]
    summary = v2_eval_summary(rerun_events)
    latency_stats = {
        "p50_ms": round(float(np.percentile(np.array(latencies, dtype=np.float32), 50)), 2) if latencies else None,
        "p95_ms": round(float(np.percentile(np.array(latencies, dtype=np.float32), 95)), 2) if latencies else None,
    }
    decision_buckets: Dict[str, int] = {"local_accept": 0, "cloud_fallback_accept": 0, "reject": 0, "unknown": 0}
    for e in rerun_events:
        key = e.get("decision_source")
        k = str(key) if isinstance(key, str) else "local_accept"
        if k not in decision_buckets:
            k = "unknown"
        decision_buckets[k] = decision_buckets.get(k, 0) + 1
    report = {
        "version": 2,
        "created_at": now_iso(),
        "engine": engine,
        "engine_id": engine,
        "dataset_role": "dev_eval_existing",
        "train_count": len(v2_train_records(manifest)),
        "eval_count": len(rerun_events),
        "summary": summary,
        "top1_score_stats": v2_top1_score_stats(rerun_events),
        "confusion_pairs": v2_confusion_pairs(rerun_events),
        "error_buckets": v2_error_buckets(rerun_events),
        "decision_buckets": decision_buckets,
        "unknown_summary": v2_run_unknown_eval(engine),
        "latency": latency_stats,
        "latency_stats": latency_stats,
        "events": rerun_events,
    }
    report_path = V2_REPORTS_DIR / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.json"
    save_json(report_path, report)
    report["report_file"] = report_path.relative_to(ROOT_DIR).as_posix()
    return report


def get_v2_engine_registry() -> List[Dict[str, Any]]:
    active = load_active_v2_engine()
    ssl_available = torch is not None and AutoFeatureExtractor is not None and AutoModel is not None
    ssl_status = "available" if ssl_available else "missing torch/transformers"
    if _SSL_MODEL_ERROR:
        ssl_status = f"last ssl load error: {_SSL_MODEL_ERROR}"
    return [
        {
            "engine_id": ENGINE_V3_PERSONALIZED_TOP1,
            "label": "V3 个体化主引擎 (SSL+DTW Top1)",
            "available": True,
            "dependency_status": (
                f"backend={SSL_BACKEND}; ssl_fusion -> ssl_proto -> mfcc_safe fallback chain"
                if V3_SSL_ENABLED
                else "ssl unavailable -> mfcc_safe fallback"
            ),
            "active": active.get("engine_id") == ENGINE_V3_PERSONALIZED_TOP1,
        },
        {
            "engine_id": "mfcc_dtw_baseline",
            "label": "MFCC + DTW 基线",
            "available": True,
            "dependency_status": "built-in librosa baseline",
            "active": active.get("engine_id") == "mfcc_dtw_baseline",
        },
        {
            "engine_id": MFCC_SAFE_ENGINE_ID,
            "label": "MFCC + DTW 安全拒识",
            "available": True,
            "dependency_status": "baseline ranking with stricter open-set gates; safer for demo",
            "active": active.get("engine_id") == MFCC_SAFE_ENGINE_ID,
        },
        {
            "engine_id": MFCC_LENIENT_ENGINE_ID,
            "label": "MFCC + DTW 宽松拒识",
            "available": True,
            "dependency_status": "same ranking as baseline, relaxed open-set gates",
            "active": active.get("engine_id") == MFCC_LENIENT_ENGINE_ID,
        },
        {
            "engine_id": MFCC_CLASSIFIER_ENGINE_ID,
            "label": "MFCC 轻量分类器",
            "available": True,
            "dependency_status": "sklearn logistic classifier on cached MFCC stats",
            "active": active.get("engine_id") == MFCC_CLASSIFIER_ENGINE_ID,
        },
        {
            "engine_id": SSL_PROTO_ENGINE_ID,
            "label": "SSL 原型检索",
            "available": bool(ssl_available),
            "dependency_status": ssl_status,
            "active": active.get("engine_id") == SSL_PROTO_ENGINE_ID,
        },
        {
            "engine_id": SSL_FUSION_ENGINE_ID,
            "label": "SSL + DTW 融合",
            "available": bool(ssl_available),
            "dependency_status": ssl_status,
            "active": active.get("engine_id") == SSL_FUSION_ENGINE_ID,
        },
        {
            "engine_id": SSL_CLASSIFIER_ENGINE_ID,
            "label": "SSL 轻量分类器",
            "available": bool(ssl_available),
            "dependency_status": ssl_status,
            "active": active.get("engine_id") == SSL_CLASSIFIER_ENGINE_ID,
        },
    ]


def normalize_engine_list(payload: Dict[str, Any]) -> List[str]:
    raw = payload.get("engines")
    if raw is None:
        raw = payload.get("engine")
    if raw is None:
        return list(V2_DEFAULT_COMPARE_ENGINES)
    if isinstance(raw, str):
        engines = [raw]
    elif isinstance(raw, list):
        engines = [str(x) for x in raw]
    else:
        raise ValueError("engines must be a string or list")
    unknown = [e for e in engines if e not in V2_ENGINE_IDS]
    if unknown:
        raise ValueError(f"unknown engines: {', '.join(unknown)}")
    return list(dict.fromkeys(engines))


def normalize_engine_list_v3(payload: Dict[str, Any]) -> List[str]:
    return normalize_engine_list(payload)


def report_compact_metrics(report: Dict[str, Any]) -> Dict[str, Any]:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    buckets = report.get("error_buckets", {}).get("counts", {}) if isinstance(report.get("error_buckets"), dict) else {}
    unknown_summary = report.get("unknown_summary", {}) if isinstance(report.get("unknown_summary"), dict) else {}
    return {
        "engine_id": report.get("engine_id", report.get("engine")),
        "top1_rate": summary.get("top1_rate"),
        "top2_rate": summary.get("top2_rate"),
        "reject_rate": summary.get("reject_rate"),
        "accepted_wrong": buckets.get("accepted_wrong", 0),
        "unknown_false_accept_rate": unknown_summary.get("false_accept_rate"),
        "unknown_false_accepts": unknown_summary.get("false_accepts", 0),
        "unknown_total": unknown_summary.get("total", 0),
        "total": summary.get("total", 0),
    }


def choose_best_engine(reports_by_engine: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    ranked: List[Dict[str, Any]] = []
    for engine_id, report in reports_by_engine.items():
        metrics = report_compact_metrics(report)
        ranked.append(
            {
                "engine_id": engine_id,
                "top1_rate": float(metrics.get("top1_rate") or 0.0),
                "top2_rate": float(metrics.get("top2_rate") or 0.0),
                "reject_rate": float(1.0 if metrics.get("reject_rate") is None else metrics.get("reject_rate")),
                "accepted_wrong": int(metrics.get("accepted_wrong") or 0),
                "unknown_false_accept_rate": float(metrics.get("unknown_false_accept_rate") or 0.0),
                "unknown_false_accepts": int(metrics.get("unknown_false_accepts") or 0),
                "unknown_total": int(metrics.get("unknown_total") or 0),
            }
        )
    ranked.sort(
        key=lambda row: (
            0 if row["accepted_wrong"] == 0 else 1,
            0 if row["unknown_false_accepts"] == 0 else 1,
            row["unknown_false_accept_rate"],
            -row["top1_rate"],
            -row["top2_rate"],
            row["reject_rate"],
        )
    )
    baseline = next((row for row in ranked if row["engine_id"] == "mfcc_dtw_baseline"), None)
    best = ranked[0] if ranked else {"engine_id": "mfcc_dtw_baseline"}
    should_switch = False
    if baseline and best["engine_id"] != "mfcc_dtw_baseline":
        should_switch = (
            best["top1_rate"] > baseline["top1_rate"]
            and best["top2_rate"] >= baseline["top2_rate"]
            and best["accepted_wrong"] == 0
            and best["unknown_false_accepts"] == 0
        )
    return {"best_engine_id": best.get("engine_id"), "ranked": ranked, "baseline": baseline, "should_switch_from_baseline": should_switch}


def compare_v2_engines(engines: Optional[List[str]] = None) -> Dict[str, Any]:
    engine_ids = engines or list(V2_ENGINE_IDS)
    reports_by_engine: Dict[str, Dict[str, Any]] = {}
    failures: Dict[str, str] = {}
    for engine_id in engine_ids:
        if engine_id not in V2_ENGINE_IDS:
            failures[engine_id] = "unknown engine"
            continue
        try:
            reports_by_engine[engine_id] = v2_run_eval(engine=engine_id)
        except Exception as exc:  # noqa: BLE001
            failures[engine_id] = str(exc)
    comparison = {
        "version": 1,
        "ok": bool(reports_by_engine),
        "created_at": now_iso(),
        "dataset_role": "dev_eval_existing",
        "requested_engines": engine_ids,
        "available_engines": list(V2_ENGINE_IDS),
        "reports_by_engine": reports_by_engine,
        "failures": failures,
        "selection_hint": choose_best_engine(reports_by_engine),
    }
    report_path = V2_REPORTS_DIR / f"model_compare_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.json"
    save_json(report_path, comparison)
    comparison["report_file"] = report_path.relative_to(ROOT_DIR).as_posix()
    return comparison


def load_model_compare_report(report_file: str) -> Dict[str, Any]:
    if not isinstance(report_file, str) or not report_file.strip():
        raise ValueError("report_file is required")
    path = (ROOT_DIR / report_file).resolve()
    reports_root = V2_REPORTS_DIR.resolve()
    if reports_root not in path.parents:
        raise ValueError("report_file must live under data_v2/reports")
    if not path.exists():
        raise FileNotFoundError("report_file not found")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid report")
    return payload


def select_v2_active_engine(engine_id: str, report_file: str) -> Dict[str, Any]:
    if engine_id not in V2_ENGINE_IDS:
        return {"ok": False, "error": "unknown engine"}
    try:
        report = load_model_compare_report(report_file)
    except FileNotFoundError:
        return {"ok": False, "error": "report_file_not_found"}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": str(exc)}
    reports = report.get("reports_by_engine", {})
    if not isinstance(reports, dict) or engine_id not in reports:
        return {"ok": False, "error": "engine_not_in_report"}
    metrics = report_compact_metrics(reports[engine_id])
    if NO_REJECT_MODE:
        payload = {
            "engine_id": engine_id,
            "updated_at": now_iso(),
            "selected_from_report": report_file,
            "metrics": metrics,
            "selection_mode": "force_top1_no_reject",
        }
        save_json(V2_ACTIVE_ENGINE_PATH, payload)
        return {"ok": True, "active_engine": payload}
    if int(metrics.get("accepted_wrong") or 0) > 0:
        return {"ok": False, "error": "unsafe_engine_has_accepted_wrong", "metrics": metrics}
    if int(metrics.get("unknown_false_accepts") or 0) > 0:
        return {"ok": False, "error": "unsafe_engine_has_unknown_false_accepts", "metrics": metrics}
    payload = {
        "engine_id": engine_id,
        "updated_at": now_iso(),
        "selected_from_report": report_file,
        "metrics": metrics,
        "selection_mode": "reject_then_correct",
    }
    save_json(V2_ACTIVE_ENGINE_PATH, payload)
    return {"ok": True, "active_engine": payload, "selected_from_report": report_file}


def v2_model_card() -> Dict[str, Any]:
    manifest = load_v2_manifest()
    meta = load_v2_index_meta()
    active = load_active_v2_engine()
    prototype_ref = meta.get("prototype_cache", {}) if isinstance(meta, dict) else {}
    model_name = None
    if isinstance(prototype_ref, dict):
        model_name = prototype_ref.get("model")
    if not isinstance(model_name, str) or not model_name:
        model_name = _SSL_MODEL_NAME or WAVLM_MODEL_NAME
    return {
        "ok": True,
        "active_engine": active.get("engine_id"),
        "engine_payload": active,
        "feature_model": {
            "preferred": WAVLM_MODEL_NAME,
            "fallback": HUBERT_MODEL_NAME,
            "current": model_name,
            "device": HUBERT_DEVICE,
            "ssl_enabled": V3_SSL_ENABLED,
        },
        "train_sample_count": len(v2_train_records(manifest)),
        "eval_sample_count": len(v2_eval_records(manifest)),
        "correction_sample_count": len(manifest.get("corrections", [])) if isinstance(manifest.get("corrections"), list) else 0,
        "last_index_updated_at": meta.get("updated_at"),
    }


def v3_build_index() -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_build_index()


def v3_process_demo_audio(data: bytes) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_process_demo_audio(data)


def v3_maybe_batch_rebuild(phrase_id: str) -> Dict[str, Any]:
    manifest = load_v3_manifest()
    phrase_train_count = sum(
        1
        for row in manifest.get("samples", [])
        if isinstance(row, dict) and row.get("split") == "train" and row.get("phrase_id") == phrase_id
    )
    should_rebuild = phrase_train_count > 0 and phrase_train_count % VB_BATCH_REBUILD_EVERY == 0
    if not should_rebuild:
        return {
            "triggered": False,
            "reason": "batch_not_reached",
            "phrase_train_count": phrase_train_count,
            "batch_every": VB_BATCH_REBUILD_EVERY,
            "index_state": "pending_rebuild",
        }
    started = time.perf_counter()
    result = v3_build_index()
    duration_ms = round((time.perf_counter() - started) * 1000.0, 2)
    return {
        "triggered": True,
        "reason": "batch_rebuild",
        "ok": bool(result.get("ok")),
        "template_count": result.get("template_count", 0),
        "duration_ms": duration_ms,
        "phrase_train_count": phrase_train_count,
        "batch_every": VB_BATCH_REBUILD_EVERY,
        "index_state": "ready" if result.get("ok") else "rebuild_failed",
    }


def v3_hybrid_process_demo_audio(data: bytes) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    runtime_cfg = runtime_config_snapshot()
    quality = analyze_wav_bytes(data)
    if not quality.get("ok"):
        raw_path = save_media_file("raw", data, "wav")
        return {
            "ok": False,
            "error": "quality_gate_failed",
            "quality": quality,
            "raw_audio_url": f"/audio/{raw_path.name}",
            "final_text": None,
            "normalized_phrase_id": None,
            "normalized_phrase_text": None,
            "decision_source": "reject",
            "reason": "quality_gate_failed",
            "local_debug": {"reject_reason": "audio_too_short" if "too_short" in quality.get("quality_flags", []) else "quality_gate_failed"},
            "cloud_debug": {"used": False},
            "latency_ms": {},
            "runtime_config": runtime_cfg,
        }

    raw_path = save_media_file("raw", data, "wav")
    started = time.perf_counter()
    with v3_runtime_context():
        matched_phrase, local_debug = v2_match_audio(raw_path, engine=ENGINE_V3_PERSONALIZED_TOP1)
    local_elapsed = round((time.perf_counter() - started) * 1000.0, 2)

    low_conf = is_low_confidence(local_debug)
    cloud_used = False
    cloud_text = ""
    cloud_error = None
    cloud_elapsed = None
    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if VB_ENABLE_CLOUD_FALLBACK and VB_FALLBACK_POLICY == "low_conf_only" and low_conf and api_key:
        cloud_started = time.perf_counter()
        cloud_used = True
        try:
            cloud_text, _ = transcribe_with_fallback(raw_path, api_key)
        except Exception as exc:  # noqa: BLE001
            cloud_error = str(exc)
        cloud_elapsed = round((time.perf_counter() - cloud_started) * 1000.0, 2)

    final_text = matched_phrase if isinstance(matched_phrase, str) and matched_phrase else ""
    decision_source = "local_accept"
    reason = "local_confident"
    normalized_phrase_id = None
    normalized_phrase_text = None

    if cloud_used and cloud_text:
        normalized_phrase_id = normalize_to_phrase_id(cloud_text)
        if normalized_phrase_id:
            normalized_phrase_text = PHRASE_ID_TO_TEXT.get(normalized_phrase_id, normalized_phrase_id)
            final_text = normalized_phrase_text
            decision_source = "cloud_fallback_accept"
            reason = "low_confidence_local_cloud_normalized"
        else:
            final_text = cloud_text
            decision_source = "cloud_fallback_accept"
            reason = "low_confidence_local_cloud_raw"
    elif low_conf and cloud_used and not cloud_text:
        decision_source = "reject"
        reason = "low_confidence_cloud_empty"
    elif low_conf and not cloud_used:
        if not api_key and VB_ENABLE_CLOUD_FALLBACK:
            reason = "low_confidence_cloud_key_missing_local_fallback"
        elif not VB_ENABLE_CLOUD_FALLBACK:
            reason = "low_confidence_cloud_disabled_local_fallback"
        else:
            reason = "low_confidence_local_fallback"

    if not final_text:
        decision_source = "reject"

    tts_audio_url = None
    release_to_tts = None
    if final_text:
        tts_path = save_media_file("tts", b"", "mp3")
        try:
            run_tts_sync(final_text, tts_path)
            tts_audio_url = f"/audio/{tts_path.name}"
            release_to_tts = round((time.perf_counter() - started) * 1000.0, 2)
        except Exception as exc:  # noqa: BLE001
            tts_path.unlink(missing_ok=True)
            cloud_error = (cloud_error + "; " if cloud_error else "") + f"tts failed: {exc}"

    return {
        "ok": bool(final_text),
        "quality": quality,
        "raw_audio_url": f"/audio/{raw_path.name}",
        "final_text": final_text or None,
        "matched_phrase": final_text or None,
        "normalized_phrase_id": normalized_phrase_id,
        "normalized_phrase_text": normalized_phrase_text,
        "decision_source": decision_source,
        "reason": reason,
        "local_debug": local_debug,
        "cloud_debug": {
            "used": cloud_used,
            "text": cloud_text or None,
            "error": cloud_error,
            "latency_ms": cloud_elapsed,
        },
        "tts_audio_url": tts_audio_url,
        "latency_ms": {
            "local_ms": local_elapsed,
            "cloud_ms": cloud_elapsed,
            "release_to_verdict": round((time.perf_counter() - started) * 1000.0, 2),
            "release_to_tts": release_to_tts,
        },
        "runtime_config": runtime_cfg,
    }


def v3_run_eval(engine: str = ENGINE_V3_PERSONALIZED_TOP1) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_run_eval(engine=engine)


def v3_run_unknown_eval(engine: str = ENGINE_V3_PERSONALIZED_TOP1) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_run_unknown_eval(engine=engine)


def compare_v3_engines(engines: Optional[List[str]] = None) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        selected = engines
        if selected is None:
            selected = ["mfcc_dtw_baseline", ENGINE_V3_PERSONALIZED_TOP1]
        return compare_v2_engines(selected)


def select_v3_active_engine(engine_id: str, report_file: str) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return select_v2_active_engine(engine_id, report_file)


def v3_model_card() -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        card = v2_model_card()
    card["dataset"] = "data_v3"
    card["feature_model"]["backend"] = SSL_BACKEND
    return card


def get_v3_engine_registry() -> List[Dict[str, Any]]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return get_v2_engine_registry()


def v3_store_audio_sample(
    data: bytes,
    phrase_id: str,
    split: str,
    source: str,
    allow_duplicate: bool = False,
    speaker_id: str = "speaker_default",
) -> Tuple[bool, Dict[str, Any]]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_store_audio_sample(
            data,
            phrase_id=phrase_id,
            split=split,
            source=source,
            allow_duplicate=allow_duplicate,
            speaker_id=speaker_id,
        )


def v3_store_unknown_audio(data: bytes, source: str = "unknown_capture") -> Tuple[bool, Dict[str, Any]]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_store_unknown_audio(data, source=source)


def v3_append_eval_event(sample_row: Dict[str, Any], matched_phrase: Optional[str], debug: Dict[str, Any]) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_append_eval_event(sample_row, matched_phrase, debug)


def v3_append_unknown_event(sample_row: Dict[str, Any], matched_phrase: Optional[str], debug: Dict[str, Any]) -> Dict[str, Any]:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        return v2_append_unknown_event(sample_row, matched_phrase, debug)


def v3_is_error_eval_event(event: Dict[str, Any]) -> bool:
    truth_phrase_id = event.get("truth_phrase_id")
    predicted_phrase_id = event.get("predicted_phrase_id") if isinstance(event.get("predicted_phrase_id"), str) else None
    truth_text = event.get("truth_text") if isinstance(event.get("truth_text"), str) else None
    predicted_text = event.get("predicted_text") if isinstance(event.get("predicted_text"), str) else None
    reject_reason = str(event.get("reject_reason") or "unknown")
    id_mismatch = isinstance(predicted_phrase_id, str) and predicted_phrase_id != truth_phrase_id
    text_mismatch = isinstance(predicted_text, str) and isinstance(truth_text, str) and predicted_text != truth_text
    return reject_reason != "none" or id_mismatch or text_mismatch


def v3_add_corrections_from_eval_event_ids(
    event_ids: List[str],
    speaker_id: str = "speaker_default",
) -> Dict[str, Any]:
    manifest = load_v3_manifest()
    events = v3_eval_records(manifest)
    events_by_id: Dict[str, Dict[str, Any]] = {
        str(ev.get("event_id")): ev for ev in events if isinstance(ev, dict) and isinstance(ev.get("event_id"), str)
    }
    existing_hashes: Set[str] = set()
    for bucket in ("samples", "corrections"):
        rows = manifest.get(bucket, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, dict) and isinstance(row.get("sha256"), str):
                existing_hashes.add(str(row.get("sha256")))
    existing_from_event_ids: Set[str] = set()
    corr_rows = manifest.get("corrections", [])
    if isinstance(corr_rows, list):
        for row in corr_rows:
            if not isinstance(row, dict):
                continue
            from_event_id = row.get("from_event_id")
            if isinstance(from_event_id, str):
                existing_from_event_ids.add(from_event_id)
            from_eval_event_id = row.get("from_eval_event_id")
            if isinstance(from_eval_event_id, str):
                existing_from_event_ids.add(from_eval_event_id)

    added = 0
    skipped: List[Dict[str, Any]] = []
    errors: List[Dict[str, Any]] = []
    affected_phrase_ids: set[str] = set()
    processed_pairs: set[Tuple[str, str]] = set()

    for event_id in event_ids:
        event = events_by_id.get(event_id)
        if not isinstance(event, dict):
            errors.append({"event_id": event_id, "error": "event_not_found"})
            continue
        if event_id in existing_from_event_ids:
            skipped.append({"event_id": event_id, "reason": "already_added_to_corrections"})
            continue
        truth_phrase_id = event.get("truth_phrase_id")
        if not isinstance(truth_phrase_id, str) or truth_phrase_id not in PHRASE_ID_TO_TEXT:
            errors.append({"event_id": event_id, "error": "invalid_truth_phrase_id"})
            continue
        predicted_phrase_id = event.get("predicted_phrase_id") if isinstance(event.get("predicted_phrase_id"), str) else None
        if not v3_is_error_eval_event(event):
            skipped.append({"event_id": event_id, "reason": "not_error_event"})
            continue
        pair = (event_id, truth_phrase_id)
        if pair in processed_pairs:
            skipped.append({"event_id": event_id, "reason": "duplicate_event_id_in_request"})
            continue
        processed_pairs.add(pair)

        file_rel = event.get("file_rel")
        if not isinstance(file_rel, str) or not file_rel:
            errors.append({"event_id": event_id, "error": "missing_file_rel"})
            continue
        try:
            src = v3_path_for_rel(file_rel)
        except ValueError:
            errors.append({"event_id": event_id, "error": "invalid_file_rel"})
            continue
        if not src.exists():
            errors.append({"event_id": event_id, "error": "audio_not_found"})
            continue

        data = src.read_bytes()
        # For eval-to-corrections flow, keep every error event even if audio hash already exists.
        # Dedupe by from_event_id is enough; content-level dedupe here causes missed error collection.
        quality = analyze_wav_bytes(data, None)
        if not quality.get("ok"):
            skipped.append(
                {
                    "event_id": event_id,
                    "reason": "quality_gate_failed",
                    "quality_flags": quality.get("quality_flags"),
                }
            )
            continue

        sample_id = uuid4().hex
        target_dir = DATA_V3_DIR / "corrections" / truth_phrase_id
        target_dir.mkdir(parents=True, exist_ok=True)
        path = target_dir / f"correction_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{sample_id[:8]}.wav"
        path.write_bytes(data)
        rel = path.relative_to(ROOT_DIR).as_posix()
        row = {
            "sample_id": sample_id,
            "speaker_id": speaker_id or "speaker_default",
            "phrase_id": truth_phrase_id,
            "truth_phrase_id": truth_phrase_id,
            "phrase_text": PHRASE_ID_TO_TEXT.get(truth_phrase_id, truth_phrase_id),
            "split": "corrections",
            "source": "eval_misclassified",
            "from_event_id": event_id,
            "from_eval_event_id": event_id,
            "predicted_phrase_id": predicted_phrase_id,
            "raw_audio_url": f"/api/v3/audio?file_rel={rel}",
            "file_rel": rel,
            "duration_sec": quality.get("duration_sec"),
            "quality_flags": quality.get("quality_flags", []),
            "sha256": quality.get("sha256"),
            "created_at": now_iso(),
        }
        manifest["corrections"].append(row)
        if isinstance(row.get("sha256"), str):
            existing_hashes.add(str(row["sha256"]))
        existing_from_event_ids.add(event_id)
        affected_phrase_ids.add(truth_phrase_id)
        added += 1

    save_v3_manifest(manifest)
    index_meta = load_v3_index_meta()
    index_state = "ready" if isinstance(index_meta.get("templates"), list) and index_meta.get("templates") else "pending_rebuild"
    return {
        "ok": True,
        "added": added,
        "skipped": skipped,
        "errors": errors,
        "merged_count": 0,
        "affected_phrase_ids": sorted(affected_phrase_ids),
        "corrections_count": len(manifest.get("corrections", [])) if isinstance(manifest.get("corrections"), list) else 0,
        "index_state": index_state,
        "phrase_counts": v3_phrase_counts(manifest),
    }


def audit_audio_tree(base_dir: Path) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    by_hash: Dict[str, List[str]] = {}
    manifest_files: set[str] = set()
    if MANIFEST_PATH.exists():
        try:
            old_manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
            for node in old_manifest.get("phrases", {}).values():
                if isinstance(node, dict):
                    for sample in node.get("samples", []):
                        if isinstance(sample, dict) and isinstance(sample.get("file_rel"), str):
                            manifest_files.add(sample["file_rel"].replace("\\", "/"))
        except Exception:
            pass

    for path in sorted(base_dir.rglob("*.wav")) if base_dir.exists() else []:
        rel = path.relative_to(ROOT_DIR).as_posix()
        data = path.read_bytes()
        quality = analyze_wav_bytes(data)
        sha = str(quality.get("sha256"))
        by_hash.setdefault(sha, []).append(rel)
        rows.append(
            {
                "file_rel": rel,
                "in_manifest": rel in manifest_files if base_dir == DATA_DIR else None,
                **quality,
            }
        )
    duplicates = {sha: rels for sha, rels in by_hash.items() if len(rels) > 1}
    for row in rows:
        sha = row.get("sha256")
        row["duplicate"] = isinstance(sha, str) and sha in duplicates
    orphan_files = [row["file_rel"] for row in rows if row.get("in_manifest") is False]
    report = {
        "created_at": now_iso(),
        "base_dir": base_dir.relative_to(ROOT_DIR).as_posix() if base_dir.exists() else str(base_dir),
        "total_wav": len(rows),
        "bad_count": sum(1 for row in rows if row.get("quality_flags")),
        "duplicate_groups": duplicates,
        "orphan_files": orphan_files,
        "rows": rows,
    }
    return report


def v2_dataset_audit() -> Dict[str, Any]:
    manifest = load_v2_manifest()
    counts = v2_phrase_counts(manifest)
    train_count = len(v2_train_records(manifest))
    eval_count = len(v2_eval_records(manifest))
    unknown_count = len(v2_unknown_records(manifest))
    corrections_count = len(manifest.get("corrections", [])) if isinstance(manifest.get("corrections"), list) else 0
    rejected_count = len(manifest.get("rejected", [])) if isinstance(manifest.get("rejected"), list) else 0
    index_meta = load_v2_index_meta()
    report = {
        "ok": True,
        "created_at": now_iso(),
        "data_v2": {
            "train_count": train_count,
            "eval_count": eval_count,
            "unknown_count": unknown_count,
            "corrections_count": corrections_count,
            "rejected_count": rejected_count,
            "phrase_counts": counts,
            "index_ready": bool(index_meta.get("templates")),
            "index_template_count": len(index_meta.get("templates", [])) if isinstance(index_meta.get("templates"), list) else 0,
            "active_engine": load_active_v2_engine(),
        },
        "legacy_data": audit_audio_tree(DATA_DIR) if DATA_DIR.exists() else None,
    }
    save_json(V2_AUDIT_REPORT_PATH, report)
    return report


def v3_manifest_from_v2_migration() -> Dict[str, Any]:
    ensure_v3_storage()
    v2 = load_v2_manifest()
    out = {
        "version": 3,
        "created_at": now_iso(),
        "phrase_pack": PHRASE_PACK,
        "samples": [],
        "events": [],
        "corrections": [],
        "rejected": [],
        "unknown_events": [],
        "migration": {"source": "data_v2", "created_at": now_iso()},
    }
    seen_hash: set[str] = set()

    def add_row(container: List[Dict[str, Any]], row: Dict[str, Any]) -> None:
        r = dict(row)
        r["phrase_text"] = PHRASE_ID_TO_TEXT.get(str(r.get("phrase_id") or r.get("truth_phrase_id") or ""), r.get("phrase_text"))
        r["quality_flags"] = list(r.get("quality_flags") or [])
        r["speaker_id"] = str(r.get("speaker_id") or "speaker_default")
        container.append(r)

    for row in v2.get("samples", []):
        if not isinstance(row, dict):
            continue
        pid = row.get("phrase_id")
        if pid not in PHRASE_ID_TO_TEXT:
            continue
        split = str(row.get("split") or "train")
        if split != "train":
            continue
        src = str(row.get("source") or "")
        sha = str(row.get("sha256") or "")
        new_row = dict(row)
        rel = new_row.get("file_rel")
        if isinstance(rel, str) and rel.startswith("data_v2/"):
            new_row["file_rel"] = "data_v3/" + rel[len("data_v2/") :]
        if src.startswith("eval_promoted_"):
            new_row["split"] = "corrections"
            new_row["source"] = "migrated_eval_promoted"
            add_row(out["corrections"], new_row)
            continue
        if sha and sha in seen_hash:
            rej = dict(new_row)
            rej["split"] = "rejected"
            rej["reason"] = "duplicate_sha256"
            add_row(out["rejected"], rej)
            continue
        if sha:
            seen_hash.add(sha)
        new_row["split"] = "train"
        new_row["source"] = "migrated_train_capture"
        add_row(out["samples"], new_row)

    for event in v2.get("events", []):
        if isinstance(event, dict):
            ev = dict(event)
            rel = ev.get("file_rel")
            if isinstance(rel, str) and rel.startswith("data_v2/"):
                ev["file_rel"] = "data_v3/" + rel[len("data_v2/") :]
            ev["split"] = "eval"
            add_row(out["events"], ev)

    for corr in v2.get("corrections", []):
        if isinstance(corr, dict):
            c = dict(corr)
            rel = c.get("file_rel")
            if isinstance(rel, str) and rel.startswith("data_v2/"):
                c["file_rel"] = "data_v3/" + rel[len("data_v2/") :]
            c["split"] = "corrections"
            add_row(out["corrections"], c)

    for rej in v2.get("rejected", []):
        if isinstance(rej, dict):
            r = dict(rej)
            rel = r.get("file_rel")
            if isinstance(rel, str) and rel.startswith("data_v2/"):
                r["file_rel"] = "data_v3/" + rel[len("data_v2/") :]
            r["split"] = "rejected"
            add_row(out["rejected"], r)

    for unk in v2.get("unknown_events", []):
        if isinstance(unk, dict):
            u = dict(unk)
            rel = u.get("file_rel")
            if isinstance(rel, str) and rel.startswith("data_v2/"):
                u["file_rel"] = "data_v3/" + rel[len("data_v2/") :]
            u["split"] = "unknown"
            out["unknown_events"].append(u)

    return out


def ensure_v3_dataset(seed_from_v2: bool = True) -> Dict[str, Any]:
    ensure_v3_storage()
    manifest = load_v3_manifest()
    if seed_from_v2 and not manifest.get("samples") and V2_MANIFEST_PATH.exists():
        manifest = v3_manifest_from_v2_migration()
        save_v3_manifest(manifest)
    return manifest


def reset_v3_dataset_hard() -> Dict[str, Any]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = ROOT_DIR / f"data_v3_legacy_{ts}"
    if DATA_V3_DIR.exists():
        shutil.copytree(DATA_V3_DIR, backup_dir)
        shutil.rmtree(DATA_V3_DIR, ignore_errors=True)
    ensure_v3_storage()
    manifest = {
        "version": 3,
        "created_at": now_iso(),
        "phrase_pack": PHRASE_PACK,
        "samples": [],
        "events": [],
        "corrections": [],
        "rejected": [],
        "unknown_events": [],
        "reset_from": "hard_reset",
        "reset_at": now_iso(),
    }
    save_v3_manifest(manifest)
    return {"ok": True, "backup_dir": backup_dir.relative_to(ROOT_DIR).as_posix(), "data_dir": DATA_V3_DIR.relative_to(ROOT_DIR).as_posix()}


def sync_v3_files_from_v2() -> Dict[str, Any]:
    ensure_v3_storage()
    copied = 0
    missing = 0
    for split in ("train", "eval", "corrections", "rejected"):
        src_root = DATA_V2_DIR / split
        dst_root = DATA_V3_DIR / split
        if not src_root.exists():
            continue
        for src in src_root.rglob("*.wav"):
            rel = src.relative_to(src_root)
            dst = dst_root / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
    src_unknown = DATA_V2_DIR / "unknown"
    dst_unknown = DATA_V3_DIR / "unknown"
    if src_unknown.exists():
        for src in src_unknown.rglob("*.wav"):
            rel = src.relative_to(src_unknown)
            dst = dst_unknown / rel
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1

    manifest = load_v3_manifest()
    for bucket in ("samples", "events", "corrections", "rejected", "unknown_events"):
        rows = manifest.get(bucket, [])
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            rel = row.get("file_rel")
            if not isinstance(rel, str):
                continue
            p = (ROOT_DIR / rel).resolve()
            if not p.exists():
                missing += 1
    return {"ok": True, "copied_files": copied, "missing_refs": missing}


def v3_dataset_audit() -> Dict[str, Any]:
    manifest = ensure_v3_dataset(seed_from_v2=False)
    counts = v3_phrase_counts(manifest)
    samples = manifest.get("samples", [])
    events = manifest.get("events", [])
    rejected = manifest.get("rejected", [])
    corrections = manifest.get("corrections", [])

    duplicate_groups: Dict[str, List[str]] = {}
    hash_to_rows: Dict[str, List[str]] = {}
    for row in samples if isinstance(samples, list) else []:
        if not isinstance(row, dict):
            continue
        sha = row.get("sha256")
        rel = row.get("file_rel")
        if isinstance(sha, str) and sha and isinstance(rel, str):
            hash_to_rows.setdefault(sha, []).append(rel)
    for sha, rels in hash_to_rows.items():
        if len(rels) > 1:
            duplicate_groups[sha] = rels

    report = {
        "ok": True,
        "created_at": now_iso(),
        "dataset": "data_v3",
        "counts": {
            "train": len(v3_train_records(manifest)),
            "eval_events": len(v3_eval_records(manifest)),
            "unknown_events": len(v3_unknown_records(manifest)),
            "corrections": len(corrections) if isinstance(corrections, list) else 0,
            "rejected": len(rejected) if isinstance(rejected, list) else 0,
        },
        "phrase_counts": counts,
        "duplicate_groups": duplicate_groups,
    }
    save_json(V3_AUDIT_REPORT_PATH, report)
    return report


def safe_float(value: float) -> Optional[float]:
    return float(value) if np.isfinite(value) else None


def safe_div(num: float, den: float) -> float:
    if den <= 0:
        return 0.0
    return round(float(num) / float(den), 4)


def percentile_stats(values: List[float]) -> Dict[str, Optional[float]]:
    if not values:
        return {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None}
    arr = np.array(values, dtype=np.float32)
    return {
        "min": safe_float(float(np.min(arr))),
        "p10": safe_float(float(np.percentile(arr, 10))),
        "p50": safe_float(float(np.percentile(arr, 50))),
        "p90": safe_float(float(np.percentile(arr, 90))),
        "p95": safe_float(float(np.percentile(arr, 95))),
        "max": safe_float(float(np.max(arr))),
    }


def compute_phrase_calibration(sample_paths: List[Path], all_phrase_paths: Dict[str, List[Path]], phrase_id: str) -> Dict[str, Any]:
    if len(sample_paths) < 3:
        return {
            "threshold_ready": False,
            "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
            "min_gap": DEFAULT_MIN_GAP,
            "ratio_min": DEFAULT_RATIO_MIN,
            "intra_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
            "inter_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
        }

    waves: List[np.ndarray] = []
    for path in sample_paths:
        y = load_wave(path)
        if y is not None:
            waves.append(y)

    if len(waves) < 3:
        return {
            "threshold_ready": False,
            "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
            "min_gap": DEFAULT_MIN_GAP,
            "ratio_min": DEFAULT_RATIO_MIN,
            "intra_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
            "inter_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
        }

    dists_pos: List[float] = []
    for i in range(len(waves)):
        for j in range(i + 1, len(waves)):
            try:
                d = mfcc_dtw_distance_from_waves(waves[i], waves[j])
            except Exception:
                continue
            if np.isfinite(d):
                dists_pos.append(float(d))

    dists_neg: List[float] = []
    for other_id, paths in all_phrase_paths.items():
        if other_id == phrase_id:
            continue
        for p in paths[: min(6, len(paths))]:
            oy = load_wave(p)
            if oy is None:
                continue
            for wy in waves[: min(6, len(waves))]:
                try:
                    d = mfcc_dtw_distance_from_waves(wy, oy)
                except Exception:
                    continue
                if np.isfinite(d):
                    dists_neg.append(float(d))

    stats = percentile_stats(dists_pos)
    pos_p90 = stats.get("p90") or DEFAULT_DISTANCE_THRESHOLD
    pos_p95 = stats.get("p95") or pos_p90
    neg_stats = percentile_stats(dists_neg)
    neg_p10 = neg_stats.get("p10") if isinstance(neg_stats, dict) else None
    if neg_p10 is None:
        threshold = max(pos_p95 + 0.0010, pos_p90 + 0.0015)
    else:
        threshold = min(float(neg_p10), max(pos_p95 + 0.0008, pos_p90 + 0.0012))
    min_gap = max(0.0008, 0.12 * threshold)
    ratio_min = DEFAULT_RATIO_MIN

    return {
        "threshold_ready": len(dists_pos) >= 3,
        "distance_threshold": round(float(threshold), 6),
        "min_gap": round(float(min_gap), 6),
        "ratio_min": round(float(ratio_min), 6),
        "intra_stats": stats,
        "inter_stats": neg_stats,
    }


def compute_fusion_calibration_for_phrase(
    phrase_id: str,
    phrase_recs: List[Dict[str, Any]],
    all_recs: List[Dict[str, Any]],
    sample_hubert: Dict[str, np.ndarray],
) -> Dict[str, Any]:
    if len(phrase_recs) < 4:
        return {
            "threshold_ready": False,
            "fusion_threshold": None,
            "fusion_min_gap": None,
            "fusion_ratio_min": None,
            "fusion_pos_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
            "fusion_neg_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
        }

    phrase_hubs: List[np.ndarray] = []
    phrase_waves: List[np.ndarray] = []
    for rec in phrase_recs:
        rel = rec.get("file_rel")
        if not isinstance(rel, str):
            continue
        hv = sample_hubert.get(rel)
        w = load_wave(ROOT_DIR / rel)
        if hv is None or w is None:
            continue
        phrase_hubs.append(hv)
        phrase_waves.append(w)

    if len(phrase_hubs) < 4 or len(phrase_waves) < 4:
        return {
            "threshold_ready": False,
            "fusion_threshold": None,
            "fusion_min_gap": None,
            "fusion_ratio_min": None,
            "fusion_pos_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
            "fusion_neg_stats": {"min": None, "p10": None, "p50": None, "p90": None, "p95": None, "max": None},
        }

    mean_proto = l2_normalize(np.mean(np.vstack(phrase_hubs), axis=0))
    center_d = [cosine_distance(v, mean_proto) for v in phrase_hubs]
    keep_idx = np.argsort(np.array(center_d, dtype=np.float32))[: max(1, int(np.ceil(0.8 * len(phrase_hubs))))]
    robust_proto = l2_normalize(np.mean(np.vstack(phrase_hubs)[keep_idx], axis=0))

    fusion_pos: List[float] = []
    for i, hv in enumerate(phrase_hubs):
        p_dist = min(cosine_distance(hv, mean_proto), cosine_distance(hv, robust_proto))
        dtw_vals: List[float] = []
        for j, w in enumerate(phrase_waves):
            if i == j:
                continue
            d = mfcc_dtw_distance_from_waves(phrase_waves[i], w)
            if np.isfinite(d):
                dtw_vals.append(float(d))
        if not dtw_vals:
            continue
        dtw_vals.sort()
        d_dist = float(np.mean(dtw_vals[: min(PROTO_DTW_TOPK_PER_PHRASE, len(dtw_vals))]))
        fusion_pos.append(float(FUSION_WEIGHT_PROTO * p_dist + FUSION_WEIGHT_DTW * d_dist))

    fusion_neg: List[float] = []
    other_recs = [r for r in all_recs if r.get("phrase_id") != phrase_id]
    for rec in other_recs[: min(32, len(other_recs))]:
        rel = rec.get("file_rel")
        if not isinstance(rel, str):
            continue
        hv = sample_hubert.get(rel)
        w = load_wave(ROOT_DIR / rel)
        if hv is None or w is None:
            continue
        p_dist = min(cosine_distance(hv, mean_proto), cosine_distance(hv, robust_proto))
        dtw_vals: List[float] = []
        for pw in phrase_waves[: min(8, len(phrase_waves))]:
            d = mfcc_dtw_distance_from_waves(w, pw)
            if np.isfinite(d):
                dtw_vals.append(float(d))
        if not dtw_vals:
            continue
        dtw_vals.sort()
        d_dist = float(np.mean(dtw_vals[: min(PROTO_DTW_TOPK_PER_PHRASE, len(dtw_vals))]))
        fusion_neg.append(float(FUSION_WEIGHT_PROTO * p_dist + FUSION_WEIGHT_DTW * d_dist))

    pos_stats = percentile_stats(fusion_pos)
    neg_stats = percentile_stats(fusion_neg)
    pos_p95 = pos_stats.get("p95")
    pos_p90 = pos_stats.get("p90")
    neg_p10 = neg_stats.get("p10")
    if pos_p95 is None or pos_p90 is None:
        return {
            "threshold_ready": False,
            "fusion_threshold": None,
            "fusion_min_gap": None,
            "fusion_ratio_min": None,
            "fusion_pos_stats": pos_stats,
            "fusion_neg_stats": neg_stats,
        }

    base_th = max(float(pos_p95) + 0.002, float(pos_p90) + 0.003)
    if neg_p10 is not None:
        threshold = min(base_th, float(neg_p10) - 0.001)
    else:
        threshold = base_th
    threshold = max(threshold, 0.02)
    min_gap = max(0.0012, 0.09 * threshold)
    ratio_min = 1.05

    return {
        "threshold_ready": True,
        "fusion_threshold": round(float(threshold), 6),
        "fusion_min_gap": round(float(min_gap), 6),
        "fusion_ratio_min": round(float(ratio_min), 6),
        "fusion_pos_stats": pos_stats,
        "fusion_neg_stats": neg_stats,
    }


def build_faiss_index(manifest: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], Dict[str, Any]]:
    if faiss is None:
        return False, {}, {"error": "faiss module missing"}

    vectors: List[np.ndarray] = []
    records: List[Dict[str, Any]] = []
    sample_waves: Dict[str, np.ndarray] = {}
    sample_hubert: Dict[str, np.ndarray] = {}
    all_phrase_paths: Dict[str, List[Path]] = {p["id"]: [] for p in PHRASE_PACK}
    phrase_nodes = manifest.get("phrases", {})

    for phrase in PHRASE_PACK:
        phrase_id = phrase["id"]
        phrase_text = phrase["text"]
        node = phrase_nodes.get(phrase_id, {})
        samples = node.get("samples", []) if isinstance(node, dict) else []
        if not isinstance(samples, list):
            continue

        for sample in samples:
            rel = sample.get("file_rel")
            if not isinstance(rel, str):
                continue
            sample_path = ROOT_DIR / rel
            if not sample_path.exists():
                continue
            wave_y = load_wave(sample_path)
            if wave_y is None:
                continue
            all_phrase_paths[phrase_id].append(sample_path)
            emb = extract_embedding(wave_y)
            if emb is None:
                continue
            hub_emb = extract_hubert_embedding(wave_y)
            sample_waves[rel.replace("\\", "/")] = wave_y
            if hub_emb is not None:
                sample_hubert[rel.replace("\\", "/")] = hub_emb

            vectors.append(emb)
            records.append(
                {
                    "sample_id": sample.get("sample_id") or uuid4().hex,
                    "phrase_id": phrase_id,
                    "phrase_text": phrase_text,
                    "file_rel": rel.replace("\\", "/"),
                }
            )

    if not vectors:
        return False, {}, {"error": "no valid vectors built from samples"}

    mat = np.vstack(vectors).astype(np.float32)
    dim = int(mat.shape[1])
    index = faiss.IndexFlatL2(dim)
    index.add(mat)
    serialized = faiss.serialize_index(index)
    FAISS_INDEX_PATH.write_bytes(np.asarray(serialized, dtype=np.uint8).tobytes())

    calibration: Dict[str, Any] = {}
    cal_stats: Dict[str, Any] = {}
    prototype_cache: Dict[str, Dict[str, Any]] = {}
    records_by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        pid = rec.get("phrase_id")
        if isinstance(pid, str):
            records_by_phrase.setdefault(pid, []).append(rec)
    for phrase in PHRASE_PACK:
        phrase_id = phrase["id"]
        phrase_text = phrase["text"]
        sample_paths: List[Path] = []
        hubert_vecs: List[np.ndarray] = []
        for rec in records:
            if rec["phrase_id"] == phrase_id:
                sample_paths.append(ROOT_DIR / rec["file_rel"])
                hv = sample_hubert.get(rec["file_rel"])
                if hv is not None:
                    hubert_vecs.append(hv)

        cal = compute_phrase_calibration(sample_paths, all_phrase_paths, phrase_id)
        fusion_cal = compute_fusion_calibration_for_phrase(
            phrase_id=phrase_id,
            phrase_recs=records_by_phrase.get(phrase_id, []),
            all_recs=records,
            sample_hubert=sample_hubert,
        )
        cal["phrase_id"] = phrase_id
        cal["phrase_text"] = phrase_text
        cal["valid_samples"] = len(sample_paths)
        cal["last_calibrated_at"] = now_iso()
        cal["fusion_threshold"] = fusion_cal.get("fusion_threshold")
        cal["fusion_min_gap"] = fusion_cal.get("fusion_min_gap")
        cal["fusion_ratio_min"] = fusion_cal.get("fusion_ratio_min")
        cal["fusion_threshold_ready"] = bool(fusion_cal.get("threshold_ready", False))
        cal["fusion_pos_stats"] = fusion_cal.get("fusion_pos_stats")
        cal["fusion_neg_stats"] = fusion_cal.get("fusion_neg_stats")
        if hubert_vecs:
            hub_stack = np.vstack(hubert_vecs).astype(np.float32)
            mean_proto = l2_normalize(np.mean(hub_stack, axis=0))
            center_d = [cosine_distance(v, mean_proto) for v in hubert_vecs]
            keep_idx = np.argsort(np.array(center_d, dtype=np.float32))[: max(1, int(np.ceil(0.8 * len(hubert_vecs))))]
            robust_proto = l2_normalize(np.mean(hub_stack[keep_idx], axis=0))
            proto_var = float(np.mean(center_d)) if center_d else None
            prototype_cache[phrase_id] = {
                "phrase_id": phrase_id,
                "phrase_text": phrase_text,
                "mean_proto": mean_proto.tolist(),
                "robust_proto": robust_proto.tolist(),
                "proto_var": round(proto_var, 6) if proto_var is not None else None,
                "count": len(hubert_vecs),
            }
        else:
            prototype_cache[phrase_id] = {
                "phrase_id": phrase_id,
                "phrase_text": phrase_text,
                "mean_proto": None,
                "robust_proto": None,
                "proto_var": None,
                "count": 0,
            }
        calibration[phrase_id] = cal
        cal_stats[phrase_id] = {
            "text": phrase_text,
            "valid_samples": len(sample_paths),
            "threshold_ready": bool(cal["threshold_ready"]),
            "distance_threshold": cal["distance_threshold"],
            "min_gap": cal["min_gap"],
            "ratio_min": cal["ratio_min"],
            "fusion_threshold": cal.get("fusion_threshold"),
            "fusion_min_gap": cal.get("fusion_min_gap"),
            "fusion_ratio_min": cal.get("fusion_ratio_min"),
            "fusion_threshold_ready": cal.get("fusion_threshold_ready"),
            "prototype_count": prototype_cache[phrase_id]["count"],
            "intra_stats": cal["intra_stats"],
            "inter_stats": cal["inter_stats"],
        }

    cache_payload: Dict[str, Any] = {
        "version": 1,
        "updated_at": now_iso(),
        "embedding_model": HUBERT_MODEL_NAME,
        "device": HUBERT_DEVICE,
        "prototypes": prototype_cache,
    }
    save_json(PROTOTYPE_CACHE_PATH.with_suffix(".json"), cache_payload)

    meta = {
        "version": 1,
        "updated_at": now_iso(),
        "feature": {
            "name": "mfcc_stats_v1",
            "sample_rate": TARGET_SAMPLE_RATE,
            "n_mfcc": EMBED_N_MFCC,
            "trim_top_db": TRIM_TOP_DB,
        },
        "faiss": {
            "path": str(FAISS_INDEX_PATH.relative_to(ROOT_DIR)).replace("\\", "/"),
            "dimension": dim,
            "count": int(len(records)),
        },
        "defaults": {
            "distance_threshold": DEFAULT_DISTANCE_THRESHOLD,
            "min_gap": DEFAULT_MIN_GAP,
            "ratio_min": DEFAULT_RATIO_MIN,
            "fusion_weight_proto": FUSION_WEIGHT_PROTO,
            "fusion_weight_dtw": FUSION_WEIGHT_DTW,
        },
        "samples": records,
        "calibration": calibration,
        "prototype_cache": {
            "path": str(PROTOTYPE_CACHE_PATH.with_suffix(".json").relative_to(ROOT_DIR)).replace("\\", "/"),
            "model": _SSL_MODEL_NAME or WAVLM_MODEL_NAME,
            "device": HUBERT_DEVICE,
            "active_phrase_ids": ACTIVE_PHRASE_IDS_DEFAULT,
        },
    }

    save_json(INDEX_META_PATH, meta)
    return True, meta, {"calibration_stats": cal_stats}


def load_prototype_cache(meta: Dict[str, Any]) -> Dict[str, Any]:
    cache_ref = meta.get("prototype_cache", {}) if isinstance(meta, dict) else {}
    if not isinstance(cache_ref, dict):
        return {}
    rel = cache_ref.get("path")
    if not isinstance(rel, str):
        return {}
    path = ROOT_DIR / rel
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        return {}
    return {}


def select_active_phrase_ids(meta: Dict[str, Any]) -> List[str]:
    cache_ref = meta.get("prototype_cache", {}) if isinstance(meta, dict) else {}
    configured = cache_ref.get("active_phrase_ids", []) if isinstance(cache_ref, dict) else []
    if isinstance(configured, list):
        picked = [x for x in configured if isinstance(x, str) and x in PHRASE_ID_TO_TEXT]
        if picked:
            return picked
    return [p["id"] for p in PHRASE_PACK]


def audio_primary_match_mfcc_fallback(query_path: Path, meta: Dict[str, Any], min_gap_profile: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "reject_reason": "none",
        "best_phrase": None,
        "best_phrase_id": None,
        "second_phrase": None,
        "second_phrase_id": None,
        "best_dist": None,
        "second_dist": None,
        "gap": None,
        "ratio": None,
        "phrase_threshold": None,
        "min_gap": None,
        "base_min_gap": None,
        "effective_min_gap": None,
        "min_gap_multiplier_global": None,
        "min_gap_multiplier_phrase": None,
        "ratio_min": None,
        "candidates": [],
        "engine": "mfcc_fallback",
    }
    if faiss is None:
        debug["reject_reason"] = "index_not_ready"
        debug["error"] = "faiss module missing"
        return None, debug
    samples = meta.get("samples", [])
    if not isinstance(samples, list) or not samples or not FAISS_INDEX_PATH.exists():
        debug["reject_reason"] = "no_templates"
        return None, debug
    query_wave = load_wave(query_path)
    if query_wave is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug
    query_emb = extract_embedding(query_wave)
    if query_emb is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug

    index_bytes = FAISS_INDEX_PATH.read_bytes()
    index = faiss.deserialize_index(np.frombuffer(index_bytes, dtype=np.uint8))
    top_k = min(INDEX_TOP_K, len(samples))
    D, I = index.search(query_emb.reshape(1, -1), top_k)
    phrase_candidates: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
    for rank in range(top_k):
        idx = int(I[0, rank])
        if idx < 0 or idx >= len(samples):
            continue
        rec = samples[idx]
        pid = rec.get("phrase_id")
        if not isinstance(pid, str):
            continue
        phrase_candidates.setdefault(pid, []).append((float(D[0, rank]), rec))
    if not phrase_candidates:
        debug["reject_reason"] = "no_templates"
        return None, debug

    ranked_phrase_ids = sorted(
        phrase_candidates.keys(),
        key=lambda pid: min(item[0] for item in phrase_candidates[pid]),
    )[:MAX_CANDIDATE_PHRASES]

    phrase_scores: List[Dict[str, Any]] = []
    for pid in ranked_phrase_ids:
        coarse_sorted = sorted(phrase_candidates[pid], key=lambda item: item[0])
        selected = coarse_sorted[:DTW_TOPK_PER_PHRASE]
        dtw_vals: List[float] = []
        used_samples: List[str] = []
        for _, rec in selected:
            rel = rec.get("file_rel")
            if not isinstance(rel, str):
                continue
            sample_path = ROOT_DIR / rel
            sample_wave = load_wave(sample_path)
            if sample_wave is None:
                continue
            try:
                dist = mfcc_dtw_distance_from_waves(query_wave, sample_wave)
            except Exception:
                continue
            if not np.isfinite(dist):
                continue
            dtw_vals.append(float(dist))
            used_samples.append(rel)
        if not dtw_vals:
            continue
        dtw_vals.sort()
        score = float(np.mean(dtw_vals[: min(DTW_TOPK_PER_PHRASE, len(dtw_vals))]))
        phrase_scores.append(
            {
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "score": score,
                "dtw_values": [round(v, 6) for v in dtw_vals],
                "used_samples": used_samples,
            }
        )

    if not phrase_scores:
        debug["reject_reason"] = "no_templates"
        return None, debug

    phrase_scores.sort(key=lambda x: x["score"])
    best = phrase_scores[0]
    second = phrase_scores[1] if len(phrase_scores) > 1 else None
    best_dist = float(best["score"])
    second_dist = float(second["score"]) if second else float("inf")
    gap = second_dist - best_dist if np.isfinite(second_dist) else float("inf")
    ratio = (second_dist / best_dist) if (best_dist > 1e-9 and np.isfinite(second_dist)) else float("inf")

    calibration = meta.get("calibration", {}) if isinstance(meta, dict) else {}
    defaults = meta.get("defaults", {}) if isinstance(meta, dict) else {}
    best_cal = calibration.get(best["phrase_id"], {}) if isinstance(calibration, dict) else {}
    phrase_threshold = float(
        best_cal.get(
            "fusion_threshold",
            best_cal.get("distance_threshold", defaults.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)),
        )
    )
    base_min_gap = float(
        best_cal.get(
            "fusion_min_gap",
            best_cal.get("min_gap", defaults.get("min_gap", DEFAULT_MIN_GAP)),
        )
    )
    ratio_min = float(
        best_cal.get(
            "fusion_ratio_min",
            best_cal.get("ratio_min", defaults.get("ratio_min", DEFAULT_RATIO_MIN)),
        )
    )
    global_mul = float(min_gap_profile.get("global_multiplier", 1.0))
    per_phrase = min_gap_profile.get("per_phrase_multiplier", {})
    phrase_mul = float(per_phrase.get(best["phrase_id"], 1.0)) if isinstance(per_phrase, dict) else 1.0
    min_gap = base_min_gap * global_mul * phrase_mul

    debug.update(
        {
            "best_phrase": best["phrase_text"],
            "best_phrase_id": best["phrase_id"],
            "second_phrase": second["phrase_text"] if second else None,
            "second_phrase_id": second["phrase_id"] if second else None,
            "best_dist": round(best_dist, 6),
            "second_dist": safe_float(round(second_dist, 6) if np.isfinite(second_dist) else second_dist),
            "gap": safe_float(round(gap, 6) if np.isfinite(gap) else gap),
            "ratio": safe_float(round(ratio, 6) if np.isfinite(ratio) else ratio),
            "phrase_threshold": round(phrase_threshold, 6),
            "min_gap": round(min_gap, 6),
            "base_min_gap": round(base_min_gap, 6),
            "effective_min_gap": round(min_gap, 6),
            "min_gap_multiplier_global": round(global_mul, 6),
            "min_gap_multiplier_phrase": round(phrase_mul, 6),
            "ratio_min": round(ratio_min, 6),
            "candidates": phrase_scores,
        }
    )
    if NO_REJECT_MODE:
        debug["reject_reason"] = "none"
        return str(best["phrase_text"]), debug
    if best_dist > phrase_threshold:
        debug["reject_reason"] = "distance_too_high"
        return None, debug
    if second is not None and (gap < min_gap or ratio < ratio_min):
        debug["reject_reason"] = "separation_too_low"
        return None, debug
    debug["reject_reason"] = "none"
    return str(best["phrase_text"]), debug


def audio_primary_match(query_path: Path, meta: Dict[str, Any], min_gap_profile: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "reject_reason": "none",
        "best_phrase": None,
        "best_phrase_id": None,
        "second_phrase": None,
        "second_phrase_id": None,
        "best_dist": None,
        "second_dist": None,
        "gap": None,
        "ratio": None,
        "final_score": None,
        "best_proto_dist": None,
        "best_dtw_dist": None,
        "second_proto_dist": None,
        "second_dtw_dist": None,
        "phrase_threshold": None,
        "min_gap": None,
        "base_min_gap": None,
        "effective_min_gap": None,
        "min_gap_multiplier_global": None,
        "min_gap_multiplier_phrase": None,
        "ratio_min": None,
        "candidates": [],
        "engine": "hubert_proto_dtw",
    }

    if faiss is None:
        debug["reject_reason"] = "index_not_ready"
        debug["error"] = "faiss module missing"
        return None, debug

    if not meta or not FAISS_INDEX_PATH.exists():
        debug["reject_reason"] = "index_not_ready"
        return None, debug

    samples = meta.get("samples", [])
    if not isinstance(samples, list) or not samples:
        debug["reject_reason"] = "no_templates"
        return None, debug

    query_wave = load_wave(query_path)
    if query_wave is None:
        debug["reject_reason"] = "audio_too_short"
        return None, debug

    query_hubert = extract_hubert_embedding(query_wave)
    if query_hubert is None:
        return audio_primary_match_mfcc_fallback(query_path, meta, min_gap_profile)

    prototype_payload = load_prototype_cache(meta)
    prototypes = prototype_payload.get("prototypes", {}) if isinstance(prototype_payload, dict) else {}
    active_ids = select_active_phrase_ids(meta)

    proto_candidates: List[Dict[str, Any]] = []
    for pid in active_ids:
        row = prototypes.get(pid, {}) if isinstance(prototypes, dict) else {}
        if not isinstance(row, dict):
            continue
        m = row.get("mean_proto")
        r = row.get("robust_proto")
        if not isinstance(m, list) or not isinstance(r, list):
            continue
        try:
            mv = np.array(m, dtype=np.float32)
            rv = np.array(r, dtype=np.float32)
        except Exception:
            continue
        if mv.size == 0 or rv.size == 0:
            continue
        dist_mean = cosine_distance(query_hubert, mv)
        dist_robust = cosine_distance(query_hubert, rv)
        proto_dist = float(min(dist_mean, dist_robust))
        proto_candidates.append(
            {
                "phrase_id": pid,
                "phrase_text": PHRASE_ID_TO_TEXT.get(pid, pid),
                "proto_dist": proto_dist,
            }
        )

    if not proto_candidates:
        return audio_primary_match_mfcc_fallback(query_path, meta, min_gap_profile)

    proto_candidates.sort(key=lambda x: x["proto_dist"])
    ranked_phrase_ids = [x["phrase_id"] for x in proto_candidates[:PROTOTYPE_TOP_K]]

    phrase_scores: List[Dict[str, Any]] = []
    samples_by_phrase: Dict[str, List[Dict[str, Any]]] = {}
    for rec in samples:
        pid = rec.get("phrase_id")
        if isinstance(pid, str):
            samples_by_phrase.setdefault(pid, []).append(rec)

    for phrase_id in ranked_phrase_ids:
        phrase_samples = samples_by_phrase.get(phrase_id, [])
        if not phrase_samples:
            continue

        proto_dist = next((x["proto_dist"] for x in proto_candidates if x["phrase_id"] == phrase_id), None)
        if proto_dist is None:
            continue
        dtw_vals: List[float] = []
        used_samples: List[str] = []
        for rec in phrase_samples[: min(PROTO_DTW_TOPK_PER_PHRASE, len(phrase_samples))]:
            rel = rec.get("file_rel")
            if not isinstance(rel, str):
                continue
            sample_path = ROOT_DIR / rel
            sample_wave = load_wave(sample_path)
            if sample_wave is None:
                continue
            try:
                dist = mfcc_dtw_distance_from_waves(query_wave, sample_wave)
            except Exception:
                continue
            if not np.isfinite(dist):
                continue
            dtw_vals.append(float(dist))
            used_samples.append(rel)

        if not dtw_vals:
            continue

        dtw_vals.sort()
        dtw_score = float(np.mean(dtw_vals[: min(PROTO_DTW_TOPK_PER_PHRASE, len(dtw_vals))]))
        final_score = float(FUSION_WEIGHT_PROTO * proto_dist + FUSION_WEIGHT_DTW * dtw_score)
        phrase_scores.append(
            {
                "phrase_id": phrase_id,
                "phrase_text": PHRASE_ID_TO_TEXT.get(phrase_id, phrase_id),
                "score": final_score,
                "proto_dist": round(float(proto_dist), 6),
                "dtw_dist": round(float(dtw_score), 6),
                "dtw_values": [round(v, 6) for v in dtw_vals],
                "used_samples": used_samples,
            }
        )

    if not phrase_scores:
        debug["reject_reason"] = "no_templates"
        return None, debug

    phrase_scores.sort(key=lambda x: x["score"])
    best = phrase_scores[0]
    second = phrase_scores[1] if len(phrase_scores) > 1 else None

    best_dist = float(best["score"])
    second_dist = float(second["score"]) if second else float("inf")
    gap = second_dist - best_dist if np.isfinite(second_dist) else float("inf")
    ratio = (second_dist / best_dist) if (best_dist > 1e-9 and np.isfinite(second_dist)) else float("inf")

    calibration = meta.get("calibration", {}) if isinstance(meta, dict) else {}
    defaults = meta.get("defaults", {}) if isinstance(meta, dict) else {}
    best_cal = calibration.get(best["phrase_id"], {}) if isinstance(calibration, dict) else {}

    phrase_threshold = float(best_cal.get("distance_threshold", defaults.get("distance_threshold", DEFAULT_DISTANCE_THRESHOLD)))
    base_min_gap = float(best_cal.get("min_gap", defaults.get("min_gap", DEFAULT_MIN_GAP)))
    global_mul = float(min_gap_profile.get("global_multiplier", 1.0))
    per_phrase = min_gap_profile.get("per_phrase_multiplier", {})
    phrase_mul = 1.0
    if isinstance(per_phrase, dict):
        phrase_mul = float(per_phrase.get(best["phrase_id"], 1.0))
    min_gap = base_min_gap * global_mul * phrase_mul
    ratio_min = float(best_cal.get("ratio_min", defaults.get("ratio_min", DEFAULT_RATIO_MIN)))

    debug.update(
        {
            "best_phrase": best["phrase_text"],
            "best_phrase_id": best["phrase_id"],
            "second_phrase": second["phrase_text"] if second else None,
            "second_phrase_id": second["phrase_id"] if second else None,
            "best_dist": round(best_dist, 6),
            "second_dist": safe_float(round(second_dist, 6) if np.isfinite(second_dist) else second_dist),
            "gap": safe_float(round(gap, 6) if np.isfinite(gap) else gap),
            "ratio": safe_float(round(ratio, 6) if np.isfinite(ratio) else ratio),
            "final_score": round(best_dist, 6),
            "best_proto_dist": best.get("proto_dist"),
            "best_dtw_dist": best.get("dtw_dist"),
            "second_proto_dist": second.get("proto_dist") if second else None,
            "second_dtw_dist": second.get("dtw_dist") if second else None,
            "phrase_threshold": round(phrase_threshold, 6),
            "min_gap": round(min_gap, 6),
            "base_min_gap": round(base_min_gap, 6),
            "effective_min_gap": round(min_gap, 6),
            "min_gap_multiplier_global": round(global_mul, 6),
            "min_gap_multiplier_phrase": round(phrase_mul, 6),
            "ratio_min": round(ratio_min, 6),
            "candidates": phrase_scores,
        }
    )
    if NO_REJECT_MODE:
        debug["reject_reason"] = "none"
        return str(best["phrase_text"]), debug

    if best_dist > phrase_threshold:
        debug["reject_reason"] = "distance_too_high"
        return None, debug

    if second is not None and gap < min_gap:
        debug["reject_reason"] = "separation_too_low"
        return None, debug

    if second is not None and ratio < ratio_min:
        debug["reject_reason"] = "separation_too_low"
        return None, debug

    debug["reject_reason"] = "none"
    return str(best["phrase_text"]), debug


def process_audio_pipeline(audio_path: Path) -> Dict[str, Any]:
    started = time.perf_counter()
    steps: List[Dict[str, Any]] = []
    result: Dict[str, Any] = {
        "ok": False,
        "steps": steps,
        "raw_text": "",
        "cleaned_signature": "",
        "matched_phrase": None,
        "match_source": "none",
        "tts_audio_url": None,
        "audio_match_debug": {},
        "latency_ms": {},
    }

    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if api_key:
        raw_text, asr_steps = transcribe_with_fallback(audio_path, api_key)
        steps.extend(asr_steps)
        result["raw_text"] = raw_text
        cleaned = clean_text(raw_text)
        result["cleaned_signature"] = cleaned
        steps.append({"stage": "clean", "status": "ok", "message": f"cleaned signature: {cleaned or '<empty>'}"})
    else:
        steps.append({"stage": "asr", "status": "warn", "message": "SILICONFLOW_API_KEY missing, skip diagnostic ASR"})

    meta = load_index_meta()
    min_gap_profile = load_min_gap_profile()
    matched_phrase, debug = audio_primary_match(audio_path, meta, min_gap_profile)
    result["audio_match_debug"] = debug

    if matched_phrase:
        result["matched_phrase"] = matched_phrase
        engine = debug.get("engine", "audio_match")
        result["match_source"] = str(engine)
        steps.append(
            {
                "stage": "match_audio",
                "status": "ok",
                "message": f"audio hit: {matched_phrase} (engine={engine}, score={debug.get('final_score', debug.get('best_dist'))}, gap={debug.get('gap')})",
            }
        )
    else:
        steps.append(
            {
                "stage": "match_audio",
                "status": "warn",
                "message": f"audio miss: reason={debug.get('reject_reason')} best_dist={debug.get('best_dist')}",
            }
        )
        result["latency_ms"]["release_to_verdict"] = round((time.perf_counter() - started) * 1000.0, 2)
        return result

    tts_path = save_media_file("tts", b"", "mp3")
    tts_start = time.perf_counter()
    try:
        run_tts_sync(matched_phrase, tts_path)
    except Exception as exc:
        steps.append({"stage": "tts", "status": "error", "message": f"tts failed: {exc}"})
        tts_path.unlink(missing_ok=True)
        result["latency_ms"]["release_to_verdict"] = round((tts_start - started) * 1000.0, 2)
        return result

    result["ok"] = True
    result["tts_audio_url"] = f"/audio/{tts_path.name}"
    steps.append({"stage": "tts", "status": "ok", "message": f"tts generated: {tts_path.name}"})

    verdict_time = tts_start - started
    tts_time = time.perf_counter() - tts_start
    result["latency_ms"]["release_to_verdict"] = round(verdict_time * 1000.0, 2)
    result["latency_ms"]["verdict_to_tts"] = round(tts_time * 1000.0, 2)
    result["latency_ms"]["release_to_tts"] = round((verdict_time + tts_time) * 1000.0, 2)
    return result


def add_sample_to_manifest(phrase_id: str, file_rel: str, duration_sec: float, size_bytes: int) -> Dict[str, Any]:
    manifest = load_manifest()
    record = {
        "sample_id": uuid4().hex,
        "file_rel": file_rel.replace("\\", "/"),
        "duration_sec": round(duration_sec, 4),
        "size_bytes": size_bytes,
        "created_at": now_iso(),
    }
    manifest["phrases"][phrase_id]["samples"].append(record)
    save_manifest(manifest)
    return record


def list_samples_for_phrase(phrase_id: str) -> List[Dict[str, Any]]:
    manifest = load_manifest()
    node = manifest.get("phrases", {}).get(phrase_id, {})
    samples = node.get("samples", []) if isinstance(node, dict) else []
    if not isinstance(samples, list):
        return []

    rows: List[Dict[str, Any]] = []
    for idx, sample in enumerate(samples, start=1):
        if not isinstance(sample, dict):
            continue
        rows.append(
            {
                "index": idx,
                "sample_id": sample.get("sample_id"),
                "file_rel": sample.get("file_rel"),
                "duration_sec": sample.get("duration_sec"),
                "created_at": sample.get("created_at"),
                "size_bytes": sample.get("size_bytes"),
                "audio_url": f"/api/samples/{sample.get('sample_id')}/audio?phrase_id={phrase_id}",
            }
        )
    return rows


def get_sample_file_path(phrase_id: str, sample_id: str) -> Optional[Path]:
    manifest = load_manifest()
    node = manifest.get("phrases", {}).get(phrase_id, {})
    samples = node.get("samples", []) if isinstance(node, dict) else []
    if not isinstance(samples, list):
        return None
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        if sample.get("sample_id") != sample_id:
            continue
        rel = sample.get("file_rel")
        if not isinstance(rel, str):
            return None
        path = ROOT_DIR / rel
        return path if path.exists() else None
    return None


def delete_single_sample(phrase_id: str, sample_id: str) -> Dict[str, Any]:
    manifest = load_manifest()
    node = manifest.get("phrases", {}).get(phrase_id, {})
    samples = node.get("samples", []) if isinstance(node, dict) else []
    if not isinstance(samples, list):
        raise HTTPException(status_code=404, detail="not_found")

    found: Optional[Dict[str, Any]] = None
    kept: List[Dict[str, Any]] = []
    for sample in samples:
        if isinstance(sample, dict) and sample.get("sample_id") == sample_id:
            found = sample
            continue
        if isinstance(sample, dict):
            kept.append(sample)

    if found is None:
        raise HTTPException(status_code=404, detail="not_found")

    warnings: List[str] = []
    rel = found.get("file_rel")
    if isinstance(rel, str):
        sample_path = ROOT_DIR / rel
        if sample_path.exists():
            try:
                sample_path.unlink()
            except Exception as exc:  # noqa: BLE001
                warnings.append(f"delete_file_failed:{exc}")
        else:
            warnings.append("file_missing")
    else:
        warnings.append("file_rel_missing")

    manifest["phrases"][phrase_id]["samples"] = kept
    save_manifest(manifest)

    rebuild_result = rebuild_indexes()
    rebuild_ok = bool(rebuild_result.get("ok"))
    if not rebuild_ok:
        warnings.append(f"rebuild_failed:{rebuild_result.get('error', 'unknown')}")

    new_manifest = load_manifest()
    signature_db = load_signature_db()
    index_meta = load_index_meta()
    phrase_stats = get_phrase_stats(new_manifest, signature_db, index_meta)
    current_count = len(new_manifest["phrases"][phrase_id]["samples"])

    return {
        "ok": True,
        "deleted_sample_id": sample_id,
        "phrase_id": phrase_id,
        "phrase_sample_count": current_count,
        "phrases_stats": phrase_stats,
        "rebuild_ok": rebuild_ok,
        "warnings": warnings,
    }


def parse_raw_audio_url(raw_audio_url: str) -> Path:
    if not isinstance(raw_audio_url, str) or not raw_audio_url.strip():
        raise ValueError("raw audio url is required")
    url = raw_audio_url.strip()
    if not url.startswith("/audio/"):
        raise ValueError("raw audio url must start with /audio/")
    name = Path(url.split("?", 1)[0]).name
    if not name.startswith("raw_") or not name.endswith(".wav"):
        raise ValueError("raw audio url must point to /audio/raw_*.wav")
    path = (AUDIO_DIR / name).resolve()
    audio_root = AUDIO_DIR.resolve()
    if audio_root not in path.parents:
        raise ValueError("raw audio path out of bounds")
    return path


def blindtest_correct_and_rebuild(raw_audio_url: str, truth_phrase_id: str) -> Dict[str, Any]:
    if truth_phrase_id not in PHRASE_ID_TO_TEXT:
        return {"ok": False, "error": "invalid truth_phrase_id"}

    try:
        source_path = parse_raw_audio_url(raw_audio_url)
    except ValueError as exc:
        return {"ok": False, "error": str(exc)}

    if not source_path.exists():
        return {"ok": False, "error": "not_found"}

    data = source_path.read_bytes()
    valid, reason, duration = validate_audio_sample(data)
    if not valid:
        return {"ok": False, "error": reason}
    if duration < MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:
        return {"ok": False, "error": f"sample too short ({duration:.3f}s), min {MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:.1f}s"}

    phrase_dir = PHRASE_DATA_DIR / truth_phrase_id
    phrase_dir.mkdir(parents=True, exist_ok=True)
    sample_name = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.wav"
    sample_path = phrase_dir / sample_name
    sample_path.write_bytes(data)
    rel = sample_path.relative_to(ROOT_DIR).as_posix()
    _ = add_sample_to_manifest(truth_phrase_id, rel, duration, len(data))

    warnings: List[str] = []
    rebuild_result = rebuild_indexes()
    rebuild_ok = bool(rebuild_result.get("ok"))
    if not rebuild_ok:
        warnings.append(f"rebuild_failed:{rebuild_result.get('error', 'unknown')}")

    manifest = load_manifest()
    count = len(manifest["phrases"][truth_phrase_id]["samples"])
    return {
        "ok": True,
        "truth_phrase_id": truth_phrase_id,
        "phrase_sample_count": count,
        "rebuild_ok": rebuild_ok,
        "warnings": warnings,
    }


def rebuild_signature_index(manifest: Dict[str, Any], api_key: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    phrase_nodes = manifest.get("phrases", {})
    db = {"version": 2, "updated_at": now_iso(), "phrases": {}}
    stats: Dict[str, Any] = {}

    for phrase in PHRASE_PACK:
        phrase_id = phrase["id"]
        phrase_text = phrase["text"]
        node = phrase_nodes.get(phrase_id, {})
        samples = node.get("samples", []) if isinstance(node, dict) else []

        counts: Dict[str, int] = {}
        ok_count = 0
        empty_count = 0
        fail_count = 0
        skipped = 0

        if not api_key:
            skipped = len(samples) if isinstance(samples, list) else 0
        else:
            for sample in samples:
                rel = sample.get("file_rel")
                if not isinstance(rel, str):
                    fail_count += 1
                    continue
                sample_path = ROOT_DIR / rel
                if not sample_path.exists():
                    fail_count += 1
                    continue

                raw_text, asr_steps = transcribe_with_fallback(sample_path, api_key)
                if not raw_text and any(step.get("status") == "error" for step in asr_steps):
                    fail_count += 1
                    continue

                cleaned = clean_text(raw_text)
                if cleaned:
                    counts[cleaned] = counts.get(cleaned, 0) + 1
                    ok_count += 1
                else:
                    empty_count += 1

        db["phrases"][phrase_text] = {
            "phrase_id": phrase_id,
            "signature_counts": dict(sorted(counts.items(), key=lambda x: x[1], reverse=True)),
        }
        stats[phrase_id] = {
            "text": phrase_text,
            "sample_total": len(samples) if isinstance(samples, list) else 0,
            "asr_ok": ok_count,
            "asr_empty": empty_count,
            "asr_fail": fail_count,
            "asr_skipped": skipped,
            "signature_count": len(counts),
        }

    save_json(SIGNATURE_DB_PATH, db)
    return db, stats


def rebuild_indexes() -> Dict[str, Any]:
    manifest = load_manifest()

    api_key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    _, asr_stats = rebuild_signature_index(manifest, api_key)

    ok, _, faiss_stats = build_faiss_index(manifest)
    if not ok:
        return {
            "ok": False,
            "error": faiss_stats.get("error", "failed to build faiss index"),
            "asr_stats": asr_stats,
            "calibration_stats": {},
        }

    return {
        "ok": True,
        "asr_stats": asr_stats,
        "calibration_stats": faiss_stats.get("calibration_stats", {}),
    }


ensure_storage()
app = FastAPI(title="VoiceBridge Test UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.mount("/audio", StaticFiles(directory=AUDIO_DIR), name="audio")


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/phrases")
def api_list_phrases() -> JSONResponse:
    manifest = load_manifest()
    signature_db = load_signature_db()
    index_meta = load_index_meta()
    return JSONResponse({"ok": True, "phrases": get_phrase_stats(manifest, signature_db, index_meta)})


@app.get("/api/samples")
def api_list_samples(phrase_id: str = Query(...)) -> JSONResponse:
    if phrase_id not in PHRASE_ID_TO_TEXT:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid phrase_id"})
    rows = list_samples_for_phrase(phrase_id)
    return JSONResponse({"ok": True, "phrase_id": phrase_id, "samples": rows})


@app.delete("/api/samples/{sample_id}")
def api_delete_sample(sample_id: str, phrase_id: str = Query(...)) -> JSONResponse:
    if phrase_id not in PHRASE_ID_TO_TEXT:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid phrase_id"})
    try:
        result = delete_single_sample(phrase_id, sample_id)
        return JSONResponse(result)
    except HTTPException as exc:
        if exc.status_code == 404:
            return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})
        raise


@app.get("/api/samples/{sample_id}/audio")
def api_sample_audio(sample_id: str, phrase_id: str = Query(...)) -> FileResponse:
    if phrase_id not in PHRASE_ID_TO_TEXT:
        raise HTTPException(status_code=400, detail="invalid phrase_id")
    path = get_sample_file_path(phrase_id, sample_id)
    if path is None:
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="audio/wav")


@app.post("/api/samples/upload")
async def api_upload_sample(
    phrase_id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    if phrase_id not in PHRASE_ID_TO_TEXT:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid phrase_id"})

    data = await file.read()
    valid, reason, duration = validate_audio_sample(data)
    if not valid:
        return JSONResponse(status_code=400, content={"ok": False, "error": reason})
    if duration < MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": f"sample too short ({duration:.3f}s), min {MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:.1f}s",
            },
        )

    phrase_dir = PHRASE_DATA_DIR / phrase_id
    phrase_dir.mkdir(parents=True, exist_ok=True)
    sample_name = f"sample_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.wav"
    sample_path = phrase_dir / sample_name
    sample_path.write_bytes(data)
    rel = sample_path.relative_to(ROOT_DIR).as_posix()
    record = add_sample_to_manifest(phrase_id, rel, duration, len(data))

    manifest = load_manifest()
    count = len(manifest["phrases"][phrase_id]["samples"])
    return JSONResponse(
        {
            "ok": True,
            "phrase_id": phrase_id,
            "phrase_text": PHRASE_ID_TO_TEXT[phrase_id],
            "sample_record": record,
            "phrase_sample_count": count,
        }
    )


@app.post("/api/samples/reset")
def api_reset_samples() -> JSONResponse:
    if PHRASE_DATA_DIR.exists():
        shutil.rmtree(PHRASE_DATA_DIR, ignore_errors=True)
    if INDEX_DIR.exists():
        shutil.rmtree(INDEX_DIR, ignore_errors=True)
    MANIFEST_PATH.unlink(missing_ok=True)
    SIGNATURE_DB_PATH.unlink(missing_ok=True)
    ensure_storage()
    return JSONResponse({"ok": True, "message": "samples reset"})


@app.post("/api/index/rebuild")
def api_rebuild_index() -> JSONResponse:
    result = rebuild_indexes()
    code = 200 if result.get("ok") else 400
    return JSONResponse(status_code=code, content=result)


@app.get("/api/min-gap-profile")
def api_get_min_gap_profile() -> JSONResponse:
    profile = load_min_gap_profile()
    return JSONResponse({"ok": True, "profile": profile})


@app.post("/api/min-gap-profile")
def api_set_min_gap_profile(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    try:
        profile = normalize_min_gap_profile(payload)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})

    save_min_gap_profile(profile)
    return JSONResponse({"ok": True, "profile": profile})


@app.post("/api/blindtest/correct")
def api_blindtest_correct(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    raw_audio_url = payload.get("raw_audio_url", "")
    truth_phrase_id = payload.get("truth_phrase_id", "")
    result = blindtest_correct_and_rebuild(raw_audio_url, truth_phrase_id)
    code = 200 if result.get("ok") else (404 if result.get("error") == "not_found" else 400)
    return JSONResponse(status_code=code, content=result)


@app.post("/api/process-audio")
async def api_process_audio(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    valid, reason, duration = validate_audio_sample(data)
    if not valid:
        return JSONResponse(
            status_code=400,
            content={
                "ok": False,
                "error": reason,
                "audio_match_debug": {"reject_reason": "audio_too_short" if "too short" in reason else "none"},
                "steps": [{"stage": "validate", "status": "error", "message": reason}],
            },
        )

    temp_path = ROOT_DIR / "temp.wav"
    temp_path.write_bytes(data)
    raw_path = save_media_file("raw", data, "wav")

    result = process_audio_pipeline(temp_path)
    result["raw_audio_url"] = f"/audio/{raw_path.name}"
    result["saved_as"] = str(temp_path)
    result["size_bytes"] = len(data)
    result["duration_sec"] = duration
    result["received_content_type"] = file.content_type
    return JSONResponse(result)


@app.post("/api/batch-eval/upload")
async def api_batch_eval_upload(
    truth_phrase_id: str = Form(...),
    file: UploadFile = File(...),
) -> JSONResponse:
    if truth_phrase_id not in PHRASE_ID_TO_TEXT:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid truth_phrase_id"})
    data = await file.read()
    valid, reason, duration = validate_audio_sample(data)
    if not valid:
        return JSONResponse(status_code=400, content={"ok": False, "error": reason})
    if duration < MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:
        return JSONResponse(
            status_code=400,
            content={"ok": False, "error": f"sample too short ({duration:.3f}s), min {MIN_SAMPLE_DURATION_SEC_FOR_SAMPLE:.1f}s"},
        )

    batch_audio_dir = BATCH_DIR / truth_phrase_id
    batch_audio_dir.mkdir(parents=True, exist_ok=True)
    name = f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:8]}.wav"
    path = batch_audio_dir / name
    path.write_bytes(data)
    rel = path.relative_to(ROOT_DIR).as_posix()

    meta = load_index_meta()
    min_gap_profile = load_min_gap_profile()
    matched_phrase, debug = audio_primary_match(path, meta, min_gap_profile)
    event = {
        "ts": now_iso(),
        "mode": "batch_eval_upload",
        "truth_phrase_id": truth_phrase_id,
        "truth_text": PHRASE_ID_TO_TEXT[truth_phrase_id],
        "predicted_phrase_id": debug.get("best_phrase_id"),
        "predicted_text": matched_phrase or debug.get("best_phrase"),
        "second_phrase": debug.get("second_phrase"),
        "reject_reason": debug.get("reject_reason"),
        "final_score": debug.get("final_score"),
        "gap": debug.get("gap"),
        "ratio": debug.get("ratio"),
        "engine": debug.get("engine"),
        "duration_sec": duration,
        "file_rel": rel,
        "raw_audio_url": f"/audio/{name}",
    }
    append_batch_event(event)
    count = count_batch_events_for_phrase(truth_phrase_id)
    return JSONResponse({"ok": True, "event": event, "truth_phrase_count": count})


@app.post("/api/batch-eval/record")
def api_batch_eval_record(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    truth_phrase_id = payload.get("truth_phrase_id")
    if not isinstance(truth_phrase_id, str) or truth_phrase_id not in PHRASE_ID_TO_TEXT:
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid truth_phrase_id"})
    append_batch_event(payload)
    return JSONResponse({"ok": True})


@app.get("/api/batch-eval/export")
def api_batch_eval_export(limit: int = Query(2000)) -> JSONResponse:
    rows = read_batch_events(limit=limit)
    return JSONResponse({"ok": True, "count": len(rows), "events": rows})


@app.post("/api/batch-eval/reset")
def api_batch_eval_reset() -> JSONResponse:
    if BATCH_EVENTS_PATH.exists():
        BATCH_EVENTS_PATH.unlink(missing_ok=True)
    return JSONResponse({"ok": True})


@app.get("/api/v2/phrases")
def api_v2_phrases() -> JSONResponse:
    manifest = load_v2_manifest()
    counts = v2_phrase_counts(manifest)
    index_meta = load_v2_index_meta()
    templates_by_phrase = index_meta.get("templates_by_phrase", {}) if isinstance(index_meta, dict) else {}
    phrases = []
    for phrase in PHRASE_PACK:
        pid = phrase["id"]
        phrases.append(
            {
                "phrase_id": pid,
                "text": phrase["text"],
                "stage_pack": pid in set(ACTIVE_PHRASE_IDS_DEFAULT),
                **counts.get(pid, {"train": 0, "eval": 0, "corrections": 0, "rejected": 0}),
                "template_count": templates_by_phrase.get(pid, 0) if isinstance(templates_by_phrase, dict) else 0,
            }
        )
    return JSONResponse({"ok": True, "phrases": phrases})


@app.get("/api/v2/engines")
def api_v2_engines() -> JSONResponse:
    return JSONResponse({"ok": True, "active_engine": load_active_v2_engine(), "engines": get_v2_engine_registry()})


@app.get("/api/v2/model/card")
def api_v2_model_card() -> JSONResponse:
    return JSONResponse(v2_model_card())


@app.get("/api/data/audit")
def api_data_audit() -> JSONResponse:
    return JSONResponse(v2_dataset_audit())


@app.post("/api/dataset/archive-legacy")
def api_archive_legacy() -> JSONResponse:
    ensure_v2_storage()
    if not DATA_DIR.exists():
        return JSONResponse({"ok": True, "message": "legacy data directory missing, nothing to archive"})
    target = ROOT_DIR / f"data_legacy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    shutil.copytree(DATA_DIR, target)
    return JSONResponse({"ok": True, "archive_dir": target.relative_to(ROOT_DIR).as_posix()})


@app.post("/api/v2/samples/upload")
async def api_v2_upload_train_sample(
    phrase_id: str = Form(...),
    speaker_id: str = Form("speaker_default"),
    file: UploadFile = File(...),
) -> JSONResponse:
    data = await file.read()
    ok, payload = v2_store_audio_sample(data, phrase_id=phrase_id, split="train", source="train_capture", speaker_id=speaker_id)
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    rebuild = v2_build_index()
    manifest = load_v2_manifest()
    payload["phrase_counts"] = v2_phrase_counts(manifest)
    payload["rebuild_ok"] = bool(rebuild.get("ok"))
    payload["template_count"] = rebuild.get("template_count", 0)
    return JSONResponse(payload)


@app.post("/api/v2/eval/upload")
async def api_v2_upload_eval_sample(
    truth_phrase_id: str = Form(...),
    speaker_id: str = Form("speaker_default"),
    file: UploadFile = File(...),
) -> JSONResponse:
    data = await file.read()
    ok, payload = v2_store_audio_sample(data, phrase_id=truth_phrase_id, split="eval", source="eval_capture", speaker_id=speaker_id)
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    sample = payload["sample"]
    path = v2_path_for_rel(sample["file_rel"])
    matched_phrase, debug = v2_match_audio(path)
    sample["raw_audio_url"] = f"/api/v2/audio?file_rel={sample['file_rel']}"
    event = v2_append_eval_event(sample, matched_phrase, debug)
    manifest = load_v2_manifest()
    return JSONResponse(
        {
            "ok": True,
            "sample": sample,
            "event": event,
            "summary": v2_eval_summary(v2_eval_records(manifest)),
            "phrase_counts": v2_phrase_counts(manifest),
        }
    )


@app.post("/api/v2/unknown/upload")
async def api_v2_upload_unknown_sample(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    ok, payload = v2_store_unknown_audio(data, source="unknown_capture")
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    sample = payload["sample"]
    path = v2_path_for_rel(sample["file_rel"])
    matched_phrase, debug = v2_match_audio(path)
    event = v2_append_unknown_event(sample, matched_phrase, debug)
    manifest = load_v2_manifest()
    return JSONResponse(
        {
            "ok": True,
            "sample": sample,
            "event": event,
            "summary": v2_unknown_summary(v2_unknown_records(manifest)),
            "phrase_counts": v2_phrase_counts(manifest),
        }
    )


@app.post("/api/v2/demo/process")
async def api_v2_demo_process(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    return JSONResponse(v2_process_demo_audio(data))


@app.post("/api/v2/corrections/upload")
async def api_v2_correction_upload(
    truth_phrase_id: str = Form(...),
    speaker_id: str = Form("speaker_default"),
    predicted_phrase_id: str = Form(""),
    raw_audio_url: str = Form(""),
    file: UploadFile = File(...),
) -> JSONResponse:
    data = await file.read()
    ok, payload = v2_store_audio_sample(data, phrase_id=truth_phrase_id, split="corrections", source="demo_correction", speaker_id=speaker_id)
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    manifest = load_v2_manifest()
    correction = manifest["corrections"][-1]
    correction["predicted_phrase_id"] = predicted_phrase_id or None
    correction["raw_audio_url"] = raw_audio_url or None
    save_v2_manifest(manifest)
    return JSONResponse({"ok": True, "correction": correction, "phrase_counts": v2_phrase_counts(manifest)})


@app.post("/api/v2/corrections/confirm")
def api_v2_confirm_corrections(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    phrase_id = payload.get("phrase_id")
    manifest = load_v2_manifest()
    corrections = manifest.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []
    keep: List[Dict[str, Any]] = []
    moved = 0
    affected_phrase_ids: set[str] = set()
    for row in corrections:
        if not isinstance(row, dict):
            continue
        truth = row.get("truth_phrase_id") or row.get("phrase_id")
        if phrase_id and truth != phrase_id:
            keep.append(row)
            continue
        rel = row.get("file_rel")
        if not isinstance(rel, str) or not isinstance(truth, str) or truth not in PHRASE_ID_TO_TEXT:
            keep.append(row)
            continue
        src = v2_path_for_rel(rel)
        data = src.read_bytes() if src.exists() else b""
        speaker_id = str(row.get("speaker_id") or "speaker_default")
        ok, payload2 = v2_store_audio_sample(
            data,
            phrase_id=truth,
            split="train",
            source="confirmed_correction",
            allow_duplicate=True,
            speaker_id=speaker_id,
        )
        if ok:
            moved += 1
            affected_phrase_ids.add(truth)
        else:
            keep.append(row)
    manifest = load_v2_manifest()
    manifest["corrections"] = keep
    save_v2_manifest(manifest)
    rebuild_ok = None
    rebuild_ms = None
    rebuild_template_count = 0
    purify_result: Optional[Dict[str, Any]] = None
    if moved > 0:
        rb_started = time.perf_counter()
        rebuild = v2_build_index()
        rebuild_ok = bool(rebuild.get("ok"))
        rebuild_ms = round((time.perf_counter() - rb_started) * 1000.0, 2)
        rebuild_template_count = int(rebuild.get("template_count", 0) or 0)
        purify_result = v2_auto_purify_train_samples(load_v2_manifest(), trigger_reason="confirm_post_rebuild")
        if isinstance(purify_result, dict) and purify_result.get("ok"):
            # Rebuild once again so disabled samples are immediately excluded from index.
            rb2_started = time.perf_counter()
            rebuild2 = v2_build_index()
            rebuild_ok = bool(rebuild2.get("ok"))
            rebuild_ms = round((rebuild_ms or 0.0) + (time.perf_counter() - rb2_started) * 1000.0, 2)
            rebuild_template_count = int(rebuild2.get("template_count", rebuild_template_count) or rebuild_template_count)
    return JSONResponse(
        {
            "ok": True,
            "moved": moved,  # backward compatibility
            "merged_count": moved,
            "affected_phrase_ids": sorted(affected_phrase_ids),
            "rebuild_ok": rebuild_ok,
            "rebuild_ms": rebuild_ms,
            "rebuild_template_count": rebuild_template_count,
            "auto_purify": purify_result,
            "phrase_counts": v2_phrase_counts(load_v2_manifest()),
        }
    )


@app.post("/api/v2/index/rebuild")
def api_v2_rebuild_index(engine: Optional[str] = Query(default=None)) -> JSONResponse:
    result = v2_build_index()
    selected_engine = engine or str(load_active_v2_engine().get("engine_id") or ENGINE_V3_PERSONALIZED_TOP1)
    if selected_engine not in V2_ENGINE_IDS:
        return JSONResponse(status_code=400, content={"ok": False, "error": f"unknown engine: {selected_engine}"})
    if result.get("ok"):
        payload = {
            "engine_id": selected_engine,
            "updated_at": now_iso(),
            "reason": "manual_select_after_rebuild",
            "selection_mode": "force_top1_no_reject" if NO_REJECT_MODE else "standard",
        }
        save_json(V2_ACTIVE_ENGINE_PATH, payload)
        result["active_engine"] = payload
    result["requested_engine"] = selected_engine
    code = 200 if result.get("ok") else 400
    return JSONResponse(status_code=code, content=result)


@app.post("/api/v2/eval/run")
def api_v2_run_eval(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    try:
        engines = normalize_engine_list(payload)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    comparison = compare_v2_engines(engines)
    code = 200 if comparison.get("ok") else 400
    return JSONResponse(status_code=code, content=comparison)


@app.post("/api/v2/model/select")
def api_v2_select_model(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    engine_id = str(payload.get("engine_id", ""))
    report_file = str(payload.get("report_file", ""))
    result = select_v2_active_engine(engine_id, report_file)
    code = 200 if result.get("ok") else 400
    return JSONResponse(status_code=code, content=result)


@app.get("/api/v2/eval/export")
def api_v2_eval_export() -> JSONResponse:
    # Export should reflect the current model/index, not stale predictions stored when
    # the eval clip was first recorded.
    report = v2_run_eval(engine=str(load_active_v2_engine().get("engine_id") or "mfcc_dtw_baseline"))
    events = report.get("events", [])
    return JSONResponse({"ok": True, "count": len(events), "summary": report.get("summary"), "events": events, "report_file": report.get("report_file")})


@app.get("/api/v2/unknown/export")
def api_v2_unknown_export() -> JSONResponse:
    engine = str(load_active_v2_engine().get("engine_id") or "mfcc_dtw_baseline")
    summary = v2_run_unknown_eval(engine)
    events = v2_unknown_records(load_v2_manifest())
    return JSONResponse({"ok": True, "count": len(events), "summary": summary, "events": events, "engine_id": engine})


@app.post("/api/v2/eval/reset")
def api_v2_eval_reset() -> JSONResponse:
    manifest = load_v2_manifest()
    manifest["events"] = []
    save_v2_manifest(manifest)
    return JSONResponse({"ok": True, "phrase_counts": v2_phrase_counts(manifest)})


@app.post("/api/v2/unknown/reset")
def api_v2_unknown_reset() -> JSONResponse:
    manifest = load_v2_manifest()
    manifest["unknown_events"] = []
    save_v2_manifest(manifest)
    return JSONResponse({"ok": True, "summary": v2_unknown_summary([])})


@app.get("/api/v2/audio")
def api_v2_audio(file_rel: str = Query(...)) -> FileResponse:
    try:
        path = v2_path_for_rel(file_rel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/v3/phrases")
def api_v3_phrases() -> JSONResponse:
    manifest = ensure_v3_dataset(seed_from_v2=False)
    counts = v3_phrase_counts(manifest)
    index_meta = load_v3_index_meta()
    templates_by_phrase = index_meta.get("templates_by_phrase", {}) if isinstance(index_meta, dict) else {}
    phrases = []
    for phrase in PHRASE_PACK:
        pid = phrase["id"]
        phrases.append(
            {
                "phrase_id": pid,
                "text": phrase["text"],
                "stage_pack": pid in set(ACTIVE_PHRASE_IDS_DEFAULT),
                **counts.get(pid, {"train": 0, "eval": 0, "corrections": 0, "rejected": 0}),
                "template_count": templates_by_phrase.get(pid, 0) if isinstance(templates_by_phrase, dict) else 0,
                "train_target_count": TRAIN_TARGET_PER_PHRASE,
                "eval_target_count": EVAL_TARGET_PER_PHRASE,
            }
        )
    return JSONResponse({"ok": True, "phrases": phrases})


@app.get("/api/v3/engines")
def api_v3_engines() -> JSONResponse:
    return JSONResponse({"ok": True, "active_engine": load_active_v3_engine(), "engines": get_v3_engine_registry()})


@app.get("/api/v3/model/card")
def api_v3_model_card() -> JSONResponse:
    return JSONResponse(v3_model_card())


@app.get("/api/v3/runtime/config")
def api_v3_runtime_config() -> JSONResponse:
    return JSONResponse({"ok": True, "config": runtime_config_snapshot()})


@app.get("/api/v3/data/audit")
def api_v3_data_audit() -> JSONResponse:
    return JSONResponse(v3_dataset_audit())


@app.post("/api/v3/dataset/reset")
def api_v3_dataset_reset() -> JSONResponse:
    result = reset_v3_dataset_hard()
    return JSONResponse(result)


@app.post("/api/v3/samples/upload")
async def api_v3_upload_train_sample(
    phrase_id: str = Form(...),
    speaker_id: str = Form("speaker_default"),
    rebuild_policy: str = Form("deferred"),
    file: UploadFile = File(...),
) -> JSONResponse:
    data = await file.read()
    ok, payload = v3_store_audio_sample(data, phrase_id=phrase_id, split="train", source="train_capture", speaker_id=speaker_id)
    if not ok:
        return JSONResponse(status_code=400, content=payload)

    policy = normalize_rebuild_policy(rebuild_policy)
    started = time.perf_counter()
    rebuild_info: Dict[str, Any]
    if policy == "immediate":
        rebuild = v3_build_index()
        rebuild_info = {
            "triggered": True,
            "reason": "immediate_policy",
            "ok": bool(rebuild.get("ok")),
            "template_count": rebuild.get("template_count", 0),
            "duration_ms": round((time.perf_counter() - started) * 1000.0, 2),
            "index_state": "ready" if rebuild.get("ok") else "rebuild_failed",
        }
    else:
        rebuild_info = v3_maybe_batch_rebuild(phrase_id)

    manifest = load_v3_manifest()
    payload["phrase_counts"] = v3_phrase_counts(manifest)
    payload["rebuild_policy"] = policy
    payload["rebuild_triggered"] = bool(rebuild_info.get("triggered"))
    payload["rebuild_ok"] = bool(rebuild_info.get("ok")) if rebuild_info.get("triggered") else None
    payload["template_count"] = rebuild_info.get("template_count", 0)
    payload["duration_ms"] = rebuild_info.get("duration_ms")
    payload["index_state"] = rebuild_info.get("index_state", "pending_rebuild")
    payload["batch_rebuild"] = rebuild_info
    return JSONResponse(payload)


@app.post("/api/v3/eval/upload")
async def api_v3_upload_eval_sample(
    truth_phrase_id: str = Form(...),
    speaker_id: str = Form("speaker_default"),
    file: UploadFile = File(...),
) -> JSONResponse:
    data = await file.read()
    ok, payload = v3_store_audio_sample(data, phrase_id=truth_phrase_id, split="eval", source="eval_capture", speaker_id=speaker_id)
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    sample = payload["sample"]
    path = v3_path_for_rel(sample["file_rel"])
    with v3_runtime_context():
        matched_phrase, debug = v2_match_audio(path)
    sample["raw_audio_url"] = f"/api/v3/audio?file_rel={sample['file_rel']}"
    event = v3_append_eval_event(sample, matched_phrase, debug)
    is_error_event = v3_is_error_eval_event(event)
    auto_add_result = None
    event_id = event.get("event_id")
    if is_error_event and isinstance(event_id, str):
        # Auto-stage misclassified eval clips into corrections to avoid manual-loss cases.
        auto_add_result = v3_add_corrections_from_eval_event_ids([event_id], speaker_id=speaker_id)
    manifest = load_v3_manifest()
    return JSONResponse(
        {
            "ok": True,
            "sample": sample,
            "event": event,
            "is_error_event": bool(is_error_event),
            "next_action": "rerecord_truth" if is_error_event else "none",
            "next_truth_phrase_id": truth_phrase_id if is_error_event else None,
            "auto_correction": auto_add_result,
            "summary": v2_eval_summary(v3_eval_records(manifest)),
            "phrase_counts": v3_phrase_counts(manifest),
        }
    )


@app.post("/api/v3/unknown/upload")
async def api_v3_upload_unknown_sample(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    ok, payload = v3_store_unknown_audio(data, source="unknown_capture")
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    sample = payload["sample"]
    path = v3_path_for_rel(sample["file_rel"])
    with v3_runtime_context():
        matched_phrase, debug = v2_match_audio(path)
    event = v3_append_unknown_event(sample, matched_phrase, debug)
    manifest = load_v3_manifest()
    return JSONResponse(
        {
            "ok": True,
            "sample": sample,
            "event": event,
            "summary": v2_unknown_summary(v3_unknown_records(manifest)),
            "phrase_counts": v3_phrase_counts(manifest),
        }
    )


@app.post("/api/v3/demo/process")
async def api_v3_demo_process(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    return JSONResponse(v3_process_demo_audio(data))


@app.post("/api/v3/hybrid/process")
async def api_v3_hybrid_process(file: UploadFile = File(...)) -> JSONResponse:
    data = await file.read()
    return JSONResponse(v3_hybrid_process_demo_audio(data))


@app.post("/api/v3/corrections/upload")
async def api_v3_correction_upload(
    truth_phrase_id: str = Form(...),
    speaker_id: str = Form("speaker_default"),
    predicted_phrase_id: str = Form(""),
    raw_audio_url: str = Form(""),
    rerecord_batch_id: str = Form(""),
    from_eval_event_id: str = Form(""),
    file: UploadFile = File(...),
) -> JSONResponse:
    data = await file.read()
    source = "rerecord_correction" if (rerecord_batch_id or from_eval_event_id) else "demo_correction"
    ok, payload = v3_store_audio_sample(data, phrase_id=truth_phrase_id, split="corrections", source=source, speaker_id=speaker_id)
    if not ok:
        return JSONResponse(status_code=400, content=payload)
    manifest = load_v3_manifest()
    correction = manifest["corrections"][-1]
    correction["predicted_phrase_id"] = predicted_phrase_id or None
    correction["raw_audio_url"] = raw_audio_url or None
    correction["rerecord_batch_id"] = rerecord_batch_id or None
    correction["from_eval_event_id"] = from_eval_event_id or None
    save_v3_manifest(manifest)
    return JSONResponse({"ok": True, "correction": correction, "phrase_counts": v3_phrase_counts(manifest)})


@app.post("/api/v3/corrections/confirm")
def api_v3_confirm_corrections(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        resp = api_v2_confirm_corrections(payload)
    body = json.loads(resp.body.decode("utf-8"))
    if isinstance(body, dict):
        body["phrase_counts"] = v3_phrase_counts(load_v3_manifest())
    return JSONResponse(status_code=resp.status_code, content=body)


@app.post("/api/v3/index/rebuild")
def api_v3_rebuild_index(payload: Dict[str, Any] = Body(default_factory=dict), engine: Optional[str] = Query(default=None)) -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    started = time.perf_counter()
    reason = str(payload.get("reason") or "manual_rebuild")
    with v3_runtime_context():
        result = v2_build_index()
        selected_engine = engine or str(load_active_v3_engine().get("engine_id") or ENGINE_V3_PERSONALIZED_TOP1)
        if selected_engine not in V2_ENGINE_IDS:
            return JSONResponse(status_code=400, content={"ok": False, "error": f"unknown engine: {selected_engine}"})
        if result.get("ok"):
            payload = {
                "engine_id": selected_engine,
                "updated_at": now_iso(),
                "reason": "manual_select_after_rebuild",
                "selection_mode": "force_top1_no_reject" if NO_REJECT_MODE else "standard",
            }
            save_json(V3_ACTIVE_ENGINE_PATH, payload)
            result["active_engine"] = payload
        result["requested_engine"] = selected_engine
        result["reason"] = reason
        result["templates_built"] = result.get("template_count", 0)
        result["duration_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        result["index_state"] = "ready" if result.get("ok") else "rebuild_failed"
    code = 200 if result.get("ok") else 400
    return JSONResponse(status_code=code, content=result)


@app.post("/api/v3/eval/run")
def api_v3_run_eval(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    try:
        engines = normalize_engine_list_v3(payload)
    except ValueError as exc:
        return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
    comparison = compare_v3_engines(engines)
    reports = comparison.get("reports_by_engine", {})
    if isinstance(reports, dict):
        local = reports.get(ENGINE_V3_PERSONALIZED_TOP1)
        if isinstance(local, dict):
            summary = local.get("summary", {}) if isinstance(local.get("summary"), dict) else {}
            error_buckets = local.get("error_buckets", {}) if isinstance(local.get("error_buckets"), dict) else {}
            counts = error_buckets.get("counts", {}) if isinstance(error_buckets.get("counts"), dict) else {}
            comparison["bucket_metrics"] = {
                "local_hit_rate": summary.get("top1_rate"),
                "cloud_fallback_hit_rate": None,
                "reject_rate": summary.get("reject_rate"),
                "accepted_wrong": counts.get("accepted_wrong", 0),
                "notes": "cloud fallback buckets require events from /api/v3/hybrid/process",
            }
    code = 200 if comparison.get("ok") else 400
    return JSONResponse(status_code=code, content=comparison)


@app.post("/api/v3/model/select")
def api_v3_select_model(payload: Dict[str, Any] = Body(...)) -> JSONResponse:
    engine_id = str(payload.get("engine_id", ""))
    report_file = str(payload.get("report_file", ""))
    result = select_v3_active_engine(engine_id, report_file)
    code = 200 if result.get("ok") else 400
    return JSONResponse(status_code=code, content=result)


@app.get("/api/v3/eval/export")
def api_v3_eval_export() -> JSONResponse:
    report = v3_run_eval(engine=str(load_active_v3_engine().get("engine_id") or ENGINE_V3_PERSONALIZED_TOP1))
    events = report.get("events", [])
    return JSONResponse({"ok": True, "count": len(events), "summary": report.get("summary"), "events": events, "report_file": report.get("report_file")})


@app.get("/api/v3/eval/errors")
def api_v3_eval_errors(low_margin_threshold: Optional[float] = Query(default=None)) -> JSONResponse:
    manifest = load_v3_manifest()
    events = v3_eval_records(manifest)
    payload = build_v3_eval_error_payload(events, low_margin_threshold=low_margin_threshold)
    return JSONResponse(
        {
            "ok": True,
            "count": len(payload.get("items", [])),
            "total_eval_events": len(events),
            "low_margin_threshold": VB_FALLBACK_MARGIN_THRESHOLD if low_margin_threshold is None else float(low_margin_threshold),
            **payload,
        }
    )


@app.get("/api/v3/train/health")
def api_v3_train_health() -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    with v3_runtime_context():
        report = v2_train_health_report(load_v2_manifest())
    return JSONResponse(report)


@app.post("/api/v3/samples/status")
def api_v3_samples_status(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    sample_ids_raw = payload.get("sample_ids")
    if not isinstance(sample_ids_raw, list):
        return JSONResponse(status_code=400, content={"ok": False, "error": "sample_ids must be string[]"})
    sample_ids = [str(x) for x in sample_ids_raw if isinstance(x, str) and x.strip()]
    status = str(payload.get("status") or "").strip().lower()
    reason = str(payload.get("reason") or "manual_status_update")
    with v3_runtime_context():
        result = v2_update_samples_status(sample_ids, status=status, reason=reason)
        if result.get("ok") and int(result.get("updated") or 0) > 0:
            rebuild = v2_build_index()
            result["rebuild_ok"] = bool(rebuild.get("ok"))
            result["template_count"] = int(rebuild.get("template_count", 0) or 0)
        else:
            result["rebuild_ok"] = None
            result["template_count"] = None
        result["health"] = v2_train_health_report(load_v2_manifest())
    code = 200 if result.get("ok") else 400
    return JSONResponse(status_code=code, content=result)


@app.post("/api/v3/corrections/from_eval")
def api_v3_corrections_from_eval(payload: Dict[str, Any] = Body(default_factory=dict)) -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    event_ids_raw = payload.get("event_ids")
    if not isinstance(event_ids_raw, list):
        return JSONResponse(status_code=400, content={"ok": False, "error": "event_ids must be string[]"})
    event_ids = [str(x) for x in event_ids_raw if isinstance(x, str) and x.strip()]
    if not event_ids:
        return JSONResponse(status_code=400, content={"ok": False, "error": "event_ids is empty"})
    speaker_id = str(payload.get("speaker_id") or "speaker_default")
    result = v3_add_corrections_from_eval_event_ids(event_ids, speaker_id=speaker_id)
    return JSONResponse(result)


@app.get("/api/v3/corrections/pending")
def api_v3_corrections_pending(
    truth_phrase_id: Optional[str] = Query(default=None),
    rerecord_batch_id: Optional[str] = Query(default=None),
) -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    manifest = load_v3_manifest()
    corrections = manifest.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []

    truth_filter = str(truth_phrase_id).strip() if isinstance(truth_phrase_id, str) else ""
    batch_filter = str(rerecord_batch_id).strip() if isinstance(rerecord_batch_id, str) else ""

    items: List[Dict[str, Any]] = []
    for row in corrections:
        if not isinstance(row, dict):
            continue
        sample_id = row.get("sample_id")
        if not isinstance(sample_id, str) or not sample_id:
            continue
        row_truth = row.get("truth_phrase_id") or row.get("phrase_id")
        if truth_filter and str(row_truth or "") != truth_filter:
            continue
        row_batch = row.get("rerecord_batch_id")
        if batch_filter and str(row_batch or "") != batch_filter:
            continue
        items.append(
            {
                "sample_id": sample_id,
                "truth_phrase_id": row_truth if isinstance(row_truth, str) else None,
                "created_at": row.get("created_at") if isinstance(row.get("created_at"), str) else None,
                "from_eval_event_id": row.get("from_eval_event_id") if isinstance(row.get("from_eval_event_id"), str) else None,
                "rerecord_batch_id": row_batch if isinstance(row_batch, str) else None,
                "raw_audio_url": row.get("raw_audio_url") if isinstance(row.get("raw_audio_url"), str) else None,
            }
        )
    items.sort(key=lambda x: str(x.get("created_at") or ""))
    return JSONResponse({"ok": True, "count": len(items), "items": items})


@app.get("/api/v3/corrections/pending/")
def api_v3_corrections_pending_with_trailing_slash(
    truth_phrase_id: Optional[str] = Query(default=None),
    rerecord_batch_id: Optional[str] = Query(default=None),
) -> JSONResponse:
    # Compat alias: some clients append a trailing slash.
    return api_v3_corrections_pending(truth_phrase_id=truth_phrase_id, rerecord_batch_id=rerecord_batch_id)


@app.delete("/api/v3/corrections/pending/{sample_id}")
def api_v3_delete_pending_correction(sample_id: str) -> JSONResponse:
    ensure_v3_dataset(seed_from_v2=False)
    if not isinstance(sample_id, str) or not sample_id.strip():
        return JSONResponse(status_code=400, content={"ok": False, "error": "invalid_sample_id"})
    target_id = sample_id.strip()
    manifest = load_v3_manifest()
    corrections = manifest.get("corrections", [])
    if not isinstance(corrections, list):
        corrections = []

    target: Optional[Dict[str, Any]] = None
    keep: List[Dict[str, Any]] = []
    for row in corrections:
        if isinstance(row, dict) and target is None and str(row.get("sample_id") or "") == target_id:
            target = row
            continue
        if isinstance(row, dict):
            keep.append(row)

    if target is None:
        return JSONResponse(status_code=404, content={"ok": False, "error": "not_found"})

    file_rel = target.get("file_rel")
    if isinstance(file_rel, str) and file_rel:
        try:
            path = v3_path_for_rel(file_rel)
            if path.exists():
                path.unlink()
        except Exception:
            # Keep manifest deletion successful even if file cleanup fails.
            pass

    manifest["corrections"] = keep
    save_v3_manifest(manifest)
    return JSONResponse(
        {
            "ok": True,
            "deleted_sample_id": target_id,
            "corrections_count": len(keep),
            "phrase_counts": v3_phrase_counts(manifest),
        }
    )


@app.get("/api/v3/unknown/export")
def api_v3_unknown_export() -> JSONResponse:
    engine = str(load_active_v3_engine().get("engine_id") or ENGINE_V3_PERSONALIZED_TOP1)
    summary = v3_run_unknown_eval(engine)
    events = v3_unknown_records(load_v3_manifest())
    return JSONResponse({"ok": True, "count": len(events), "summary": summary, "events": events, "engine_id": engine})


@app.post("/api/v3/eval/reset")
def api_v3_eval_reset() -> JSONResponse:
    manifest = load_v3_manifest()
    manifest["events"] = []
    save_v3_manifest(manifest)
    return JSONResponse({"ok": True, "phrase_counts": v3_phrase_counts(manifest)})


@app.post("/api/v3/unknown/reset")
def api_v3_unknown_reset() -> JSONResponse:
    manifest = load_v3_manifest()
    manifest["unknown_events"] = []
    save_v3_manifest(manifest)
    return JSONResponse({"ok": True, "summary": v2_unknown_summary([])})


@app.get("/api/v3/audio")
def api_v3_audio(file_rel: str = Query(...)) -> FileResponse:
    try:
        path = v3_path_for_rel(file_rel)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not path.exists():
        raise HTTPException(status_code=404, detail="not_found")
    return FileResponse(path, media_type="audio/wav")
