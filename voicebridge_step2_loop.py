# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import threading
import time
import wave
from pathlib import Path
from typing import Dict, List

import edge_tts
import keyboard
import pyaudio
import requests


RATE = 16000
CHANNELS = 1
FORMAT = pyaudio.paInt16
CHUNK = 1024

TEMP_WAV_PATH = Path("temp.wav")
TEMP_TTS_PATH = Path("temp_tts.mp3")
SIGNATURE_DB_PATH = Path("signature_db.json")

SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1/audio/transcriptions"
ASR_PRIMARY_MODEL = "FunAudioLLM/SenseVoiceSmall"
ASR_FALLBACK_MODEL = "TeleAI/TeleSpeechASR"
TTS_VOICE = "zh-CN-YunxiNeural"

DEFAULT_SIGNATURE_DB = {
    "\u4f60\u597d": ["\u55ef\u55ef", "\u6069\u6069"],
    "\u6211\u60f3\u559d\u6c34": ["\u554a\u554a\u554a"],
}

ALNUM_CJK_PATTERN = re.compile(r"[\u4e00-\u9fffA-Za-z0-9]+")


def log(message: str, fallback: str | None = None) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(fallback or message.encode("ascii", "ignore").decode("ascii"), flush=True)


def clean_text(raw_text: str) -> str:
    if not raw_text:
        return ""
    tokens = ALNUM_CJK_PATTERN.findall(raw_text)
    return "".join(tokens)


def ensure_signature_db(path: Path) -> None:
    if path.exists():
        return
    path.write_text(
        json.dumps(DEFAULT_SIGNATURE_DB, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_signature_db(path: Path) -> Dict[str, List[str]]:
    ensure_signature_db(path)
    raw = json.loads(path.read_text(encoding="utf-8"))

    signature_db: Dict[str, List[str]] = {}
    # v1: { "你好": ["嗯嗯", ...] }
    if all(isinstance(k, str) for k in raw.keys()):
        for canonical_phrase, signatures in raw.items():
            if not isinstance(canonical_phrase, str):
                continue
            if isinstance(signatures, list):
                normalized = []
                for item in signatures:
                    if not isinstance(item, str):
                        continue
                    cleaned = clean_text(item)
                    if cleaned:
                        normalized.append(cleaned)
                if normalized:
                    signature_db[canonical_phrase] = sorted(set(normalized))

    # v2: { "phrases": { "你好": {"signature_counts": {"嗯嗯": 3}} } }
    phrases_node = raw.get("phrases") if isinstance(raw, dict) else None
    if isinstance(phrases_node, dict):
        for canonical_phrase, detail in phrases_node.items():
            if not isinstance(canonical_phrase, str) or not isinstance(detail, dict):
                continue
            signature_counts = detail.get("signature_counts", {})
            if not isinstance(signature_counts, dict):
                continue
            normalized = []
            for sig, count in signature_counts.items():
                if not isinstance(sig, str) or not isinstance(count, int) or count <= 0:
                    continue
                cleaned = clean_text(sig)
                if cleaned:
                    normalized.append(cleaned)
            if normalized:
                signature_db[canonical_phrase] = sorted(set(normalized))
    return signature_db


def build_signature_lookup(signature_db: Dict[str, List[str]]) -> Dict[str, str]:
    lookup: Dict[str, str] = {}
    for canonical_phrase, signatures in signature_db.items():
        for signature in signatures:
            lookup[signature] = canonical_phrase
    return lookup


def play_audio_file(file_path: Path) -> None:
    if os.name == "nt":
        os.startfile(str(file_path))  # type: ignore[attr-defined]
        return

    # Keep simple cross-platform behavior for hackathon demo.
    if hasattr(os, "uname") and os.uname().sysname.lower() == "darwin":
        subprocess.run(["afplay", str(file_path)], check=False)
        return
    subprocess.run(["xdg-open", str(file_path)], check=False)


async def synthesize_tts_async(text: str, output_path: Path) -> None:
    communicator = edge_tts.Communicate(text=text, voice=TTS_VOICE)
    await communicator.save(str(output_path))


def synthesize_tts_with_fallback(text: str, output_path: Path) -> None:
    try:
        asyncio.run(synthesize_tts_async(text, output_path))
        return
    except Exception as exc:
        log(f"[i] edge-tts Python API 失败，尝试 CLI 回退: {exc}", "[i] fallback to edge-tts cli")

    subprocess.run(
        [
            str(Path(".venv") / "Scripts" / "edge-tts.exe"),
            "--text",
            text,
            "--voice",
            TTS_VOICE,
            "--write-media",
            str(output_path),
        ],
        check=True,
    )


class SiliconFlowASR:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key.strip()
        self.enabled = bool(self.api_key)
        if not self.enabled:
            log("[!] 未检测到 SILICONFLOW_API_KEY，ASR 将跳过。", "[!] SILICONFLOW_API_KEY missing; ASR disabled")

    def _transcribe_once(self, audio_path: Path, model_name: str) -> str:
        with audio_path.open("rb") as f:
            files = {
                "file": (audio_path.name, f, "audio/wav"),
            }
            data = {
                "model": model_name,
                "language": "zh",
                "temperature": "0.0",
            }
            headers = {"Authorization": f"Bearer {self.api_key}"}
            resp = requests.post(
                SILICONFLOW_BASE_URL,
                headers=headers,
                data=data,
                files=files,
                timeout=30,
            )
        resp.raise_for_status()
        payload = resp.json()
        return self._extract_text(payload)

    @staticmethod
    def _extract_text(payload: dict) -> str:
        # Primary shape: {"text": "..."}
        text = payload.get("text")
        if isinstance(text, str):
            return text.strip()

        # Compatibility fallback shapes.
        choices = payload.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                candidate = first.get("text")
                if isinstance(candidate, str):
                    return candidate.strip()
                message = first.get("message")
                if isinstance(message, dict):
                    content = message.get("content")
                    if isinstance(content, str):
                        return content.strip()
        return ""

    def transcribe(self, audio_path: Path) -> str:
        if not self.enabled:
            return ""

        try:
            return self._transcribe_once(audio_path, ASR_PRIMARY_MODEL)
        except Exception as primary_exc:
            log(
                f"[i] 主模型失败，尝试回退模型: {primary_exc}",
                "[i] primary model failed; trying fallback",
            )
            return self._transcribe_once(audio_path, ASR_FALLBACK_MODEL)


class VoiceBridgeLoop:
    def __init__(self) -> None:
        api_key = os.environ.get("SILICONFLOW_API_KEY", "")
        self.asr = SiliconFlowASR(api_key=api_key)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames: List[bytes] = []
        self.is_recording = False
        self._stop_event = threading.Event()
        self._recording_thread: threading.Thread | None = None
        self._state_lock = threading.Lock()

        self.signature_db = load_signature_db(SIGNATURE_DB_PATH)
        self.signature_lookup = build_signature_lookup(self.signature_db)
        log(f"[i] 已加载签名库: {len(self.signature_db)} 条标准句", f"[i] loaded signature db: {len(self.signature_db)} phrases")

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                data = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(data)
        except Exception as exc:
            log(f"[!] 录音流异常: {exc}", f"[!] audio stream error: {exc}")
        finally:
            if self.stream is not None:
                try:
                    self.stream.stop_stream()
                except Exception:
                    pass
                try:
                    self.stream.close()
                except Exception:
                    pass
                self.stream = None

    def start_recording(self, _event=None) -> None:
        with self._state_lock:
            if self.is_recording:
                return
            self.frames = []
            self._stop_event.clear()
            try:
                self.stream = self.audio.open(
                    format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )
            except Exception as exc:
                log(f"[!] 麦克风打开失败: {exc}", f"[!] microphone open failed: {exc}")
                return

            self.is_recording = True
            self._recording_thread = threading.Thread(
                target=self._capture_loop,
                name="voicebridge-recorder",
                daemon=True,
            )
            self._recording_thread.start()
            log("[🔴 录音中...]", "[REC] recording...")

    def stop_recording(self, _event=None) -> None:
        with self._state_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._stop_event.set()
            worker = self._recording_thread

        if worker is not None and worker.is_alive():
            worker.join(timeout=2)

        if not self.frames:
            log("[i] 空录音，跳过本轮", "[i] empty audio, skip round")
            return

        self._save_temp_wav()
        log("[🔴 物理特征提取完毕]", "[OK] captured")
        self._asr_match_and_speak()

    def _save_temp_wav(self) -> None:
        with wave.open(str(TEMP_WAV_PATH), "wb") as wf:
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.audio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.frames))

    def _asr_match_and_speak(self) -> None:
        try:
            raw_text = self.asr.transcribe(TEMP_WAV_PATH)
        except Exception as exc:
            log(f"[!] ASR 调用失败: {exc}", f"[!] asr failed: {exc}")
            return

        if not raw_text:
            if self.asr.enabled:
                log("[🦻 ASR 原始结果]: <empty>", "[ASR] empty")
            else:
                log("[🦻 ASR 原始结果]: <skipped, no api key>", "[ASR] skipped (no key)")
            return

        log(f"[🦻 ASR 原始结果]: {raw_text}", f"[ASR] {raw_text}")
        cleaned = clean_text(raw_text)
        log(f"[🧹 清洗后签名]: {cleaned or '<empty>'}", f"[CLEAN] {cleaned or '<empty>'}")

        if not cleaned:
            log("[❌ 未命中]: 清洗后为空", "[MISS] cleaned empty")
            return

        matched = self.signature_lookup.get(cleaned)
        if not matched:
            log(f"[❌ 未命中]: {cleaned}", f"[MISS] {cleaned}")
            return

        log(f"[🎯 命中特征]: {matched}", f"[HIT] {matched}")
        log("[🔊 替身发音中...]", "[SPEAK] speaking...")

        try:
            synthesize_tts_with_fallback(matched, TEMP_TTS_PATH)
            play_audio_file(TEMP_TTS_PATH)
        except Exception as exc:
            log(f"[!] TTS 失败: {exc}", f"[!] tts failed: {exc}")

    def close(self) -> None:
        try:
            self.stop_recording()
        finally:
            self.audio.terminate()


def main() -> None:
    log("按住空格录音，松开后自动 ASR/签名命中/TTS。按 Ctrl+C 退出。")
    engine = VoiceBridgeLoop()

    keyboard.on_press_key("space", engine.start_recording)
    keyboard.on_release_key("space", engine.stop_recording)

    try:
        while True:
            time.sleep(0.2)
    except KeyboardInterrupt:
        log("\n[i] 收到退出信号，正在关闭...", "\n[i] exiting...")
    finally:
        keyboard.unhook_all()
        engine.close()


if __name__ == "__main__":
    main()
