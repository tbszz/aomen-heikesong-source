import threading
import wave
from pathlib import Path
import sys

import keyboard
import pyaudio


RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH_FORMAT = pyaudio.paInt16
CHUNK = 1024
OUTPUT_FILE = "temp.wav"


def log(message: str, fallback: str | None = None) -> None:
    try:
        print(message, flush=True)
    except UnicodeEncodeError:
        print(fallback or message.encode("ascii", "ignore").decode("ascii"), flush=True)


class SpaceHoldRecorder:
    def __init__(self) -> None:
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.frames: list[bytes] = []
        self.is_recording = False
        self._recording_thread: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._state_lock = threading.Lock()

    def _capture_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                chunk = self.stream.read(CHUNK, exception_on_overflow=False)
                self.frames.append(chunk)
        except Exception as exc:
            log(f"[!] 录音流发生异常: {exc}", f"[!] audio stream error: {exc}")
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
                    format=SAMPLE_WIDTH_FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK,
                )
            except Exception as exc:
                log(f"[!] 无法打开默认麦克风: {exc}", f"[!] cannot open default microphone: {exc}")
                return

            self.is_recording = True
            log("[🔴 录音中...]", "[REC] recording...")
            self._recording_thread = threading.Thread(
                target=self._capture_loop,
                name="space-hold-recorder",
                daemon=True,
            )
            self._recording_thread.start()

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
            log("[i] 本次未采集到有效音频，未生成 temp.wav", "[i] empty recording, temp.wav not generated")
            return

        output_path = Path.cwd() / OUTPUT_FILE
        try:
            with wave.open(str(output_path), "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(self.audio.get_sample_size(SAMPLE_WIDTH_FORMAT))
                wf.setframerate(RATE)
                wf.writeframes(b"".join(self.frames))
            log(f"[✅ 已保存] {output_path}", f"[OK] saved: {output_path}")
        except Exception as exc:
            log(f"[!] 写入 WAV 失败: {exc}", f"[!] failed to write wav: {exc}")

    def close(self) -> None:
        try:
            self.stop_recording()
        finally:
            self.audio.terminate()


def main() -> None:
    # Make console output robust on Windows terminals with legacy encodings.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(errors="replace")
        except Exception:
            pass

    recorder = SpaceHoldRecorder()

    log("按住空格开始录音，松开空格停止并保存为 temp.wav。")
    log("按 Ctrl+C 退出。")

    keyboard.on_press_key("space", recorder.start_recording)
    keyboard.on_release_key("space", recorder.stop_recording)

    try:
        keyboard.wait()
    except KeyboardInterrupt:
        log("\n[i] 收到退出信号，正在关闭...", "\n[i] exiting...")
    finally:
        recorder.close()
        keyboard.unhook_all()


if __name__ == "__main__":
    main()
