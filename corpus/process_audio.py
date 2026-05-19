# -*- coding: utf-8 -*-
"""语料音频处理脚本 - 20Hz~1kHz带通滤波 + 归一化"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
from pathlib import Path
from typing import Dict, List, Optional

import librosa
import numpy as np
import soundfile as sf
from scipy import signal


# 处理参数
TARGET_SAMPLE_RATE = 16000
NORMALIZE_DB = -20  # 归一化到 -20dB RMS
HIGH_PASS_HZ = 10   # 高通滤波截止频率（降低以保留更多信号）
BAND_LOW_HZ = 10    # 带通下限
BAND_HIGH_HZ = 8000 # 带通上限（提高到8kHz以保留语音）
TRIM_TOP_DB = 20    # 静音去除阈值（降低敏感度）


def design_bandpass_filter(sr: int, low_hz: float, high_hz: float, order: int = 8) -> tuple:
    """设计带通滤波器"""
    nyq = sr / 2
    low = low_hz / nyq
    high = high_hz / nyq
    # 防止频率超出范围
    low = max(0.001, min(0.999, low))
    high = max(0.001, min(0.999, high))
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def design_highpass_filter(sr: int, cutoff_hz: float, order: int = 8) -> tuple:
    """设计高通滤波器（消除基线漂移）"""
    nyq = sr / 2
    cutoff = cutoff_hz / nyq
    cutoff = max(0.001, min(0.999, cutoff))
    b, a = signal.butter(order, cutoff, btype='high')
    return b, a


def normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """归一化音频到指定dB RMS"""
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-10:
        return audio
    target_rms = 10 ** (target_db / 20.0)
    return audio * (target_rms / rms)


def process_single_audio(input_path: Path, output_path: Path, sr: int = TARGET_SAMPLE_RATE) -> Optional[Dict]:
    """处理单个音频文件"""
    try:
        # 1. 读取音频
        audio, orig_sr = librosa.load(input_path, sr=sr, mono=True)

        if len(audio) < sr * 0.1:  # 小于0.1秒跳过
            print(f"  [跳过] {input_path.name} - 音频太短")
            return None

        # 2. 基线漂移消除 - 高通滤波 20Hz
        b, a = design_highpass_filter(sr, HIGH_PASS_HZ)
        audio = signal.filtfilt(b, a, audio)

        # 3. 带通滤波 - 20Hz ~ 1000Hz
        b, a = design_bandpass_filter(sr, BAND_LOW_HZ, BAND_HIGH_HZ)
        audio = signal.filtfilt(b, a, audio)

        # 4. 预加重（提升高频）
        audio = librosa.effects.preemphasis(audio)

        # 5. 端点检测 - 移除静音
        audio, _ = librosa.effects.trim(audio, top_db=TRIM_TOP_DB)

        # 确保音频不为空
        if len(audio) < sr * 0.1:
            print(f"  [跳过] {input_path.name} - 静音去除后为空")
            return None

        # 6. 归一化
        audio = normalize_audio(audio, NORMALIZE_DB)

        # 7. 保存
        sf.write(output_path, audio, sr)

        return {
            "file": input_path.name,
            "duration_sec": len(audio) / sr,
            "samples": len(audio),
        }

    except Exception as e:
        print(f"  [错误] {input_path.name}: {e}")
        return None


def process_all_audios():
    """处理所有语料音频"""
    corpus_raw = Path("corpus/raw")
    corpus_processed = Path("corpus/processed")
    manifest_path = Path("corpus/manifest.json")

    # 确保输出目录存在
    corpus_processed.mkdir(parents=True, exist_ok=True)

    # 短语映射（生成ID）
    phrase_folders = sorted([d for d in corpus_raw.iterdir() if d.is_dir()])
    phrase_to_id = {d.name: f"p{i:02d}_{d.name[:4]}" for i, d in enumerate(phrase_folders)}

    print(f"发现 {len(phrase_folders)} 个短语类别")

    all_records = []
    total_processed = 0
    total_skipped = 0

    for phrase_folder in phrase_folders:
        phrase_text = phrase_folder.name
        phrase_id = phrase_to_id[phrase_text]

        # 创建输出目录
        output_folder = corpus_processed / phrase_text
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"\n[{phrase_id}] {phrase_text}")

        # 处理每个音频
        wav_files = sorted(phrase_folder.glob("*.wav"))
        for wav_file in wav_files:
            output_path = output_folder / wav_file.name

            result = process_single_audio(wav_file, output_path)

            if result:
                all_records.append({
                    "phrase_id": phrase_id,
                    "phrase_text": phrase_text,
                    "original_file": result["file"],
                    "processed_file": str(output_path.relative_to("corpus")),
                    "duration_sec": result["duration_sec"],
                    "samples": result["samples"],
                })
                total_processed += 1
            else:
                total_skipped += 1

    # 保存manifest
    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "total_original": total_processed + total_skipped,
        "total_processed": total_processed,
        "total_skipped": total_skipped,
        "phrase_count": len(phrase_folders),
        "process_params": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "normalize_db": NORMALIZE_DB,
            "high_pass_hz": HIGH_PASS_HZ,
            "band_low_hz": BAND_LOW_HZ,
            "band_high_hz": BAND_HIGH_HZ,
            "trim_top_db": TRIM_TOP_DB,
        },
        "phrase_mapping": phrase_to_id,
        "records": all_records,
    }

    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n=== 处理完成 ===")
    print(f"处理成功: {total_processed}")
    print(f"跳过: {total_skipped}")
    print(f"清单已保存: {manifest_path}")

    return manifest


if __name__ == "__main__":
    from datetime import datetime
    process_all_audios()