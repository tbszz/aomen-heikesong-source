# -*- coding: utf-8 -*-
"""语料音频处理脚本 - 简化版"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import soundfile as sf


TARGET_SAMPLE_RATE = 16000
NORMALIZE_DB = -20


def process_audio_simple(input_path: Path, output_folder: Path, phrase_text: str, phrase_id: str):
    """简化处理：只做归一化和基础滤波"""
    try:
        # 读取
        audio, sr = sf.read(input_path)

        # 转单声道
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # 重采样
        if sr != TARGET_SAMPLE_RATE:
            import librosa
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SAMPLE_RATE)
            sr = TARGET_SAMPLE_RATE

        # 基础高通滤波（去除直流偏移）
        audio = audio - np.mean(audio)

        # 归一化到 -20dB RMS
        rms = np.sqrt(np.mean(audio ** 2))
        if rms > 1e-10:
            target_rms = 10 ** (NORMALIZE_DB / 20.0)
            audio = audio * (target_rms / rms)

        # 保存
        output_path = output_folder / input_path.name
        sf.write(output_path, audio, sr)

        return {
            "phrase_id": phrase_id,
            "phrase_text": phrase_text,
            "original": input_path.name,
            "processed": output_path.name,
            "duration_sec": len(audio) / sr,
        }

    except Exception as e:
        print(f"  [错误] {input_path.name}: {e}")
        return None


def main():
    corpus_raw = Path("corpus/raw")
    corpus_processed = Path("corpus/processed")
    corpus_processed.mkdir(parents=True, exist_ok=True)

    # 获取短语列表
    phrase_folders = sorted([d for d in corpus_raw.iterdir() if d.is_dir()])

    all_records = []
    total = 0

    for i, phrase_folder in enumerate(phrase_folders):
        phrase_text = phrase_folder.name
        phrase_id = f"p{i:02d}_{phrase_text[:4]}"

        output_folder = corpus_processed / phrase_text
        output_folder.mkdir(parents=True, exist_ok=True)

        print(f"[{phrase_id}] {phrase_text}")

        for wav_file in phrase_folder.glob("*.wav"):
            result = process_audio_simple(wav_file, output_folder, phrase_text, phrase_id)
            if result:
                all_records.append(result)
                total += 1
                if total % 50 == 0:
                    print(f"  已处理 {total} 个...")

    # 保存manifest
    manifest = {
        "version": 1,
        "created_at": datetime.now().isoformat(),
        "total_processed": total,
        "phrase_count": len(phrase_folders),
        "process_params": {
            "sample_rate": TARGET_SAMPLE_RATE,
            "normalize_db": NORMALIZE_DB,
        },
        "records": all_records,
    }

    manifest_path = Path("corpus/manifest.json")
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n=== 完成 ===")
    print(f"处理成功: {total}")
    print(f"清单: {manifest_path}")


if __name__ == "__main__":
    main()