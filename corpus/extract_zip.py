# -*- coding: utf-8 -*-
"""解压语料库zip文件"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import zipfile
import os
from pathlib import Path

SOURCE_DIR = Path(r"C:\Users\MY PC\Documents\xwechat_files\wxid_noktrzk3wa8r22_47d7\msg\attach\9e20f478899dc29eb19741386f9343c8\2026-05\Rec\10ad6f458aa1e135\F")
CORPUS_RAW = Path("corpus/raw")

def extract_all_zips():
    CORPUS_RAW.mkdir(parents=True, exist_ok=True)

    # 遍历 0-12 号目录
    for i in range(13):
        subdir = SOURCE_DIR / str(i)
        if not subdir.exists():
            print(f"[跳过] {subdir} 不存在")
            continue

        # 找zip文件
        zip_files = list(subdir.glob("*.zip"))
        if not zip_files:
            print(f"[跳过] {subdir} 无zip文件")
            continue

        for zip_file in zip_files:
            # 从zip文件名提取短语（去掉✅和.zip）
            phrase_name = zip_file.stem.replace("✅", "").strip()
            print(f"[解压] {phrase_name} <- {zip_file.name}")

            # 解压到对应目录
            extract_dir = CORPUS_RAW / phrase_name
            extract_dir.mkdir(parents=True, exist_ok=True)

            try:
                with zipfile.ZipFile(zip_file, 'r') as zf:
                    for name in zf.namelist():
                        # 跳过目录
                        if name.endswith('/'):
                            continue
                        # 只处理wav文件
                        if not name.endswith('.wav'):
                            continue
                        zf.extract(name, extract_dir)

                # 处理嵌套目录：将音频文件移动到正确位置
                for nested_dir in extract_dir.iterdir():
                    if nested_dir.is_dir():
                        for wav_file in nested_dir.glob("*.wav"):
                            # 移动到父目录
                            dst = extract_dir / wav_file.name
                            wav_file.rename(dst)
                        # 删除空嵌套目录
                        try:
                            nested_dir.rmdir()
                        except:
                            pass

                print(f"  -> 已解压到 {extract_dir}")
            except Exception as e:
                print(f"  -> 解压失败: {e}")

if __name__ == "__main__":
    extract_all_zips()
    print("\n=== 解压完成 ===")

    # 统计
    total_files = 0
    for d in CORPUS_RAW.iterdir():
        if d.is_dir():
            wavs = list(d.glob("*.wav"))
            total_files += len(wavs)
            print(f"  {d.name}: {len(wavs)} 个音频")
    print(f"总计: {total_files} 个音频文件")