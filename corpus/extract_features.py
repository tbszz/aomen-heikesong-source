# -*- coding: utf-8 -*-
"""特征提取 + SVM 分类器训练"""
import sys
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import json
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline


SAMPLE_RATE = 16000
N_MFCC = 13


def extract_features(audio_path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """提取37维特征向量"""
    try:
        y, sr = librosa.load(audio_path, sr=sr, mono=True)
    except Exception as e:
        print(f"  [错误] 加载失败 {audio_path.name}: {e}")
        return None

    features = []

    # 1. MFCC (13维 -> 26维统计: 均值+标准差)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
    features.extend(np.mean(mfcc, axis=1))   # 13
    features.extend(np.std(mfcc, axis=1))    # 13

    # 2. 过零率 (1维 -> 2维: 均值+标准差)
    zcr = librosa.feature.zero_crossing_rate(y)
    features.append(np.mean(zcr))
    features.append(np.std(zcr))

    # 3. 短时能量 (2维: 总能量 + 平均能量)
    energy = np.sum(y**2)
    features.append(energy)
    features.append(energy / len(y))

    # 4. 频谱质心 (2维: 均值+标准差)
    cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    features.append(np.mean(cent))
    features.append(np.std(cent))

    # 5. 频谱带宽 (2维: 均值+标准差)
    band = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    features.append(np.mean(band))
    features.append(np.std(band))

    # 6. 频谱滚降点 (2维: 均值+标准差)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)
    features.append(np.mean(rolloff))
    features.append(np.std(rolloff))

    # 7. 频谱平坦度 (1维: 均值)
    flatness = librosa.feature.spectral_flatness(y=y)
    features.append(np.mean(flatness))

    return np.array(features)


def main():
    corpus_processed = Path("corpus/processed")

    # 收集所有音频和标签
    print("=== 1. 收集音频文件 ===")
    audio_files = []
    labels = []

    phrase_folders = sorted([d for d in corpus_processed.iterdir() if d.is_dir()])
    for phrase_folder in phrase_folders:
        phrase_text = phrase_folder.name
        for wav_file in phrase_folder.glob("*.wav"):
            audio_files.append(wav_file)
            labels.append(phrase_text)

    print(f"找到 {len(audio_files)} 个音频，{len(set(labels))} 个类别")

    # 提取特征
    print("\n=== 2. 提取特征 ===")
    X = []
    y = []
    valid_files = []

    for i, (audio_path, label) in enumerate(zip(audio_files, labels)):
        if (i + 1) % 50 == 0:
            print(f"  处理中... {i+1}/{len(audio_files)}")

        feat = extract_features(audio_path)
        if feat is not None:
            X.append(feat)
            y.append(label)
            valid_files.append(audio_path.name)

    X = np.array(X)
    y = np.array(y)

    print(f"特征矩阵: {X.shape}")
    print(f"类别: {np.unique(y)}")

    # 标签编码
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # 划分数据集
    print("\n=== 3. 划分数据集 ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    print(f"训练集: {len(X_train)}, 测试集: {len(X_test)}")

    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 训练SVM
    print("\n=== 4. 训练SVM分类器 ===")
    svm = SVC(kernel='rbf', C=10, gamma='scale', random_state=42, decision_function_shape='ovr')
    svm.fit(X_train_scaled, y_train)

    # 评估
    print("\n=== 5. 评估模型 ===")
    train_acc = svm.score(X_train_scaled, y_train)
    test_acc = svm.score(X_test_scaled, y_test)
    print(f"训练准确率: {train_acc:.4f}")
    print(f"测试准确率: {test_acc:.4f}")

    # 交叉验证
    X_all_scaled = scaler.fit_transform(X)
    cv_scores = cross_val_score(svm, X_all_scaled, y_encoded, cv=5)
    print(f"5折交叉验证: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

    # 详细分类报告
    y_pred = svm.predict(X_test_scaled)
    print("\n分类报告:")
    print(classification_report(
        y_test, y_pred,
        target_names=le.classes_,
        zero_division=0
    ))

    # 保存模型
    print("\n=== 6. 保存模型 ===")
    model_data = {
        "svm": svm,
        "scaler": scaler,
        "label_encoder": le,
        "feature_dim": X.shape[1],
        "classes": list(le.classes_),
        "created_at": datetime.now().isoformat(),
    }

    model_path = Path("corpus/svm_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    print(f"模型已保存: {model_path}")

    # 保存特征矩阵（用于后续分析）
    features_path = Path("corpus/features.npz")
    np.savez(features_path, X=X, y=y_encoded, labels=y)
    print(f"特征已保存: {features_path}")

    print("\n=== 完成 ===")


if __name__ == "__main__":
    main()