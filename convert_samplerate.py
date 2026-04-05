#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 WAV 파일의 샘플레이트를 48000Hz로 변환
경로: Stimuli/day{n}block{n}/mixture/ 하위 전체 스캔
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly
from math import gcd

# =========================================================
# 설정
# =========================================================
STIMULI_DIR = r"/ASA/Stimuli"
OUTPUT_DIR  = r"/ASA/Stimuli_48k"  # 변환 결과 저장 폴더
TARGET_SR = 48000

# =========================================================
# 변환 함수
# =========================================================
def resample_wav(filepath, target_sr, input_root, output_root):
    orig_sr, data = wavfile.read(filepath)

    # 원본과 동일한 상대경로로 출력 폴더에 저장
    rel_path = os.path.relpath(filepath, input_root)
    out_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if orig_sr == target_sr:
        # 샘플레이트 같아도 출력 폴더에 복사
        import shutil
        shutil.copy2(filepath, out_path)
        print(f"  [COPY] 이미 {target_sr}Hz (복사만): {os.path.basename(filepath)}")
        return False

    # 원본 dtype 저장
    orig_dtype = data.dtype

    # float32로 변환 후 리샘플
    if np.issubdtype(orig_dtype, np.integer):
        data_f = data.astype(np.float64) / np.iinfo(orig_dtype).max
    else:
        data_f = data.astype(np.float64)

    # 업/다운샘플 비율 계산
    g = gcd(target_sr, orig_sr)
    up = target_sr // g
    down = orig_sr // g

    if data_f.ndim == 1:
        resampled = resample_poly(data_f, up, down)
    else:
        # 스테레오
        resampled = np.stack([
            resample_poly(data_f[:, ch], up, down)
            for ch in range(data_f.shape[1])
        ], axis=1)

    # 원본 dtype으로 복원
    if np.issubdtype(orig_dtype, np.integer):
        max_val = np.iinfo(orig_dtype).max
        resampled = np.clip(resampled * max_val, -max_val - 1, max_val).astype(orig_dtype)
    else:
        resampled = resampled.astype(orig_dtype)

    wavfile.write(out_path, target_sr, resampled)
    print(f"  [OK] {orig_sr}Hz → {target_sr}Hz: {os.path.basename(filepath)}")
    return True


# =========================================================
# 메인: Stimuli 하위 모든 mixture 폴더 스캔
# =========================================================
def main():
    converted = 0
    skipped = 0
    errors = 0

    print(f"입력: {STIMULI_DIR}")
    print(f"출력: {OUTPUT_DIR}")

    for root, dirs, files in os.walk(STIMULI_DIR):
        wav_files = [f for f in files if f.lower().endswith('.wav')]
        if not wav_files:
            continue

        print(f"\n📁 {root}")
        for fname in sorted(wav_files):
            fpath = os.path.join(root, fname)
            try:
                result = resample_wav(fpath, TARGET_SR, STIMULI_DIR, OUTPUT_DIR)
                if result:
                    converted += 1
                else:
                    skipped += 1
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
                errors += 1

    print("\n" + "="*50)
    print(f"완료! 변환: {converted}개 / 스킵: {skipped}개 / 오류: {errors}개")
    print("="*50)


if __name__ == "__main__":
    main()