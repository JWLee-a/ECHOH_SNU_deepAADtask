#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모든 WAV 파일 앞에 COS WAV + 1초 무음 삽입
구조: [COS][1초 무음][원본 오디오]
입력: Stimuli_48k 하위 모든 wav
출력: Stimuli_48k_cos 폴더에 동일 구조로 저장
"""

import os
import numpy as np
from scipy.io import wavfile

# =========================================================
# 설정
# =========================================================
STIMULI_DIR = r"C:\Users\KIST\PycharmProjects\SNUexp\ASA\Stimuli_48k"
OUTPUT_DIR  = r"C:\Users\KIST\PycharmProjects\SNUexp\ASA\Stimuli_48k_cos"
COS_WAV     = r"C:\Users\KIST\PycharmProjects\SNUexp\ASA\Stimuli_48k\cosine_100ms_48k.wav"
SILENCE_SEC = 1.0  # COS 이후 무음 길이(초)

# =========================================================
# COS WAV 로드
# =========================================================
cos_sr, cos_data = wavfile.read(COS_WAV)
print(f"COS WAV 로드: {cos_sr}Hz, shape={cos_data.shape}, dtype={cos_data.dtype}")


# =========================================================
# 삽입 함수
# =========================================================
def prepend_cos(filepath, input_root, output_root):
    sr, data = wavfile.read(filepath)

    # 출력 경로 설정
    rel_path = os.path.relpath(filepath, input_root)
    out_path = os.path.join(output_root, rel_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # 샘플레이트 확인
    if sr != cos_sr:
        print(f"  [ERROR] SR 불일치 ({sr}Hz): {os.path.basename(filepath)}")
        return False

    orig_dtype = data.dtype

    # 모노/스테레오 통일
    def match_channels(d, ref):
        if d.ndim == 1 and ref.ndim == 2:
            return np.stack([d, d], axis=1)
        if d.ndim == 2 and ref.ndim == 1:
            return d[:, 0]
        return d

    cos_matched = match_channels(cos_data, data)

    # 1초 무음 생성
    n_silence = int(sr * SILENCE_SEC)
    if data.ndim == 1:
        silence = np.zeros(n_silence, dtype=orig_dtype)
    else:
        silence = np.zeros((n_silence, data.shape[1]), dtype=orig_dtype)

    # [COS][무음][원본] 합치기
    combined = np.concatenate([cos_matched, silence, data], axis=0)

    wavfile.write(out_path, sr, combined)
    print(f"  [OK] {os.path.basename(filepath)}")
    return True


# =========================================================
# 메인
# =========================================================
def main():
    done = 0
    errors = 0

    print(f"입력: {STIMULI_DIR}")
    print(f"출력: {OUTPUT_DIR}")
    print(f"구조: [COS({cos_data.shape[0]/cos_sr*1000:.0f}ms)] + [무음 {SILENCE_SEC}s] + [원본]\n")

    for root, dirs, files in os.walk(STIMULI_DIR):
        wav_files = [f for f in files if f.lower().endswith('.wav')
                     and f != os.path.basename(COS_WAV)]  # COS 파일 자체는 제외
        if not wav_files:
            continue

        print(f"\n📁 {root}")
        for fname in sorted(wav_files):
            fpath = os.path.join(root, fname)
            try:
                result = prepend_cos(fpath, STIMULI_DIR, OUTPUT_DIR)
                if result:
                    done += 1
            except Exception as e:
                print(f"  [ERROR] {fname}: {e}")
                errors += 1

    print("\n" + "="*50)
    print(f"완료! 처리: {done}개 / 오류: {errors}개")
    print("="*50)


if __name__ == "__main__":
    main()