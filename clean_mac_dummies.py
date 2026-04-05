import os

base = r"C:\Users\KIST\Desktop\Matrix3"

deleted = 0
for root, dirs, files in os.walk(base):
    for fname in files:
        if fname.startswith('._'):
            fpath = os.path.join(root, fname)
            os.remove(fpath)
            print(f"삭제: {fpath}")
            deleted += 1

print(f"\n총 {deleted}개 삭제 완료!")