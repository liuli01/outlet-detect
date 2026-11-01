"""从 videos/ 抽 5 连帧组到 images/排污/<视频名>/，供正负样本验证。"""
import glob
import os

import cv2

os.makedirs("images/排污", exist_ok=True)

for p in sorted(glob.glob("videos/*.mp4")):
    stem = os.path.splitext(os.path.basename(p))[0]
    out_dir = f"images/排污/{stem}"
    if os.path.isdir(out_dir):
        print(f"跳过已存在: {stem}")
        continue
    cap = cv2.VideoCapture(p)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 250:
        print(f"打不开/太短: {stem} ({total} 帧)")
        cap.release()
        continue
    # 取视频中段一组 5 连帧（间隔 50 帧≈2 秒，与在线 burst 节奏一致）
    start = max(0, total // 2 - 125)
    os.makedirs(out_dir)
    n_saved = 0
    for k in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, start + k * 50)
        ok, f = cap.read()
        if ok:
            # cv2.imwrite 不支持中文路径，用 imencode + tofile
            cv2.imencode(".jpg", f)[1].tofile(f"{out_dir}/{k}.jpg")
            n_saved += 1
    cap.release()
    print(f"{stem}: 存 {n_saved} 帧 → {out_dir}")
