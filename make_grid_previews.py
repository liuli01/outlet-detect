"""给每个排污样本组生成带坐标网格的首帧预览，用于人工标排口点。"""
import glob
import os

import cv2
import numpy as np


def imread_cn(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


for g in sorted(glob.glob("images/排污/*/")):
    name = os.path.basename(os.path.normpath(g))
    f0 = imread_cn(os.path.join(g, "0.jpg"))
    if f0 is None:
        continue
    h, w = f0.shape[:2]
    out = f0.copy()
    step = 200
    for x in range(0, w, step):
        cv2.line(out, (x, 0), (x, h), (0, 0, 255), 2)
        cv2.putText(out, str(x), (x + 5, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    for y in range(0, h, step):
        cv2.line(out, (0, y), (w, y), (0, 0, 255), 2)
        cv2.putText(out, str(y), (5, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    cv2.imencode(".jpg", out)[1].tofile(f"grid_{name[:12]}.jpg")
    print(f"{name}: {w}x{h} → grid_{name[:12]}.jpg")
