"""诊断验证中的异常样本：分割覆盖率 / 帧差 / ROI 大小逐步检查。"""
import glob
import os

import cv2
import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根入 path，保证能 import outlet_detect

from outlet_detect.rois import build_rois
from outlet_detect.water_seg import HttpWaterSegmenter


def imread_cn(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


seg = HttpWaterSegmenter(erode_px=25)

SAMPLES = {
    "bba6cc73a8fda9ed46db7555acc95304": (550, 150),
    "创新河春申路河道球机_1687161684788": (1780, 750),
    "创新河春申路河道球机_1687161745744": (200, 500),
    "微信视频_20260128161856": (700, 120),
    "01521d22e3e199b46ac42aebcc68ab79": (730, 150),
}

for name, pt in SAMPLES.items():
    print(f"\n=== {name} 排口{pt} ===")
    paths = sorted(glob.glob(f"images/排污/{name}/*.jpg"))
    frames = [imread_cn(p) for p in paths]
    print(f"帧数 {len(frames)}, 尺寸 {[f.shape[:2] for f in frames]}")
    # 帧差检查：5 帧是否有差异
    diffs = []
    for a, b in zip(frames, frames[1:]):
        if a.shape != b.shape:
            diffs.append(-1)
            continue
        diffs.append(float(np.abs(a.astype(int) - b.astype(int)).mean()))
    print(f"相邻帧差均值: {[round(d, 2) for d in diffs]}")
    # 分割
    try:
        mask = seg.segment(frames[0])
        cov = mask.sum() / 255 / mask.size
        print(f"水面覆盖率: {cov:.1%}")
        if cov == 0:
            print("→ 无水面，pipeline 判 sample_error")
            continue
        rois = build_rois(mask, pt)
        print(f"ROI 像素: 排口={rois.outfall.sum()//255} 扩散={rois.plume.sum()//255} 参考={rois.reference.sum()//255}")
        # 排口点是否在 mask 内
        print(f"排口点在水面内: {bool(mask[pt[1], pt[0]])}")
    except Exception as e:
        print(f"分割异常: {type(e).__name__}: {e}")
