"""漂浮物检测验证：images/有漂浮物（正样本）vs images/无漂浮物（负样本）。

流程与在线一致：水面分割（erode 25px）→ 漂浮物检测 → 水面范围过滤。
每张图输出：原始检出数 / 过滤后保留数 / 各类别置信度，并保存标注图到 data/float_check/。
"""
import glob
import os
import sys

import cv2
import numpy as np

from outlet_detect.float_detector import HttpFloatDetector, filter_to_water
from outlet_detect.reporter import annotate_float
from outlet_detect.water_seg import HttpWaterSegmenter

FLOAT_ENGINE = "floating/20261030_145216"
FLOAT_CONF = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3
ERODE_PX = 25
SEG_CONF = 0.25
OUT_DIR = "data/float_check"


def imread_cn(path):
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


def imwrite_cn(path, img):
    cv2.imencode(".jpg", img)[1].tofile(path)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    seg = HttpWaterSegmenter(erode_px=ERODE_PX, confidence_threshold=SEG_CONF)
    det = HttpFloatDetector(engine_id=FLOAT_ENGINE, confidence=FLOAT_CONF)

    for label in ("有漂浮物", "无漂浮物"):
        print(f"\n=== {label} ===")
        for p in sorted(glob.glob(f"images/{label}/*.jpg")):
            img = imread_cn(p)
            if img is None:
                print(f"{p} 读不了，跳过")
                continue
            name = os.path.basename(p)
            mask = seg.segment(img)
            water_pct = mask.sum() / 255 / mask.size * 100
            dets_all = det.detect(img)
            dets = filter_to_water(dets_all, mask)
            verdict = "有漂浮物" if dets else "干净"
            print(f"{name[:40]:42s} 水面{water_pct:5.1f}%  检出{len(dets_all)}→保留{len(dets)}  {verdict}")
            for d in dets_all:
                kept = d in dets
                x1, y1, x2, y2 = map(int, d.bbox)
                print(f"    {'✓' if kept else '✗'} {d.class_name} {d.confidence:.2f} "
                      f"bbox=({x1},{y1},{x2},{y2})")
            if dets_all:
                ann = annotate_float(img, dets_all)
                overlay = ann.copy()
                overlay[mask > 0] = (overlay[mask > 0] * 0.6
                                     + np.array([0, 255, 0]) * 0.4).astype(np.uint8)
                imwrite_cn(os.path.join(OUT_DIR, f"{label}_{name}"), overlay)


if __name__ == "__main__":
    main()
