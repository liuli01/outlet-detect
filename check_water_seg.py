"""验证 8005 水面分割服务：直接调接口，打印原始返回，保存 mask 叠加图。"""
import base64
import json
import sys

import cv2
import numpy as np
import requests

video = sys.argv[1] if len(sys.argv) > 1 else "videos/东城河阳光桥球机_1687161142095.mp4"

cap = cv2.VideoCapture(video)
ok, frame = cap.read()
cap.release()
assert ok, "打不开视频"
h, w = frame.shape[:2]
print(f"视频: {video}  首帧 {w}x{h}")

b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
resp = requests.post(
    "http://127.0.0.1:8005/infer",
    json={"engine_id": "river/20261030_132330", "image": b64, "confidence_threshold": 0.25},
    timeout=60,
)
data = resp.json()
print(f"HTTP {resp.status_code}  推理耗时 {data.get('inference_time_ms')}ms")
dets = data.get("detections", [])
print(f"detections: {len(dets)} 个")
for i, d in enumerate(dets):
    pts = d.get("points") or []
    print(f"  [{i}] class={d.get('class_name')} conf={d.get('confidence'):.3f} "
          f"bbox={d.get('bbox')} 多边形顶点数={len(pts)}")

# 栅格化 + 叠加显示
mask = np.zeros((h, w), dtype=np.uint8)
for d in dets:
    pts = d.get("points") or []
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)], 255)
water_px = int(mask.sum() / 255)
print(f"水面像素: {water_px} / {h*w}（{100*water_px/(h*w):.1f}%）")

overlay = frame.copy()
overlay[mask > 0] = (overlay[mask > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
cv2.imwrite("water_mask_overlay.jpg", overlay)
cv2.imwrite("water_mask.png", mask)
print("已保存 water_mask_overlay.jpg（绿色=识别为水面）和 water_mask.png（纯 mask）")
