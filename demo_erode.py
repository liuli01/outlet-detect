"""演示形态学腐蚀收缩水面边缘的效果。"""
import base64

import cv2
import numpy as np
import requests

cap = cv2.VideoCapture("videos/东城河阳光桥球机_1687161142095.mp4")
ok, frame = cap.read()
cap.release()
h, w = frame.shape[:2]
b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()

data = requests.post(
    "http://127.0.0.1:8005/infer",
    json={"engine_id": "river/20261029_152643", "image": b64, "confidence_threshold": 0.25},
    timeout=60,
).json()

mask = np.zeros((h, w), dtype=np.uint8)
for d in data["detections"]:
    pts = d.get("points") or []
    if len(pts) >= 3:
        cv2.fillPoly(mask, [np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)], 255)

for erode_px in (0, 15, 30):
    if erode_px:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * erode_px + 1,) * 2)
        m = cv2.erode(mask, k)
    else:
        m = mask
    cov = m.sum() / 255 / (h * w)
    overlay = frame.copy()
    overlay[m > 0] = (overlay[m > 0] * 0.4 + np.array([0, 255, 0]) * 0.6).astype(np.uint8)
    cv2.imwrite(f"water_erode_{erode_px}.jpg", overlay)
    print(f"腐蚀 {erode_px}px: 覆盖率 {cov:.1%}  →  water_erode_{erode_px}.jpg")
