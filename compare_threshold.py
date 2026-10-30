"""对比 confidence_threshold 对水面分割结果的影响。"""
import base64

import cv2
import numpy as np
import requests

cap = cv2.VideoCapture("videos/东城河阳光桥球机_1687161142095.mp4")
ok, frame = cap.read()
cap.release()
assert ok
h, w = frame.shape[:2]
b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()

for thr in (0.25, 0.5, 0.7, 0.75):
    data = requests.post(
        "http://127.0.0.1:8005/infer",
        json={"engine_id": "river/20261029_152643", "image": b64, "confidence_threshold": thr},
        timeout=60,
    ).json()
    dets = data.get("detections", [])
    if not dets:
        print(f"阈值 {thr}: 无 detection（整片水面被过滤掉）")
        continue
    mask = np.zeros((h, w), dtype=np.uint8)
    for d in dets:
        pts = d.get("points") or []
        if len(pts) >= 3:
            cv2.fillPoly(mask, [np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)], 255)
    cov = mask.sum() / 255 / (h * w)
    verts = sum(len(d.get("points") or []) for d in dets)
    print(f"阈值 {thr}: {len(dets)} 个 detection, conf={dets[0]['confidence']:.3f}, "
          f"顶点 {verts}, 覆盖率 {cov:.1%}")
