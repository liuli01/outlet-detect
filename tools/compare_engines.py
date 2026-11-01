"""对比新旧水面分割引擎在黑臭水体样本上的表现。"""
import base64

import cv2
import numpy as np
import requests

SAMPLES = [
    "bba6cc73a8fda9ed46db7555acc95304",
    "创新河春申路河道球机_1687161684788",
    "创新河春申路河道球机_1687161745744",
    "东城河阳光桥球机_1687161142095",
]
ENGINES = ["river/20261029_152643", "river/20261030_132330"]


def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


for name in SAMPLES:
    f = imread_cn(f"images/排污/{name}/0.jpg")
    h, w = f.shape[:2]
    b64 = base64.b64encode(cv2.imencode(".jpg", f)[1]).decode()
    print(f"\n=== {name} ({w}x{h}) ===")
    for eng in ENGINES:
        d = requests.post(
            "http://127.0.0.1:8005/infer",
            json={"engine_id": eng, "image": b64, "confidence_threshold": 0.25},
            timeout=60,
        ).json()
        dets = d.get("detections", [])
        mask = np.zeros((h, w), dtype=np.uint8)
        for det in dets:
            pts = det.get("points") or []
            if len(pts) >= 3:
                cv2.fillPoly(mask, [np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)], 255)
        cov = mask.sum() / 255 / (h * w)
        confs = [round(x["confidence"], 3) for x in dets]
        tag = "旧" if "1029" in eng else "新"
        print(f"  {tag}引擎: {len(dets)} 个 detection, conf={confs}, 覆盖率={cov:.1%}, "
              f"耗时={d.get('inference_time_ms', 0):.0f}ms")
