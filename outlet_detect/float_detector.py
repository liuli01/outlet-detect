"""漂浮物检测：复用 8005 推理服务（检测任务，返回 bbox）。"""
import os
from dataclasses import dataclass


@dataclass
class FloatDetection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


def filter_to_water(detections, water_mask):
    """只保留 bbox 中心落在水面 mask 内的检出（岸上误检过滤）。"""
    h, w = water_mask.shape[:2]
    out = []
    for d in detections:
        x1, y1, x2, y2 = d.bbox
        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
        if 0 <= cx < w and 0 <= cy < h and water_mask[cy, cx] > 0:
            out.append(d)
    return out


class HttpFloatDetector:
    def __init__(self, endpoint=None, engine_id="",
                 confidence=0.6, ignore_classes=(), timeout=30.0, post_fn=None):
        # 推理服务地址：显式参数 > 环境变量 INFER_ENDPOINT > 本机默认
        self.endpoint = endpoint or os.environ.get(
            "INFER_ENDPOINT", "http://127.0.0.1:8005/infer")
        self.engine_id = engine_id      # 空 = 关闭（模型未部署时的默认态）
        self.confidence = confidence
        self.ignore_classes = set(ignore_classes)
        self.timeout = timeout
        self.post_fn = post_fn or self._default_post

    @property
    def enabled(self):
        return bool(self.engine_id)

    @staticmethod
    def _default_post(url, payload, timeout):
        import requests
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def detect(self, frame):
        import base64
        import cv2
        try:
            ok, buf = cv2.imencode(".jpg", frame)
        except Exception as e:
            raise RuntimeError(f"漂浮物检测帧编码失败: {e}") from e
        if not ok:
            raise RuntimeError("漂浮物检测帧编码失败: cv2.imencode 返回 False")
        b64 = base64.b64encode(buf).decode()
        payload = {"engine_id": self.engine_id, "image": b64,
                   "confidence_threshold": self.confidence}
        try:
            data = self.post_fn(self.endpoint, payload, self.timeout)
        except Exception as e:
            raise RuntimeError(f"漂浮物检测服务调用失败: {e}") from e
        if not isinstance(data, dict) or "detections" not in data:
            raise RuntimeError(f"漂浮物检测响应格式异常: {str(data)[:200]}")
        out = []
        try:
            for det in data["detections"]:
                cls = str(det.get("class_name", ""))
                conf = float(det.get("confidence", 0.0))
                if cls in self.ignore_classes or conf < self.confidence:
                    continue
                bbox = det.get("bbox")
                if bbox is None:
                    continue  # 缺 bbox 的条目跳过，不伪造全零框
                out.append(FloatDetection(cls, conf, tuple(bbox)))
        except Exception as e:
            raise RuntimeError(f"漂浮物检测响应格式异常: {e}") from e
        return out
