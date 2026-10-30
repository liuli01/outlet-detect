import base64

import numpy as np
import pytest

from outlet_detect.float_detector import HttpFloatDetector

RESP = {"detections": [
    {"class_name": "bottle", "confidence": 0.88, "bbox": [10, 20, 50, 60]},
    {"class_name": "aeration", "confidence": 0.95, "bbox": [100, 100, 200, 200]},
    {"class_name": "bag", "confidence": 0.4, "bbox": [5, 5, 15, 15]},  # 低于阈值
], "engine_id": "float/1", "inference_time_ms": 50}


def make_detector(response=RESP, captured=None, **kw):
    def fake_post(url, payload, timeout):
        if captured is not None:
            captured.update(payload)
        return response
    return HttpFloatDetector(engine_id="float/1", post_fn=fake_post, **kw)


def test_disabled_when_no_engine_id():
    assert HttpFloatDetector(engine_id="").enabled is False
    assert HttpFloatDetector(engine_id="float/1").enabled is True


def test_detect_filters_ignore_and_threshold():
    # 注意：confidence_threshold 已随请求发给服务端，这里 0.4 的 bag 模拟服务端漏网，
    # 本地按 self.confidence 再过滤一次兜底
    det = make_detector(confidence=0.6, ignore_classes=["aeration"])
    out = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].class_name == "bottle" and out[0].confidence == 0.88
    assert out[0].bbox == (10, 20, 50, 60)


def test_empty_detections_returns_empty():
    det = make_detector(response={"detections": []})
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_service_error_raises():
    def bad_post(url, payload, timeout):
        raise ConnectionError("refused")
    det = HttpFloatDetector(engine_id="float/1", post_fn=bad_post)
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_malformed_response_raises():
    det = make_detector(response={"bad": "shape"})
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_payload_fields():
    captured = {}
    det = make_detector(captured=captured, confidence=0.6)
    det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
    assert captured["engine_id"] == "float/1"
    assert captured["confidence_threshold"] == 0.6
    assert len(base64.b64decode(captured["image"])) > 0  # image 可 base64 解码


def test_malformed_entries_raise_or_skip():
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    # null 条目 → RuntimeError
    det = make_detector(response={"detections": [None]})
    with pytest.raises(RuntimeError):
        det.detect(frame)
    # confidence 非数值 → RuntimeError
    det = make_detector(response={"detections": [
        {"class_name": "bottle", "confidence": "高", "bbox": [1, 2, 3, 4]}]})
    with pytest.raises(RuntimeError):
        det.detect(frame)
    # 缺 bbox 的合法条目 → 跳过，不伪造全零框
    det = make_detector(response={"detections": [
        {"class_name": "bottle", "confidence": 0.9},
        {"class_name": "bag", "confidence": 0.9, "bbox": [1, 2, 3, 4]}]})
    out = det.detect(frame)
    assert len(out) == 1 and out[0].class_name == "bag" and out[0].bbox == (1, 2, 3, 4)
