import numpy as np
import pytest

from outlet_detect.float_detector import HttpFloatDetector

RESP = {"detections": [
    {"class_name": "bottle", "confidence": 0.88, "bbox": [10, 20, 50, 60]},
    {"class_name": "aeration", "confidence": 0.95, "bbox": [100, 100, 200, 200]},
    {"class_name": "bag", "confidence": 0.4, "bbox": [5, 5, 15, 15]},  # 低于阈值
], "engine_id": "float/1", "inference_time_ms": 50}


def make_detector(response=RESP, **kw):
    def fake_post(url, payload, timeout):
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
