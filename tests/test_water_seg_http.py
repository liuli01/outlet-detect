import json
from pathlib import Path

import numpy as np
import pytest

from outlet_detect.water_seg import HttpWaterSegmenter

FIXTURE = json.loads(
    Path(__file__).parent.joinpath("fixtures/water_infer_response.json").read_text(encoding="utf-8")
)

def make_segmenter(response):
    calls = []
    def fake_post(url, payload, timeout):
        calls.append(payload)
        return response
    return HttpWaterSegmenter(post_fn=fake_post), calls

def test_segment_rasterizes_polygon():
    seg, calls = make_segmenter(FIXTURE)
    frame = np.zeros((150, 200, 3), dtype=np.uint8)
    mask = seg.segment(frame)
    assert mask.shape == (150, 200) and mask.dtype == np.uint8
    # 多边形区域 (10..190, 100..140) 为水面
    assert mask[120, 100] == 255
    assert mask[50, 100] == 0  # 多边形外
    # 请求体契约
    assert calls[0]["engine_id"] == "river/20261030_132330"
    assert calls[0]["confidence_threshold"] == 0.25
    assert isinstance(calls[0]["image"], str) and len(calls[0]["image"]) > 100
    assert "image_url" not in calls[0]

def test_empty_detections_returns_zero_mask():
    seg, _ = make_segmenter({"detections": []})
    mask = seg.segment(np.zeros((100, 100, 3), dtype=np.uint8))
    assert mask.sum() == 0

def test_service_error_raises():
    def bad_post(url, payload, timeout):
        raise ConnectionError("refused")
    seg = HttpWaterSegmenter(post_fn=bad_post)
    with pytest.raises(RuntimeError):
        seg.segment(np.zeros((100, 100, 3), dtype=np.uint8))

def test_malformed_response_raises():
    seg, _ = make_segmenter({"unexpected": "shape"})
    with pytest.raises(RuntimeError):
        seg.segment(np.zeros((100, 100, 3), dtype=np.uint8))
