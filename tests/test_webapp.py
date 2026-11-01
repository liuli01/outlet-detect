import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from outlet_detect import webapp
from outlet_detect.float_detector import FloatDetection


@pytest.fixture
def client():
    return TestClient(webapp.app)


@pytest.fixture
def jpg_bytes():
    img = np.full((200, 300, 3), 128, dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


class FakeSeg:
    def segment(self, frame):
        h, w = frame.shape[:2]
        m = np.zeros((h, w), dtype=np.uint8)
        m[h // 2:, :] = 255
        return m


def test_index_page(client):
    r = client.get("/")
    assert r.status_code == 200 and "测试台" in r.text


def test_upload_runs_detection(client, monkeypatch, jpg_bytes):
    monkeypatch.setattr(webapp, "make_segmenter", lambda: FakeSeg())
    monkeypatch.setattr(webapp, "make_detector", lambda e, c: _FakeDet())
    r = client.post("/test", files={"image": ("a.jpg", jpg_bytes, "image/jpeg")},
                    data={"outfall_x": "150", "outfall_y": "150"})
    assert r.status_code == 200
    assert "检测结果" in r.text
    assert "综合评分" in r.text
    assert "bottle" in r.text           # 检出进入结果表
    assert "data:image/jpeg;base64" in r.text  # 标注图


class _FakeDet:
    def detect(self, frame):
        return [FloatDetection("bottle", 0.88, (100, 120, 160, 180))]  # 中心在下半幅水面


def test_seg_service_down_returns_502(client, monkeypatch, jpg_bytes):
    class BadSeg:
        def segment(self, frame):
            raise ConnectionError("8005 down")
    monkeypatch.setattr(webapp, "make_segmenter", lambda: BadSeg())
    r = client.post("/test", files={"image": ("a.jpg", jpg_bytes, "image/jpeg)" )},
                    data={})
    assert r.status_code == 502
    assert "水面分割服务异常" in r.text


def test_bad_image_returns_400(client):
    r = client.post("/test", files={"image": ("a.jpg", b"not an image", "image/jpeg")},
                    data={})
    assert r.status_code == 400
