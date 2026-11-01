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
                    data={"outfall_x": "150", "outfall_y": "150",
                          "float_engine": "floating/test"})
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


@pytest.fixture
def sample_tree(tmp_path, monkeypatch):
    """构造假样本库：images/排污/g1/5帧 + images/无漂浮物/1张。"""
    import outlet_detect.webapp as w
    root = tmp_path / "images"
    g1 = root / "排污" / "g1"
    g1.mkdir(parents=True)
    img = np.full((100, 150, 3), 100, dtype=np.uint8)
    for i in range(5):
        # 中文路径 cv2.imwrite 静默失败，用 imencode+tofile
        cv2.imencode(".jpg", img + i)[1].tofile(str(g1 / f"{i}.jpg"))
    (root / "无漂浮物").mkdir()
    cv2.imencode(".jpg", img)[1].tofile(str(root / "无漂浮物" / "a.jpg"))
    monkeypatch.setattr(w, "IMAGES_ROOT", root)
    return root


def test_index_lists_samples(client, sample_tree):
    r = client.get("/")
    assert "排污·g1" in r.text and "无漂浮物·a" in r.text
    assert "/sample_img?path=" in r.text


def test_sample_img_serves_and_guards(client, sample_tree):
    r = client.get("/sample_img", params={"path": "无漂浮物/a.jpg"})
    assert r.status_code == 200 and r.content[:2] == b"\xff\xd8"  # JPEG
    assert client.get("/sample_img", params={"path": "../pyproject.toml"}).status_code == 400
    assert client.get("/sample_img", params={"path": "排污/g1"}).status_code == 404  # 目录不是文件


def test_test_with_sample_path(client, monkeypatch, sample_tree):
    monkeypatch.setattr(webapp, "make_segmenter", lambda: FakeSeg())
    monkeypatch.setattr(webapp, "make_detector", lambda e, c: _FakeDet())
    r = client.post("/test", data={"sample_path": "排污/g1",
                                   "float_engine": "floating/test"})
    assert r.status_code == 200
    assert "样本组 g1（5 帧）" in r.text  # 真实5连帧而非复制
    assert "检测结果" in r.text
