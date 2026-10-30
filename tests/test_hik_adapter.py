import numpy as np
import pytest

from outlet_detect.hik_adapter import HikApi, make_capture_fn, make_download_fn


class FakeResp:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def test_hik_api_signs_and_posts(monkeypatch):
    calls = []
    def fake_post(url, data=None, headers=None, verify=None, timeout=None):
        calls.append({"url": url, "headers": headers})
        return FakeResp({"code": "0", "msg": "success", "data": {"picUrl": "http://x/1.jpg"}})
    monkeypatch.setattr("requests.post", fake_post)
    api = HikApi("https://hik:14443", "/artemis/api/video/v1/manualCapture", "KEY", "SECRET")
    resp = api.post({"cameraIndexCode": "abc"})
    assert resp["data"]["picUrl"] == "http://x/1.jpg"
    h = calls[0]["headers"]
    assert h["x-ca-key"] == "KEY" and "x-ca-signature" in h and "x-ca-nonce" in h
    assert calls[0]["url"] == "https://hik:14443/artemis/api/video/v1/manualCapture"


def test_capture_fn_returns_pic_url(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp(
        {"msg": "success", "data": {"picUrl": "http://x/p.jpg"}}))
    cap = make_capture_fn("https://hik:14443", "K", "S")
    assert cap({"cameraIndexCode": "abc"}) == "http://x/p.jpg"


def test_capture_fn_raises_on_failure(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp({"msg": "denied"}))
    cap = make_capture_fn("https://hik:14443", "K", "S")
    with pytest.raises(RuntimeError):
        cap({"cameraIndexCode": "abc"})


def test_download_fn_decodes_image(monkeypatch):
    import cv2
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: FakeResp(content=cv2.imencode(".jpg", img)[1].tobytes()))
    out = make_download_fn()("http://x/1.jpg")
    assert out.shape == (10, 10, 3)


def test_download_fn_raises_on_bad_bytes(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(content=b"not an image"))
    with pytest.raises(RuntimeError):
        make_download_fn()("http://x/bad")
