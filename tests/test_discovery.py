import numpy as np
import pytest

from outlet_detect.config import CameraConfig, load_camera_config, save_camera_config
from outlet_detect.discovery import auto_register


@pytest.fixture
def cam_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    return tmp_path


def _frame():
    return np.full((60, 80, 3), 128, dtype=np.uint8)


def test_auto_register_creates_config_with_centroid_mode(cam_dir):
    created = auto_register(["camA"], lambda body: "http://x/pic.jpg",
                            lambda url: _frame(), ref_dir=str(cam_dir / "ref"),
                            float_engine_id="floating/1")
    assert created == ["camA"]
    cfg = load_camera_config("camA")
    assert cfg.outfall_point is None                  # 质心兜底
    assert cfg.hik_body["cameraIndexCode"] == "camA"
    assert cfg.float_engine_id == "floating/1"
    assert (cam_dir / "ref" / "camA.jpg").exists()    # 参照帧已落盘


def test_auto_register_skips_existing(cam_dir):
    save_camera_config(CameraConfig(camera_id="camA",
                                    hik_body={"cameraIndexCode": "camA"}))
    called = []
    created = auto_register(["camA", "camB"],
                            lambda body: called.append(body) or "http://x/p.jpg",
                            lambda url: _frame(), ref_dir=str(cam_dir / "ref"))
    assert created == ["camB"]                        # camA 已有标定，不抓图不覆盖
    assert len(called) == 1


def test_auto_register_survives_capture_failure(cam_dir):
    def bad_capture(body):
        raise RuntimeError("平台不可达")
    created = auto_register(["camA", "camB"], bad_capture,
                            lambda url: _frame(), ref_dir=str(cam_dir / "ref"))
    assert created == []                              # 失败跳过，不抛异常
