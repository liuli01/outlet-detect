import json
from pathlib import Path
from outlet_detect.config import CameraConfig, load_camera_config, save_camera_config, list_camera_ids

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    cfg = CameraConfig(
        camera_id="cam01",
        outfall_point=(640, 300),
        reference_frame="archive/cam01/ref.jpg",
        hik_body={"cameraIndexCode": "abc"},
    )
    save_camera_config(cfg)
    loaded = load_camera_config("cam01")
    assert loaded.camera_id == "cam01"
    assert loaded.outfall_point == (640, 300)
    assert loaded.score_low == 0.4
    assert loaded.score_high == 0.7

def test_list_camera_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    save_camera_config(CameraConfig(camera_id="a", outfall_point=(0, 0),
                                    reference_frame="", hik_body={}))
    save_camera_config(CameraConfig(camera_id="b", outfall_point=(0, 0),
                                    reference_frame="", hik_body={}))
    assert sorted(list_camera_ids()) == ["a", "b"]
