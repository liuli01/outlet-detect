# tests/test_calibrate_tool.py
import numpy as np
from outlet_detect.calibrate_tool import generate_config
from outlet_detect.water_seg import StubWaterSegmenter
from outlet_detect.config import load_camera_config

def test_generate_config(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr("outlet_detect.calibrate_tool.REF_DIR", tmp_path / "refs")
    frame = np.full((150, 200, 3), 128, dtype=np.uint8)
    cfg = generate_config("cam01", frame, (100, 100), StubWaterSegmenter(),
                          hik_body={"cameraIndexCode": "x"})
    assert cfg.outfall_point == (100, 100)
    loaded = load_camera_config("cam01")
    assert loaded.camera_id == "cam01"
    assert loaded.reference_frame.endswith("cam01_ref.jpg")
