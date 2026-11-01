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


def test_float_fields_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    cfg = CameraConfig(camera_id="cam02", outfall_point=(1, 2), reference_frame="",
                       hik_body={}, float_engine_id="float/1",
                       ignore_classes=["aeration"])
    save_camera_config(cfg)
    loaded = load_camera_config("cam02")
    assert loaded.float_engine_id == "float/1"
    assert loaded.float_confidence == 0.6
    assert loaded.ignore_classes == ["aeration"]


def test_old_config_json_loads_with_defaults(tmp_path, monkeypatch):
    """旧版 JSON（无新字段）加载不报错，取默认值。"""
    import json
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    (tmp_path / "old.json").write_text(json.dumps({
        "camera_id": "old", "outfall_point": [0, 0], "reference_frame": "",
        "hik_body": {}}), encoding="utf-8")
    loaded = load_camera_config("old")
    assert loaded.float_engine_id == "" and loaded.ignore_classes == []


def test_missing_outfall_point_loads_as_none(tmp_path, monkeypatch):
    """排口点缺省/null → None（运行时水面质心兜底）；保存 None 也不报错。"""
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    (tmp_path / "c1.json").write_text(json.dumps({
        "camera_id": "c1", "reference_frame": "",
        "hik_body": {"cameraIndexCode": "x"}}), encoding="utf-8")
    assert load_camera_config("c1").outfall_point is None
    (tmp_path / "c2.json").write_text(json.dumps({
        "camera_id": "c2", "outfall_point": None, "reference_frame": "",
        "hik_body": {}}), encoding="utf-8")
    assert load_camera_config("c2").outfall_point is None
    save_camera_config(CameraConfig(camera_id="c3"))
    assert load_camera_config("c3").outfall_point is None
