import json

import pytest

from main import build_pipelines

ENV = {
    "HIK_BASE_URL": "https://hik:14443", "HIK_APP_KEY": "K", "HIK_APP_SECRET": "S",
    "MINIO_ENDPOINT": "m:9000", "MINIO_AK": "a", "MINIO_SK": "s",
    "REPORT_URL": "http://r/api", "EXTERNAL_HOST": "ext",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # 防止 load_dotenv 读到仓库根的真实 .env
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def test_build_pipelines_assembles(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    (cam_dir / "cam01.json").write_text(json.dumps({
        "camera_id": "cam01", "outfall_point": [640, 300],
        "reference_frame": "", "hik_body": {"cameraIndexCode": "abc"},
        "float_engine_id": "float/1"}), encoding="utf-8")
    (cam_dir / "vlm.json").write_text(json.dumps({
        "camera_id": "vlm", "outfall_point": [0, 0], "reference_frame": "",
        "hik_body": {}, "vlm_base_url": "http://v", "vlm_api_key": "k",
        "vlm_model": "", "vlm_envelope": True}), encoding="utf-8")
    pipes = build_pipelines()
    assert list(pipes) == ["cam01"]  # vlm.json 被跳过
    p = pipes["cam01"]
    assert p.float_detector.enabled is True
    assert p.vlm is not None and p.reporter is not None


def test_build_pipelines_corrupt_reference_frame(tmp_path, monkeypatch):
    """参照帧文件存在但损坏（非图片内容）：不抛异常，该摄像头降级为 _NoRefAligner。"""
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    bad_ref = tmp_path / "bad_ref.jpg"
    bad_ref.write_bytes(b"not an image")
    (cam_dir / "cam02.json").write_text(json.dumps({
        "camera_id": "cam02", "outfall_point": [640, 300],
        "reference_frame": str(bad_ref),
        "hik_body": {"cameraIndexCode": "def"},
        "float_engine_id": "float/1"}), encoding="utf-8")
    pipes = build_pipelines()  # 不抛异常
    assert "cam02" in pipes
    ar = pipes["cam02"].aligner.align(None)  # _NoRefAligner 行为
    assert ar.ok is False


def test_build_pipelines_missing_env_raises(tmp_path, monkeypatch):
    """缺环境变量时报错信息指出缺哪个键。"""
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    monkeypatch.delenv("HIK_APP_SECRET")
    with pytest.raises(RuntimeError, match="HIK_APP_SECRET"):
        build_pipelines()
