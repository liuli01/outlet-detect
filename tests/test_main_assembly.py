import json
from pathlib import Path

import pytest

from main import build_pipelines

ENV = {
    "HIK_BASE_URL": "https://hik:14443", "HIK_APP_KEY": "K", "HIK_APP_SECRET": "S",
    "MINIO_ENDPOINT": "m:9000", "MINIO_AK": "a", "MINIO_SK": "s",
    "REPORT_URL": "http://r/api", "EXTERNAL_HOST": "ext",
}


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    # load_dotenv 的 find_dotenv 从调用文件向上查找，chdir 挡不住仓库根的真实 .env，
    # 直接打桩成 no-op，保证测试只认 monkeypatch 的环境变量
    monkeypatch.setattr("dotenv.load_dotenv", lambda *a, **k: False)
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)


def test_build_pipelines_assembles(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    monkeypatch.setenv("VLM_BASE_URL", "http://v")
    monkeypatch.setenv("VLM_API_KEY", "k")
    monkeypatch.setenv("VLM_MODEL", "")
    monkeypatch.setenv("VLM_ENVELOPE", "true")
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    (cam_dir / "cam01.json").write_text(json.dumps({
        "camera_id": "cam01", "outfall_point": [640, 300],
        "reference_frame": "", "hik_body": {"cameraIndexCode": "abc"},
        "float_engine_id": "float/1"}), encoding="utf-8")
    pipes = build_pipelines()
    assert list(pipes) == ["cam01"]
    p = pipes["cam01"]
    assert p.float_detector.enabled is True
    assert p.vlm is not None and p.reporter is not None


def test_vlm_disabled_when_no_base_url(tmp_path, monkeypatch):
    """VLM_BASE_URL 留空 → vlm=None（退化为规则+状态机）。"""
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    monkeypatch.delenv("VLM_BASE_URL", raising=False)
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    (cam_dir / "cam01.json").write_text(json.dumps({
        "camera_id": "cam01", "outfall_point": [640, 300],
        "reference_frame": "", "hik_body": {"cameraIndexCode": "abc"}}),
        encoding="utf-8")
    pipes = build_pipelines()
    assert pipes["cam01"].vlm is None


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


def test_auto_register_via_camera_list_url(tmp_path, monkeypatch):
    """CAMERA_LIST_URL 设置后：清单里未标定的摄像头自动建档并装配（质心模式）。"""
    import numpy as np
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    monkeypatch.setenv("CAMERA_LIST_URL", "http://platform/api/cameraindexcode")
    monkeypatch.setenv("FLOAT_ENGINE_ID", "floating/1")
    monkeypatch.setattr("outlet_detect.hik_adapter.get_index_code_list",
                        lambda url: ["camA"])
    monkeypatch.setattr("outlet_detect.hik_adapter.make_capture_fn",
                        lambda *a: lambda body: "http://x/p.jpg")
    monkeypatch.setattr("outlet_detect.hik_adapter.make_download_fn",
                        lambda *a, **k: lambda url: np.full((60, 80, 3), 128,
                                                            dtype=np.uint8))
    pipes = build_pipelines()
    assert "camA" in pipes                     # 自动建档 + 装配成功
    assert pipes["camA"].config.outfall_point is None
    assert pipes["camA"].float_detector.enabled
    assert Path("data/reference/camA.jpg").exists()  # cwd 已是 tmp_path


def test_camera_list_failure_keeps_existing(tmp_path, monkeypatch):
    """清单接口挂掉不影响已有标定摄像头。"""
    import json
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    monkeypatch.setenv("CAMERA_LIST_URL", "http://platform/api/cameraindexcode")
    def boom(url):
        raise ConnectionError("平台挂了")
    monkeypatch.setattr("outlet_detect.hik_adapter.get_index_code_list", boom)
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    (cam_dir / "camA.json").write_text(json.dumps({
        "camera_id": "camA", "hik_body": {"cameraIndexCode": "a"}}),
        encoding="utf-8")
    pipes = build_pipelines()                  # 不抛异常
    assert list(pipes) == ["camA"]


def test_main_reads_run_interval_env(tmp_path, monkeypatch):
    """RUN_INTERVAL_SEC 环境变量控制调度间隔；.env 在 build_pipelines 内加载。"""
    import main as m
    seen = {}
    monkeypatch.setenv("RUN_INTERVAL_SEC", "600")
    monkeypatch.setattr(m, "build_pipelines", lambda: {})
    def fake_run_loop(pipes, active_hours=None, interval_sec=300, **kw):
        seen["interval"] = interval_sec
        seen["hours"] = active_hours
    monkeypatch.setattr("outlet_detect.scheduler.run_loop", fake_run_loop)
    monkeypatch.setattr("sys.argv", ["main.py"])
    m.main()
    assert seen == {"interval": 600, "hours": (10, 15)}


def test_main_default_interval_300(tmp_path, monkeypatch):
    import main as m
    seen = {}
    monkeypatch.delenv("RUN_INTERVAL_SEC", raising=False)
    monkeypatch.setattr(m, "build_pipelines", lambda: {})
    monkeypatch.setattr("outlet_detect.scheduler.run_loop",
                        lambda pipes, active_hours=None, interval_sec=300, **kw:
                        seen.update(interval=interval_sec))
    monkeypatch.setattr("sys.argv", ["main.py"])
    m.main()
    assert seen["interval"] == 300
