import json

from main import build_pipelines


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
    deploy = tmp_path / "deploy.json"
    deploy.write_text(json.dumps({
        "hik_base_url": "https://hik:14443", "app_key": "K", "app_secret": "S",
        "minio_endpoint": "m:9000", "minio_ak": "a", "minio_sk": "s",
        "report_url": "http://r/api", "external_host": "ext"}), encoding="utf-8")
    pipes = build_pipelines(str(deploy))
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
    deploy = tmp_path / "deploy.json"
    deploy.write_text(json.dumps({
        "hik_base_url": "https://hik:14443", "app_key": "K", "app_secret": "S",
        "minio_endpoint": "m:9000", "minio_ak": "a", "minio_sk": "s",
        "report_url": "http://r/api", "external_host": "ext"}), encoding="utf-8")
    pipes = build_pipelines(str(deploy))  # 不抛异常
    assert "cam02" in pipes
    ar = pipes["cam02"].aligner.align(None)  # _NoRefAligner 行为
    assert ar.ok is False
