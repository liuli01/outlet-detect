import cv2
import numpy as np
from outlet_detect.replay_tool import extract_sample_frames, replay

def make_video(tmp_path, n_frames=600, fps=30):
    path = str(tmp_path / "t.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (200, 150))
    for i in range(n_frames):
        w.write(np.full((150, 200, 3), i % 255, dtype=np.uint8))
    w.release()
    return path

def test_extract_sample_frames(tmp_path):
    path = make_video(tmp_path, n_frames=600, fps=30)  # 20秒
    frames = extract_sample_frames(path, interval_sec=10, burst=3, burst_gap_frames=5)
    # 20秒视频、每10秒一轮、每轮3帧 → 2轮 × 3帧 = 6帧
    assert len(frames) == 6

def test_replay_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cfg")
    path = make_video(tmp_path, n_frames=1200, fps=30)  # 40秒 → 4轮 × 5帧 = 20帧
    results = replay(path, camera_id="r1", interval_sec=10,
                     storage_dir=str(tmp_path))
    assert len(results) == 4  # 20帧 ÷ 5帧/轮 = 4轮
    assert all(r in ("normal", "suspect", "alert", "gray",
                     "view_error", "sample_error") for r in results)
