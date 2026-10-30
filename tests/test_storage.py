# tests/test_storage.py
import numpy as np
from outlet_detect.storage import Storage

def make_frame():
    return np.full((100, 100, 3), 128, dtype=np.uint8)

def test_record_and_baseline(tmp_path):
    s = Storage(db_path=str(tmp_path / "t.db"), archive_root=str(tmp_path / "arc"))
    feats = {"rel_darkness": 0.9, "flow_energy": 1.2, "color_dev": 0.1, "baseline_dev": 0.0}
    s.record_sample("cam01", "2026-10-28T10:00:00", feats, "normal", None, None,
                    "normal", ["a.jpg"], 120)
    s.record_sample("cam01", "2026-10-27T10:05:00", {**feats, "rel_darkness": 0.8},
                    "normal", None, None, "normal", ["b.jpg"], 100)
    baseline = s.get_baseline_darkness("cam01", hour=10, before_ts="2026-10-28T10:00:00", days=7)
    assert baseline == 0.8  # 排除了今天 10:00 之外昨天同时刻的值

def test_record_report(tmp_path):
    s = Storage(db_path=str(tmp_path / "t.db"), archive_root=str(tmp_path / "arc"))
    s.record_report("cam01", "2026-10-30T12:00:00", "float", 0.88, 0.0, 0.0,
                    "http://x/out.jpg", 1, "")
    s.record_report("cam01", "2026-10-30T12:00:30", "discharge", 0.0, 0.72, 0.72,
                    "", 0, "minio down")
    rows = s.conn.execute(
        "SELECT camera_id,ts,event_type,result_a,result_b,result_c,output_url,ok,error"
        " FROM reports ORDER BY id").fetchall()
    assert len(rows) == 2
    assert rows[0] == ("cam01", "2026-10-30T12:00:00", "float", 0.88, 0.0, 0.0,
                       "http://x/out.jpg", 1, "")
    assert rows[1][2] == "discharge" and rows[1][7] == 0 and rows[1][8] == "minio down"

def test_archive_and_labeled(tmp_path):
    s = Storage(db_path=str(tmp_path / "t.db"), archive_root=str(tmp_path / "arc"))
    paths = s.archive_frames("cam01", "2026-10-28T10-00-00", [make_frame()], keep=True)
    assert len(paths) == 1 and (tmp_path / "arc").exists()
    s.save_labeled_sample("cam01", "2026-10-28T10:00:00", paths[0],
                          "discharge", "vlm", "排口发黑")
    row = s.conn.execute(
        "SELECT camera_id, ts FROM labeled_samples").fetchone()
    assert row[0] == "cam01" and row[1].startswith("2026-10-28")
    # 再存一条正常帧（keep=False），purge 应只删正常帧
    tmp_paths = s.archive_frames("cam01", "2026-10-28T11-00-00", [make_frame()], keep=False)
    deleted = s.purge_old_frames(days=0)
    assert deleted >= 1
    import os
    assert os.path.exists(paths[0])  # 保留帧不删
