# outlet_detect/storage.py
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import cv2

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, camera_id TEXT,
    rel_darkness REAL, flow_energy REAL, color_dev REAL, baseline_dev REAL,
    score REAL, rule_verdict TEXT, cnn_prob REAL, vlm_json TEXT,
    status TEXT, frame_paths TEXT, duration_ms INTEGER
);
CREATE TABLE IF NOT EXISTS labeled_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, camera_id TEXT, frame_path TEXT,
    label TEXT, source TEXT, reason TEXT
);
"""

class Storage:
    def __init__(self, db_path: str = "data/samples.db", archive_root: str = "archive"):
        self.db_path = db_path
        self.archive_root = Path(archive_root)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def record_sample(self, camera_id, ts, features, rule_verdict, cnn_prob,
                      vlm_json, status, frame_paths, duration_ms):
        self.conn.execute(
            "INSERT INTO samples (ts,camera_id,rel_darkness,flow_energy,color_dev,"
            "baseline_dev,score,rule_verdict,cnn_prob,vlm_json,status,frame_paths,"
            "duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, camera_id, features.get("rel_darkness"), features.get("flow_energy"),
             features.get("color_dev"), features.get("baseline_dev"),
             features.get("score"), rule_verdict, cnn_prob, vlm_json, status,
             json.dumps(frame_paths), duration_ms))
        self.conn.commit()

    def archive_frames(self, camera_id, ts, frames, keep):
        day = ts[:10]
        tier = "keep" if keep else "tmp"
        d = self.archive_root / camera_id / day / tier
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, f in enumerate(frames):
            p = d / f"{ts[11:].replace(':', '-')}_{i}.jpg"
            cv2.imwrite(str(p), f)
            paths.append(str(p))
        return paths

    def save_snapshot(self, camera_id, ts, frame, tag):
        """标注图/快照存 archive_root/<cid>/<date>/<ts>_<tag>.jpg，返回路径字符串。"""
        d = self.archive_root / camera_id / ts[:10]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{ts.replace(':', '-')}_{tag}.jpg"
        cv2.imwrite(str(p), frame)
        return str(p)

    def get_baseline_darkness(self, camera_id, hour, before_ts, days=7):
        start = (datetime.fromisoformat(before_ts) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT rel_darkness FROM samples WHERE camera_id=? AND ts<? AND ts>? "
            "AND CAST(substr(ts,12,2) AS INTEGER)=? AND rel_darkness IS NOT NULL",
            (camera_id, before_ts, start, hour)).fetchall()
        if not rows:
            return None
        vals = sorted(r[0] for r in rows)
        return vals[len(vals) // 2]  # 中位数

    def save_labeled_sample(self, camera_id, ts, frame_path, label, source, reason):
        self.conn.execute(
            "INSERT INTO labeled_samples (ts,camera_id,frame_path,label,source,reason)"
            " VALUES (?,?,?,?,?,?)",
            (ts, camera_id, frame_path, label, source, reason))
        self.conn.commit()

    def purge_old_frames(self, days=7):
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        deleted = 0
        for cam_dir in self.archive_root.glob("*/*/tmp"):
            day = cam_dir.parent.name
            if day < cutoff:
                for f in cam_dir.glob("*.jpg"):
                    os.remove(f)
                    deleted += 1
        return deleted
