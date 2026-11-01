"""批量回放全部视频样本（一次性效果验证）。"""
import glob
import os
import sqlite3

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根入 path，保证能 import outlet_detect

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.replay_tool import extract_sample_frames
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.view_align import AlignResult
from outlet_detect.water_seg import HttpWaterSegmenter


class FA:
    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)


seg = HttpWaterSegmenter()
for i, p in enumerate(sorted(glob.glob("videos/*.mp4"))):
    cid = f"v{i:02d}"
    name = os.path.basename(p)
    try:
        frames = extract_sample_frames(p, interval_sec=8)
    except Exception as e:
        print(f"{cid} {name}: 抽帧异常 {e}", flush=True)
        continue
    if not frames:
        print(f"{cid} {name}: 打不开/无帧", flush=True)
        continue
    cfg = CameraConfig(camera_id=cid, outfall_point=(0, 0), reference_frame="", hik_body={})
    pipe = Pipeline(
        cfg, ListFrameSampler(frames), seg, FA(), CNNClassifier(None), None,
        Storage("data/replay.db", "data/replay_arc"),
        DischargeStateMachine(cid), LogAlerter("data/replay_alerts.log"),
    )
    res = [pipe.run_once() for _ in range(len(frames) // 5)]
    con = sqlite3.connect("data/replay.db")
    scores = [r[0] for r in con.execute(
        "SELECT score FROM samples WHERE camera_id=? AND score IS NOT NULL", (cid,))]
    dist = {x: res.count(x) for x in set(res)}
    print(f"{cid} {name[:30]:32s} 轮={len(res)} 判定={dist} scores={[round(s, 2) for s in scores]}",
          flush=True)
