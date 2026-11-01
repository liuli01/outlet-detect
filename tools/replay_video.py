"""自定义回放测试：指定排口坐标与水面分割方式，回放真实视频验证检测效果。

用法示例：
  # 最简：Stub 假水面 + 排口点取画面中心
  uv run python tools/replay_video.py videos/xxx.mp4

  # 真实：HTTP 水面分割服务 + 指定排口像素坐标
  uv run python tools/replay_video.py videos/xxx.mp4 --http-seg --point 960,700 --interval 30

  # 先抽一帧看画面、确定排口坐标：
  uv run python tools/replay_video.py videos/xxx.mp4 --peek

结果查看：
  判定列表直接打印；每轮特征值/score 在 data/replay.db 的 samples 表；
  截图归档在 data/replay_arc/（tmp=正常帧，keep=疑似/告警帧）。
"""
import argparse
import sqlite3
import sys

import cv2

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
from outlet_detect.water_seg import HttpWaterSegmenter, StubWaterSegmenter


class FixedAligner:
    """回放不做视角配准：排口点固定为用户指定坐标。"""

    def __init__(self, point):
        self.point = point

    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, self.point or (w // 2, h // 2), 1.0, True)


def peek(video_path, out="peek_frame.jpg"):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"打不开视频: {video_path}")
    cv2.imwrite(out, frame)
    h, w = frame.shape[:2]
    print(f"首帧已存 {out}（{w}x{h}）——用看图工具确定排口的像素坐标后加 --point x,y 重跑")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--point", default=None, help="排口像素坐标 x,y（默认画面中心）")
    ap.add_argument("--interval", type=float, default=60, help="采样间隔（视频秒）")
    ap.add_argument("--http-seg", action="store_true",
                    help="用 127.0.0.1:8005 HTTP 水面分割（默认 Stub 假水面）")
    ap.add_argument("--erode", type=int, default=0,
                    help="水面 mask 边缘腐蚀像素数，削岸边误检（建议 20~30，0=不腐蚀）")
    ap.add_argument("--camera-id", default="replay")
    ap.add_argument("--storage", default="data")
    ap.add_argument("--peek", action="store_true", help="只抽首帧存图，不跑回放")
    args = ap.parse_args()

    if args.peek:
        peek(args.video)
        return

    point = tuple(map(int, args.point.split(","))) if args.point else None
    frames = extract_sample_frames(args.video, interval_sec=args.interval)
    if not frames:
        raise SystemExit("未抽到帧（视频太短或 interval 太大，试试 --interval 10）")

    cfg = CameraConfig(camera_id=args.camera_id, outfall_point=(0, 0),
                       reference_frame="", hik_body={})
    seg = HttpWaterSegmenter(erode_px=args.erode) if args.http_seg else StubWaterSegmenter()
    pipe = Pipeline(
        config=cfg,
        sampler=ListFrameSampler(frames),
        segmenter=seg,
        aligner=FixedAligner(point),
        cnn=CNNClassifier(None),
        vlm=None,
        storage=Storage(f"{args.storage}/replay.db", f"{args.storage}/replay_arc"),
        state_machine=DischargeStateMachine(args.camera_id),
        alerter=LogAlerter(f"{args.storage}/replay_alerts.log"),
    )
    n = len(frames) // 5
    results = [pipe.run_once() for _ in range(n)]
    print(f"抽帧 {len(frames)}，回放 {n} 轮，判定: {results}")

    con = sqlite3.connect(f"{args.storage}/replay.db")
    q = ("SELECT ts, status, ROUND(rel_darkness,3), ROUND(flow_energy,3), "
         "ROUND(color_dev,3), ROUND(score,3) FROM samples WHERE camera_id=? "
         "ORDER BY ts DESC LIMIT ?")
    print(f"\n最近 {n} 轮特征（黑度/波动/色偏/score）：")
    for row in con.execute(q, (args.camera_id, n)):
        print("  ", row)


if __name__ == "__main__":
    sys.exit(main())
