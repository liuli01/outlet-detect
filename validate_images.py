"""正负样本验证：images/排污（5连帧组，带人工标注排口点）vs images/无排污（单图复制5帧）。

排污样本保留帧间波动特征；无排污照片为静止序列（波动=0，只测黑度/色偏）。
排口点来自 OUTFALL_POINTS 人工标注（看 grid_*.jpg 网格图估计）；未标注的取画面中心。
"""
import glob
import os
import sqlite3

import cv2

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.view_align import AlignResult
from outlet_detect.water_seg import HttpWaterSegmenter

ERODE_PX = 25
# 黑臭水体在分割模型下置信度极低（0.02~0.22），默认 0.25 会把排污最严重的水面过滤掉
# → 验证阈值降到 0.05。根本解法是分割模型补黑水样本重训。
SEG_CONF = 0.05

# 人工标注的排口像素坐标（按 grid_*.jpg 网格估计，原图坐标系）
OUTFALL_POINTS = {
    "01521d22e3e199b46ac42aebcc68ab79": (730, 200),    # 墙根白色水流（下移进水面 mask）
    "255407597f6d29b497ec7028542a90b4": (860, 360),    # 混凝土涵洞出水
    "bba6cc73a8fda9ed46db7555acc95304": (550, 150),    # 黑色污水源头墙根
    "东城河阳光桥球机_1687161142095": (400, 500),       # 左侧黑色水带（排口不明显）
    "创新河春申路河道球机_1687161684788": (1780, 750),  # 圆形井口黑色污水+湍流
    "创新河春申路河道球机_1687161745744": (200, 500),   # 拦污排污物聚集处
    "普惠中心河五星路球机_1687161282421": (1280, 720),  # 排口不明显，取中心
    "芦花沟河天庆路球机_1687160641609": (1750, 620),    # 台阶式排水口
    "芦花沟河天庆路球机_1687160680573": (500, 400),     # 白色泡沫水流入口
}


def imread_cn(path):
    """cv2.imread 不支持中文路径，用 imdecode 兜底。"""
    import numpy as np
    return cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)


class FixedAligner:
    def __init__(self, point):
        self.point = point

    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, self.point or (w // 2, h // 2), 1.0, True)


def run_sample(cid, frames, seg, point=None, vlm=None):
    cfg = CameraConfig(camera_id=cid, outfall_point=(0, 0), reference_frame="", hik_body={})
    pipe = Pipeline(
        cfg, ListFrameSampler(frames), seg, FixedAligner(point), CNNClassifier(None), vlm,
        Storage("data/validate.db", "data/validate_arc"),
        DischargeStateMachine(cid), LogAlerter("data/validate_alerts.log"),
    )
    return pipe.run_once()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--vlm", action="store_true",
                    help="灰色地带送 VLM 终判（读 config/cameras/vlm.json，需先配置）")
    args = ap.parse_args()

    vlm = None
    if args.vlm:
        from outlet_detect.config import load_camera_config
        from outlet_detect.vlm_judge import VLMJudge
        vc = load_camera_config("vlm")
        vlm = VLMJudge(vc.vlm_base_url, vc.vlm_api_key, vc.vlm_model,
                       envelope=vc.vlm_envelope)
        print(f"VLM 已接入: {vc.vlm_base_url}")

    # 清掉上次运行的数据，避免累积
    for f in ("data/validate.db", "data/validate_alerts.log"):
        if os.path.exists(f):
            os.remove(f)

    seg = HttpWaterSegmenter(erode_px=ERODE_PX, confidence_threshold=SEG_CONF)
    rows = []  # (类别, cid, 显示名, status)

    groups = sorted(glob.glob("images/排污/*/"))
    for g in groups:
        name = os.path.basename(os.path.normpath(g))
        cid = "pw" + str(groups.index(g)).zfill(2)
        frames = [imread_cn(f) for f in sorted(glob.glob(g + "*.jpg"))[:5]]
        frames = [f for f in frames if f is not None]
        if len(frames) < 5:
            print(f"{name}: 帧不足 {len(frames)}，跳过")
            continue
        status = run_sample(cid, frames, seg, OUTFALL_POINTS.get(name), vlm=vlm)
        rows.append(("排污", cid, name[:22], status))

    photos = sorted(glob.glob("images/无排污/*.jpg") + glob.glob("images/无排污/*.jpeg"))
    for i, p in enumerate(photos):
        cid = f"nw{i:02d}"
        f = imread_cn(p)
        if f is None:
            print(f"{p} 读不了，跳过")
            continue
        status = run_sample(cid, [f] * 5, seg, vlm=vlm)  # 静止序列：波动=0
        rows.append(("无排污", cid, os.path.basename(p), status))

    con = sqlite3.connect("data/validate.db")
    print(f"\n{'类别':5s} {'样本':24s} {'判定':13s} {'score':>6s}  黑度  波动  色偏")
    for label, cid, name, status in rows:
        r = con.execute(
            "SELECT score, rel_darkness, flow_energy, color_dev FROM samples "
            "WHERE camera_id=? ORDER BY ts DESC LIMIT 1", (cid,)).fetchone()
        if r and r[0] is not None:
            print(f"{label:5s} {name:24s} {status:13s} {r[0]:6.3f}  {r[1]:.2f}  {r[2]:.2f}  {r[3]:.2f}")
        else:
            print(f"{label:5s} {name:24s} {status:13s}      -")

    def scores(lb):
        cids = [c for l, c, _, _ in rows if l == lb]
        return [r[0] for c in cids for r in
                [con.execute("SELECT score FROM samples WHERE camera_id=? AND score IS NOT NULL",
                             (c,)).fetchone()] if r]

    pw, nw = scores("排污"), scores("无排污")
    print(f"\n排污 score（{len(pw)} 个有效）: {[round(s, 2) for s in pw]}")
    print(f"无排污 score（{len(nw)} 个）: {[round(s, 2) for s in nw]}")
    if pw and nw:
        print(f"排污 min={min(pw):.2f} / 无排污 max={max(nw):.2f} → 分离裕度 {min(pw) - max(nw):+.2f}")


if __name__ == "__main__":
    main()
