"""离线回放：把视频按在线采样节奏抽帧，跑完整管线，验证算法与调阈值。"""
import logging

import cv2

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.water_seg import StubWaterSegmenter

log = logging.getLogger(__name__)

class _PassThroughAligner:
    """回放模式不做视角配准：排口点取画面中心。"""
    def align(self, frame):
        from outlet_detect.view_align import AlignResult
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)

def extract_sample_frames(video_path, interval_sec=300, burst=5, burst_gap_frames=50):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    pos = 0
    step = int(interval_sec * fps)
    while pos + (burst - 1) * burst_gap_frames < total:
        for k in range(burst):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + k * burst_gap_frames)
            ok, f = cap.read()
            if ok:
                frames.append(f)
        pos += step
    cap.release()
    return frames

def replay(video_path, camera_id="replay", interval_sec=300, storage_dir="data"):
    frames = extract_sample_frames(video_path, interval_sec)
    if not frames:
        raise RuntimeError(f"{video_path} 未抽到任何帧")
    cfg = CameraConfig(camera_id=camera_id, outfall_point=(0, 0),
                       reference_frame="", hik_body={})
    pipe = Pipeline(
        config=cfg,
        sampler=ListFrameSampler(frames),
        segmenter=StubWaterSegmenter(),  # 回放验证时替换为真实 OnnxWaterSegmenter
        aligner=_PassThroughAligner(),
        cnn=CNNClassifier(None),
        vlm=None,                        # 回放默认不调 VLM，省钱；需验证时注入真实 VLMJudge
        storage=Storage(f"{storage_dir}/replay.db", f"{storage_dir}/replay_arc"),
        state_machine=DischargeStateMachine(camera_id),
        alerter=LogAlerter(f"{storage_dir}/replay_alerts.log"),
    )
    n_rounds = len(frames) // 5
    results = [pipe.run_once() for _ in range(n_rounds)]
    log.info("回放完成 %s: %d 轮, 结果分布: %s", video_path, n_rounds,
             {r: results.count(r) for r in set(results)})
    return results
