"""VLM 故障容错：VLM 挂掉时规则判定的 suspect 结果保留，状态机照常推进报警。"""
import numpy as np

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.view_align import AlignResult
from outlet_detect.vlm_judge import VLMResponseError
from outlet_detect.water_seg import StubWaterSegmenter


class _Aligner:
    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)


class DeadVLM:
    """VLM 完全无返回。"""

    def judge(self, *a, **k):
        raise VLMResponseError("timeout")


def make_pipe(tmp_path):
    # 灰色区样本：排口区黑且逐帧波动
    frames = []
    for i in range(5):
        f = np.full((400, 600, 3), 200, dtype=np.uint8)
        f[200:, 270:330, :] = (60 + i * 30) % 255
        frames.append(f)
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={}, score_low=0.2, score_high=0.9)  # 宽灰色带确保进 gray
    return Pipeline(cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
                    CNNClassifier(None), DeadVLM(),
                    Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
                    DischargeStateMachine("cam1", confirm_count=2),
                    LogAlerter(str(tmp_path / "a.log")))


def test_gray_sample_still_suspect_when_vlm_down(tmp_path):
    """VLM 挂掉时灰色样本保持 suspect（不被静默否掉）。"""
    pipe = make_pipe(tmp_path)
    assert pipe.run_once() == "suspect"


def test_alert_fires_after_confirm_when_vlm_down(tmp_path):
    """VLM 全程故障，连续 2 轮 suspect 后照常 alert。"""
    pipe = make_pipe(tmp_path)
    pipe.run_once()
    assert pipe.run_once() == "alert"
