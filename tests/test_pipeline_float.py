import numpy as np

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.float_detector import FloatDetection
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.view_align import AlignResult
from outlet_detect.water_seg import StubWaterSegmenter


class _Aligner:
    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)


class FakeFloatDetector:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    @property
    def enabled(self):
        return True

    def detect(self, frame):
        self.calls += 1
        return self.dets


class FakeReporter:
    def __init__(self):
        self.events = []

    def report(self, ev):
        self.events.append(ev)
        return "http://x/out.jpg"


def make_pipe(tmp_path, dets, reporter=None):
    frames = [np.full((400, 600, 3), 120 + i, dtype=np.uint8) for i in range(5)]
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={})
    return Pipeline(
        cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
        CNNClassifier(None), None,
        Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
        DischargeStateMachine("cam1"), LogAlerter(str(tmp_path / "a.log")),
        float_detector=FakeFloatDetector(dets),
        float_sm=DischargeStateMachine("cam1_float", confirm_count=2),
        reporter=reporter)


def test_float_detect_runs_on_frame1(tmp_path):
    pipe = make_pipe(tmp_path, [])
    pipe.run_once()
    assert pipe.float_detector.calls == 1


def test_float_report_after_confirm(tmp_path):
    rep = FakeReporter()
    pipe = make_pipe(tmp_path, [FloatDetection("bottle", 0.88, (10, 20, 50, 60))],
                     reporter=rep)
    pipe.run_once()          # 第 1 次：观察
    assert rep.events == []
    pipe.run_once()          # 第 2 次：confirm_count=2 → 上报
    assert len(rep.events) == 1
    assert rep.events[0].result_a == 0.88
    assert rep.events[0].result_b == 0.0  # 非排污事件


def test_float_failure_does_not_break_discharge(tmp_path):
    class BadDetector:
        enabled = True

        def detect(self, frame):
            raise ConnectionError("8005 down")

    frames = [np.full((400, 600, 3), 120 + i, dtype=np.uint8) for i in range(5)]
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={})
    pipe = Pipeline(cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
                    CNNClassifier(None), None,
                    Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
                    DischargeStateMachine("cam1"), LogAlerter(str(tmp_path / "a.log")),
                    float_detector=BadDetector(),
                    float_sm=DischargeStateMachine("cam1_float"))
    # 漂浮物检测异常不影响排污主线返回正常状态
    assert pipe.run_once() in ("normal", "suspect", "alert")


def test_discharge_alert_reports(tmp_path):
    """排污 alert 时 reporter 收到 result_b=result_c=置信度的事件。"""
    rep = FakeReporter()
    # 排口区纯黑+剧烈波动 → score 高，confirm_count=1 立即 alert
    frames = []
    for i in range(5):
        f = np.full((400, 600, 3), 230, dtype=np.uint8)  # 亮参考区
        f[200:, 270:330, :] = (i * 40) % 255             # 排口区黑且逐帧变化
        frames.append(f)
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={}, score_high=0.3)     # 降低阈值确保 suspect
    pipe = Pipeline(cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
                    CNNClassifier(None), None,
                    Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
                    DischargeStateMachine("cam1", confirm_count=1),
                    LogAlerter(str(tmp_path / "a.log")),
                    reporter=rep)
    status = pipe.run_once()
    assert status == "alert"
    assert len(rep.events) == 1
    ev = rep.events[0]
    assert ev.result_b == ev.result_c and ev.result_b > 0
    assert ev.result_a == 0.0
