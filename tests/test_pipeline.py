import numpy as np
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.water_seg import StubWaterSegmenter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.alerter import LogAlerter
from outlet_detect.storage import Storage

class FakeAlignerOK:
    def __init__(self):
        from outlet_detect.view_align import AlignResult
        self.result = AlignResult(True, (300, 300), 0.9, True)
    def align(self, frame):
        return self.result

class FakeAlignerBad:
    def align(self, frame):
        from outlet_detect.view_align import AlignResult
        return AlignResult(False, None, 0.1, False)

class FakeVLM:
    def __init__(self, is_discharge):
        from outlet_detect.vlm_judge import VLMResult
        self.result = VLMResult(is_discharge, 0.9, "测试", False)
    def judge(self, *a, **kw):
        return self.result

def normal_frames():
    # 全画面均匀亮度 → 分数低。尺寸 600x400 以保证默认 ROI 半径
    #（r_ref=250）内存在参考水面像素
    return [np.full((400, 600, 3), 120, dtype=np.uint8) for _ in range(5)]

def suspect_frames():
    # StubWaterSegmenter 视下半幅(y>=200)为水面，排口点(300,300)处又黑又闪烁
    frames = []
    for i in range(5):
        f = np.full((400, 600, 3), 180, dtype=np.uint8)
        f[250:350, 250:350] = 20 if i % 2 == 0 else 120
        frames.append(f)
    return frames

def make_pipeline(tmp_path, frames, aligner, vlm, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cfg")
    cfg = CameraConfig(camera_id="cam01", outfall_point=(300, 300),
                       reference_frame="", hik_body={})
    storage = Storage(str(tmp_path / "t.db"), str(tmp_path / "arc"))
    return Pipeline(
        config=cfg,
        sampler=ListFrameSampler(frames),
        segmenter=StubWaterSegmenter(),
        aligner=aligner,
        cnn=CNNClassifier(None),          # 冷启动：CNN 关闭
        vlm=vlm,
        storage=storage,
        state_machine=DischargeStateMachine("cam01"),
        alerter=LogAlerter(str(tmp_path / "alerts.log")),
    )

def test_normal_path(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, normal_frames(), FakeAlignerOK(), FakeVLM(False), monkeypatch)
    assert p.run_once(0) == "normal"

def test_suspect_path_vlm_confirms(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, suspect_frames(), FakeAlignerOK(), FakeVLM(True), monkeypatch)
    assert p.run_once(0) == "suspect"
    assert p.run_once(300) == "suspect"
    assert p.run_once(600) == "alert"     # 第三次 → 告警

def test_suspect_path_vlm_rejects(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, suspect_frames(), FakeAlignerOK(), FakeVLM(False), monkeypatch)
    assert p.run_once(0) == "normal"      # VLM 否决 → 按正常

def test_view_error_suspends(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, normal_frames(), FakeAlignerBad(), FakeVLM(False), monkeypatch)
    assert p.run_once(0) == "view_error"
    p.aligner = FakeAlignerOK()           # 摄像头转回
    assert p.run_once(300) == "normal"    # 自动恢复
