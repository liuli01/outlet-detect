# tests/test_vlm_judge.py
import json
import numpy as np
from outlet_detect.vlm_judge import VLMJudge, build_prompt, parse_vlm_response

def test_parse_valid_json():
    r = parse_vlm_response('{"is_discharge": true, "confidence": 0.9,'
                           ' "reason": "排口发黑", "shadow_suspect": false}')
    assert r.is_discharge and r.confidence == 0.9 and not r.shadow_suspect

def test_parse_garbage_returns_conservative():
    r = parse_vlm_response("抱歉我无法分析这张图片")
    assert not r.is_discharge and r.confidence == 0.0

def test_parse_json_embedded_in_text():
    r = parse_vlm_response('分析结果：{"is_discharge": true, "confidence": 0.8,'
                           ' "reason": "x", "shadow_suspect": true} 以上。')
    assert r.is_discharge and r.shadow_suspect

def test_judge_calls_client():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    class FakeMsg:
        content = '{"is_discharge": false, "confidence": 0.9, "reason": "正常", "shadow_suspect": false}'
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]
    class FakeChat:
        def create(self, **kw):
            self.kw = kw
            return FakeResp()
    class FakeClient:
        def __init__(self):
            self.chat = type("C", (), {"completions": FakeChat()})()
    judge = VLMJudge("http://x", "k", "model-x", client=FakeClient())
    r = judge.judge(img, img, img, {"rel_darkness": 0.8, "flow_energy": 1.0,
                                    "color_dev": 0.05, "score": 0.5})
    assert not r.is_discharge

def test_build_prompt_contains_features():
    p = build_prompt({"rel_darkness": 0.8, "flow_energy": 2.0, "color_dev": 0.1, "score": 0.65})
    assert "0.80" in p and "阴影" in p
