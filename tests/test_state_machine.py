# tests/test_state_machine.py
from outlet_detect.state_machine import DischargeStateMachine

def test_three_consecutive_triggers_alert():
    sm = DischargeStateMachine("cam01")
    assert sm.update(True, 0) is None     # 第1次疑似
    assert sm.state == "watching"
    assert sm.update(True, 300) is None   # 第2次
    ev = sm.update(True, 600)             # 第3次 → 告警
    assert ev is not None and ev.consecutive == 3
    assert sm.state == "cooldown"

def test_normal_resets_count():
    sm = DischargeStateMachine("cam01")
    sm.update(True, 0)
    sm.update(True, 300)
    sm.update(False, 600)                 # 中间一次正常
    assert sm.state == "normal"
    assert sm.update(True, 900) is None   # 重新计数

def test_cooldown_suppresses_repeat_alert():
    sm = DischargeStateMachine("cam01")
    for t in (0, 300, 600):
        sm.update(True, t)
    assert sm.update(True, 900) is None   # 冷却期内不重复告警
    ev = sm.update(True, 600 + 1801)      # 冷却结束后再次确认
    assert ev is not None

def test_suspend_resume():
    sm = DischargeStateMachine("cam01")
    sm.suspend()
    assert sm.update(True, 0) is None and sm.state == "normal"
    sm.resume()
    sm.update(True, 300)
    assert sm.state == "watching"
