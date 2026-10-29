# outlet_detect/state_machine.py
from dataclasses import dataclass

@dataclass
class AlertEvent:
    camera_id: str
    ts: float
    consecutive: int

class DischargeStateMachine:
    """时序确认状态机。

    state 三值契约：``"normal" | "watching" | "cooldown"``。
    视角异常时可通过 suspend()/resume() 挂起与恢复（挂起期间 state 恒为 "normal"）。
    """

    def __init__(self, camera_id, confirm_count=3, cooldown_sec=1800):
        self.camera_id = camera_id
        self.confirm_count = confirm_count
        self.cooldown_sec = cooldown_sec
        self.consecutive = 0
        self.last_alert_ts = None
        self.suspended = False
        self._last_in_cooldown = False

    @property
    def state(self):
        """当前状态："normal" | "watching" | "cooldown"。"""
        if self.suspended:
            return "normal"
        if self._last_in_cooldown:
            return "cooldown"
        return "watching" if self.consecutive > 0 else "normal"

    def suspend(self):
        self.suspended = True

    def resume(self):
        self.suspended = False
        self.consecutive = 0

    def update(self, is_suspect, ts):
        if self.suspended:
            return None
        if is_suspect:
            self.consecutive += 1
        else:
            self.consecutive = 0
        in_cooldown = (self.last_alert_ts is not None
                       and ts - self.last_alert_ts < self.cooldown_sec)
        if self.last_alert_ts is not None:
            self._last_in_cooldown = in_cooldown
        if self.consecutive >= self.confirm_count and not in_cooldown:
            self.last_alert_ts = ts
            self._last_in_cooldown = True
            return AlertEvent(self.camera_id, ts, self.consecutive)
        return None
