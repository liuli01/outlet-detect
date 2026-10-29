# outlet_detect/state_machine.py
from dataclasses import dataclass

@dataclass
class AlertEvent:
    camera_id: str
    ts: float
    consecutive: int

class DischargeStateMachine:
    def __init__(self, camera_id, confirm_count=3, cooldown_sec=1800):
        self.camera_id = camera_id
        self.confirm_count = confirm_count
        self.cooldown_sec = cooldown_sec
        self.consecutive = 0
        self.last_alert_ts = None
        self.suspended = False

    @property
    def state(self):
        if self.suspended:
            return "normal"
        if self.last_alert_ts is not None and self.consecutive >= self.confirm_count:
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
        if self.consecutive >= self.confirm_count and not in_cooldown:
            self.last_alert_ts = ts
            return AlertEvent(self.camera_id, ts, self.consecutive)
        return None
