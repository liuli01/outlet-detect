# outlet_detect/alerter.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class AlertPayload:
    camera_id: str
    ts: str
    frame_paths: list
    vlm_reason: str
    features: dict
    consecutive: int

class LogAlerter:
    """默认告警实现：追加 JSON 行到日志文件。
    对接推送渠道时保持 alert(payload) 签名不变即可。"""
    def __init__(self, log_path="data/alerts.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def alert(self, payload):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(payload), ensure_ascii=False) + "\n")
