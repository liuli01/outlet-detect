# tests/test_alerter.py
import json
from outlet_detect.alerter import AlertPayload, LogAlerter

def test_log_alerter_writes_json_line(tmp_path):
    log = tmp_path / "alerts.log"
    a = LogAlerter(str(log))
    p = AlertPayload(camera_id="cam01", ts="2026-10-28T10:00:00",
                     frame_paths=["a.jpg"], vlm_reason="排口发黑",
                     features={"score": 0.8}, consecutive=3)
    a.alert(p)
    line = json.loads(log.read_text().strip())
    assert line["camera_id"] == "cam01" and line["consecutive"] == 3
