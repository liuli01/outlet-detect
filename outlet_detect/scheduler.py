# outlet_detect/scheduler.py
import logging
import time
from datetime import datetime

log = logging.getLogger(__name__)

def run_loop(pipelines, interval_sec=300, stagger_sec=10, stop_flag=None,
             active_hours=None):
    """每 interval_sec 对所有摄像头各跑一轮；摄像头间错峰 stagger_sec。
    active_hours=(10, 15) 时只在 [10:00, 15:00) 内执行；窗口外每 60 秒重查。"""
    while stop_flag is None or not stop_flag.is_set():
        if active_hours is not None:
            h = datetime.now().hour
            if not (active_hours[0] <= h < active_hours[1]):
                log.info("当前 %d 时在运行窗口 %s 外，休眠", h, active_hours)
                time.sleep(60)
                continue
        for cid, pipe in pipelines.items():
            if stop_flag is not None and stop_flag.is_set():
                return
            try:
                result = pipe.run_once()
                log.info("%s -> %s", cid, result)
            except Exception:
                log.exception("%s 本轮执行异常", cid)
            time.sleep(stagger_sec)
        time.sleep(interval_sec)
