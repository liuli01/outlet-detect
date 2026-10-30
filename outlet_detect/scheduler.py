import logging
import time

log = logging.getLogger(__name__)

def run_loop(pipelines, interval_sec=300, stagger_sec=10, stop_flag=None):
    """每 interval_sec 对所有摄像头各跑一轮；摄像头间错峰 stagger_sec。"""
    while stop_flag is None or not stop_flag.is_set():
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
