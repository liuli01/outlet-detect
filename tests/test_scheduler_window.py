# tests/test_scheduler_window.py
import threading

from outlet_detect.scheduler import run_loop


class FakePipe:
    def __init__(self):
        self.calls = 0

    def run_once(self):
        self.calls += 1
        return "normal"


def _run_with_hour(hour):
    """在指定小时下跑一轮 run_loop 后停。"""
    pipe = FakePipe()
    stop = threading.Event()

    class FakeDatetime:
        @classmethod
        def now(cls):
            import datetime as dt
            return dt.datetime(2026, 10, 30, hour, 0, 0)

    import outlet_detect.scheduler as sch
    orig = sch.datetime
    sch.datetime = FakeDatetime
    orig_sleep = sch.time.sleep
    sch.time.sleep = lambda s: stop.set()  # 第一次 sleep 后停
    try:
        run_loop({"cam1": pipe}, interval_sec=300, stagger_sec=0,
                 stop_flag=stop, active_hours=(10, 15))
    finally:
        sch.datetime = orig
        sch.time.sleep = orig_sleep
    return pipe.calls


def test_inside_window_runs():
    assert _run_with_hour(11) == 1


def test_outside_window_skips():
    assert _run_with_hour(8) == 0
    assert _run_with_hour(15) == 0  # 15:00 已出窗口


def test_no_window_always_runs():
    pipe = FakePipe()
    stop = threading.Event()
    import outlet_detect.scheduler as sch
    orig_sleep = sch.time.sleep
    sch.time.sleep = lambda s: stop.set()
    try:
        run_loop({"cam1": pipe}, interval_sec=300, stagger_sec=0, stop_flag=stop)
    finally:
        sch.time.sleep = orig_sleep
    assert pipe.calls == 1
