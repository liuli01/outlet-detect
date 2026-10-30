import threading
import time
from outlet_detect.scheduler import run_loop

def test_run_loop_calls_each_pipeline():
    calls = {"a": 0, "b": 0}
    class FakePipeline:
        def __init__(self, name):
            self.name = name
        def run_once(self, now_ts=None):
            calls[self.name] += 1
            return "normal"
    stop = threading.Event()
    t = threading.Thread(target=run_loop, kwargs=dict(
        pipelines={"a": FakePipeline("a"), "b": FakePipeline("b")},
        interval_sec=0, stagger_sec=0, stop_flag=stop))
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=5)
    assert calls["a"] >= 2 and calls["b"] >= 2

def test_run_loop_survives_exception():
    class BadPipeline:
        def run_once(self, now_ts=None):
            raise RuntimeError("boom")
    stop = threading.Event()
    t = threading.Thread(target=run_loop, kwargs=dict(
        pipelines={"x": BadPipeline()}, interval_sec=0, stagger_sec=0,
        stop_flag=stop))
    t.start()
    time.sleep(0.2)
    alive = t.is_alive()
    stop.set()
    t.join(timeout=5)
    assert alive  # 异常没有杀死循环
