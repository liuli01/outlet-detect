import time

class HikFrameSampler:
    """通过海康 manualCapture 连续抓 n 帧。capture_fn/download_fn 由用户现有海康代码提供。"""
    def __init__(self, capture_fn, download_fn, interval_sec=2.0):
        self.capture_fn = capture_fn
        self.download_fn = download_fn
        self.interval_sec = interval_sec

    def capture(self, camera_id, n=5, body=None):
        frames = []
        for i in range(n):
            if i > 0:
                time.sleep(self.interval_sec)
            url = self.capture_fn(body)
            if not url:
                raise RuntimeError(f"{camera_id} 第{i+1}帧抓图失败")
            frames.append(self.download_fn(url))
        return frames

class ListFrameSampler:
    """测试与离线回放用：从给定帧序列循环取帧，不睡眠。"""
    def __init__(self, frames):
        self.frames = frames
        self.pos = 0

    def capture(self, camera_id, n=5, body=None):
        batch = []
        for _ in range(n):
            batch.append(self.frames[self.pos % len(self.frames)])
            self.pos += 1
        return batch
