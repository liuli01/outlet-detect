import numpy as np
from outlet_detect.sampler import HikFrameSampler, ListFrameSampler

def test_hik_sampler_uses_injected_fns():
    calls = []
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    def fake_capture(body):
        calls.append(body)
        return "http://x/pic.jpg"
    def fake_download(url):
        return img
    s = HikFrameSampler(capture_fn=fake_capture, download_fn=fake_download,
                        interval_sec=0)
    frames = s.capture("cam01", n=5, body={"cameraIndexCode": "abc"})
    assert len(frames) == 5
    assert len(calls) == 5
    assert frames[0].shape == (10, 10, 3)

def test_list_sampler_cycles_frames():
    frames_src = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    s = ListFrameSampler(frames_src)
    batch = s.capture("cam01", n=5)
    assert len(batch) == 5
    assert batch[0][0, 0, 0] == 0 and batch[4][0, 0, 0] == 4
