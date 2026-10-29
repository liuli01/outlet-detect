import numpy as np
from outlet_detect.water_seg import StubWaterSegmenter, mask_coverage

def test_stub_mask_shape_and_values():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = StubWaterSegmenter().segment(frame)
    assert mask.shape == (100, 200)
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[50, 100] == 255   # 中心是水面
    assert mask[0, 0] == 0        # 角落不是

def test_mask_coverage():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5:, :] = 255
    roi = np.ones((10, 10), dtype=np.uint8) * 255
    assert mask_coverage(mask, roi) == 0.5
