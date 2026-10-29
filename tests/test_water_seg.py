import numpy as np
from outlet_detect.water_seg import OnnxWaterSegmenter, StubWaterSegmenter, mask_coverage

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

def _make_segmenter_without_model(conf_threshold=0.5):
    seg = OnnxWaterSegmenter.__new__(OnnxWaterSegmenter)
    seg.conf_threshold = conf_threshold
    return seg

def test_onnx_decode_synthetic():
    # det: (4 框 + 1 置信 + 32 系数, 5 个检测)；proto: (32, 160, 160)
    det = np.zeros((37, 5), dtype=np.float32)
    proto = np.zeros((32, 160, 160), dtype=np.float32)
    det[4, 0] = 0.9   # 检测 0 置信度过阈
    det[5, 0] = 1.0   # 检测 0 的第 0 个原型系数
    proto[0] = 1.0    # 第 0 个原型通道全 1
    outs = (det[None], proto[None])
    mask = _make_segmenter_without_model()._decode(outs)
    assert mask.shape == (160, 160)
    assert mask.dtype == np.uint8
    assert mask.sum() > 0

def test_onnx_decode_all_low_conf_returns_zeros():
    det = np.zeros((37, 5), dtype=np.float32)
    proto = np.zeros((32, 160, 160), dtype=np.float32)
    proto[0] = 1.0
    outs = (det[None], proto[None])
    mask = _make_segmenter_without_model()._decode(outs)
    assert mask.shape == (160, 160)
    assert mask.dtype == np.uint8
    assert mask.sum() == 0
