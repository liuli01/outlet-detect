# tests/test_rois.py
import numpy as np
from outlet_detect.rois import build_rois

def test_build_rois_on_half_water():
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[200:, :] = 255  # 下半幅是水面
    rois = build_rois(mask, (300, 300), r_outfall=40, r_plume=120, r_ref=200)
    # 排口区非空且全是水面
    assert rois.outfall.sum() > 0
    assert (rois.outfall[mask == 0] == 0).all()
    # 三区互不重叠
    assert ((rois.outfall > 0) & (rois.plume > 0)).sum() == 0
    assert ((rois.plume > 0) & (rois.reference > 0)).sum() == 0
    # 参考区应落在远离排口的水面
    ys, xs = np.nonzero(rois.reference)
    assert len(xs) > 0
    dists = np.hypot(xs - 300, ys - 300)
    assert dists.min() >= 200

def test_build_rois_no_water():
    mask = np.zeros((100, 100), dtype=np.uint8)
    rois = build_rois(mask, (50, 50))
    assert rois.outfall.sum() == 0 and rois.reference.sum() == 0

def test_snap_to_water_moves_outside_point():
    import numpy as np
    from outlet_detect.rois import snap_to_water, build_rois
    mask = np.zeros((300, 400), dtype=np.uint8)
    mask[150:, :] = 255  # 下半幅水面
    # 点在水面外上方 30px → 吸附到最近水面像素 (200, 150)
    assert snap_to_water(mask, (200, 120)) == (200, 150)
    # 点在水面内不动
    assert snap_to_water(mask, (200, 200)) == (200, 200)
    # 全图无水面返回原点
    assert snap_to_water(np.zeros((10, 10), dtype=np.uint8), (5, 5)) == (5, 5)
    # build_rois 对 mask 外排口点也能生成非空排口区
    rois = build_rois(mask, (200, 120))
    assert rois.outfall.sum() > 0
