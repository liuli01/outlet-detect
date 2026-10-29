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
