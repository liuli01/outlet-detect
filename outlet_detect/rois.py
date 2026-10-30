# outlet_detect/rois.py
from dataclasses import dataclass

import numpy as np

@dataclass
class ROIs:
    outfall: np.ndarray
    plume: np.ndarray
    reference: np.ndarray

def snap_to_water(water_mask, point, max_dist=200):
    """排口点不在水面 mask 内时，吸附到最近的水面像素。

    排口物理上就在岸线，分割 mask 的水线边界抖动/边缘腐蚀会让排口点
    落在 mask 外，若直接用原点会导致排口区 ROI 为空、特征全灭。
    找不到水面（全图无水面）时返回原点，由上层按无水面处理。
    """
    h, w = water_mask.shape
    x, y = int(point[0]), int(point[1])
    if 0 <= y < h and 0 <= x < w and water_mask[y, x]:
        return (x, y)
    for r in range(10, max_dist + 1, 10):
        x0, x1 = max(0, x - r), min(w, x + r + 1)
        y0, y1 = max(0, y - r), min(h, y + r + 1)
        sub = water_mask[y0:y1, x0:x1]
        ys, xs = np.nonzero(sub)
        if len(ys):
            d2 = (xs + x0 - x) ** 2 + (ys + y0 - y) ** 2
            i = int(np.argmin(d2))
            return (int(xs[i] + x0), int(ys[i] + y0))
    return (x, y)

def build_rois(water_mask, outfall_point, r_outfall=60, r_plume=180, r_ref=250):
    """基于水面 mask 与排口点生成三个互斥 ROI（均为水面像素的子集）。
    排口区：半径 r_outfall 内水面；扩散区：r_outfall~r_plume 环带水面；
    参考区：距离 > r_ref 的水面。排口点不在水面内时先吸附到最近水面。"""
    outfall_point = snap_to_water(water_mask, outfall_point)
    h, w = water_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - outfall_point[0], yy - outfall_point[1])
    water = water_mask > 0
    return ROIs(
        outfall=((dist <= r_outfall) & water).astype(np.uint8) * 255,
        plume=(((dist > r_outfall) & (dist <= r_plume)) & water).astype(np.uint8) * 255,
        reference=((dist > r_ref) & water).astype(np.uint8) * 255,
    )
