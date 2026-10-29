# outlet_detect/rois.py
from dataclasses import dataclass

import numpy as np

@dataclass
class ROIs:
    outfall: np.ndarray
    plume: np.ndarray
    reference: np.ndarray

def build_rois(water_mask, outfall_point, r_outfall=60, r_plume=180, r_ref=250):
    """基于水面 mask 与排口点生成三个互斥 ROI（均为水面像素的子集）。
    排口区：半径 r_outfall 内水面；扩散区：r_outfall~r_plume 环带水面；
    参考区：距离 > r_ref 的水面。"""
    h, w = water_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - outfall_point[0], yy - outfall_point[1])
    water = water_mask > 0
    return ROIs(
        outfall=((dist <= r_outfall) & water).astype(np.uint8) * 255,
        plume=(((dist > r_outfall) & (dist <= r_plume)) & water).astype(np.uint8) * 255,
        reference=((dist > r_ref) & water).astype(np.uint8) * 255,
    )
