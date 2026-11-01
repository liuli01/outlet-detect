from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class AlignResult:
    ok: bool
    mapped_point: tuple | None
    confidence: float
    in_frame: bool

class ViewAligner:
    """ORB 特征匹配 + RANSAC 单应矩阵，把标定排口点映射到当前帧。
    outfall_point 为 None（未标定排口）时只做视角校验，不映射点（mapped_point=None），
    由 pipeline 用分割 mask 的水面质心兜底。"""
    def __init__(self, reference_frame_path, outfall_point, min_matches=12):
        self.ref = cv2.imread(reference_frame_path)
        if self.ref is None:
            raise FileNotFoundError(reference_frame_path)
        self.outfall_point = outfall_point
        self.min_matches = min_matches
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref, None)

    def align(self, frame):
        h, w = frame.shape[:2]
        kp, des = self.orb.detectAndCompute(frame, None)
        if des is None or self.des_ref is None:
            return AlignResult(False, None, 0.0, False)
        matches = self.matcher.knnMatch(self.des_ref, des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < self.min_matches:
            return AlignResult(False, None, len(good) / self.min_matches, False)
        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return AlignResult(False, None, 0.0, False)
        ratio = float(inliers.sum()) / len(good)
        if ratio < 0.4:  # 内点率过低说明是误匹配（无关场景），拒绝
            return AlignResult(False, None, ratio, False)
        if self.outfall_point is None:
            return AlignResult(True, None, ratio, True)  # 无排口点可映射，视角校验通过
        p = H @ np.array([self.outfall_point[0], self.outfall_point[1], 1.0])
        mx, my = p[0] / p[2], p[1] / p[2]
        in_frame = bool(0 <= mx < w and 0 <= my < h)
        return AlignResult(True, (int(mx), int(my)), ratio, in_frame)
