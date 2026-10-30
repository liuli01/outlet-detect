# outlet_detect/rule_engine.py
import cv2
import numpy as np

# 验证集实测：波动是唯一拉开差距的特征（排污1.4~5.3 vs 静止1.0），
# 黑度比值对"污水已扩散全河"场景失效（排口区与参考区一样黑），故权重向波动倾斜。
DEFAULT_WEIGHTS = {"dark": 0.2, "flow": 0.5, "color": 0.2, "base": 0.1}

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def _mean_in_roi(img, roi_mask):
    sel = img[roi_mask > 0]
    if sel.size == 0:
        return None
    if sel.ndim == 1:
        return float(sel.mean())  # 单通道输入返回标量
    return sel.reshape(-1, sel.shape[-1]).mean(axis=0)

def compute_features(frames, rois, baseline_darkness):
    lab = cv2.cvtColor(frames[0], cv2.COLOR_BGR2Lab)
    l_out = _mean_in_roi(lab[:, :, 0], rois.outfall)
    l_ref = _mean_in_roi(lab[:, :, 0], rois.reference)
    if l_out is None or l_ref is None or l_ref == 0:
        rel_darkness = 1.0  # ROI 失效时按正常处理，避免误报
    else:
        rel_darkness = float(l_out / l_ref)

    # 波动能量：相邻帧灰度差分的均值
    def flow_in(roi):
        energies = []
        for a, b in zip(frames[:-1], frames[1:]):
            ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            diff = cv2.absdiff(ga, gb)
            v = _mean_in_roi(diff, roi)
            if v is not None:
                energies.append(float(v))
        return float(np.mean(energies)) if energies else 0.0
    f_out, f_ref = flow_in(rois.outfall), flow_in(rois.reference)
    flow_energy = f_out / f_ref if f_ref > 0.5 else (1.0 if f_out < 1.0 else 3.0)

    hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0; hsv[:, :, 1] /= 255.0  # H、S 归一化
    c_out = _mean_in_roi(hsv[:, :, :2], rois.outfall)
    c_ref = _mean_in_roi(hsv[:, :, :2], rois.reference)
    color_dev = float(np.linalg.norm(c_out - c_ref)) if c_out is not None and c_ref is not None else 0.0

    baseline_dev = (baseline_darkness - rel_darkness) if baseline_darkness is not None else 0.0

    return {
        "rel_darkness": rel_darkness,
        "flow_energy": flow_energy,
        "color_dev": color_dev,
        "baseline_dev": baseline_dev,
        "dark_sus": _clamp((1.0 - rel_darkness) / 0.4),
        "flow_sus": _clamp((flow_energy - 1.2) / 1.5),
        "color_sus": _clamp(color_dev / 0.3),
        "base_sus": _clamp(baseline_dev / 0.15),
    }

def score(features, weights=None):
    w = weights or DEFAULT_WEIGHTS
    s = (w["dark"] * features["dark_sus"] + w["flow"] * features["flow_sus"]
         + w["color"] * features["color_sus"] + w["base"] * features["base_sus"])
    features["score"] = s
    return s

def verdict(s, low=0.4, high=0.7):
    if s < low:
        return "normal"
    if s > high:
        return "suspect"
    return "gray"
