# outlet_detect/cnn_classifier.py
import os

import cv2
import numpy as np

def crop_for_cnn(frame, rois, size=224):
    ys, xs = np.nonzero((rois.outfall > 0) | (rois.plume > 0))
    if len(xs) == 0:
        return cv2.resize(frame, (size, size))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))

class CNNClassifier:
    """MobileNetV3-Small 二分类 ONNX 推理；无模型文件时关闭（冷启动）。"""
    def __init__(self, model_path):
        self.enabled = bool(model_path) and os.path.exists(model_path)
        if self.enabled:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name

    def classify(self, crop):
        if not self.enabled:
            raise RuntimeError("CNN 未启用（无模型文件）")
        blob = crop.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        logits = self.session.run(None, {self.input_name: blob})[0][0]
        exp = np.exp(logits - logits.max())
        return float(exp[1] / exp.sum())  # 类别1 = 排污
