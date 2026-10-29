import cv2
import numpy as np

def mask_coverage(water_mask, roi_mask):
    """ROI 中被判定为水面的像素占比，用于检测漂浮物遮挡等异常。"""
    roi_pixels = int((roi_mask > 0).sum())
    if roi_pixels == 0:
        return 0.0
    both = int(((water_mask > 0) & (roi_mask > 0)).sum())
    return both / roi_pixels

class StubWaterSegmenter:
    """测试用：画面下半部分视为水面。"""
    def segment(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2:, :] = 255
        return mask

class OnnxWaterSegmenter:
    """用户已有 YOLO 水面分割模型的 ONNX 封装（单类别：水面）。"""
    def __init__(self, model_path, input_size=640, conf_threshold=0.5):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale

    def segment(self, frame):
        # 注意：此处后处理按 YOLOv8-seg 输出结构实现（检测框 + mask 原型系数）。
        # 接入时须用真实模型跑一次，按实际输出张量维度调整本方法的解析逻辑。
        canvas, scale = self._letterbox(frame)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = self.session.run(None, {self.input_name: blob})
        mask_small = self._decode(outs)  # (input_size, input_size) 0/255
        h, w = frame.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        mask = cv2.resize(mask_small, (nw, nh), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((h, w), dtype=np.uint8)
        full[:nh, :nw] = mask
        return full

    def _decode(self, outs):
        """解析 YOLOv8-seg 输出并合成单类别 mask。接入真实模型时按输出结构调整。"""
        det, proto = outs[0][0], outs[1][0]  # det: (4+1+32, N), proto: (32, mh, mw)
        boxes, confs, coeffs = det[:4], det[4], det[5:]
        keep = confs > self.conf_threshold
        if not keep.any():
            mh, mw = proto.shape[1:]
            return np.zeros((mh, mw), dtype=np.uint8)
        coeffs = coeffs[:, keep]
        mask = np.einsum("kn,kmn->mn", coeffs.T, proto)
        mask = (1 / (1 + np.exp(-mask)) > 0.5).astype(np.uint8) * 255
        return mask
