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
        coeffs = coeffs[:, keep]  # (32, K)：32 个原型通道 × K 个过阈检测实例
        mask = np.einsum("kd,kmn->mn", coeffs, proto)
        mask = (1 / (1 + np.exp(-mask)) > 0.5).astype(np.uint8) * 255
        return mask

class HttpWaterSegmenter:
    """已部署水面分割 HTTP 服务的适配器（/infer，多边形响应栅格化）。"""
    def __init__(self, endpoint="http://127.0.0.1:8005/infer",
                 engine_id="river/20261029_152643",
                 confidence_threshold=0.25, timeout=30.0, post_fn=None,
                 erode_px=0):
        self.endpoint = endpoint
        self.engine_id = engine_id
        self.confidence_threshold = confidence_threshold
        self.timeout = timeout
        self.post_fn = post_fn or self._default_post
        # 分割服务的 confidence_threshold 是 detection 级开关，无法修边缘误检；
        # 用形态学腐蚀收缩 mask 边缘，削掉岸边/护坡的误判像素。0 = 不腐蚀。
        self.erode_px = erode_px

    @staticmethod
    def _default_post(url, payload, timeout):
        import requests
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def segment(self, frame):
        import base64
        b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
        payload = {"engine_id": self.engine_id, "image": b64,
                   "confidence_threshold": self.confidence_threshold}
        try:
            data = self.post_fn(self.endpoint, payload, self.timeout)
        except Exception as e:
            raise RuntimeError(f"水面分割服务调用失败: {e}") from e
        if not isinstance(data, dict) or "detections" not in data:
            raise RuntimeError(f"水面分割服务响应格式异常: {str(data)[:200]}")
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for det in data["detections"]:
            pts = det.get("points") or []
            if len(pts) >= 3:
                poly = np.array([[p["x"], p["y"]] for p in pts], dtype=np.int32)
                cv2.fillPoly(mask, [poly], 255)
        if self.erode_px > 0 and mask.any():
            k = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (2 * self.erode_px + 1,) * 2)
            mask = cv2.erode(mask, k)
        return mask
