"""结果上报：标注图绘制 + minio 上传 + 业务平台上报（沿用 hik-sdk.py 链路）。

字段约定：ResultA=漂浮物置信度，ResultB=ResultC=排污置信度（排污与水体颜色共享）。
上报失败抛 ReporterError，由调用方记日志——不阻断检测主线。
"""
import logging
from dataclasses import dataclass
from datetime import datetime
from urllib import parse

import cv2

log = logging.getLogger(__name__)


class ReporterError(RuntimeError):
    pass


@dataclass
class ReportEvent:
    index_code: str
    input_path: str     # 本地原图路径，或 http(s) URL（如海康 picUrl）
    output_path: str    # 本地标注图路径，或 http(s) URL
    result_a: float = 0.0   # 漂浮物置信度
    result_b: float = 0.0   # 排污置信度
    result_c: float = 0.0   # 水体颜色置信度（排污时同 result_b）


def _clamp01(v):
    return max(0.0, min(1.0, float(v)))


def annotate_float(frame, detections):
    """漂浮物标注图：红 bbox + 类别置信度。"""
    out = frame.copy()
    for d in detections:
        x1, y1, x2, y2 = map(int, d.bbox)
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 3)
        cv2.putText(out, f"{d.class_name} {d.confidence:.2f}", (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
    return out


def annotate_discharge(frame, rois, score):
    """排污标注图：排口红/扩散黄/参考绿 ROI 轮廓 + score（中文 reason 不进图，
    cv2.putText 不支持中文；完整理由在 SQLite samples 表与告警日志里）。"""
    out = frame.copy()
    for mask, color in ((rois.outfall, (0, 0, 255)), (rois.plume, (0, 255, 255)),
                        (rois.reference, (0, 255, 0))):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)
    cv2.putText(out, f"discharge score={score:.2f}", (20, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
    return out


class MinioReporter:
    def __init__(self, minio_endpoint, access_key, secret_key,
                 report_url, external_host="", client=None, post_fn=None):
        self.report_url = report_url
        self.external_host = external_host
        if client is not None:
            self.client = client
        else:
            from minio import Minio
            self.client = Minio(endpoint=minio_endpoint, access_key=access_key,
                                secret_key=secret_key, secure=False)
        self._minio_host = minio_endpoint  # 完整 endpoint（host:port），外网替换时整体换掉
        self.post_fn = post_fn or self._default_post

    @staticmethod
    def _default_post(url, data):
        import requests
        r = requests.post(url,
                          headers={"Content-Type": "application/x-www-form-urlencoded"},
                          data=parse.urlencode(data, True), timeout=30)
        r.raise_for_status()
        return r

    def _upload(self, bucket, object_name, file_path):
        self.client.fput_object(bucket, object_name, file_path,
                                content_type="image/jpeg")
        address = self.client.presigned_get_object(bucket, object_name).split("?")[0]
        if self.external_host:
            address = address.replace(self._minio_host, self.external_host)
        return address

    def _url_or_upload(self, bucket, path, object_name):
        if path.startswith(("http://", "https://")):
            return path
        return self._upload(bucket, object_name, path)

    def report(self, event):
        object_name = f"{event.index_code}-{datetime.now():%Y-%m-%d-%H-%M}.jpg"
        try:
            input_url = self._url_or_upload("input-image", event.input_path, object_name)
            output_url = self._url_or_upload("ai-image", event.output_path, object_name)
        except Exception as e:
            raise ReporterError(f"minio 上传失败: {e}") from e
        data = {"IndexCode": event.index_code, "InputPicture": input_url,
                "OutputPicture": output_url, "ResultA": _clamp01(event.result_a),
                "ResultB": _clamp01(event.result_b), "ResultC": _clamp01(event.result_c)}
        try:
            self.post_fn(self.report_url, data)
        except Exception as e:
            raise ReporterError(f"上报接口失败: {e}") from e
        log.info("已上报 %s: A=%.2f B=%.2f C=%.2f", event.index_code,
                 data["ResultA"], data["ResultB"], data["ResultC"])
        return output_url
