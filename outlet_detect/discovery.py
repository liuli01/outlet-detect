"""摄像头自动接入：平台清单里未标定的摄像头，启动时自动建档。

自动建档 = 抓一帧当参照帧 + 排口点留空（运行时水面质心兜底）。
抓图失败的摄像头跳过本轮（下次启动重试），不影响已有摄像头。
"""
import logging
from pathlib import Path

import cv2

from outlet_detect.config import CameraConfig, list_camera_ids, save_camera_config

log = logging.getLogger(__name__)


def auto_register(codes, capture_fn, download_fn, ref_dir="data/reference",
                  float_engine_id="", float_confidence=0.3,
                  ignore_classes=None):
    """对清单里还没有标定文件的摄像头自动建档，返回新建的 camera_id 列表。

    codes: 平台 indexCode 全量列表；capture_fn(body)->picUrl；download_fn(url)->BGR。
    ignore_classes: 漂浮物检测要排除的类别（如 ["aeration"] 曝气泡沫）。
    """
    existing = set(list_camera_ids())
    created = []
    Path(ref_dir).mkdir(parents=True, exist_ok=True)
    for code in codes:
        if code in existing:
            continue
        try:
            url = capture_fn({"cameraIndexCode": code})
            frame = download_fn(url)
            ref_path = str(Path(ref_dir) / f"{code}.jpg")
            cv2.imencode(".jpg", frame)[1].tofile(ref_path)  # imwrite 中文路径会静默失败
            save_camera_config(CameraConfig(
                camera_id=code, outfall_point=None,  # 水面质心兜底
                reference_frame=ref_path,
                hik_body={"cameraIndexCode": code},
                float_engine_id=float_engine_id,
                float_confidence=float_confidence,
                ignore_classes=ignore_classes or []))
            created.append(code)
            log.info("自动接入摄像头 %s（参照帧 %s，排口点=水面质心）", code, ref_path)
        except Exception as e:
            log.warning("摄像头 %s 自动建档失败，跳过（下次启动重试）: %s", code, e)
    return created
