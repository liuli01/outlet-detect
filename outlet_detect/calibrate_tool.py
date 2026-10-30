# outlet_detect/calibrate_tool.py
"""标定工具：python -m outlet_detect.calibrate_tool <camera_id> <图片路径>
鼠标左键点击排口位置，回车确认，ESC 取消。"""
import sys
from pathlib import Path

import cv2
import numpy as np

from outlet_detect.config import CameraConfig, save_camera_config
from outlet_detect.rois import build_rois

REF_DIR = Path("data/refs")

def generate_config(camera_id, frame, outfall_point, segmenter, hik_body):
    REF_DIR.mkdir(parents=True, exist_ok=True)
    ref_path = REF_DIR / f"{camera_id}_ref.jpg"
    cv2.imwrite(str(ref_path), frame)
    cfg = CameraConfig(camera_id=camera_id, outfall_point=tuple(outfall_point),
                       reference_frame=str(ref_path), hik_body=hik_body)
    save_camera_config(cfg)
    return cfg

def overlay_rois(frame, rois, outfall_point):
    vis = frame.copy()
    for roi, color in ((rois.outfall, (0, 0, 255)), (rois.plume, (0, 165, 255)),
                       (rois.reference, (0, 255, 0))):
        vis[roi > 0] = (0.5 * vis[roi > 0] + 0.5 * np.array(color)).astype(np.uint8)
    cv2.drawMarker(vis, outfall_point, (255, 0, 255), cv2.MARKER_CROSS, 20, 2)
    return vis

def main():
    camera_id, image_path = sys.argv[1], sys.argv[2]
    frame = cv2.imread(image_path)
    assert frame is not None, f"读不到图片 {image_path}"
    clicked = []
    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            clicked.clear(); clicked.append((x, y))
    cv2.namedWindow("calibrate")
    cv2.setMouseCallback("calibrate", on_mouse)
    while True:
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            cv2.destroyAllWindows(); return
        if key == 13 and clicked:
            break
    cv2.destroyAllWindows()
    point = clicked[0]
    from outlet_detect.water_seg import OnnxWaterSegmenter
    import os
    model = os.environ.get("WATER_SEG_MODEL", "models/water_seg.onnx")
    segmenter = OnnxWaterSegmenter(model)
    mask = segmenter.segment(frame)
    rois = build_rois(mask, point)
    preview = overlay_rois(frame, rois, point)
    preview_path = Path(f"config/cameras/{camera_id}_preview.jpg")
    cv2.imwrite(str(preview_path), preview)
    hik_body = {"cameraIndexCode": input("cameraIndexCode: ")}
    generate_config(camera_id, frame, point, segmenter, hik_body)
    print(f"配置已保存，ROI 预览: {preview_path}")

if __name__ == "__main__":
    main()
