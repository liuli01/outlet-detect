"""在线模式入口：python main.py
回放模式入口：python main.py --replay videos/xxx.mp4
"""
import argparse
import json
import logging
from pathlib import Path

from outlet_detect.view_align import AlignResult

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


class _NoRefAligner:
    """未标定或参照帧损坏的摄像头的占位对齐器：
    恒返回对齐失败 → pipeline 按 view_error 挂起，不阻断组装与调度。"""
    def align(self, frame):
        return AlignResult(False, None, 0.0, False)


def build_pipelines(deploy_path="config/deploy.json"):
    """组装各摄像头的真实组件。摄像头配置：config/cameras/*.json 中
    hik_body.cameraIndexCode 非空的条目（vlm.json 等辅助配置自动跳过）。"""
    from outlet_detect.alerter import LogAlerter
    from outlet_detect.cnn_classifier import CNNClassifier
    from outlet_detect.config import load_camera_config, list_camera_ids
    from outlet_detect.float_detector import HttpFloatDetector
    from outlet_detect.hik_adapter import make_capture_fn, make_download_fn
    from outlet_detect.pipeline import Pipeline
    from outlet_detect.reporter import MinioReporter
    from outlet_detect.sampler import HikFrameSampler
    from outlet_detect.state_machine import DischargeStateMachine
    from outlet_detect.storage import Storage
    from outlet_detect.view_align import ViewAligner
    from outlet_detect.vlm_judge import VLMJudge
    from outlet_detect.water_seg import HttpWaterSegmenter

    deploy = json.loads(Path(deploy_path).read_text(encoding="utf-8"))
    capture_fn = make_capture_fn(deploy["hik_base_url"], deploy["app_key"],
                                 deploy["app_secret"])
    download_fn = make_download_fn()
    reporter = MinioReporter(deploy["minio_endpoint"], deploy["minio_ak"],
                             deploy["minio_sk"], deploy["report_url"],
                             external_host=deploy.get("external_host", ""))
    vlm = None
    if "vlm" in list_camera_ids():
        vc = load_camera_config("vlm")
        vlm = VLMJudge(vc.vlm_base_url, vc.vlm_api_key, vc.vlm_model,
                       envelope=vc.vlm_envelope)

    pipes = {}
    for cid in list_camera_ids():
        cfg = load_camera_config(cid)
        if not cfg.hik_body.get("cameraIndexCode"):
            continue  # vlm.json 等辅助配置
        if cfg.reference_frame and Path(cfg.reference_frame).exists():
            try:
                aligner = ViewAligner(cfg.reference_frame, cfg.outfall_point)
            except FileNotFoundError:
                # 文件存在但内容损坏（截断 jpg 等）：cv2.imread 返回 None，
                # ViewAligner 抛 FileNotFoundError。单摄像头标定文件损坏
                # 不应瘫痪全局——降级为占位对齐器（view_error 挂起）
                aligner = _NoRefAligner()
        else:
            # 未标定（参照帧为空/缺失）同样用占位对齐器挂起，待标定后生效
            aligner = _NoRefAligner()
        pipes[cid] = Pipeline(
            config=cfg,
            sampler=HikFrameSampler(capture_fn, download_fn),
            segmenter=HttpWaterSegmenter(),
            aligner=aligner,
            cnn=CNNClassifier("models/discharge_cnn.onnx"),
            vlm=vlm,
            storage=Storage(f"data/{cid}.db", f"data/archive/{cid}"),
            state_machine=DischargeStateMachine(cid),
            alerter=LogAlerter(f"data/{cid}_alerts.log"),
            float_detector=HttpFloatDetector(
                engine_id=cfg.float_engine_id, confidence=cfg.float_confidence,
                ignore_classes=cfg.ignore_classes),
            float_sm=DischargeStateMachine(cid + "_float"),
            reporter=reporter,
        )
    return pipes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", help="离线回放视频路径")
    args = parser.parse_args()
    if args.replay:
        from outlet_detect.replay_tool import replay
        replay(args.replay)
    else:
        from outlet_detect.scheduler import run_loop
        run_loop(build_pipelines(), active_hours=(10, 15))


if __name__ == "__main__":
    main()
