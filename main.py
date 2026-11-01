"""在线模式入口：python main.py
回放模式入口：python main.py --replay videos/xxx.mp4
测试台入口：python -m outlet_detect.webapp（FastAPI 页面上传照片测试）
"""
import argparse
import logging
import os
from pathlib import Path

from outlet_detect.view_align import AlignResult

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


class _NoRefAligner:
    """未标定或参照帧损坏的摄像头的占位对齐器：
    恒返回对齐失败 → pipeline 按 view_error 挂起，不阻断组装与调度。"""
    def align(self, frame):
        return AlignResult(False, None, 0.0, False)


def _env(key):
    v = os.environ.get(key)
    if not v:
        raise RuntimeError(f"缺少环境变量 {key}（参考 .env.example 配置 .env）")
    return v


def build_pipelines():
    """组装各摄像头的真实组件。部署配置从 .env / 环境变量读取（见 .env.example）。
    摄像头配置：config/cameras/*.json 中 hik_body.cameraIndexCode 非空的条目
    （无 cameraIndexCode 的辅助配置自动跳过）。"""
    from dotenv import load_dotenv

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

    load_dotenv()  # 读 .env（不覆盖已有环境变量）
    capture_fn = make_capture_fn(_env("HIK_BASE_URL"), _env("HIK_APP_KEY"),
                                 _env("HIK_APP_SECRET"))
    download_fn = make_download_fn()
    reporter = MinioReporter(_env("MINIO_ENDPOINT"), _env("MINIO_AK"),
                             _env("MINIO_SK"), _env("REPORT_URL"),
                             external_host=os.environ.get("EXTERNAL_HOST", ""))
    # VLM 终判：VLM_BASE_URL 留空 = 关闭（系统退化为规则+状态机）
    vlm = None
    if os.environ.get("VLM_BASE_URL"):
        vlm = VLMJudge(_env("VLM_BASE_URL"), _env("VLM_API_KEY"),
                       os.environ.get("VLM_MODEL", ""),
                       envelope=os.environ.get("VLM_ENVELOPE", "").lower()
                       in ("1", "true", "yes"))

    pipes = {}
    for cid in list_camera_ids():
        cfg = load_camera_config(cid)
        if not cfg.hik_body.get("cameraIndexCode"):
            continue  # 无 cameraIndexCode 的辅助配置
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
