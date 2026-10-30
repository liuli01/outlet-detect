import json
import logging
import time
from datetime import datetime

import cv2

from outlet_detect.rois import build_rois
from outlet_detect.rule_engine import compute_features, score, verdict

log = logging.getLogger(__name__)

class Pipeline:
    def __init__(self, config, sampler, segmenter, aligner, cnn, vlm,
                 storage, state_machine, alerter):
        self.config = config
        self.sampler = sampler
        self.segmenter = segmenter
        self.aligner = aligner
        self.cnn = cnn
        self.vlm = vlm
        self.storage = storage
        self.sm = state_machine
        self.alerter = alerter
        self._prev_frame = None

    def run_once(self, now_ts=None):
        t0 = time.time()
        ts_epoch = now_ts if now_ts is not None else time.time()
        ts = datetime.fromtimestamp(ts_epoch).isoformat(timespec="seconds")
        cid = self.config.camera_id
        feats, frames, vlm_json, cnn_prob = {}, None, None, None
        status = "sample_error"

        try:
            frames = self.sampler.capture(cid, n=5, body=self.config.hik_body)
            mask = self.segmenter.segment(frames[0])
            if mask.sum() == 0:
                log.warning("%s 分割结果无水面，按采样失败处理", cid)
                self.storage.record_sample(cid, ts, feats, "error", None, None,
                                           status, [], int((time.time() - t0) * 1000))
                return status
        except Exception as e:
            log.exception("%s 采样/分割失败: %s", cid, e)
            self.storage.record_sample(cid, ts, feats, "error", None, None,
                                       status, [], int((time.time() - t0) * 1000))
            return status

        try:
            ar = self.aligner.align(frames[0])
            if not ar.ok or not ar.in_frame:
                self.sm.suspend()
                self.storage.record_sample(cid, ts, feats, "view_error", None, None,
                                           "view_error", [], int((time.time() - t0) * 1000))
                log.warning("%s 视角异常(conf=%.2f)，暂停判定", cid, ar.confidence)
                return "view_error"
        except Exception as e:
            log.exception("%s 对齐失败: %s", cid, e)
            self.storage.record_sample(cid, ts, feats, "error", None, None,
                                       status, [], int((time.time() - t0) * 1000))
            return status
        if self.sm.suspended:
            log.info("%s 视角恢复", cid)
            self.sm.resume()

        try:
            rois = build_rois(mask, ar.mapped_point)
            baseline = self.storage.get_baseline_darkness(cid, hour=int(ts[11:13]), before_ts=ts)
            feats = compute_features(frames, rois, baseline)
            s = score(feats)
            v = verdict(s, self.config.score_low, self.config.score_high)

            suspect = (v == "suspect")
            if v == "gray":
                if self.cnn.enabled:
                    from outlet_detect.cnn_classifier import crop_for_cnn
                    try:
                        cnn_prob = self.cnn.classify(crop_for_cnn(frames[0], rois))
                        suspect = cnn_prob >= 0.5
                    except Exception as e:
                        log.exception("%s CNN 复判失败，按疑似送 VLM: %s", cid, e)
                        suspect = True
                else:
                    suspect = True  # 冷启动：灰色直接送 VLM

            vlm_result = None
            if suspect and self.vlm is not None:
                try:
                    ref = None
                    if self.config.reference_frame:
                        ref = cv2.imread(self.config.reference_frame)
                    if ref is None:
                        log.warning("%s 参考帧不可用(%s)，退化为当前帧",
                                    cid, self.config.reference_frame)
                        ref = frames[0]
                    prev = self._prev_frame if self._prev_frame is not None else frames[-1]
                    vlm_result = self.vlm.judge(frames[0], ref, prev, feats)
                    vlm_json = json.dumps(vlm_result.__dict__, ensure_ascii=False)
                    suspect = vlm_result.is_discharge
                except Exception as e:
                    # VLM 无效（超时/乱码/空返回）：保留规则判定的 suspect，
                    # 状态机照常推进——VLM 全挂时系统退化为 规则引擎+状态机 两级
                    log.exception("%s VLM 调用失败，按规则结果处理: %s", cid, e)
            self._prev_frame = frames[0]

            keep = suspect
            frame_paths = self.storage.archive_frames(cid, ts.replace(":", "-"), frames, keep)
            if vlm_result is not None:
                self.storage.save_labeled_sample(
                    cid, ts, frame_paths[0],
                    "discharge" if vlm_result.is_discharge else "normal",
                    "vlm", vlm_result.reason)

            ev = self.sm.update(suspect, ts_epoch)
            status = "normal"
            if ev is not None:
                from outlet_detect.alerter import AlertPayload
                self.alerter.alert(AlertPayload(
                    camera_id=cid, ts=ts, frame_paths=frame_paths,
                    vlm_reason=vlm_result.reason if vlm_result else "",
                    features=feats, consecutive=ev.consecutive))
                status = "alert"
            elif suspect:
                status = "suspect"
            self.storage.record_sample(cid, ts, feats, v, cnn_prob, vlm_json,
                                       status, frame_paths, int((time.time() - t0) * 1000))
            return status
        except Exception as e:
            log.exception("%s 判定流程异常: %s", cid, e)
            self.storage.record_sample(cid, ts, feats, "error", None, None,
                                       "sample_error", [], int((time.time() - t0) * 1000))
            return "sample_error"
