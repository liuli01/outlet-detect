"""VLM 连通性测试：用排污样本调公司多模态大模型终判。
手动运行：uv run python tools/test_vlm.py
配置从 .env 读（VLM_BASE_URL/VLM_API_KEY/VLM_MODEL/VLM_ENVELOPE，见 .env.example）。
注意：本文件不是 pytest 用例（模块名仅沿用习惯），主体在 __main__ 守卫内。
"""
import os

import numpy as np
import cv2
from dotenv import load_dotenv

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # 仓库根入 path，保证能 import outlet_detect

from outlet_detect.vlm_judge import VLMJudge


def imread_cn(p):
    return cv2.imdecode(np.fromfile(p, dtype=np.uint8), cv2.IMREAD_COLOR)


# 创新河排污样本（明显黑色污水+湍流）与无排污照片各测一张
# ref 用无排污照片模拟"该点位历史正常画面"（线上来自标定参照帧）
CASES = [
    ("排污-创新河1684788", "images/排污/创新河春申路河道球机_1687161684788/0.jpg",
     "images/无排污/0.jpg",
     {"rel_darkness": 1.07, "flow_energy": 2.88, "color_dev": 0.20, "score": 0.48}),
    ("无排污-18.jpg(阴影嫌疑)", "images/无排污/18.jpg", "images/无排污/18.jpg",
     {"rel_darkness": 0.75, "flow_energy": 1.00, "color_dev": 0.39, "score": 0.42}),
]


def main():
    load_dotenv()
    judge = VLMJudge(os.environ["VLM_BASE_URL"], os.environ["VLM_API_KEY"],
                     os.environ.get("VLM_MODEL", ""),
                     envelope=os.environ.get("VLM_ENVELOPE", "").lower()
                     in ("1", "true", "yes"))
    for name, path, ref_path, feats in CASES:
        img = imread_cn(path)
        ref = imread_cn(ref_path)
        assert img is not None and ref is not None, path
        try:
            r = judge.judge(img, ref, img, feats)
            print(f"[{name}]")
            print(f"  is_discharge={r.is_discharge} confidence={r.confidence} "
                  f"shadow_suspect={r.shadow_suspect}")
            print(f"  reason: {r.reason}\n")
        except Exception as e:
            print(f"[{name}] 调用失败: {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
