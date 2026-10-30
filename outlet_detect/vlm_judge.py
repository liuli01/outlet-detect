# outlet_detect/vlm_judge.py
import base64
import json
import re
from dataclasses import dataclass

import cv2

@dataclass
class VLMResult:
    is_discharge: bool
    confidence: float
    reason: str
    shadow_suspect: bool

def build_prompt(features):
    return (
        "你是河道排污识别专家。给你三张图：图1是当前画面（已标注排口区/扩散区/参考水面区），"
        "图2是该点位历史正常画面，图3是上一采样点画面。\n"
        "算法特征值：排口区相对黑度={rel_darkness:.2f}（<1表示排口区比参考区暗），"
        "波动能量比={flow_energy:.2f}（>1.2表示排口区水面扰动明显强于参考区），"
        "颜色偏离={color_dev:.2f}，综合评分={score:.2f}。\n"
        "判断是否存在污水排放。特别注意区分阴影/光照变化造成的假阳性"
        "（阴影通常静止、边界与岸上物体对应、不随时间扩散）。\n"
        "只返回JSON，不要其他内容：\n"
        '{{"is_discharge": bool, "confidence": 0~1, "reason": "一句话依据", '
        '"shadow_suspect": bool}}'
    ).format(**features)

def parse_vlm_response(text):
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return VLMResult(False, 0.0, "VLM返回无法解析: " + text[:100], False)
    try:
        d = json.loads(m.group(0))
        return VLMResult(bool(d["is_discharge"]), float(d["confidence"]),
                         str(d.get("reason", "")), bool(d.get("shadow_suspect", False)))
    except (json.JSONDecodeError, KeyError, ValueError):
        return VLMResult(False, 0.0, "VLM返回无法解析: " + text[:100], False)

def _b64(img):
    return base64.b64encode(cv2.imencode(".jpg", img)[1]).decode()

class VLMJudge:
    def __init__(self, base_url, api_key, model, client=None, envelope=False):
        self.model = model
        self.envelope = envelope
        self._base_url = base_url
        self._api_key = api_key
        if client is not None:
            self.client = client
        elif not envelope:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def _post_envelope(self, content):
        """公司网关包装格式：响应为 {code, message, data: {...OpenAI响应...}}。
        该端点必须带图（纯文本请求返回 S0202）；judge 始终带图，不受影响。"""
        import requests
        r = requests.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={"model": self.model, "temperature": 0, "max_tokens": 3000,
                  "messages": [{"role": "user", "content": content}]},
            timeout=120,
        )
        d = r.json()
        if d.get("code") != "00000":
            raise RuntimeError(f"VLM 服务错误: {d.get('code')} {d.get('message')}")
        return d["data"]["choices"][0]["message"]["content"]

    def judge(self, current_frame, ref_frame, prev_frame, features):
        content = [{"type": "text", "text": build_prompt(features)}]
        for img in (current_frame, ref_frame, prev_frame):
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_b64(img)}"}})
        if self.envelope:
            text = self._post_envelope(content)
        else:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": content}],
                temperature=0,
            )
            text = resp.choices[0].message.content
        return parse_vlm_response(text)
