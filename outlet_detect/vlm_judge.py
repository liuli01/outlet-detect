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

class VLMResponseError(RuntimeError):
    """VLM 响应无效：超时/网关错误/空返回/JSON 乱码。
    调用方（pipeline）捕获后保留规则判定的 suspect 结果，继续推进状态机。"""

def build_prompt(features):
    return (
        "你是河道异常排水识别专家。给你三张图：图1是当前画面（已标注排口区/扩散区/参考水面区），"
        "图2是该点位历史正常画面，图3是上一采样点画面。\n"
        "算法特征值（仅供参考，不要当作判决规则）：排口区相对黑度={rel_darkness:.2f}，"
        "波动能量比={flow_energy:.2f}（>1.2表示排口区水面扰动明显强于参考区），"
        "颜色偏离={color_dev:.2f}，综合评分={score:.2f}。\n"
        "判定标准：排口/管道口正在排出水体、或水面出现正在扩散的异常水带，即为排放事件——"
        "污水常呈黑灰色，但也可能带白色浪花泡沫；雨污混排同样算排放事件。"
        "重点看：排口是否有水流涌出、水面是否有异常扩散带、与图2历史正常画面相比水体是否变黑变浑。\n"
        "需要排除的干扰：阴影/光照变化（静止、边界与岸上物体对应、不随时间扩散）、"
        "整个画面均匀的自然波纹（无排口处局部增强）。\n"
        "只返回JSON，不要其他内容：\n"
        '{{"is_discharge": bool, "confidence": 0~1, "reason": "一句话依据", '
        '"shadow_suspect": bool}}'
    ).format(**features)

def parse_vlm_response(text):
    if not text:
        raise VLMResponseError("VLM返回为空")
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        raise VLMResponseError("VLM返回无法解析: " + text[:100])
    try:
        d = json.loads(m.group(0))
        return VLMResult(bool(d["is_discharge"]), float(d["confidence"]),
                         str(d.get("reason", "")), bool(d.get("shadow_suspect", False)))
    except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
        raise VLMResponseError(f"VLM返回JSON无效: {e} | {text[:100]}") from e

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
        if not text:
            raise VLMResponseError("VLM返回为空")
        return parse_vlm_response(text)
