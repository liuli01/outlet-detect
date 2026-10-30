# 漂浮物检测 + 结果上报 + 时段调度 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** 在排污检测系统上扩展漂浮物检测线、minio+业务平台结果上报、10:00~15:00 时段调度。

**Architecture:** 漂浮物检测复用 8005 推理服务（检测任务 bbox），排污 5 连帧的第 1 帧兼做漂浮物输入；两类事件各自三连确认状态机，共用 MinioReporter 上报；scheduler 加运行时段窗口。

**Tech Stack:** Python 3.12（uv）、opencv-python、numpy、requests、minio、openai SDK、pytest

## Global Constraints

- 设计文档：`docs/superpowers/specs/2026-10-30-float-detection-reporting-design.md`（字段约定 ResultA=漂浮物置信度、ResultB=ResultC=排污置信度）
- 测试命令：`uv run pytest`；pytest.ini `pythonpath = ["."]`
- 上报失败、漂浮物引擎失败均不得阻断排污主线（记日志降级）
- 漂浮物 engine_id 为空 = 检测关闭（模型训练中，部署后填配置启用）
- 中文路径：cv2.imread/imwrite 不支持，图片读写用 imdecode/imencode+tofile（摄像头 cid 为海康 indexCode 纯 ASCII，archive 路径无中文）
- 提交只暂存本任务文件，不 `git add -A`

---

### Task 1: 海康 API 适配器

**Files:**
- Create: `outlet_detect/hik_adapter.py`
- Test: `tests/test_hik_adapter.py`

**Interfaces:**
- Produces: `HikApi(base_url, api_path, app_key, app_secret).post(body) -> dict`
- Produces: `make_capture_fn(base_url, app_key, app_secret) -> capture_fn(body) -> str(picUrl)`（失败抛 RuntimeError）
- Produces: `make_download_fn(timeout=30) -> download_fn(url) -> np.ndarray`（BGR；失败抛 RuntimeError）
- Consumed by: Task 7（main.py 组装 HikFrameSampler）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_hik_adapter.py
import numpy as np
import pytest

from outlet_detect.hik_adapter import HikApi, make_capture_fn, make_download_fn


class FakeResp:
    def __init__(self, json_data=None, content=b""):
        self._json = json_data
        self.content = content

    def json(self):
        return self._json

    def raise_for_status(self):
        pass


def test_hik_api_signs_and_posts(monkeypatch):
    calls = []
    def fake_post(url, data=None, headers=None, verify=None, timeout=None):
        calls.append({"url": url, "headers": headers})
        return FakeResp({"code": "0", "msg": "success", "data": {"picUrl": "http://x/1.jpg"}})
    monkeypatch.setattr("requests.post", fake_post)
    api = HikApi("https://hik:14443", "/artemis/api/video/v1/manualCapture", "KEY", "SECRET")
    resp = api.post({"cameraIndexCode": "abc"})
    assert resp["data"]["picUrl"] == "http://x/1.jpg"
    h = calls[0]["headers"]
    assert h["x-ca-key"] == "KEY" and "x-ca-signature" in h and "x-ca-nonce" in h
    assert calls[0]["url"] == "https://hik:14443/artemis/api/video/v1/manualCapture"


def test_capture_fn_returns_pic_url(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp(
        {"msg": "success", "data": {"picUrl": "http://x/p.jpg"}}))
    cap = make_capture_fn("https://hik:14443", "K", "S")
    assert cap({"cameraIndexCode": "abc"}) == "http://x/p.jpg"


def test_capture_fn_raises_on_failure(monkeypatch):
    monkeypatch.setattr("requests.post", lambda *a, **k: FakeResp({"msg": "denied"}))
    cap = make_capture_fn("https://hik:14443", "K", "S")
    with pytest.raises(RuntimeError):
        cap({"cameraIndexCode": "abc"})


def test_download_fn_decodes_image(monkeypatch):
    import cv2
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    monkeypatch.setattr("requests.get",
                        lambda *a, **k: FakeResp(content=cv2.imencode(".jpg", img)[1].tobytes()))
    out = make_download_fn()("http://x/1.jpg")
    assert out.shape == (10, 10, 3)


def test_download_fn_raises_on_bad_bytes(monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: FakeResp(content=b"not an image"))
    with pytest.raises(RuntimeError):
        make_download_fn()("http://x/bad")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_hik_adapter.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 hik_adapter**

```python
# outlet_detect/hik_adapter.py
"""海康 Artemis 平台适配：AK/SK 签名抓图 + 摄像头列表（移植自 hik-sdk.py）。"""
import base64
import hashlib
import hmac
import json
import time
import uuid
from urllib import parse

import cv2
import numpy as np
import requests

requests.packages.urllib3.disable_warnings()  # Artemis 自签证书


class HikApi:
    """Artemis AK/SK 签名 POST 客户端。api_path 需含 /artemis 前缀。"""

    def __init__(self, base_url, api_path, app_key, app_secret):
        self.base_url = base_url
        self.api_path = api_path
        self.app_key = app_key
        self.app_secret = app_secret

    def post(self, body):
        nonce = str(uuid.uuid4())
        ts = str(int(round(time.time() * 1000)))
        sign_str = ("POST\n*/*\napplication/json" + "\nx-ca-key:" + self.app_key
                    + "\nx-ca-nonce:" + nonce + "\nx-ca-timestamp:" + ts + "\n"
                    + self.api_path)
        signature = base64.b64encode(
            hmac.new(self.app_secret.encode(), sign_str.encode(),
                     digestmod=hashlib.sha256).digest()).decode()
        headers = {
            "Accept": "*/*",
            "Content-Type": "application/json",
            "x-ca-key": self.app_key,
            "x-ca-signature-headers": "x-ca-key,x-ca-nonce,x-ca-timestamp",
            "x-ca-signature": signature,
            "x-ca-timestamp": ts,
            "x-ca-nonce": nonce,
        }
        r = requests.post(self.base_url + self.api_path, data=json.dumps(body),
                          headers=headers, verify=False, timeout=30)
        return r.json()


def make_capture_fn(base_url, app_key, app_secret):
    """capture_fn(body) -> picUrl（manualCapture）；失败抛 RuntimeError。"""
    api = HikApi(base_url, "/artemis/api/video/v1/manualCapture", app_key, app_secret)

    def capture_fn(body):
        resp = api.post(body)
        if resp.get("msg") == "success":
            return resp["data"]["picUrl"]
        raise RuntimeError(f"manualCapture 失败: {resp}")

    return capture_fn


def make_download_fn(timeout=30):
    """download_fn(url) -> BGR ndarray；解码失败抛 RuntimeError。"""

    def download_fn(url):
        r = requests.get(url, verify=False, timeout=timeout)
        r.raise_for_status()
        img = cv2.imdecode(np.frombuffer(r.content, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"图片解码失败: {url[:100]}")
        return img

    return download_fn


def get_index_code_list(platform_url="http://192.166.250.224:17082/api/jyly/cameraindexcode"):
    """从业务平台获取摄像头 IndexCode 列表。"""
    r = requests.post(platform_url,
                      headers={"Content-Type": "application/x-www-form-urlencoded"},
                      data=parse.urlencode({}, True), timeout=30)
    return [i["indexCode"] for i in r.json()["data"]]
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_hik_adapter.py -v`
Expected: 5 passed

```bash
git add outlet_detect/hik_adapter.py tests/test_hik_adapter.py
git commit -m "feat: 海康Artemis适配器（签名抓图/下载/摄像头列表）"
```

---

### Task 2: 漂浮物检测器

**Files:**
- Create: `outlet_detect/float_detector.py`
- Test: `tests/test_float_detector.py`

**Interfaces:**
- Produces: `FloatDetection(class_name: str, confidence: float, bbox: tuple)`
- Produces: `HttpFloatDetector(endpoint="http://127.0.0.1:8005/infer", engine_id="", confidence=0.6, ignore_classes=(), timeout=30.0, post_fn=None)`
  - `.enabled`：engine_id 非空
  - `.detect(frame) -> list[FloatDetection]`：8005 /infer，过滤置信度与 ignore 类别；服务异常/格式异常抛 RuntimeError
- Consumed by: Task 6（Pipeline 漂浮物分支）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_float_detector.py
import numpy as np
import pytest

from outlet_detect.float_detector import HttpFloatDetector

RESP = {"detections": [
    {"class_name": "bottle", "confidence": 0.88, "bbox": [10, 20, 50, 60]},
    {"class_name": "aeration", "confidence": 0.95, "bbox": [100, 100, 200, 200]},
    {"class_name": "bag", "confidence": 0.4, "bbox": [5, 5, 15, 15]},  # 低于阈值
], "engine_id": "float/1", "inference_time_ms": 50}


def make_detector(response=RESP, **kw):
    def fake_post(url, payload, timeout):
        return response
    return HttpFloatDetector(engine_id="float/1", post_fn=fake_post, **kw)


def test_disabled_when_no_engine_id():
    assert HttpFloatDetector(engine_id="").enabled is False
    assert HttpFloatDetector(engine_id="float/1").enabled is True


def test_detect_filters_ignore_and_threshold():
    # 注意：confidence_threshold 已随请求发给服务端，这里 0.4 的 bag 模拟服务端漏网，
    # 本地按 self.confidence 再过滤一次兜底
    det = make_detector(confidence=0.6, ignore_classes=["aeration"])
    out = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert len(out) == 1
    assert out[0].class_name == "bottle" and out[0].confidence == 0.88
    assert out[0].bbox == (10, 20, 50, 60)


def test_empty_detections_returns_empty():
    det = make_detector(response={"detections": []})
    assert det.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []


def test_service_error_raises():
    def bad_post(url, payload, timeout):
        raise ConnectionError("refused")
    det = HttpFloatDetector(engine_id="float/1", post_fn=bad_post)
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))


def test_malformed_response_raises():
    det = make_detector(response={"bad": "shape"})
    with pytest.raises(RuntimeError):
        det.detect(np.zeros((10, 10, 3), dtype=np.uint8))
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_float_detector.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 float_detector**

```python
# outlet_detect/float_detector.py
"""漂浮物检测：复用 8005 推理服务（检测任务，返回 bbox）。"""
from dataclasses import dataclass


@dataclass
class FloatDetection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


class HttpFloatDetector:
    def __init__(self, endpoint="http://127.0.0.1:8005/infer", engine_id="",
                 confidence=0.6, ignore_classes=(), timeout=30.0, post_fn=None):
        self.endpoint = endpoint
        self.engine_id = engine_id      # 空 = 关闭（模型未部署时的默认态）
        self.confidence = confidence
        self.ignore_classes = set(ignore_classes)
        self.timeout = timeout
        self.post_fn = post_fn or self._default_post

    @property
    def enabled(self):
        return bool(self.engine_id)

    @staticmethod
    def _default_post(url, payload, timeout):
        import requests
        resp = requests.post(url, json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    def detect(self, frame):
        import base64
        import cv2
        b64 = base64.b64encode(cv2.imencode(".jpg", frame)[1]).decode()
        payload = {"engine_id": self.engine_id, "image": b64,
                   "confidence_threshold": self.confidence}
        try:
            data = self.post_fn(self.endpoint, payload, self.timeout)
        except Exception as e:
            raise RuntimeError(f"漂浮物检测服务调用失败: {e}") from e
        if not isinstance(data, dict) or "detections" not in data:
            raise RuntimeError(f"漂浮物检测响应格式异常: {str(data)[:200]}")
        out = []
        for det in data["detections"]:
            cls = str(det.get("class_name", ""))
            conf = float(det.get("confidence", 0.0))
            if cls in self.ignore_classes or conf < self.confidence:
                continue
            out.append(FloatDetection(cls, conf, tuple(det.get("bbox", (0, 0, 0, 0)))))
        return out
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_float_detector.py -v`
Expected: 5 passed

```bash
git add outlet_detect/float_detector.py tests/test_float_detector.py
git commit -m "feat: 漂浮物检测器（8005服务，可关闭/类别过滤）"
```

---

### Task 3: 结果上报器（标注图 + minio + 业务平台）

**Files:**
- Create: `outlet_detect/reporter.py`
- Test: `tests/test_reporter.py`
- Modify: `pyproject.toml`（`uv add minio`）

**Interfaces:**
- Produces: `ReporterError(RuntimeError)`
- Produces: `ReportEvent(index_code, input_path, output_path, result_a=0.0, result_b=0.0, result_c=0.0)`
- Produces: `annotate_float(frame, detections) -> frame`（bbox+类别置信度）
- Produces: `annotate_discharge(frame, rois, score) -> frame`（三色 ROI 轮廓+score 文字）
- Produces: `MinioReporter(minio_endpoint, access_key, secret_key, report_url, external_host="", client=None, post_fn=None).report(event) -> output_url`
  - 上传：本地原图→input-image 桶、标注图→ai-image 桶；input_path 已是 http(s) URL 时直接用
  - URL 外网替换：`presigned_get_object` 地址的 minio 内网 host 换成 external_host
  - result_a/b/c 上报前 clamp 到 [0,1]
  - 上传/上报失败抛 ReporterError
- Consumed by: Task 6（Pipeline 上报）、Task 7（main.py）

- [ ] **Step 1: 装依赖 + 写失败测试**

```bash
uv add minio
```

```python
# tests/test_reporter.py
import numpy as np
import pytest

from outlet_detect.reporter import (MinioReporter, ReportEvent, ReporterError,
                                    annotate_float)


class FakeMinio:
    def __init__(self):
        self.uploads = []

    def fput_object(self, bucket, object_name, file_path, content_type=None):
        self.uploads.append((bucket, object_name, file_path))

    def presigned_get_object(self, bucket, object_name):
        return f"http://192.166.250.223:17077/{bucket}/{object_name}?sig=x"


def make_reporter(**kw):
    posts = []

    def fake_post(url, data):
        posts.append((url, data))

    return MinioReporter("192.166.250.223:17077", "ak", "sk",
                         "http://192.166.250.224:17082/api/insert_ai_result",
                         external_host="223.108.117.18",
                         client=FakeMinio(), post_fn=fake_post, **kw), posts


def test_report_uploads_and_posts(tmp_path):
    rep, posts = make_reporter()
    img = tmp_path / "a.jpg"
    import cv2
    cv2.imwrite(str(img), np.zeros((10, 10, 3), dtype=np.uint8))
    ev = ReportEvent(index_code="cam01", input_path=str(img), output_path=str(img),
                     result_a=0.88, result_b=0.72, result_c=0.72)
    url = rep.report(ev)
    # 原图+标注图都上传（input 为本地路径时）
    buckets = [u[0] for u in rep.client.uploads]
    assert buckets == ["input-image", "ai-image"]
    # 外网地址替换
    assert url.startswith("http://223.108.117.18/")
    # 上报字段
    data = posts[0][1]
    assert data["IndexCode"] == "cam01"
    assert data["ResultA"] == 0.88 and data["ResultB"] == 0.72 and data["ResultC"] == 0.72
    assert data["InputPicture"].startswith("http://223.108.117.18/")


def test_report_uses_url_input_directly():
    rep, posts = make_reporter()
    ev = ReportEvent(index_code="cam01", input_path="http://hik/pic.jpg",
                     output_path="http://hik/pic.jpg", result_b=0.9, result_c=0.9)
    # input 是 URL 时不上传原图，只上传 output（output 此处也是 URL，跳过上传）
    rep.report(ev)
    assert rep.client.uploads == []
    assert posts[0][1]["InputPicture"] == "http://hik/pic.jpg"


def test_result_values_clamped():
    rep, posts = make_reporter()
    ev = ReportEvent(index_code="c", input_path="http://x/1.jpg",
                     output_path="http://x/2.jpg", result_a=1.5, result_b=-0.2, result_c=0.5)
    rep.report(ev)
    assert posts[0][1]["ResultA"] == 1.0 and posts[0][1]["ResultB"] == 0.0


def test_upload_failure_raises_reporter_error(tmp_path):
    class BadMinio(FakeMinio):
        def fput_object(self, *a, **k):
            raise ConnectionError("minio down")
    posts = []
    rep = MinioReporter("h:9000", "ak", "sk", "http://r/api",
                        client=BadMinio(), post_fn=lambda u, d: posts.append(d))
    ev = ReportEvent(index_code="c", input_path=str(tmp_path / "x.jpg"),
                     output_path=str(tmp_path / "x.jpg"))
    (tmp_path / "x.jpg").write_bytes(b"x")
    with pytest.raises(ReporterError):
        rep.report(ev)


def test_annotate_float_draws_boxes():
    from outlet_detect.float_detector import FloatDetection
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    out = annotate_float(frame, [FloatDetection("bottle", 0.88, (10, 20, 50, 60))])
    assert out.shape == frame.shape
    assert out.sum() > 0  # 画了内容
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_reporter.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 reporter**

```python
# outlet_detect/reporter.py
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
        self._minio_host = minio_endpoint.split(":")[0]
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
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_reporter.py -v`
Expected: 5 passed

```bash
git add outlet_detect/reporter.py tests/test_reporter.py pyproject.toml uv.lock
git commit -m "feat: 结果上报器（标注图/minio上传/业务平台上报）"
```

---

### Task 4: 配置字段扩展 + 调度时段窗口

**Files:**
- Modify: `outlet_detect/config.py`（CameraConfig 加 3 字段）
- Modify: `outlet_detect/scheduler.py`（active_hours 窗口）
- Test: `tests/test_scheduler_window.py`、修改 `tests/test_config.py`（补新字段 roundtrip 断言）

**Interfaces:**
- Produces: `CameraConfig(..., float_engine_id: str = "", float_confidence: float = 0.6, ignore_classes: list = field(default_factory=list))`
- Produces: `run_loop(pipelines, interval_sec=300, stagger_sec=10, stop_flag=None, active_hours=None, now_fn=None)`——active_hours=(10,15) 时只在 [10:00,15:00) 内执行；窗口外每 60 秒重查

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scheduler_window.py
import threading
import time
from datetime import datetime

from outlet_detect.scheduler import run_loop


class FakePipe:
    def __init__(self):
        self.calls = 0

    def run_once(self):
        self.calls += 1
        return "normal"


def _run_with_hour(hour):
    """在指定小时下跑一轮 run_loop 后停。"""
    pipe = FakePipe()
    stop = threading.Event()

    class FakeDatetime:
        @classmethod
        def now(cls):
            return datetime(2026, 10, 30, hour, 0, 0)

    import outlet_detect.scheduler as sch
    orig = sch.datetime
    sch.datetime = FakeDatetime
    sleeps = []
    orig_sleep = sch.time.sleep
    sch.time.sleep = lambda s: (sleeps.append(s), stop.set())  # 第一次 sleep 后停
    try:
        run_loop({"cam1": pipe}, interval_sec=300, stagger_sec=0,
                 stop_flag=stop, active_hours=(10, 15))
    finally:
        sch.datetime = orig
        sch.time.sleep = orig_sleep
    return pipe.calls


def test_inside_window_runs():
    assert _run_with_hour(11) == 1


def test_outside_window_skips():
    assert _run_with_hour(8) == 0
    assert _run_with_hour(15) == 0  # 15:00 已出窗口


def test_no_window_always_runs():
    pipe = FakePipe()
    stop = threading.Event()
    import outlet_detect.scheduler as sch
    orig_sleep = sch.time.sleep
    sch.time.sleep = lambda s: stop.set()
    try:
        run_loop({"cam1": pipe}, interval_sec=300, stagger_sec=0, stop_flag=stop)
    finally:
        sch.time.sleep = orig_sleep
    assert pipe.calls == 1
```

config 测试追加：

```python
# tests/test_config.py 追加
def test_float_fields_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    cfg = CameraConfig(camera_id="cam02", outfall_point=(1, 2), reference_frame="",
                       hik_body={}, float_engine_id="float/1",
                       ignore_classes=["aeration"])
    save_camera_config(cfg)
    loaded = load_camera_config("cam02")
    assert loaded.float_engine_id == "float/1"
    assert loaded.float_confidence == 0.6
    assert loaded.ignore_classes == ["aeration"]


def test_old_config_json_loads_with_defaults(tmp_path, monkeypatch):
    """旧版 JSON（无新字段）加载不报错，取默认值。"""
    import json
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    (tmp_path / "old.json").write_text(json.dumps({
        "camera_id": "old", "outfall_point": [0, 0], "reference_frame": "",
        "hik_body": {}}), encoding="utf-8")
    loaded = load_camera_config("old")
    assert loaded.float_engine_id == "" and loaded.ignore_classes == []
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_scheduler_window.py tests/test_config.py -v`
Expected: FAIL（active_hours 参数不存在 / 新字段不存在）

- [ ] **Step 3: 实现**

config.py 修改（dataclass 头部加 import field，类内加字段）：

```python
from dataclasses import dataclass, asdict, field
```

```python
    vlm_envelope: bool = False  # True=公司网关包装格式 {code,data:{choices}}，False=标准OpenAI
    float_engine_id: str = ""       # 漂浮物引擎；空=关闭漂浮物检测
    float_confidence: float = 0.6
    ignore_classes: list = field(default_factory=list)  # 如 ["aeration"]
```

scheduler.py 整体替换为：

```python
# outlet_detect/scheduler.py
import logging
import time
from datetime import datetime

log = logging.getLogger(__name__)

def run_loop(pipelines, interval_sec=300, stagger_sec=10, stop_flag=None,
             active_hours=None):
    """每 interval_sec 对所有摄像头各跑一轮；摄像头间错峰 stagger_sec。
    active_hours=(10, 15) 时只在 [10:00, 15:00) 内执行；窗口外每 60 秒重查。"""
    while stop_flag is None or not stop_flag.is_set():
        if active_hours is not None:
            h = datetime.now().hour
            if not (active_hours[0] <= h < active_hours[1]):
                log.info("当前 %d 时在运行窗口 %s 外，休眠", h, active_hours)
                time.sleep(60)
                continue
        for cid, pipe in pipelines.items():
            if stop_flag is not None and stop_flag.is_set():
                return
            try:
                result = pipe.run_once()
                log.info("%s -> %s", cid, result)
            except Exception:
                log.exception("%s 本轮执行异常", cid)
            time.sleep(stagger_sec)
        time.sleep(interval_sec)
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_scheduler_window.py tests/test_config.py tests/test_scheduler.py -v`
Expected: 全部通过

```bash
git add outlet_detect/config.py outlet_detect/scheduler.py tests/test_config.py tests/test_scheduler_window.py
git commit -m "feat: 摄像头配置加漂浮物字段+调度器时段窗口"
```

---

### Task 5: VLM 容错加固（VLM 无返回时仍能识别排污）

**背景:** `parse_vlm_response` 当前在解析失败时静默返回 `VLMResult(is_discharge=False, ...)`——VLM 返回乱码/空内容时排污被静默否掉，pipeline 的 except 兜底分支走不到。本任务把"无效响应"统一改为抛 `VLMResponseError`，让管线 except 分支（保留规则判定的 suspect 继续推进状态机）真正生效，并用测试固化：VLM 完全挂掉时，灰色区样本照样能积累计数并报警。

**Files:**
- Modify: `outlet_detect/vlm_judge.py`（parse_vlm_response 改为抛异常 + judge 空响应防护）
- Test: `tests/test_vlm_judge.py`（追加）、`tests/test_pipeline_vlm_fallback.py`（新）

**Interfaces:**
- Produces: `VLMResponseError(RuntimeError)`；`parse_vlm_response(text) -> VLMResult`，无效输入抛 VLMResponseError；`VLMJudge.judge(...)` 在 content 为空/None 时抛 VLMResponseError
- Consumed by: 现有 pipeline.py 的 `except Exception` 分支（无需改 pipeline，行为已正确，本任务用测试固化）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_vlm_judge.py 追加
import pytest

from outlet_detect.vlm_judge import VLMResponseError, parse_vlm_response


def test_parse_garbage_raises():
    with pytest.raises(VLMResponseError):
        parse_vlm_response("我看到水面很正常，没有问题")


def test_parse_empty_raises():
    with pytest.raises(VLMResponseError):
        parse_vlm_response("")


def test_parse_broken_json_raises():
    with pytest.raises(VLMResponseError):
        parse_vlm_response('{"is_discharge": true, "confidence": "高"}')  # 非法值


def test_parse_valid_still_works():
    r = parse_vlm_response('{"is_discharge": true, "confidence": 0.9, "reason": "有黑水", "shadow_suspect": false}')
    assert r.is_discharge and r.confidence == 0.9


def test_judge_none_content_raises():
    """envelope 网关返回 content=None 时抛 VLMResponseError 而不是 TypeError。"""
    from outlet_detect.vlm_judge import VLMJudge
    j = VLMJudge("http://x", "k", "m", envelope=True)
    j._post_envelope = lambda content: None
    import numpy as np
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(VLMResponseError):
        j.judge(frame, frame, frame, {"rel_darkness": 1, "flow_energy": 1,
                                      "color_dev": 1, "score": 0.5})
```

```python
# tests/test_pipeline_vlm_fallback.py
"""VLM 故障容错：VLM 挂掉时规则判定的 suspect 结果保留，状态机照常推进报警。"""
import numpy as np

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.view_align import AlignResult
from outlet_detect.vlm_judge import VLMResponseError
from outlet_detect.water_seg import StubWaterSegmenter


class _Aligner:
    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)


class DeadVLM:
    """VLM 完全无返回。"""

    def judge(self, *a, **k):
        raise VLMResponseError("timeout")


def make_pipe(tmp_path):
    # 灰色区样本：排口区黑且逐帧波动，score 落在 gray（score_low < s < score_high）
    frames = []
    for i in range(5):
        f = np.full((400, 600, 3), 200, dtype=np.uint8)
        f[200:, 270:330, :] = (60 + i * 30) % 255
        frames.append(f)
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={}, score_low=0.2, score_high=0.9)  # 宽灰色带确保进 gray
    return Pipeline(cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
                    CNNClassifier(None), DeadVLM(),
                    Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
                    DischargeStateMachine("cam1", confirm_count=2),
                    LogAlerter(str(tmp_path / "a.log")))


def test_gray_sample_still_suspect_when_vlm_down(tmp_path):
    """VLM 挂掉时灰色样本保持 suspect（不被静默否掉）。"""
    pipe = make_pipe(tmp_path)
    assert pipe.run_once() == "suspect"


def test_alert_fires_after_confirm_when_vlm_down(tmp_path):
    """VLM 全程故障，连续 2 轮 suspect 后照常 alert。"""
    pipe = make_pipe(tmp_path)
    pipe.run_once()
    assert pipe.run_once() == "alert"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_vlm_judge.py tests/test_pipeline_vlm_fallback.py -v`
Expected: FAIL（VLMResponseError 不存在；现有 parse_vlm_response 静默返回 False 导致 fallback 测试失败）

- [ ] **Step 3: 实现 vlm_judge 改动**

vlm_judge.py 修改：

```python
class VLMResponseError(RuntimeError):
    """VLM 响应无效：超时/网关错误/空返回/JSON 乱码。
    调用方（pipeline）捕获后保留规则判定的 suspect 结果，继续推进状态机。"""


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
```

judge 方法末尾改为：

```python
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
```

pipeline.py 的 except 分支加一行注释说明容错语义（行为不变）：

```python
                except Exception as e:
                    # VLM 无效（超时/乱码/空返回）：保留规则判定的 suspect，
                    # 状态机照常推进——VLM 全挂时系统退化为 规则引擎+状态机 两级
                    log.exception("%s VLM 调用失败，按规则结果处理: %s", cid, e)
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_vlm_judge.py tests/test_pipeline_vlm_fallback.py -v`
Expected: 全部通过；再跑 `uv run pytest` 全量确认无回归

```bash
git add outlet_detect/vlm_judge.py outlet_detect/pipeline.py tests/test_vlm_judge.py tests/test_pipeline_vlm_fallback.py
git commit -m "fix: VLM无效响应抛异常代替静默否决，VLM全挂时规则+状态机照常报警"
```

---

### Task 6: Pipeline 接入漂浮物分支与上报

**Files:**
- Modify: `outlet_detect/pipeline.py`（__init__ 加 3 个可选依赖 + run_once 插漂浮物分支 + alert 上报）
- Modify: `outlet_detect/storage.py`（加 save_snapshot）
- Test: `tests/test_pipeline_float.py`

**Interfaces:**
- Consumes: `HttpFloatDetector.detect`（Task 2）、`MinioReporter.report`/`annotate_float`/`annotate_discharge`（Task 3）、`CameraConfig.float_*`（Task 4）
- Produces: `Pipeline(..., float_detector=None, float_sm=None, reporter=None)`（新参数全为可选，默认 None 时行为与现状完全一致）
- Produces: `Storage.save_snapshot(camera_id, ts, frame, tag) -> str`（写 archive_root/<cid>/<date>/<ts>_<tag>.jpg）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_pipeline_float.py
import numpy as np

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.float_detector import FloatDetection
from outlet_detect.pipeline import Pipeline
from outlet_detect.reporter import ReportEvent
from outlet_detect.rois import ROIs
from outlet_detect.rule_engine import score
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.view_align import AlignResult
from outlet_detect.water_seg import StubWaterSegmenter


class _Aligner:
    def align(self, frame):
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)


class FakeFloatDetector:
    def __init__(self, dets):
        self.dets = dets
        self.calls = 0

    @property
    def enabled(self):
        return True

    def detect(self, frame):
        self.calls += 1
        return self.dets


class FakeReporter:
    def __init__(self):
        self.events = []

    def report(self, ev):
        self.events.append(ev)
        return "http://x/out.jpg"


def make_pipe(tmp_path, dets, reporter=None):
    frames = [np.full((400, 600, 3), 120 + i, dtype=np.uint8) for i in range(5)]
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={})
    return Pipeline(
        cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
        CNNClassifier(None), None,
        Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
        DischargeStateMachine("cam1"), LogAlerter(str(tmp_path / "a.log")),
        float_detector=FakeFloatDetector(dets),
        float_sm=DischargeStateMachine("cam1_float", confirm_count=2),
        reporter=reporter)


def test_float_detect_runs_on_frame1(tmp_path):
    pipe = make_pipe(tmp_path, [])
    pipe.run_once()
    assert pipe.float_detector.calls == 1


def test_float_report_after_confirm(tmp_path):
    rep = FakeReporter()
    pipe = make_pipe(tmp_path, [FloatDetection("bottle", 0.88, (10, 20, 50, 60))],
                     reporter=rep)
    pipe.run_once()          # 第 1 次：观察
    assert rep.events == []
    pipe.run_once()          # 第 2 次：confirm_count=2 → 上报
    assert len(rep.events) == 1
    assert rep.events[0].result_a == 0.88
    assert rep.events[0].result_b == 0.0  # 非排污事件


def test_float_failure_does_not_break_discharge(tmp_path):
    class BadDetector:
        enabled = True

        def detect(self, frame):
            raise ConnectionError("8005 down")

    frames = [np.full((400, 600, 3), 120 + i, dtype=np.uint8) for i in range(5)]
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={})
    pipe = Pipeline(cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
                    CNNClassifier(None), None,
                    Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
                    DischargeStateMachine("cam1"), LogAlerter(str(tmp_path / "a.log")),
                    float_detector=BadDetector(),
                    float_sm=DischargeStateMachine("cam1_float"))
    # 漂浮物检测异常不影响排污主线返回正常状态
    assert pipe.run_once() in ("normal", "suspect", "alert")


def test_discharge_alert_reports(tmp_path):
    """排污 alert 时 reporter 收到 result_b=result_c=置信度的事件。"""
    rep = FakeReporter()
    # 排口区纯黑+剧烈波动 → score 高，confirm_count=1 立即 alert
    frames = []
    for i in range(5):
        f = np.full((400, 600, 3), 230, dtype=np.uint8)  # 亮参考区
        f[200:, 270:330, :] = (i * 40) % 255             # 排口区黑且逐帧变化
        frames.append(f)
    cfg = CameraConfig(camera_id="cam1", outfall_point=(0, 0), reference_frame="",
                       hik_body={}, score_high=0.3)     # 降低阈值确保 suspect
    pipe = Pipeline(cfg, ListFrameSampler(frames), StubWaterSegmenter(), _Aligner(),
                    CNNClassifier(None), None,
                    Storage(str(tmp_path / "t.db"), str(tmp_path / "arc")),
                    DischargeStateMachine("cam1", confirm_count=1),
                    LogAlerter(str(tmp_path / "a.log")),
                    reporter=rep)
    status = pipe.run_once()
    assert status == "alert"
    assert len(rep.events) == 1
    ev = rep.events[0]
    assert ev.result_b == ev.result_c and ev.result_b > 0
    assert ev.result_a == 0.0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_pipeline_float.py -v`
Expected: FAIL（Pipeline 无 float_detector 参数）

- [ ] **Step 3: 实现**

storage.py 追加：

```python
    def save_snapshot(self, camera_id, ts, frame, tag):
        """标注图/快照存 archive_root/<cid>/<date>/<ts>_<tag>.jpg，返回路径。"""
        import cv2
        d = self.archive_root / camera_id / ts[:10]
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{ts.replace(':', '-')}_{tag}.jpg"
        cv2.imwrite(str(p), frame)
        return str(p)
```

pipeline.py 修改：

```python
    def __init__(self, config, sampler, segmenter, aligner, cnn, vlm,
                 storage, state_machine, alerter,
                 float_detector=None, float_sm=None, reporter=None):
        self.config = config
        self.sampler = sampler
        self.segmenter = segmenter
        self.aligner = aligner
        self.cnn = cnn
        self.vlm = vlm
        self.storage = storage
        self.sm = state_machine
        self.alerter = alerter
        self.float_detector = float_detector
        self.float_sm = float_sm
        self.reporter = reporter
        self._prev_frame = None
```

run_once 里，采样成功（`frames = self.sampler.capture(...)`）之后、分割之前插入：

```python
            # 漂浮物检测（帧1，独立于排污主线；失败不影响排污）
            if self.float_detector is not None and self.float_detector.enabled:
                try:
                    dets = self.float_detector.detect(frames[0])
                except Exception as e:
                    log.warning("%s 漂浮物检测失败: %s", cid, e)
                else:
                    fev = self.float_sm.update(bool(dets), ts_epoch)
                    if fev is not None:
                        self._report_float(cid, ts, frames[0], dets)
```

alert 分支（`if ev is not None:` 内，`self.alerter.alert(...)` 之后）插入：

```python
                if self.reporter is not None:
                    try:
                        from outlet_detect.reporter import (ReportEvent,
                                                            annotate_discharge)
                        ann = annotate_discharge(frames[0], rois, s)
                        out_path = self.storage.save_snapshot(cid, ts, ann, "alert")
                        conf = vlm_result.confidence if vlm_result else s
                        self.reporter.report(ReportEvent(
                            index_code=cid, input_path=frame_paths[0],
                            output_path=out_path, result_b=conf, result_c=conf))
                    except Exception as e:
                        log.exception("%s 排污上报失败: %s", cid, e)
```

类尾追加方法：

```python
    def _report_float(self, cid, ts, frame, dets):
        from outlet_detect.reporter import ReportEvent, annotate_float
        conf = max(d.confidence for d in dets)
        log.warning("%s 漂浮物告警（%d 个，最高置信 %.2f）", cid, len(dets), conf)
        if self.reporter is None:
            return
        try:
            ann = annotate_float(frame, dets)
            out_path = self.storage.save_snapshot(cid, ts, ann, "float")
            raw_path = self.storage.save_snapshot(cid, ts, frame, "float_raw")
            self.reporter.report(ReportEvent(
                index_code=cid, input_path=raw_path, output_path=out_path,
                result_a=conf))
        except Exception as e:
            log.exception("%s 漂浮物上报失败: %s", cid, e)
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_pipeline_float.py tests/test_pipeline.py -v`
Expected: 全部通过（原 pipeline 测试不回归）

```bash
git add outlet_detect/pipeline.py outlet_detect/storage.py tests/test_pipeline_float.py
git commit -m "feat: Pipeline接入漂浮物检测分支与两类事件上报"
```

---

### Task 7: main.py 真实组装 + deploy.json

**Files:**
- Modify: `main.py`（实现 build_pipelines）
- Create: `config/deploy.example.json`（示例配置，进 git；真实 deploy.json 在 gitignore）
- Modify: `.gitignore`（加 `config/deploy.json`）
- Test: `tests/test_main_assembly.py`

**Interfaces:**
- Consumes: 全部前序任务
- Produces: `build_pipelines(deploy_path="config/deploy.json") -> dict[str, Pipeline]`
- deploy.json 键：`hik_base_url`、`app_key`、`app_secret`、`minio_endpoint`、`minio_ak`、`minio_sk`、`report_url`、`external_host`
- 摄像头配置识别：`config/cameras/*.json` 中 `hik_body.cameraIndexCode` 非空的为摄像头（跳过 vlm.json 等辅助配置）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_main_assembly.py
import json

from main import build_pipelines


def test_build_pipelines_assembles(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cameras")
    cam_dir = tmp_path / "cameras"
    cam_dir.mkdir()
    (cam_dir / "cam01.json").write_text(json.dumps({
        "camera_id": "cam01", "outfall_point": [640, 300],
        "reference_frame": "", "hik_body": {"cameraIndexCode": "abc"},
        "float_engine_id": "float/1"}), encoding="utf-8")
    (cam_dir / "vlm.json").write_text(json.dumps({
        "camera_id": "vlm", "outfall_point": [0, 0], "reference_frame": "",
        "hik_body": {}, "vlm_base_url": "http://v", "vlm_api_key": "k",
        "vlm_model": "", "vlm_envelope": True}), encoding="utf-8")
    deploy = tmp_path / "deploy.json"
    deploy.write_text(json.dumps({
        "hik_base_url": "https://hik:14443", "app_key": "K", "app_secret": "S",
        "minio_endpoint": "m:9000", "minio_ak": "a", "minio_sk": "s",
        "report_url": "http://r/api", "external_host": "ext"}), encoding="utf-8")
    pipes = build_pipelines(str(deploy))
    assert list(pipes) == ["cam01"]  # vlm.json 被跳过
    p = pipes["cam01"]
    assert p.float_detector.enabled is True
    assert p.vlm is not None and p.reporter is not None
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_main_assembly.py -v`
Expected: FAIL（build_pipelines 抛 NotImplementedError）

- [ ] **Step 3: 实现 main.py**

```python
"""在线模式入口：python main.py
回放模式入口：python main.py --replay videos/xxx.mp4
"""
import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")


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
        pipes[cid] = Pipeline(
            config=cfg,
            sampler=HikFrameSampler(capture_fn, download_fn),
            segmenter=HttpWaterSegmenter(),
            aligner=ViewAligner(cfg.reference_frame, cfg.outfall_point),
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
```

config/deploy.example.json：

```json
{
  "hik_base_url": "https://223.71.143.153:14443",
  "app_key": "20850564",
  "app_secret": "在此填入真实AppSecret",
  "minio_endpoint": "192.166.250.223:17077",
  "minio_ak": "minio",
  "minio_sk": "minio123456",
  "report_url": "http://192.166.250.224:17082/api/insert_ai_result",
  "external_host": "223.108.117.18:17077"
}
```

注意：`external_host` 必须带端口（整体替换 minio 内网 `host:port`），与 hik-sdk.py 参考链路行为一致（外网 NAT 转发 17077）。

.gitignore 追加一行：`config/deploy.json`

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest`（全量）
Expected: 全部通过

```bash
git add main.py config/deploy.example.json .gitignore tests/test_main_assembly.py
git commit -m "feat: main.py真实组装+deploy.json部署配置+10:00~15:00时段"
```

---

## 上线前人工步骤（不属于本计划）

1. 复制 `config/deploy.example.json` 为 `config/deploy.json` 并填入真实 AppSecret
2. 用 `calibrate_tool` 对每个摄像头标定排口点（生成 cameras/<cid>.json + 参照帧）
3. 漂浮物模型已部署到 8005（engine_id：`floating/20261030_145216`），把该值填入各摄像头配置的 `float_engine_id`
4. 联调一天检查业务平台收到上报、minio 图片可访问
