# 河道排污检测系统实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建基于定时截图的河道排污自动检测系统：YOLO 水面分割约束分析区域，规则引擎初筛 → CNN 复判 → VLM 终判的三级漏斗，时序状态机确认后告警。

**Architecture:** 每摄像头每 5 分钟通过海康 API 连续抓 5 帧；YOLO 分割出水面 mask，特征点配准映射排口点并重建三个 ROI；规则引擎三特征打分（仅水面像素），灰色地带走 CNN（冷启动期关闭），疑似走 VLM 终判；连续 3 次疑似触发告警事件；SQLite + 文件归档全程留痕。

**Tech Stack:** Python ≥3.9（uv 管理虚拟环境与依赖）, OpenCV (opencv-python), numpy, onnxruntime, openai SDK, requests, pytest。纯 CPU，无深度学习框架依赖（CNN/VLM 分别走 ONNX Runtime 和 HTTP API）。

**参考 spec:** `docs/superpowers/specs/2026-10-28-river-discharge-detection-design.md`

## Global Constraints

- Python ≥ 3.9，虚拟环境与依赖一律用 **uv** 管理（`uv add` / `uv run`）；禁止引入 torch/tensorflow 等重依赖
- 所有测试命令统一为 `uv run pytest ...`
- 所有图像处理用 opencv-python + numpy；模型推理只用 onnxruntime
- 时间戳一律本地时间 `datetime.now()`，存储为 ISO 格式字符串
- 摄像头配置文件：`config/cameras/<camera_id>.json`
- 截图归档目录：`archive/<camera_id>/<YYYY-MM-DD>/`
- 数据库：`data/samples.db`（SQLite）
- 判定阈值初值：`score < 0.4` 正常，`> 0.7` 疑似（spec 第 4 节）
- 状态机：连续 3 次疑似告警，告警冷却期 30 分钟（spec 第 7 节）
- 告警接口 `alert(event)` 默认仅日志实现，用户自行对接推送渠道
- 海康凭据等敏感信息只从环境变量或本地未入库的配置读取，不硬编码进仓库

---

### Task 1: 项目脚手架与配置模块

**Files:**
- Create: `pyproject.toml`（由 `uv init` + `uv add` 生成）
- Create: `outlet_detect/__init__.py`
- Create: `outlet_detect/config.py`
- Create: `config/cameras/.gitkeep`
- Test: `tests/test_config.py`

**Interfaces:**
- Produces: `CameraConfig` dataclass，字段：`camera_id: str`、`outfall_point: tuple[int, int]`、`reference_frame: str`、`hik_body: dict`、`score_low: float = 0.4`、`score_high: float = 0.7`、`vlm_base_url: str = ""`、`vlm_api_key: str = ""`、`vlm_model: str = ""`
- Produces: `load_camera_config(camera_id: str) -> CameraConfig`、`save_camera_config(cfg: CameraConfig) -> None`、`list_camera_ids() -> list[str]`

- [ ] **Step 1: 初始化 git 仓库与 uv 虚拟环境**

```bash
cd f:/Vaults/coding/outlet-detect && git init
uv init --bare --python 3.12
uv add opencv-python numpy onnxruntime openai requests
uv add --dev pytest
```

（`uv init --bare` 只生成 `pyproject.toml`，不覆盖现有文件；依赖写进 pyproject 并自动创建 `.venv`）

`.gitignore` 内容：

```
__pycache__/
*.pyc
.venv/
data/
archive/
config/cameras/*.json
!config/cameras/.gitkeep
frames_preview/
videos/
```

- [ ] **Step 2: 写失败测试**

```python
# tests/test_config.py
import json
from pathlib import Path
from outlet_detect.config import CameraConfig, load_camera_config, save_camera_config

def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    cfg = CameraConfig(
        camera_id="cam01",
        outfall_point=(640, 300),
        reference_frame="archive/cam01/ref.jpg",
        hik_body={"cameraIndexCode": "abc"},
    )
    save_camera_config(cfg)
    loaded = load_camera_config("cam01")
    assert loaded.camera_id == "cam01"
    assert loaded.outfall_point == (640, 300)
    assert loaded.score_low == 0.4
    assert loaded.score_high == 0.7

def test_list_camera_ids(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path)
    save_camera_config(CameraConfig(camera_id="a", outfall_point=(0, 0),
                                    reference_frame="", hik_body={}))
    save_camera_config(CameraConfig(camera_id="b", outfall_point=(0, 0),
                                    reference_frame="", hik_body={}))
    assert sorted(list_camera_ids()) == ["a", "b"]
```

- [ ] **Step 3: 运行测试确认失败**

Run: `uv run pytest tests/test_config.py -v`
Expected: FAIL，`ModuleNotFoundError: No module named 'outlet_detect.config'`

- [ ] **Step 4: 实现配置模块**

```python
# outlet_detect/config.py
import json
from dataclasses import dataclass, asdict, field
from pathlib import Path

CONFIG_DIR = Path("config/cameras")

@dataclass
class CameraConfig:
    camera_id: str
    outfall_point: tuple  # (x, y) 排口在标定参照帧中的像素坐标
    reference_frame: str  # 标定参照帧图片路径
    hik_body: dict        # 海康 API 请求体（含 cameraIndexCode）
    score_low: float = 0.4
    score_high: float = 0.7
    vlm_base_url: str = ""
    vlm_api_key: str = ""
    vlm_model: str = ""

def _path(camera_id: str) -> Path:
    return CONFIG_DIR / f"{camera_id}.json"

def save_camera_config(cfg: CameraConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    data = asdict(cfg)
    data["outfall_point"] = list(cfg.outfall_point)
    _path(cfg.camera_id).write_text(json.dumps(data, ensure_ascii=False, indent=2))

def load_camera_config(camera_id: str) -> CameraConfig:
    data = json.loads(_path(camera_id).read_text())
    data["outfall_point"] = tuple(data["outfall_point"])
    return CameraConfig(**data)

def list_camera_ids() -> list:
    if not CONFIG_DIR.exists():
        return []
    return [p.stem for p in CONFIG_DIR.glob("*.json")]
```

```python
# outlet_detect/__init__.py
```

（空文件）

- [ ] **Step 5: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_config.py -v`
Expected: 2 passed

```bash
git add -A && git commit -m "chore: 项目脚手架与摄像头配置模块"
```

---

### Task 2: 存储模块（SQLite + 截图归档）

**Files:**
- Create: `outlet_detect/storage.py`
- Test: `tests/test_storage.py`

**Interfaces:**
- Produces: `Storage(db_path: str = "data/samples.db", archive_root: str = "archive")`
  - `record_sample(camera_id: str, ts: str, features: dict, rule_verdict: str, cnn_prob: float | None, vlm_json: str | None, status: str, frame_paths: list, duration_ms: int) -> None`
  - `archive_frames(camera_id: str, ts: str, frames: list, keep: bool) -> list`（返回保存路径列表；keep=False 时存临时区供滚动删除）
  - `get_baseline_darkness(camera_id: str, hour: int, days: int = 7) -> float | None`（近 N 天同时刻黑度中位数）
  - `save_labeled_sample(camera_id: str, ts: str, frame_path: str, label: str, source: str, reason: str) -> None`
  - `purge_old_frames(days: int = 7) -> int`（删除正常帧，返回删除数）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_storage.py
import numpy as np
from outlet_detect.storage import Storage

def make_frame():
    return np.full((100, 100, 3), 128, dtype=np.uint8)

def test_record_and_baseline(tmp_path):
    s = Storage(db_path=str(tmp_path / "t.db"), archive_root=str(tmp_path / "arc"))
    feats = {"rel_darkness": 0.9, "flow_energy": 1.2, "color_dev": 0.1, "baseline_dev": 0.0}
    s.record_sample("cam01", "2026-10-28T10:00:00", feats, "normal", None, None,
                    "normal", ["a.jpg"], 120)
    s.record_sample("cam01", "2026-10-27T10:05:00", {**feats, "rel_darkness": 0.8},
                    "normal", None, None, "normal", ["b.jpg"], 100)
    baseline = s.get_baseline_darkness("cam01", hour=10, days=7)
    assert baseline == 0.8  # 排除了今天 10:00 之外昨天同时刻的值

def test_archive_and_labeled(tmp_path):
    s = Storage(db_path=str(tmp_path / "t.db"), archive_root=str(tmp_path / "arc"))
    paths = s.archive_frames("cam01", "2026-10-28T10-00-00", [make_frame()], keep=True)
    assert len(paths) == 1 and (tmp_path / "arc").exists()
    s.save_labeled_sample("cam01", "2026-10-28T10:00:00", paths[0],
                          "discharge", "vlm", "排口发黑")
    # 再存一条正常帧（keep=False），purge 应只删正常帧
    tmp_paths = s.archive_frames("cam01", "2026-10-28T11-00-00", [make_frame()], keep=False)
    deleted = s.purge_old_frames(days=0)
    assert deleted >= 1
    import os
    assert os.path.exists(paths[0])  # 保留帧不删
```

注意：baseline 查询需排除"当前这条记录"——约定 `get_baseline_darkness` 只查 `ts < 当前调用当天 0 点`之前的记录不够通用，实现上改为：传入 `before_ts` 参数，查 `ts < before_ts`。相应签名修正为 `get_baseline_darkness(camera_id, hour, before_ts, days=7)`，测试改为传 `"2026-10-28T10:00:00"`。

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_storage.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现存储模块**

```python
# outlet_detect/storage.py
import json
import os
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path

import cv2

SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, camera_id TEXT,
    rel_darkness REAL, flow_energy REAL, color_dev REAL, baseline_dev REAL,
    score REAL, rule_verdict TEXT, cnn_prob REAL, vlm_json TEXT,
    status TEXT, frame_paths TEXT, duration_ms INTEGER
);
CREATE TABLE IF NOT EXISTS labeled_samples (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT, camera_id TEXT, frame_path TEXT,
    label TEXT, source TEXT, reason TEXT
);
"""

class Storage:
    def __init__(self, db_path: str = "data/samples.db", archive_root: str = "archive"):
        self.db_path = db_path
        self.archive_root = Path(archive_root)
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.executescript(SCHEMA)

    def record_sample(self, camera_id, ts, features, rule_verdict, cnn_prob,
                      vlm_json, status, frame_paths, duration_ms):
        self.conn.execute(
            "INSERT INTO samples (ts,camera_id,rel_darkness,flow_energy,color_dev,"
            "baseline_dev,score,rule_verdict,cnn_prob,vlm_json,status,frame_paths,"
            "duration_ms) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (ts, camera_id, features.get("rel_darkness"), features.get("flow_energy"),
             features.get("color_dev"), features.get("baseline_dev"),
             features.get("score"), rule_verdict, cnn_prob, vlm_json, status,
             json.dumps(frame_paths), duration_ms))
        self.conn.commit()

    def archive_frames(self, camera_id, ts, frames, keep):
        day = ts[:10]
        tier = "keep" if keep else "tmp"
        d = self.archive_root / camera_id / day / tier
        d.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, f in enumerate(frames):
            p = d / f"{ts[11:].replace(':', '-')}_{i}.jpg"
            cv2.imwrite(str(p), f)
            paths.append(str(p))
        return paths

    def get_baseline_darkness(self, camera_id, hour, before_ts, days=7):
        start = (datetime.fromisoformat(before_ts) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT rel_darkness FROM samples WHERE camera_id=? AND ts<? AND ts>? "
            "AND CAST(substr(ts,12,2) AS INTEGER)=? AND rel_darkness IS NOT NULL",
            (camera_id, before_ts, start, hour)).fetchall()
        if not rows:
            return None
        vals = sorted(r[0] for r in rows)
        return vals[len(vals) // 2]  # 中位数

    def save_labeled_sample(self, camera_id, ts, frame_path, label, source, reason):
        self.conn.execute(
            "INSERT INTO labeled_samples (ts,camera_id,frame_path,label,source,reason)"
            " VALUES (?,?,?,?,?,?)",
            (camera_id, ts, frame_path, label, source, reason))
        self.conn.commit()

    def purge_old_frames(self, days=7):
        cutoff = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
        deleted = 0
        for cam_dir in self.archive_root.glob("*/*/tmp"):
            day = cam_dir.parent.name
            if day < cutoff:
                for f in cam_dir.glob("*.jpg"):
                    os.remove(f)
                    deleted += 1
        return deleted
```

- [ ] **Step 4: 修正测试签名并运行通过**

把 Step 1 测试中 `get_baseline_darkness("cam01", hour=10, days=7)` 改为
`get_baseline_darkness("cam01", hour=10, before_ts="2026-10-28T10:00:00", days=7)`。

Run: `uv run pytest tests/test_storage.py -v`
Expected: 2 passed

- [ ] **Step 5: 提交**

```bash
git add -A && git commit -m "feat: SQLite记录与截图归档存储模块"
```

---

### Task 3: 采样器（海康抓帧 + 可注入接口）

**Files:**
- Create: `outlet_detect/sampler.py`
- Test: `tests/test_sampler.py`

**Interfaces:**
- Consumes: `CameraConfig.hik_body`（Task 1）
- Produces: `FrameSampler` 协议：`capture(camera_id: str, n: int = 5, interval_sec: float = 2.0) -> list[np.ndarray]`
- Produces: `HikFrameSampler(capture_fn, download_fn)`——两个可注入 callable，便于测试与用户接入现有海康代码；`capture_fn(body: dict) -> str` 返回图片 URL，`download_fn(url: str) -> np.ndarray`
- Produces: `ListFrameSampler(frames: list)`——测试/回放用，每次调用从帧列表顺序取 n 帧（间隔不睡眠）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_sampler.py
import numpy as np
from outlet_detect.sampler import HikFrameSampler, ListFrameSampler

def test_hik_sampler_uses_injected_fns():
    calls = []
    img = np.full((10, 10, 3), 200, dtype=np.uint8)
    def fake_capture(body):
        calls.append(body)
        return "http://x/pic.jpg"
    def fake_download(url):
        return img
    s = HikFrameSampler(capture_fn=fake_capture, download_fn=fake_download,
                        interval_sec=0)
    frames = s.capture("cam01", n=5, body={"cameraIndexCode": "abc"})
    assert len(frames) == 5
    assert len(calls) == 5
    assert frames[0].shape == (10, 10, 3)

def test_list_sampler_cycles_frames():
    frames_src = [np.full((4, 4, 3), i, dtype=np.uint8) for i in range(10)]
    s = ListFrameSampler(frames_src)
    batch = s.capture("cam01", n=5)
    assert len(batch) == 5
    assert batch[0][0, 0, 0] == 0 and batch[4][0, 0, 0] == 4
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_sampler.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现采样器**

```python
# outlet_detect/sampler.py
import time

class HikFrameSampler:
    """通过海康 manualCapture 连续抓 n 帧。capture_fn/download_fn 由用户现有海康代码提供。"""
    def __init__(self, capture_fn, download_fn, interval_sec=2.0):
        self.capture_fn = capture_fn
        self.download_fn = download_fn
        self.interval_sec = interval_sec

    def capture(self, camera_id, n=5, body=None):
        frames = []
        for i in range(n):
            if i > 0:
                time.sleep(self.interval_sec)
            url = self.capture_fn(body)
            if not url:
                raise RuntimeError(f"{camera_id} 第{i+1}帧抓图失败")
            frames.append(self.download_fn(url))
        return frames

class ListFrameSampler:
    """测试与离线回放用：从给定帧序列循环取帧，不睡眠。"""
    def __init__(self, frames):
        self.frames = frames
        self.pos = 0

    def capture(self, camera_id, n=5, body=None):
        batch = []
        for _ in range(n):
            batch.append(self.frames[self.pos % len(self.frames)])
            self.pos += 1
        return batch
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_sampler.py -v`
Expected: 2 passed

```bash
git add -A && git commit -m "feat: 采样器（海康抓帧+列表回放）"
```

---

### Task 4: 水面分割模块

**Files:**
- Create: `outlet_detect/water_seg.py`
- Test: `tests/test_water_seg.py`

**Interfaces:**
- Produces: `WaterSegmenter` 协议：`segment(frame: np.ndarray) -> np.ndarray`（返回与输入同尺寸的 uint8 mask，水面=255，非水面=0）
- Produces: `OnnxWaterSegmenter(model_path: str, input_size: int = 640, conf_threshold: float = 0.5)`——封装用户已有 YOLO 分割模型的 ONNX 推理（letterbox 预处理 + mask 后处理，针对单类别"水面"）
- Produces: `StubWaterSegmenter()`——测试用，返回画面中央椭圆水面 mask

- [ ] **Step 1: 写失败测试**

```python
# tests/test_water_seg.py
import numpy as np
from outlet_detect.water_seg import StubWaterSegmenter, mask_coverage

def test_stub_mask_shape_and_values():
    frame = np.zeros((100, 200, 3), dtype=np.uint8)
    mask = StubWaterSegmenter().segment(frame)
    assert mask.shape == (100, 200)
    assert set(np.unique(mask)) <= {0, 255}
    assert mask[50, 100] == 255   # 中心是水面
    assert mask[0, 0] == 0        # 角落不是

def test_mask_coverage():
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[5:, :] = 255
    roi = np.ones((10, 10), dtype=np.uint8) * 255
    assert mask_coverage(mask, roi) == 0.5
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_water_seg.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现分割模块**

```python
# outlet_detect/water_seg.py
import cv2
import numpy as np

def mask_coverage(water_mask, roi_mask):
    """ROI 中被判定为水面的像素占比，用于检测漂浮物遮挡等异常。"""
    roi_pixels = int((roi_mask > 0).sum())
    if roi_pixels == 0:
        return 0.0
    both = int(((water_mask > 0) & (roi_mask > 0)).sum())
    return both / roi_pixels

class StubWaterSegmenter:
    """测试用：画面下半部分视为水面。"""
    def segment(self, frame):
        h, w = frame.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[h // 2:, :] = 255
        return mask

class OnnxWaterSegmenter:
    """用户已有 YOLO 水面分割模型的 ONNX 封装（单类别：水面）。"""
    def __init__(self, model_path, input_size=640, conf_threshold=0.5):
        import onnxruntime as ort
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.input_size = input_size
        self.conf_threshold = conf_threshold

    def _letterbox(self, img):
        h, w = img.shape[:2]
        scale = self.input_size / max(h, w)
        nh, nw = int(h * scale), int(w * scale)
        resized = cv2.resize(img, (nw, nh))
        canvas = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = resized
        return canvas, scale

    def segment(self, frame):
        # 注意：此处后处理按 YOLOv8-seg 输出结构实现（检测框 + mask 原型系数）。
        # 接入时须用真实模型跑一次，按实际输出张量维度调整本方法的解析逻辑。
        canvas, scale = self._letterbox(frame)
        blob = canvas.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        outs = self.session.run(None, {self.input_name: blob})
        mask_small = self._decode(outs)  # (input_size, input_size) 0/255
        h, w = frame.shape[:2]
        nh, nw = int(h * scale), int(w * scale)
        mask = cv2.resize(mask_small, (nw, nh), interpolation=cv2.INTER_NEAREST)
        full = np.zeros((h, w), dtype=np.uint8)
        full[:nh, :nw] = mask
        return full

    def _decode(self, outs):
        """解析 YOLOv8-seg 输出并合成单类别 mask。接入真实模型时按输出结构调整。"""
        det, proto = outs[0][0], outs[1][0]  # det: (4+1+32, N), proto: (32, mh, mw)
        boxes, confs, coeffs = det[:4], det[4], det[5:]
        keep = confs > self.conf_threshold
        if not keep.any():
            mh, mw = proto.shape[1:]
            return np.zeros((mh, mw), dtype=np.uint8)
        coeffs = coeffs[:, keep]
        mask = np.einsum("kn,kmn->mn", coeffs.T, proto)
        mask = (1 / (1 + np.exp(-mask)) > 0.5).astype(np.uint8) * 255
        return mask
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_water_seg.py -v`
Expected: 2 passed（OnnxWaterSegmenter 无模型文件不测试，接入时手动验证）

```bash
git add -A && git commit -m "feat: 水面分割模块（Stub+ONNX封装）"
```

---

### Task 5: 视角配准模块

**Files:**
- Create: `outlet_detect/view_align.py`
- Test: `tests/test_view_align.py`

**Interfaces:**
- Consumes: `CameraConfig.outfall_point`、`reference_frame`（Task 1）
- Produces: `AlignResult` dataclass：`ok: bool`、`mapped_point: tuple[int, int] | None`、`confidence: float`、`in_frame: bool`
- Produces: `ViewAligner(reference_frame_path: str, outfall_point: tuple, min_matches: int = 12)`，方法 `align(frame: np.ndarray) -> AlignResult`（ORB 特征匹配 + RANSAC 单应矩阵，把标定排口点映射到当前帧；`in_frame` 表示映射点是否仍在画面内）

- [ ] **Step 1: 写失败测试**

构造合成场景：带随机特征点的图片，用已知单应矩阵 warp，验证映射点误差。

```python
# tests/test_view_align.py
import cv2
import numpy as np
from outlet_detect.view_align import ViewAligner

def make_scene(seed=0, size=(400, 600)):
    rng = np.random.RandomState(seed)
    img = np.full((*size, 3), 200, dtype=np.uint8)
    for _ in range(300):  # 随机方块作为特征点（模拟岸边参照物）
        x, y = rng.randint(20, size[1] - 40), rng.randint(20, size[0] - 40)
        cv2.rectangle(img, (x, y), (x + 15, y + 15),
                      int(rng.randint(0, 200)), -1)
    return img

def test_align_recovers_known_transform(tmp_path):
    ref = make_scene()
    ref_path = str(tmp_path / "ref.jpg")
    cv2.imwrite(ref_path, ref)
    point = (300, 200)
    # 模拟球机小幅转动：平移+缩放
    H_true = np.array([[1.05, 0.01, -30], [-0.01, 1.05, -20], [0, 0, 1]], dtype=np.float64)
    moved = cv2.warpPerspective(ref, np.linalg.inv(H_true), (600, 400))
    aligner = ViewAligner(ref_path, point)
    result = aligner.align(moved)
    assert result.ok and result.in_frame
    mx, my = result.mapped_point
    # H_true 作用后的真实位置
    p = H_true @ np.array([300, 200, 1.0])
    ex, ey = p[0] / p[2], p[1] / p[2]
    assert abs(mx - ex) < 8 and abs(my - ey) < 8

def test_align_fails_on_unrelated_frame(tmp_path):
    ref = make_scene(seed=0)
    ref_path = str(tmp_path / "ref.jpg")
    cv2.imwrite(ref_path, ref)
    other = make_scene(seed=99)  # 完全不同的场景
    aligner = ViewAligner(ref_path, (300, 200))
    result = aligner.align(other)
    assert not result.ok
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_view_align.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现配准模块**

```python
# outlet_detect/view_align.py
from dataclasses import dataclass

import cv2
import numpy as np

@dataclass
class AlignResult:
    ok: bool
    mapped_point: tuple | None
    confidence: float
    in_frame: bool

class ViewAligner:
    """ORB 特征匹配 + RANSAC 单应矩阵，把标定排口点映射到当前帧。"""
    def __init__(self, reference_frame_path, outfall_point, min_matches=12):
        self.ref = cv2.imread(reference_frame_path)
        if self.ref is None:
            raise FileNotFoundError(reference_frame_path)
        self.outfall_point = outfall_point
        self.min_matches = min_matches
        self.orb = cv2.ORB_create(nfeatures=2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        self.kp_ref, self.des_ref = self.orb.detectAndCompute(self.ref, None)

    def align(self, frame):
        h, w = frame.shape[:2]
        kp, des = self.orb.detectAndCompute(frame, None)
        if des is None or self.des_ref is None:
            return AlignResult(False, None, 0.0, False)
        matches = self.matcher.knnMatch(self.des_ref, des, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < self.min_matches:
            return AlignResult(False, None, len(good) / self.min_matches, False)
        src = np.float32([self.kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        if H is None:
            return AlignResult(False, None, 0.0, False)
        ratio = float(inliers.sum()) / len(good)
        p = H @ np.array([self.outfall_point[0], self.outfall_point[1], 1.0])
        mx, my = p[0] / p[2], p[1] / p[2]
        in_frame = 0 <= mx < w and 0 <= my < h
        return AlignResult(True, (int(mx), int(my)), ratio, in_frame)
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_view_align.py -v`
Expected: 2 passed（若合成图特征不足导致测试 1 失败，把方块数量从 300 调到 500 或降低 `min_matches`）

```bash
git add -A && git commit -m "feat: ORB特征配准与排口点映射"
```

---

### Task 6: ROI 重建模块

**Files:**
- Create: `outlet_detect/rois.py`
- Test: `tests/test_rois.py`

**Interfaces:**
- Consumes: 水面 mask（Task 4）、映射后排口点（Task 5）
- Produces: `ROIs` dataclass：`outfall: np.ndarray`、`plume: np.ndarray`、`reference: np.ndarray`（均为 uint8 mask，已与水面的交集）
- Produces: `build_rois(water_mask: np.ndarray, outfall_point: tuple, r_outfall: int = 60, r_plume: int = 180, r_ref: int = 250) -> ROIs`

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rois.py
import numpy as np
from outlet_detect.rois import build_rois

def test_build_rois_on_half_water():
    mask = np.zeros((400, 600), dtype=np.uint8)
    mask[200:, :] = 255  # 下半幅是水面
    rois = build_rois(mask, (300, 300), r_outfall=40, r_plume=120, r_ref=200)
    # 排口区非空且全是水面
    assert rois.outfall.sum() > 0
    assert (rois.outfall[mask == 0] == 0).all()
    # 三区互不重叠
    assert ((rois.outfall > 0) & (rois.plume > 0)).sum() == 0
    assert ((rois.plume > 0) & (rois.reference > 0)).sum() == 0
    # 参考区应落在远离排口的水面
    ys, xs = np.nonzero(rois.reference)
    assert len(xs) > 0
    dists = np.hypot(xs - 300, ys - 300)
    assert dists.min() >= 200

def test_build_rois_no_water():
    mask = np.zeros((100, 100), dtype=np.uint8)
    rois = build_rois(mask, (50, 50))
    assert rois.outfall.sum() == 0 and rois.reference.sum() == 0
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_rois.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 ROI 模块**

```python
# outlet_detect/rois.py
from dataclasses import dataclass

import numpy as np

@dataclass
class ROIs:
    outfall: np.ndarray
    plume: np.ndarray
    reference: np.ndarray

def build_rois(water_mask, outfall_point, r_outfall=60, r_plume=180, r_ref=250):
    """基于水面 mask 与排口点生成三个互斥 ROI（均为水面像素的子集）。
    排口区：半径 r_outfall 内水面；扩散区：r_outfall~r_plume 环带水面；
    参考区：距离 > r_ref 的水面。"""
    h, w = water_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    dist = np.hypot(xx - outfall_point[0], yy - outfall_point[1])
    water = water_mask > 0
    return ROIs(
        outfall=((dist <= r_outfall) & water).astype(np.uint8) * 255,
        plume=(((dist > r_outfall) & (dist <= r_plume)) & water).astype(np.uint8) * 255,
        reference=((dist > r_ref) & water).astype(np.uint8) * 255,
    )
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_rois.py -v`
Expected: 2 passed

```bash
git add -A && git commit -m "feat: 基于水面mask的三ROI重建"
```

---

### Task 7: 规则引擎（三特征 + 打分）

**Files:**
- Create: `outlet_detect/rule_engine.py`
- Test: `tests/test_rule_engine.py`

**Interfaces:**
- Consumes: `ROIs`（Task 6）、5 帧图像列表、baseline 黑度（Task 2 的 `get_baseline_darkness`，可为 None）
- Produces: `compute_features(frames: list, rois: ROIs, baseline_darkness: float | None) -> dict`，返回 `{"rel_darkness", "flow_energy", "color_dev", "baseline_dev", "dark_sus", "flow_sus", "color_sus", "base_sus"}`
- Produces: `score(features: dict, weights: dict | None) -> float`，默认权重 `{"dark": 0.35, "flow": 0.35, "color": 0.2, "base": 0.1}`
- Produces: `verdict(score: float, low: float = 0.4, high: float = 0.7) -> str`，返回 `"normal" | "gray" | "suspect"`

特征定义（spec 第 4 节）：
- `rel_darkness` = 排口区 Lab-L 均值 / 参考区 Lab-L 均值；`dark_sus = clamp((1.0 - rel_darkness) / 0.4, 0, 1)`
- `flow_energy` = 排口区帧差能量均值 / 参考区帧差能量均值；`flow_sus = clamp((flow_energy - 1.2) / 1.5, 0, 1)`
- `color_dev` = 排口区与参考区 HSV 的 H、S 均值归一化欧氏距离；`color_sus = clamp(color_dev / 0.3, 0, 1)`
- `baseline_dev` = (baseline − rel_darkness)；`base_sus = clamp(baseline_dev / 0.15, 0, 1)`（baseline 为 None 时 base_sus=0）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_rule_engine.py
import numpy as np
from outlet_detect.rois import ROIs
from outlet_detect.rule_engine import compute_features, score, verdict

def make_rois():
    out = np.zeros((100, 100), dtype=np.uint8); out[40:60, 40:60] = 255
    ref = np.zeros((100, 100), dtype=np.uint8); ref[0:20, 0:20] = 255
    plume = np.zeros((100, 100), dtype=np.uint8); plume[60:80, 40:60] = 255
    return ROIs(outfall=out, plume=plume, reference=ref)

def dark_frames(n=5, outfall_l=40, ref_l=160, flicker=False):
    frames = []
    for i in range(n):
        f = np.full((100, 100, 3), 0, dtype=np.uint8)
        f[0:20, 0:20] = ref_l          # 参考区亮
        f[40:60, 40:60] = outfall_l    # 排口区黑
        if flicker:                    # 排口区帧间闪烁 = 波动
            f[40:60, 40:60] = outfall_l if i % 2 == 0 else outfall_l + 100
        frames.append(f)
    return frames

def test_dark_and_turbulent_scores_high():
    feats = compute_features(dark_frames(flicker=True), make_rois(), None)
    assert feats["rel_darkness"] < 0.5
    assert feats["flow_energy"] > 2.0
    assert score(feats) > 0.7

def test_uniform_scene_scores_low():
    frames = [np.full((100, 100, 3), 120, dtype=np.uint8) for _ in range(5)]
    feats = compute_features(frames, make_rois(), None)
    assert score(feats) < 0.4

def test_verdict_thresholds():
    assert verdict(0.3) == "normal"
    assert verdict(0.5) == "gray"
    assert verdict(0.8) == "suspect"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_rule_engine.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现规则引擎**

```python
# outlet_detect/rule_engine.py
import cv2
import numpy as np

DEFAULT_WEIGHTS = {"dark": 0.35, "flow": 0.35, "color": 0.2, "base": 0.1}

def _clamp(v, lo=0.0, hi=1.0):
    return max(lo, min(hi, v))

def _mean_in_roi(img, roi_mask):
    sel = img[roi_mask > 0]
    if sel.size == 0:
        return None
    return sel.reshape(-1, sel.shape[-1] if sel.ndim > 1 else 1).mean(axis=0)

def compute_features(frames, rois, baseline_darkness):
    lab = cv2.cvtColor(frames[0], cv2.COLOR_BGR2Lab)
    l_out = _mean_in_roi(lab[:, :, 0], rois.outfall)
    l_ref = _mean_in_roi(lab[:, :, 0], rois.reference)
    if l_out is None or l_ref is None or l_ref == 0:
        rel_darkness = 1.0  # ROI 失效时按正常处理，避免误报
    else:
        rel_darkness = float(l_out / l_ref)

    # 波动能量：相邻帧灰度差分的均值
    def flow_in(roi):
        energies = []
        for a, b in zip(frames[:-1], frames[1:]):
            ga = cv2.GaussianBlur(cv2.cvtColor(a, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            gb = cv2.GaussianBlur(cv2.cvtColor(b, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            diff = cv2.absdiff(ga, gb)
            v = _mean_in_roi(diff, roi)
            if v is not None:
                energies.append(float(v))
        return float(np.mean(energies)) if energies else 0.0
    f_out, f_ref = flow_in(rois.outfall), flow_in(rois.reference)
    flow_energy = f_out / f_ref if f_ref > 0.5 else (1.0 if f_out < 1.0 else 3.0)

    hsv = cv2.cvtColor(frames[0], cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] /= 180.0; hsv[:, :, 1] /= 255.0  # H、S 归一化
    c_out = _mean_in_roi(hsv[:, :, :2], rois.outfall)
    c_ref = _mean_in_roi(hsv[:, :, :2], rois.reference)
    color_dev = float(np.linalg.norm(c_out - c_ref)) if c_out is not None and c_ref is not None else 0.0

    baseline_dev = (baseline_darkness - rel_darkness) if baseline_darkness is not None else 0.0

    return {
        "rel_darkness": rel_darkness,
        "flow_energy": flow_energy,
        "color_dev": color_dev,
        "baseline_dev": baseline_dev,
        "dark_sus": _clamp((1.0 - rel_darkness) / 0.4),
        "flow_sus": _clamp((flow_energy - 1.2) / 1.5),
        "color_sus": _clamp(color_dev / 0.3),
        "base_sus": _clamp(baseline_dev / 0.15),
    }

def score(features, weights=None):
    w = weights or DEFAULT_WEIGHTS
    s = (w["dark"] * features["dark_sus"] + w["flow"] * features["flow_sus"]
         + w["color"] * features["color_sus"] + w["base"] * features["base_sus"])
    features["score"] = s
    return s

def verdict(s, low=0.4, high=0.7):
    if s < low:
        return "normal"
    if s > high:
        return "suspect"
    return "gray"
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_rule_engine.py -v`
Expected: 3 passed

```bash
git add -A && git commit -m "feat: 规则引擎三特征打分"
```

---

### Task 8: CNN 分类器

**Files:**
- Create: `outlet_detect/cnn_classifier.py`
- Test: `tests/test_cnn_classifier.py`

**Interfaces:**
- Produces: `CNNClassifier(model_path: str | None)`；`enabled: bool` 属性（model_path 为 None 或文件不存在时 False，即冷启动关闭）；`classify(crop: np.ndarray) -> float`（返回排污概率 0~1；未启用时抛 `RuntimeError`）
- Produces: `crop_for_cnn(frame: np.ndarray, rois: ROIs, size: int = 224) -> np.ndarray`（排口区+扩散区外接矩形裁剪并 resize）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_cnn_classifier.py
import numpy as np
import pytest
from outlet_detect.cnn_classifier import CNNClassifier, crop_for_cnn
from outlet_detect.rois import ROIs

def test_disabled_when_no_model():
    c = CNNClassifier(None)
    assert not c.enabled
    with pytest.raises(RuntimeError):
        c.classify(np.zeros((224, 224, 3), dtype=np.uint8))

def test_crop_for_cnn():
    frame = np.full((300, 400, 3), 100, dtype=np.uint8)
    out = np.zeros((300, 400), dtype=np.uint8); out[100:150, 150:200] = 255
    plume = np.zeros((300, 400), dtype=np.uint8); plume[150:200, 150:250] = 255
    crop = crop_for_cnn(frame, ROIs(outfall=out, plume=plume,
                                    reference=np.zeros((300, 400), dtype=np.uint8)))
    assert crop.shape == (224, 224, 3)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_cnn_classifier.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 CNN 模块**

```python
# outlet_detect/cnn_classifier.py
import os

import cv2
import numpy as np

def crop_for_cnn(frame, rois, size=224):
    ys, xs = np.nonzero((rois.outfall > 0) | (rois.plume > 0))
    if len(xs) == 0:
        return cv2.resize(frame, (size, size))
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    return cv2.resize(frame[y0:y1, x0:x1], (size, size))

class CNNClassifier:
    """MobileNetV3-Small 二分类 ONNX 推理；无模型文件时关闭（冷启动）。"""
    def __init__(self, model_path):
        self.enabled = bool(model_path) and os.path.exists(model_path)
        if self.enabled:
            import onnxruntime as ort
            self.session = ort.InferenceSession(model_path)
            self.input_name = self.session.get_inputs()[0].name

    def classify(self, crop):
        if not self.enabled:
            raise RuntimeError("CNN 未启用（无模型文件）")
        blob = crop.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        logits = self.session.run(None, {self.input_name: blob})[0][0]
        exp = np.exp(logits - logits.max())
        return float(exp[1] / exp.sum())  # 类别1 = 排污
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_cnn_classifier.py -v`
Expected: 2 passed

```bash
git add -A && git commit -m "feat: CNN二分类器（冷启动可关闭）"
```

---

### Task 9: VLM 终判模块

**Files:**
- Create: `outlet_detect/vlm_judge.py`
- Test: `tests/test_vlm_judge.py`

**Interfaces:**
- Produces: `VLMResult` dataclass：`is_discharge: bool`、`confidence: float`、`reason: str`、`shadow_suspect: bool`
- Produces: `VLMJudge(base_url: str, api_key: str, model: str, client=None)`（client 可注入，测试用 mock）；`judge(current_frame, ref_frame, prev_frame, features: dict) -> VLMResult`
- Produces: `build_prompt(features: dict) -> str`（单独成函数便于调优与测试）
- Produces: `parse_vlm_response(text: str) -> VLMResult`（容错解析：JSON 提取失败时返回保守结果 `is_discharge=False, confidence=0`）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_vlm_judge.py
import json
import numpy as np
from outlet_detect.vlm_judge import VLMJudge, build_prompt, parse_vlm_response

def test_parse_valid_json():
    r = parse_vlm_response('{"is_discharge": true, "confidence": 0.9,'
                           ' "reason": "排口发黑", "shadow_suspect": false}')
    assert r.is_discharge and r.confidence == 0.9 and not r.shadow_suspect

def test_parse_garbage_returns_conservative():
    r = parse_vlm_response("抱歉我无法分析这张图片")
    assert not r.is_discharge and r.confidence == 0.0

def test_parse_json_embedded_in_text():
    r = parse_vlm_response('分析结果：{"is_discharge": true, "confidence": 0.8,'
                           ' "reason": "x", "shadow_suspect": true} 以上。')
    assert r.is_discharge and r.shadow_suspect

def test_judge_calls_client():
    img = np.zeros((64, 64, 3), dtype=np.uint8)
    class FakeMsg:
        content = '{"is_discharge": false, "confidence": 0.9, "reason": "正常", "shadow_suspect": false}'
    class FakeChoice:
        message = FakeMsg()
    class FakeResp:
        choices = [FakeChoice()]
    class FakeChat:
        def create(self, **kw):
            self.kw = kw
            return FakeResp()
    class FakeClient:
        def __init__(self):
            self.chat = type("C", (), {"completions": FakeChat()})()
    judge = VLMJudge("http://x", "k", "model-x", client=FakeClient())
    r = judge.judge(img, img, img, {"rel_darkness": 0.8, "flow_energy": 1.0,
                                    "color_dev": 0.05, "score": 0.5})
    assert not r.is_discharge

def test_build_prompt_contains_features():
    p = build_prompt({"rel_darkness": 0.8, "flow_energy": 2.0, "color_dev": 0.1, "score": 0.65})
    assert "0.80" in p and "阴影" in p
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_vlm_judge.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现 VLM 模块**

```python
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
        '{"is_discharge": bool, "confidence": 0~1, "reason": "一句话依据", '
        '"shadow_suspect": bool}'
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
    def __init__(self, base_url, api_key, model, client=None):
        self.model = model
        if client is not None:
            self.client = client
        else:
            from openai import OpenAI
            self.client = OpenAI(base_url=base_url, api_key=api_key)

    def judge(self, current_frame, ref_frame, prev_frame, features):
        content = [{"type": "text", "text": build_prompt(features)}]
        for img in (current_frame, ref_frame, prev_frame):
            content.append({"type": "image_url", "image_url": {
                "url": f"data:image/jpeg;base64,{_b64(img)}"}})
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=0,
        )
        return parse_vlm_response(resp.choices[0].message.content)
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_vlm_judge.py -v`
Expected: 5 passed

```bash
git add -A && git commit -m "feat: VLM终判模块（OpenAI兼容接口）"
```

---

### Task 10: 时序状态机

**Files:**
- Create: `outlet_detect/state_machine.py`
- Test: `tests/test_state_machine.py`

**Interfaces:**
- Produces: `AlertEvent` dataclass：`camera_id`、`ts`、`consecutive: int`
- Produces: `DischargeStateMachine(camera_id: str, confirm_count: int = 3, cooldown_sec: int = 1800)`；`update(is_suspect: bool, ts: float) -> AlertEvent | None`；`state` 属性（`"normal" | "watching" | "alarming" | "cooldown"`）；`suspend() / resume()`（视角异常时挂起，spec 第 8 节）

- [ ] **Step 1: 写失败测试**

```python
# tests/test_state_machine.py
from outlet_detect.state_machine import DischargeStateMachine

def test_three_consecutive_triggers_alert():
    sm = DischargeStateMachine("cam01")
    assert sm.update(True, 0) is None     # 第1次疑似
    assert sm.state == "watching"
    assert sm.update(True, 300) is None   # 第2次
    ev = sm.update(True, 600)             # 第3次 → 告警
    assert ev is not None and ev.consecutive == 3
    assert sm.state == "cooldown"

def test_normal_resets_count():
    sm = DischargeStateMachine("cam01")
    sm.update(True, 0)
    sm.update(True, 300)
    sm.update(False, 600)                 # 中间一次正常
    assert sm.state == "normal"
    assert sm.update(True, 900) is None   # 重新计数

def test_cooldown_suppresses_repeat_alert():
    sm = DischargeStateMachine("cam01")
    for t in (0, 300, 600):
        sm.update(True, t)
    assert sm.update(True, 900) is None   # 冷却期内不重复告警
    ev = sm.update(True, 600 + 1801)      # 冷却结束后再次确认
    assert ev is not None

def test_suspend_resume():
    sm = DischargeStateMachine("cam01")
    sm.suspend()
    assert sm.update(True, 0) is None and sm.state == "normal"
    sm.resume()
    sm.update(True, 300)
    assert sm.state == "watching"
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_state_machine.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现状态机**

```python
# outlet_detect/state_machine.py
from dataclasses import dataclass

@dataclass
class AlertEvent:
    camera_id: str
    ts: float
    consecutive: int

class DischargeStateMachine:
    def __init__(self, camera_id, confirm_count=3, cooldown_sec=1800):
        self.camera_id = camera_id
        self.confirm_count = confirm_count
        self.cooldown_sec = cooldown_sec
        self.consecutive = 0
        self.last_alert_ts = None
        self.suspended = False

    @property
    def state(self):
        if self.suspended:
            return "normal"
        if self.last_alert_ts is not None and self.consecutive >= self.confirm_count:
            return "cooldown"
        return "watching" if self.consecutive > 0 else "normal"

    def suspend(self):
        self.suspended = True

    def resume(self):
        self.suspended = False
        self.consecutive = 0

    def update(self, is_suspect, ts):
        if self.suspended:
            return None
        if is_suspect:
            self.consecutive += 1
        else:
            self.consecutive = 0
        in_cooldown = (self.last_alert_ts is not None
                       and ts - self.last_alert_ts < self.cooldown_sec)
        if self.consecutive >= self.confirm_count and not in_cooldown:
            self.last_alert_ts = ts
            return AlertEvent(self.camera_id, ts, self.consecutive)
        return None
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_state_machine.py -v`
Expected: 4 passed

```bash
git add -A && git commit -m "feat: 时序确认状态机"
```

---

### Task 11: 告警接口

**Files:**
- Create: `outlet_detect/alerter.py`
- Test: `tests/test_alerter.py`

**Interfaces:**
- Consumes: `AlertEvent`（Task 10）、`VLMResult`（Task 9）
- Produces: `AlertPayload` dataclass：`camera_id`、`ts: str`、`frame_paths: list`、`vlm_reason: str`、`features: dict`、`consecutive: int`
- Produces: `LogAlerter(log_path: str = "data/alerts.log")`，方法 `alert(payload: AlertPayload) -> None`（追加写 JSON 行）——用户后续替换/包装此类对接推送渠道，接口签名不变

- [ ] **Step 1: 写失败测试**

```python
# tests/test_alerter.py
import json
from outlet_detect.alerter import AlertPayload, LogAlerter

def test_log_alerter_writes_json_line(tmp_path):
    log = tmp_path / "alerts.log"
    a = LogAlerter(str(log))
    p = AlertPayload(camera_id="cam01", ts="2026-10-28T10:00:00",
                     frame_paths=["a.jpg"], vlm_reason="排口发黑",
                     features={"score": 0.8}, consecutive=3)
    a.alert(p)
    line = json.loads(log.read_text().strip())
    assert line["camera_id"] == "cam01" and line["consecutive"] == 3
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_alerter.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现告警模块**

```python
# outlet_detect/alerter.py
import json
from dataclasses import dataclass, asdict
from pathlib import Path

@dataclass
class AlertPayload:
    camera_id: str
    ts: str
    frame_paths: list
    vlm_reason: str
    features: dict
    consecutive: int

class LogAlerter:
    """默认告警实现：追加 JSON 行到日志文件。
    对接推送渠道时保持 alert(payload) 签名不变即可。"""
    def __init__(self, log_path="data/alerts.log"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def alert(self, payload):
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(payload), ensure_ascii=False) + "\n")
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_alerter.py -v`
Expected: 1 passed

```bash
git add -A && git commit -m "feat: 告警事件接口（日志实现）"
```

---

### Task 12: 管线编排（单次采样全流程）

**Files:**
- Create: `outlet_detect/pipeline.py`
- Test: `tests/test_pipeline.py`

**Interfaces:**
- Consumes: 全部前序模块
- Produces: `Pipeline` 类，构造参数：`config: CameraConfig`、`sampler`、`segmenter`、`aligner: ViewAligner`、`cnn: CNNClassifier`、`vlm: VLMJudge | None`、`storage: Storage`、`state_machine: DischargeStateMachine`、`alerter`
- Produces: `run_once(now_ts: float | None) -> str`，返回最终状态（`"normal" | "gray" | "suspect" | "alert" | "view_error" | "sample_error"`）

流程（对应 spec 第 2 节）：
1. `sampler.capture` 抓 5 帧（失败 → `sample_error`，记录并返回）
2. `segmenter.segment(frames[0])` 得水面 mask（失败/无水面 → `sample_error`）
3. `aligner.align(frames[0])`：`ok=False` → 状态机 suspend，返回 `view_error`；`ok=True` 但 `in_frame=False` → 同上；否则若之前挂起则 resume
4. `build_rois(mask, mapped_point)` + `compute_features` + `score` + `verdict`
5. `normal` → 记录、状态机 update(False)、返回
6. `gray` → CNN 启用则复判（prob < 0.5 按 normal 处理），未启用或 prob ≥ 0.5 按疑似进 VLM
7. `suspect`（规则直判或 CNN 放行）→ VLM 终判；VLM 判正常则按 normal 处理；VLM 判定结果连同首帧存入标注样本库（`save_labeled_sample`）
8. 最终疑似 → `state_machine.update(True)`，返回 `AlertEvent` 则 `alerter.alert(...)` 并返回 `"alert"`
9. 每步记录耗时，`storage.record_sample` 全程留痕；疑似/告警帧 `keep=True` 归档

- [ ] **Step 1: 写失败测试**

用 Stub 组件搭全链路，验证三条路径：

```python
# tests/test_pipeline.py
import numpy as np
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.water_seg import StubWaterSegmenter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.alerter import LogAlerter
from outlet_detect.storage import Storage

class FakeAlignerOK:
    def __init__(self):
        from outlet_detect.view_align import AlignResult
        self.result = AlignResult(True, (300, 300), 0.9, True)
    def align(self, frame):
        return self.result

class FakeAlignerBad:
    def align(self, frame):
        from outlet_detect.view_align import AlignResult
        return AlignResult(False, None, 0.1, False)

class FakeVLM:
    def __init__(self, is_discharge):
        from outlet_detect.vlm_judge import VLMResult
        self.result = VLMResult(is_discharge, 0.9, "测试", False)
    def judge(self, *a, **kw):
        return self.result

def normal_frames():
    # 全画面均匀亮度 → 分数低。尺寸 600x400 以保证默认 ROI 半径
    #（r_ref=250）内存在参考水面像素
    return [np.full((400, 600, 3), 120, dtype=np.uint8) for _ in range(5)]

def suspect_frames():
    # StubWaterSegmenter 视下半幅(y>=200)为水面，排口点(300,300)处又黑又闪烁
    frames = []
    for i in range(5):
        f = np.full((400, 600, 3), 180, dtype=np.uint8)
        f[250:350, 250:350] = 20 if i % 2 == 0 else 120
        frames.append(f)
    return frames

def make_pipeline(tmp_path, frames, aligner, vlm, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cfg")
    cfg = CameraConfig(camera_id="cam01", outfall_point=(300, 300),
                       reference_frame="", hik_body={})
    storage = Storage(str(tmp_path / "t.db"), str(tmp_path / "arc"))
    return Pipeline(
        config=cfg,
        sampler=ListFrameSampler(frames),
        segmenter=StubWaterSegmenter(),
        aligner=aligner,
        cnn=CNNClassifier(None),          # 冷启动：CNN 关闭
        vlm=vlm,
        storage=storage,
        state_machine=DischargeStateMachine("cam01"),
        alerter=LogAlerter(str(tmp_path / "alerts.log")),
    )

def test_normal_path(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, normal_frames(), FakeAlignerOK(), FakeVLM(False), monkeypatch)
    assert p.run_once(0) == "normal"

def test_suspect_path_vlm_confirms(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, suspect_frames(), FakeAlignerOK(), FakeVLM(True), monkeypatch)
    assert p.run_once(0) == "suspect"
    assert p.run_once(300) == "suspect"
    assert p.run_once(600) == "alert"     # 第三次 → 告警

def test_suspect_path_vlm_rejects(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, suspect_frames(), FakeAlignerOK(), FakeVLM(False), monkeypatch)
    assert p.run_once(0) == "normal"      # VLM 否决 → 按正常

def test_view_error_suspends(tmp_path, monkeypatch):
    p = make_pipeline(tmp_path, normal_frames(), FakeAlignerBad(), FakeVLM(False), monkeypatch)
    assert p.run_once(0) == "view_error"
    p.aligner = FakeAlignerOK()           # 摄像头转回
    assert p.run_once(300) == "normal"    # 自动恢复
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现管线**

```python
# outlet_detect/pipeline.py
import logging
import time
from datetime import datetime

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
        except Exception as e:
            log.exception("%s 采样/分割失败: %s", cid, e)
            self.storage.record_sample(cid, ts, feats, "error", None, None,
                                       status, [], int((time.time() - t0) * 1000))
            return status

        ar = self.aligner.align(frames[0])
        if not ar.ok or not ar.in_frame:
            self.sm.suspend()
            self.storage.record_sample(cid, ts, feats, "view_error", None, None,
                                       "view_error", [], int((time.time() - t0) * 1000))
            log.warning("%s 视角异常(conf=%.2f)，暂停判定", cid, ar.confidence)
            return "view_error"
        if self.sm.suspended:
            log.info("%s 视角恢复", cid)
            self.sm.resume()

        rois = build_rois(mask, ar.mapped_point)
        baseline = self.storage.get_baseline_darkness(cid, hour=int(ts[11:13]), before_ts=ts)
        feats = compute_features(frames, rois, baseline)
        s = score(feats)
        v = verdict(s, self.config.score_low, self.config.score_high)

        suspect = (v == "suspect")
        if v == "gray":
            if self.cnn.enabled:
                from outlet_detect.cnn_classifier import crop_for_cnn
                cnn_prob = self.cnn.classify(crop_for_cnn(frames[0], rois))
                suspect = cnn_prob >= 0.5
            else:
                suspect = True  # 冷启动：灰色直接送 VLM

        vlm_result = None
        if suspect and self.vlm is not None:
            try:
                vlm_result = self.vlm.judge(frames[0], frames[0], frames[-1], feats)
                vlm_json = vlm_result.__dict__ and str(vlm_result.__dict__)
                self.storage.save_labeled_sample(
                    cid, ts, "", "discharge" if vlm_result.is_discharge else "normal",
                    "vlm", vlm_result.reason)
                suspect = vlm_result.is_discharge
            except Exception as e:
                log.exception("%s VLM 调用失败，按规则结果处理: %s", cid, e)

        keep = suspect
        frame_paths = self.storage.archive_frames(cid, ts.replace(":", "-"), frames, keep)
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
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_pipeline.py -v`
Expected: 4 passed（若 suspect_frames 分数不到 0.7，调整黑块亮度/闪烁幅度使规则分数足够高）

```bash
git add -A && git commit -m "feat: 单次采样全流程管线编排"
```

---

### Task 13: 定时调度器与入口

**Files:**
- Create: `outlet_detect/scheduler.py`
- Create: `main.py`
- Test: `tests/test_scheduler.py`

**Interfaces:**
- Consumes: `Pipeline.run_once`（Task 12）、`list_camera_ids` / `load_camera_config`（Task 1）
- Produces: `run_loop(pipelines: dict[str, Pipeline], interval_sec: int = 300, stop_flag=None) -> None`（每摄像头错峰 10 秒，异常不中断循环）
- `main.py`：组装真实组件（HikFrameSampler 的用户海康函数通过 `hik_adapter.py` 提供，本任务只留 TODO 注释指向用户代码位置——这是计划中唯一允许的人工接线点）；支持 `--replay` 走回放模式

- [ ] **Step 1: 写失败测试**

```python
# tests/test_scheduler.py
import threading
import time
from outlet_detect.scheduler import run_loop

def test_run_loop_calls_each_pipeline():
    calls = {"a": 0, "b": 0}
    class FakePipeline:
        def __init__(self, name):
            self.name = name
        def run_once(self, now_ts=None):
            calls[self.name] += 1
            return "normal"
    stop = threading.Event()
    t = threading.Thread(target=run_loop, kwargs=dict(
        pipelines={"a": FakePipeline("a"), "b": FakePipeline("b")},
        interval_sec=0, stagger_sec=0, stop_flag=stop))
    t.start()
    time.sleep(0.3)
    stop.set()
    t.join(timeout=5)
    assert calls["a"] >= 2 and calls["b"] >= 2

def test_run_loop_survives_exception():
    class BadPipeline:
        def run_once(self, now_ts=None):
            raise RuntimeError("boom")
    stop = threading.Event()
    t = threading.Thread(target=run_loop, kwargs=dict(
        pipelines={"x": BadPipeline()}, interval_sec=0, stagger_sec=0,
        stop_flag=stop))
    t.start()
    time.sleep(0.2)
    alive = t.is_alive()
    stop.set()
    t.join(timeout=5)
    assert alive  # 异常没有杀死循环
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现调度器与入口**

```python
# outlet_detect/scheduler.py
import logging
import time

log = logging.getLogger(__name__)

def run_loop(pipelines, interval_sec=300, stagger_sec=10, stop_flag=None):
    """每 interval_sec 对所有摄像头各跑一轮；摄像头间错峰 stagger_sec。"""
    while stop_flag is None or not stop_flag.is_set():
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

```python
# main.py
"""在线模式入口：python main.py
回放模式入口：python main.py --replay videos/xxx.mp4
"""
import argparse
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(name)s %(message)s")

def build_pipelines():
    """组装各摄像头的真实组件。海康抓图函数见 hik_adapter.py（用户接线点）。"""
    raise NotImplementedError("请先实现 hik_adapter.py 并在此组装 Pipeline")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--replay", help="离线回放视频路径")
    args = parser.parse_args()
    if args.replay:
        from outlet_detect.replay_tool import replay
        replay(args.replay)
    else:
        from outlet_detect.scheduler import run_loop
        run_loop(build_pipelines())

if __name__ == "__main__":
    main()
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_scheduler.py -v`
Expected: 2 passed

```bash
git add -A && git commit -m "feat: 定时调度器与程序入口"
```

---

### Task 14: 标定工具

**Files:**
- Create: `outlet_detect/calibrate_tool.py`
- Test: `tests/test_calibrate_tool.py`（只测非 GUI 部分）

**Interfaces:**
- Consumes: `save_camera_config`（Task 1）、`WaterSegmenter`（Task 4）、`build_rois`（Task 6）
- Produces: `generate_config(camera_id: str, frame: np.ndarray, outfall_point: tuple, segmenter, hik_body: dict) -> CameraConfig`（分割→建 ROI→保存参照帧与配置，供测试）
- Produces: CLI `python -m outlet_detect.calibrate_tool <camera_id> <图片路径>`：显示图片，鼠标点击排口位置，回车确认，生成配置并保存带 ROI 叠加的预览图 `config/cameras/<camera_id>_preview.jpg`

- [ ] **Step 1: 写失败测试（非 GUI 部分）**

```python
# tests/test_calibrate_tool.py
import numpy as np
from outlet_detect.calibrate_tool import generate_config
from outlet_detect.water_seg import StubWaterSegmenter
from outlet_detect.config import load_camera_config

def test_generate_config(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cfg")
    monkeypatch.setattr("outlet_detect.calibrate_tool.REF_DIR", tmp_path / "refs")
    frame = np.full((150, 200, 3), 128, dtype=np.uint8)
    cfg = generate_config("cam01", frame, (100, 100), StubWaterSegmenter(),
                          hik_body={"cameraIndexCode": "x"})
    assert cfg.outfall_point == (100, 100)
    loaded = load_camera_config("cam01")
    assert loaded.camera_id == "cam01"
    assert loaded.reference_frame.endswith("cam01_ref.jpg")
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_calibrate_tool.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现标定工具**

```python
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
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_calibrate_tool.py -v`
Expected: 1 passed

```bash
git add -A && git commit -m "feat: 交互式标定工具"
```

---

### Task 15: 离线回放器

**Files:**
- Create: `outlet_detect/replay_tool.py`
- Test: `tests/test_replay_tool.py`

**Interfaces:**
- Consumes: `Pipeline`（Task 12）、`ListFrameSampler`（Task 3）
- Produces: `extract_sample_frames(video_path: str, interval_sec: float = 300, burst: int = 5, burst_gap_frames: int = 50) -> list[np.ndarray]`（按"每 interval_sec 视频时间抽 burst 帧"展平成帧序列）
- Produces: `replay(video_path: str, camera_id: str = "replay", interval_sec: float = 300) -> list[str]`（内部用 StubWaterSegmenter + 直通 aligner 组装 Pipeline，返回每轮状态列表，打印汇总）

- [ ] **Step 1: 写失败测试**

生成一个合成小视频做测试：

```python
# tests/test_replay_tool.py
import cv2
import numpy as np
from outlet_detect.replay_tool import extract_sample_frames, replay

def make_video(tmp_path, n_frames=600, fps=30):
    path = str(tmp_path / "t.mp4")
    w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (200, 150))
    for i in range(n_frames):
        w.write(np.full((150, 200, 3), i % 255, dtype=np.uint8))
    w.release()
    return path

def test_extract_sample_frames(tmp_path):
    path = make_video(tmp_path, n_frames=600, fps=30)  # 20秒
    frames = extract_sample_frames(path, interval_sec=10, burst=3, burst_gap_frames=5)
    # 20秒视频、每10秒一轮、每轮3帧 → 2轮 × 3帧 = 6帧
    assert len(frames) == 6

def test_replay_runs(tmp_path, monkeypatch):
    monkeypatch.setattr("outlet_detect.config.CONFIG_DIR", tmp_path / "cfg")
    path = make_video(tmp_path, n_frames=1200, fps=30)  # 40秒 → 4轮 × 3帧 = 12帧
    results = replay(path, camera_id="r1", interval_sec=10,
                     storage_dir=str(tmp_path))
    assert len(results) == 2  # 12帧 ÷ 5帧/轮 = 2轮
    assert all(r in ("normal", "suspect", "alert", "gray",
                     "view_error", "sample_error") for r in results)
```

- [ ] **Step 2: 运行测试确认失败**

Run: `uv run pytest tests/test_replay_tool.py -v`
Expected: FAIL，`ModuleNotFoundError`

- [ ] **Step 3: 实现回放器**

```python
# outlet_detect/replay_tool.py
"""离线回放：把视频按在线采样节奏抽帧，跑完整管线，验证算法与调阈值。"""
import logging

import cv2

from outlet_detect.alerter import LogAlerter
from outlet_detect.cnn_classifier import CNNClassifier
from outlet_detect.config import CameraConfig
from outlet_detect.pipeline import Pipeline
from outlet_detect.sampler import ListFrameSampler
from outlet_detect.state_machine import DischargeStateMachine
from outlet_detect.storage import Storage
from outlet_detect.water_seg import StubWaterSegmenter

log = logging.getLogger(__name__)

class _PassThroughAligner:
    """回放模式不做视角配准：排口点取画面中心。"""
    def align(self, frame):
        from outlet_detect.view_align import AlignResult
        h, w = frame.shape[:2]
        return AlignResult(True, (w // 2, h // 2), 1.0, True)

def extract_sample_frames(video_path, interval_sec=300, burst=5, burst_gap_frames=50):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    pos = 0
    step = int(interval_sec * fps)
    while pos + (burst - 1) * burst_gap_frames < total:
        for k in range(burst):
            cap.set(cv2.CAP_PROP_POS_FRAMES, pos + k * burst_gap_frames)
            ok, f = cap.read()
            if ok:
                frames.append(f)
        pos += step
    cap.release()
    return frames

def replay(video_path, camera_id="replay", interval_sec=300, storage_dir="data"):
    frames = extract_sample_frames(video_path, interval_sec)
    if not frames:
        raise RuntimeError(f"{video_path} 未抽到任何帧")
    cfg = CameraConfig(camera_id=camera_id, outfall_point=(0, 0),
                       reference_frame="", hik_body={})
    pipe = Pipeline(
        config=cfg,
        sampler=ListFrameSampler(frames),
        segmenter=StubWaterSegmenter(),  # 回放验证时替换为真实 OnnxWaterSegmenter
        aligner=_PassThroughAligner(),
        cnn=CNNClassifier(None),
        vlm=None,                        # 回放默认不调 VLM，省钱；需验证时注入真实 VLMJudge
        storage=Storage(f"{storage_dir}/replay.db", f"{storage_dir}/replay_arc"),
        state_machine=DischargeStateMachine(camera_id),
        alerter=LogAlerter(f"{storage_dir}/replay_alerts.log"),
    )
    n_rounds = len(frames) // 5
    results = [pipe.run_once() for _ in range(n_rounds)]
    log.info("回放完成 %s: %d 轮, 结果分布: %s", video_path, n_rounds,
             {r: results.count(r) for r in set(results)})
    return results
```

- [ ] **Step 4: 运行测试确认通过并提交**

Run: `uv run pytest tests/test_replay_tool.py -v`
Expected: 2 passed

- [ ] **Step 5: 用真实视频做冒烟验证（手动）**

```bash
python -c "
from outlet_detect.replay_tool import replay
r = replay('videos/东城河阳光桥球机_1687161142095.mp4', interval_sec=5)
print(r)
"
```
确认管线在真实视频帧上无异常、状态输出合理。随后提交：

```bash
git add -A && git commit -m "feat: 离线回放器与真实视频冒烟验证"
```

---

## 后续人工接线点（不属于本计划的编码任务）

1. **hik_adapter.py**：把 `hik-sdk.py` 里的 `get_picture_url` 包装成 `capture_fn`，实现 `download_fn`（requests 下载图片字节 → `cv2.imdecode`），在 `main.py: build_pipelines()` 中组装
2. **水面分割模型**：把用户 YOLO 模型导出 ONNX 放 `models/water_seg.onnx`，用一张真实截图手动验证 `OnnxWaterSegmenter._decode` 的输出解析（不同导出方式张量布局可能不同）
3. **VLM 接入**：填入公司服务的 `base_url / api_key / model`，用一张排污截图手动验证
4. **推送渠道**：替换 `LogAlerter` 或在其 `alert()` 里追加推送调用
5. **阈值调优**：用回放器跑 videos/ 全部样本，人工标注排污时段，对比 `data/replay.db` 中 score 分布调整权重与双阈值
6. **VLM 定位重建 ROI（spec 第 8 节第 2 层的增强）**：视角异常持续 1 小时以上且画面内有足够水面时，调 VLM grounding 定位排口点重建 ROI。依赖公司 VLM 的 grounding 能力实测，待在线运行稳定后作为增量任务实现

---

## 最终审查后续任务清单（2026-10-30 全分支审查产出）

合入后建议的增量任务（非阻塞，按优先级）：

1. **purge_old_frames 接调度**：设计 §9 要求正常帧 7 天滚动删除，实现已存在但无人调用——在 run_loop 加每日一次调用或文档化运维步骤
2. **pipeline 路径测试补齐**：gray 分流、VLM 降级、sample_error 记录、align 异常四条路径无测试，上线试运行前补齐
3. **mask_coverage 接线**：设计 §3.2 排口区水面覆盖率过低降置信，函数已有但未接入 pipeline，回放调优期一并校准
4. **标定工具分割器解耦**：calibrate_tool 硬编码 OnnxWaterSegmenter，生产已切 HTTP 服务，支持注入 HttpWaterSegmenter
5. **设计 §12 错误处理落地**：海康抓图失败重试 2 次、连续失败通知、SQLite 写失败降级写日志
6. **重复代码清理**：base64 编码（vlm_judge/water_seg 各一份）、rule_engine GaussianBlur 重复计算，抽公共 util
