# 扩展设计：漂浮物检测 + 结果上报 + 时段调度

- 日期：2026-10-30
- 状态：已批准（用户确认）
- 前置：[2026-10-28-river-discharge-detection-design.md](2026-10-28-river-discharge-detection-design.md)（排污检测主系统，已完成）

## 1. 背景与目标

在已完成的排污检测系统上扩展三项能力：

1. **漂浮物检测**：复用 8005 推理服务的漂浮物引擎（训练中，engine_id 待部署后配置）
2. **结果上报**：检测事件通过用户业务平台接口上报（`save_model_result` + minio 图片上传，沿用 hik-sdk.py 的链路）
3. **时段调度**：每天 10:00~15:00 运行，每 5 分钟一轮，窗口外完全休眠

## 2. 总体架构

```
调度器：每天 10:00~15:00，每 5 分钟一轮（窗口外休眠到下个检查点）
  ↓ 每摄像头
采样器：manualCapture 5 连帧（间隔 2 秒）
  ├─ 帧 1 → 漂浮物引擎（8005 /infer，float engine_id）
  │        ↓ bbox 列表（本地过滤 ignore 类别如 aeration）
  │        漂浮物状态机：连续 3 次检出 → 上报，30 分钟冷却
  │
  └─ 5 帧 → 排污三级漏斗（水面分割→规则引擎→CNN→VLM→状态机，不变）
           ↓ alert
        排污告警 → 上报

上报线（两类事件共用 Reporter）：
  画标注图 → minio 上传（input-image 原图 + ai-image 标注图）→ save_model_result
  字段约定：ResultA=漂浮物置信度，ResultB=ResultC=排污置信度（排污与水体颜色共享）
```

## 3. 模块设计

### 3.1 `hik_adapter.py`（新）

从 hik-sdk.py 提取为正式模块：

- `HikApi(base_url, app_key, app_secret)`：AK/SK 签名 POST（签名逻辑原样移植：HMAC-SHA256 → base64，x-ca-* headers）
- `get_index_code_list(platform_url)`：从业务平台拿摄像头 IndexCode 列表
- `make_capture_fn(hik_base_url, app_key, app_secret)` → `capture_fn(body) -> picUrl`（manualCapture）
- `make_download_fn()` → `download_fn(url) -> np.ndarray`（requests 下载 + `cv2.imdecode`，verify=False 兼容自签证书）

### 3.2 `float_detector.py`（新）

```python
@dataclass
class FloatDetection:
    class_name: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)

class HttpFloatDetector:
    def __init__(self, endpoint="http://127.0.0.1:8005/infer", engine_id="",
                 confidence=0.6, ignore_classes=(), timeout=30.0, post_fn=None): ...
    @property
    def enabled(self) -> bool: ...  # engine_id 非空
    def detect(self, frame) -> list[FloatDetection]: ...
```

- 与 HttpWaterSegmenter 同一服务、同一请求格式（base64 image + confidence_threshold）
- 响应 detections 取 class_name/confidence/bbox（漂浮物是检测任务，无 points）
- 本地按置信度和 ignore_classes 过滤（如 aeration 曝气）
- **engine_id 为空 → enabled=False，调用方跳过**；服务异常抛 RuntimeError 由管线兜底

### 3.3 `reporter.py`（新）

```python
@dataclass
class ReportEvent:
    index_code: str
    input_picture: str       # 海康 picUrl 或本地路径
    output_path: str         # 本地标注图路径
    result_a: float          # 漂浮物置信度
    result_b: float          # 排污置信度
    result_c: float          # 水体颜色置信度（排污时与 result_b 同值）

class MinioReporter:
    def __init__(self, minio_endpoint, access_key, secret_key,
                 report_url, external_host, client=None, post_fn=None): ...
    def report(self, event: ReportEvent) -> None: ...
```

- minio 上传：原图 → `input-image` 桶，标注图 → `ai-image` 桶；URL 内网地址替换为外网 host（沿用 hik-sdk 的 `replace("192.166.250.223", "223.108.117.18")` 策略，host 对做成参数）
- 上报：form POST 到 report_url（`save_model_result` 原样字段 IndexCode/InputPicture/OutputPicture/ResultA/ResultB/ResultC）
- **上报失败不阻断**：抛 ReporterError 由调用方记日志，下轮继续

标注图绘制（`annotate.py` 并入 reporter 或独立小模块）：
- 漂浮物：bbox 矩形 + `类别 置信度` 文字
- 排污：三色 ROI 圈（排口红/扩散黄/参考绿）+ score 与 VLM reason 文字

### 3.4 `pipeline.py`（改）

`run_once` 内插入漂浮物分支（在排污判定之前）：

```python
if self.float_detector and self.float_detector.enabled:
    try:
        dets = self.float_detector.detect(frames[0])
    except Exception as e:
        log.warning("漂浮物检测失败: %s", e)  # 不影响排污主线
    else:
        evt = self.float_sm.update(bool(dets), ts_epoch)
        if evt:
            reporter.report(漂浮物事件, result_a=max(d.confidence for d in dets))
```

- 漂浮物状态机复用 `DischargeStateMachine`（三连确认+冷却，逻辑通用）
- 排污 alert 时：标注图（ROI+文字）→ 上报，result_b=result_c=VLM confidence（无 VLM 时用 score）
- reporter 为可选依赖（None 时只走 alerter，保持回放/验证路径不变）

### 3.5 `scheduler.py`（改）

`run_loop(..., active_hours=(10, 15))`：每轮触发时检查 `datetime.now().hour`，窗口外跳过本轮（睡眠 interval 后重查，无需精确睡到 10 点）。

### 3.6 `config.py`（改）

`CameraConfig` 加字段（均有默认值，向后兼容已有配置 JSON）：

- `float_engine_id: str = ""`（空=关闭漂浮物检测）
- `float_confidence: float = 0.6`
- `ignore_classes: list = field(default_factory=list)`（如 ["aeration"]）

全局部署配置（minio/上报平台/海康 AK-SK）不放摄像头配置，放 `config/deploy.json`（gitignored），由 main.py 加载。

### 3.7 `main.py`（改）

实现 `build_pipelines()`：

1. 加载 `config/deploy.json`（海康、minio、上报地址）
2. 遍历 `config/cameras/*.json`（标定过的摄像头，跳过 vlm.json 等特殊文件——按有无 `hik_body.cameraIndexCode` 区分）
3. 组装 Pipeline：HikFrameSampler（capture_fn/download_fn）+ HttpWaterSegmenter + ViewAligner + CNNClassifier + VLMJudge（envelope 按配置）+ HttpFloatDetector + Storage + 状态机×2 + LogAlerter + MinioReporter

## 4. 错误处理

| 故障 | 处理 |
|---|---|
| 漂浮物引擎未配置/调用失败 | 跳过本轮漂浮物检测记日志，排污主线不受影响 |
| minio 上传失败 | 记日志，不上报（本地告警日志仍在） |
| 上报接口失败 | 记日志不阻断，下轮继续 |
| 窗口外 | 不采样不计特征，状态机挂起状态保持 |

## 5. 存储

- 上报事件写入 SQLite 新表 `reports`（camera_id、ts、类型、result_a/b/c、图片 URL、成功与否）——便于排查上报链路
- 其余沿用现有 samples/labeled_samples/归档结构

## 6. 验证路径

- 单测：float_detector（mock post_fn）、reporter（mock minio+requests）、时段窗口（mock 时间）、pipeline 漂浮物分支
- 集成：漂浮物引擎部署后填 engine_id，用回放验证（replay 增加漂浮物开关）
- 联调：真实摄像头跑一天，检查业务平台收到上报

## 7. 非目标（本期不做）

- 漂浮物种类细分统计与趋势看板
- 上报失败的重试队列（记日志足够，事件本身不丢）
- 水色异常独立检测线（river_color 模型——其职责已被排污检测覆盖，ResultC 共享排污置信度是过渡约定）
