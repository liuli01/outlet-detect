# 部署文档

河道排污 + 漂浮物检测系统。每天 10:00~15:00 每 5 分钟一轮：海康球机 5 连帧截图 → 水面分割（8005 推理服务）→ 漂浮物检测（水面范围内）→ 排污三级漏斗（规则引擎 → CNN → VLM 终判）→ 三连确认后 minio 存图 + 业务平台上报。

## 前置依赖

| 依赖 | 说明 |
|---|---|
| 推理服务 `8005` | YOLO 推理 HTTP 服务，需加载水面分割引擎（如 `river/20261030_132330`）与漂浮物引擎（如 `floating/20261030_145216`） |
| 海康 Artemis 平台 | manualCapture 抓图（AK/SK 签名） |
| minio | 事件图片存储（桶：`input-image`、`ai-image`） |
| 业务平台 | `insert_ai_result` 上报接口 |

## 配置

### 1. 部署配置 `.env`

复制仓库根的 `.env.example` 为 `.env` 并填入真实值（该文件已 gitignore）：

```
HIK_BASE_URL=https://<artemis-host>:14443
HIK_APP_KEY=<AppKey>
HIK_APP_SECRET=<AppSecret>
MINIO_ENDPOINT=<minio-host>:9000
MINIO_AK=<access-key>
MINIO_SK=<secret-key>
REPORT_URL=http://<平台host>/api/insert_ai_result
EXTERNAL_HOST=<外网host>:<minio外网端口>
```

注意：`EXTERNAL_HOST` 必须带端口（整体替换 minio 内网 host:port，与外网 NAT 映射一致）。

### 2. 摄像头配置 `config/cameras/<indexCode>.json`

每个摄像头一个文件（已 gitignore），关键字段：

```json
{
  "camera_id": "<indexCode>",
  "outfall_point": [x, y],
  "reference_frame": "data/reference/<indexCode>.jpg",
  "hik_body": {"cameraIndexCode": "<indexCode>"},
  "float_engine_id": "floating/20261030_145216",
  "float_confidence": 0.3,
  "ignore_classes": ["aeration"]
}
```

- `outfall_point`：排口像素坐标，用标定工具在参照帧上取点
- `float_engine_id`：留空则该摄像头关闭漂浮物检测
- `float_confidence`：建议 0.3（验证集实测检出置信 0.48~0.85，默认 0.6 会漏检）
- `hik_body.cameraIndexCode` 为空的文件（如 `vlm.json`）会被当作辅助配置跳过

### 3. VLM 终判配置（在 .env 中）

```
VLM_BASE_URL=<VLM网关地址>   # 留空 = 关闭 VLM
VLM_API_KEY=<key>
VLM_MODEL=                   # 公司网关路由默认模型可留空
VLM_ENVELOPE=true            # 公司网关包装格式；标准 OpenAI 填 false
```

VLM 故障（超时/乱码/空返回）或关闭时不影响排污识别：系统退化为"规则引擎+状态机"照常报警，置信度回退为规则评分。

## 运行

### Docker（推荐）

```bash
cd deploy
docker compose up -d --build
docker compose logs -f
```

### 本地（uv）

```bash
uv sync
uv run python main.py
```

### 打 tag 发布镜像

推送 `v*` 标签触发 GitHub Actions：pytest → 构建 → 容器冒烟（接口全不可达时验证容错降级）→ 发布到 ghcr.io：

```bash
git tag v0.2.0 && git push origin v0.2.0
# 产出 ghcr.io/<owner>/outlet-detect:0.2.0 / :0.2 / :latest
```

### 测试台（可选）

上传照片在线测试漂浮物/排污检测（需能访问 8005 推理服务）：

```bash
# 本地
uv run python -m outlet_detect.webapp        # http://127.0.0.1:8090
# Docker
docker compose --profile tools up -d test-page
```

## 验证与排查

- 上报成败持久化在每个摄像头的 `data/<cid>.db` 的 `reports` 表（ok/error 字段）
- 排污判定明细在 `samples` 表（特征值/score/verdict/VLM 结果）
- 告警日志 `data/<cid>_alerts.log`，归档帧 `data/archive/<cid>/`
- 离线验证：`uv run python validate_images.py`（排污）、`uv run python validate_float.py [阈值]`（漂浮物）
- 回放调试：`uv run python main.py --replay <视频路径>`

## 上报字段约定

| 字段 | 含义 |
|---|---|
| ResultA | 漂浮物置信度（bbox 最高值） |
| ResultB | 排污置信度（VLM 置信度，VLM 故障时为规则评分） |
| ResultC | 水体颜色置信度（与 ResultB 同值） |
