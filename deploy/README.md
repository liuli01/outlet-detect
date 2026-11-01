# 部署文档

河道排污 + 漂浮物检测系统。每天 10:00~15:00（Asia/Shanghai 时区）每 5 分钟一轮：海康球机 5 连帧截图 → 水面分割（推理服务）→ 漂浮物检测（水面范围内）→ 排污三级漏斗（规则引擎 → CNN → VLM 终判）→ 三连确认后 minio 存图 + 业务平台上报。

生产部署包含两个常驻容器：

| 容器 | 作用 | 端口 |
|---|---|---|
| `outlet-detect` | 定时任务主程序（采样→检测→上报） | 无外露 |
| `outlet-detect-testpage` | 测试页面（上传照片/样本库在线测检测效果） | `8090` |

## 前置依赖

| 依赖 | 说明 |
|---|---|
| 推理服务 | YOLO 推理 HTTP 服务（`/infer`），需加载水面分割引擎（如 `river/20261030_132330`）与漂浮物引擎（如 `floating/20261030_145216`） |
| 海康 Artemis 平台 | manualCapture 抓图（AK/SK 签名） |
| minio | 事件图片存储（桶：`input-image`、`ai-image`） |
| 业务平台 | `insert_ai_result` 上报接口 |
| VLM（可选） | 多模态大模型网关，故障/关闭时系统退化为规则+状态机照常报警 |

## 生产服务器部署步骤

### 1. 准备

服务器需安装 Docker 与 Docker Compose 插件，并能访问上表所有外部服务（推理服务、海康平台、minio、上报平台）。

```bash
# 方式 A：拉源码（可自行构建镜像）
git clone https://github.com/liuli01/outlet-detect.git
cd outlet-detect

# 方式 B：不拉源码，只取部署文件（使用 ghcr.io 预构建镜像）
mkdir outlet-detect && cd outlet-detect
# 下载 deploy/docker-compose.yml、deploy/Dockerfile 与 .env.example 到对应位置
```

### 2. 配置 `.env`

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
INFER_ENDPOINT=http://<推理服务host>:8005/infer
```

注意：

- `EXTERNAL_HOST` 必须带端口（整体替换 minio 内网 host:port，与外网 NAT 映射一致）
- `INFER_ENDPOINT` 是水面分割 + 漂浮物共用的推理服务地址；推理服务不在本机时必须改成实际地址
- VLM 终判配置（可选）：

```
VLM_BASE_URL=<VLM网关地址>   # 留空 = 关闭 VLM
VLM_API_KEY=<key>
VLM_MODEL=                   # 公司网关路由默认模型可留空
VLM_ENVELOPE=true            # 公司网关包装格式；标准 OpenAI 填 false
```

### 3. 摄像头标定 `config/cameras/<indexCode>.json`

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
- `reference_frame`：标定参照帧路径（相对仓库根，放 `data/reference/` 随 data 卷持久化）；缺失/损坏时该摄像头自动挂起（view_error），不影响其他摄像头
- `float_engine_id`：留空则该摄像头关闭漂浮物检测
- `float_confidence`：建议 0.3（验证集实测检出置信 0.48~0.85，默认 0.6 会漏检）
- `hik_body.cameraIndexCode` 为空的文件会被当作辅助配置跳过

### 4. 启动

```bash
cd deploy
docker compose up -d --build     # 首次或代码更新后
docker compose ps                # 确认两个容器都在运行
docker compose logs -f           # 跟日志（Ctrl+C 退出不影响运行）
```

测试页面：浏览器打开 `http://<服务器IP>:8090`，上传照片或点选样本库即可看到分割/漂浮物/排污结果与标注图。

> **安全提示**：测试页面无鉴权，生产环境请用防火墙/安全组把 8090 限制在内网或运维网段。
> 不需要样本库时可删掉 compose 里 test-page 的 `../images:/app/images:ro` 挂载行（页面仍支持上传照片测试）。

### 5. 日常运维

```bash
docker compose restart outlet-detect   # 改了 .env 或摄像头配置后重启生效
docker compose pull && docker compose up -d   # 升级到最新镜像
docker compose logs --tail 200 outlet-detect  # 查主程序日志
docker compose down                    # 停止全部
```

- 摄像头配置、`.env` 改动都需要 `restart` 才生效（配置在启动时加载）
- `data/` 目录（SQLite、归档帧、告警日志）挂载在宿主机，升级/重建容器不丢数据
- 单容器日志已限制 10MB×3 滚动，不会撑爆磁盘

### 打 tag 发布镜像（开发侧）

推送 `v*` 标签触发 GitHub Actions：pytest → 构建 → 容器冒烟（接口全不可达时验证容错降级）→ 发布到 ghcr.io：

```bash
git tag v0.2.2 && git push origin v0.2.2
# 产出 ghcr.io/liuli01/outlet-detect:0.2.2 / :0.2 / :latest
```

生产侧固定版本：`docker compose pull` 前把 compose 里的 `:latest` 改为 `:0.2.2` 等具体版本更稳妥。

### 本地开发（uv）

```bash
uv sync
uv run python main.py                  # 在线运行
uv run python -m outlet_detect.webapp  # 测试台：http://127.0.0.1:8090
```

## 验证与排查

- 上报成败持久化在每个摄像头的 `data/<cid>.db` 的 `reports` 表（ok/error 字段）
- 排污判定明细在 `samples` 表（特征值/score/verdict/VLM 结果）
- 告警日志 `data/<cid>_alerts.log`，归档帧 `data/archive/<cid>/`
- 离线验证：`uv run python tools/validate_images.py`（排污）、`uv run python tools/validate_float.py [阈值]`（漂浮物）
- 回放调试：`uv run python main.py --replay <视频路径>`
- 容器内验证时区：`docker compose exec outlet-detect date` 应显示 CST（调度窗口按此时区判断）

## 上报字段约定

| 字段 | 含义 |
|---|---|
| ResultA | 漂浮物置信度（bbox 最高值） |
| ResultB | 排污置信度（VLM 置信度，VLM 故障时为规则评分） |
| ResultC | 水体颜色置信度（与 ResultB 同值） |
