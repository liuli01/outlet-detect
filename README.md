# outlet-detect

河道排口排污 + 水面漂浮物视频检测系统。

## 功能

- **排口排污检测**：三级漏斗——规则引擎（黑度/波动/色偏特征）→ CNN 复判 → VLM 终判，三连确认状态机，30 分钟告警冷却
- **漂浮物检测**：水面分割范围内 YOLO 目标检测，连续检出确认后上报
- **视角自适应**：参照帧对齐，镜头转动/参照帧异常时自动挂起判定
- **故障容错**：VLM 超时/乱码时退化为"规则+状态机"照常报警；分割/漂浮物/上报任一环节失败不阻断排污主线
- **定时调度**：每天 10:00~15:00 每 5 分钟一轮，窗口外休眠
- **结果上报**：标注图上传 minio，事件字段（漂浮物/排污/水色置信度）POST 业务平台，SQLite 持久化上报记录

## 架构

```
海康球机 --5分钟截图--> 采样器(5连帧)
   |
   ├─ 帧1 → 漂浮物引擎(8005) → 水面过滤 → 状态机 → 上报
   └─ 5帧 → 水面分割 → 视角对齐 → ROI → 规则评分
              ├─ 低分 → normal
              ├─ 高分 → suspect → 状态机 → alert → 上报
              └─ 灰色 → CNN/VLM 终判 → 状态机 → alert → 上报
```

## 快速开始

```bash
uv sync
uv run pytest                          # 94 个测试
cp .env.example .env                   # 填入真实配置
uv run python main.py                  # 在线运行（详见 deploy/README.md）
uv run python -m outlet_detect.webapp  # 测试台：http://127.0.0.1:8090 上传照片测检测
```

部署与配置详见 [deploy/README.md](deploy/README.md)。

## 目录

- `outlet_detect/` —— 核心包（采样/分割/对齐/ROI/规则引擎/CNN/VLM/状态机/存储/上报/调度）
- `main.py` —— 在线入口；`--replay` 回放模式
- `tools/` —— 手动工具脚本（离线验证/回放/连通性测试/样本诊断等）
- `deploy/` —— Dockerfile、docker-compose、部署文档
- `tests/` —— pytest 套件
- `docs/superpowers/` —— 设计文档与实施计划

## 发布

推送 `v*` 标签触发 GitHub Actions 构建镜像并发布到 ghcr.io：

```bash
git tag v0.2.0 && git push origin v0.2.0
```
