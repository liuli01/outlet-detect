# 河道排污/漂浮物检测系统镜像
# 构建上下文为仓库根目录：docker build -f deploy/Dockerfile .
FROM python:3.12-slim

# 时区数据：调度窗口 10:00~15:00 按系统本地时间判断，容器默认 UTC 会偏 8 小时
RUN apt-get update && apt-get install -y --no-install-recommends tzdata \
    && rm -rf /var/lib/apt/lists/*
ENV TZ=Asia/Shanghai

# uv 包管理器（虚拟环境为 /app/.venv）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先装依赖（锁文件层缓存）
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 再拷代码
COPY outlet_detect/ outlet_detect/
COPY main.py ./

# 运行时配置：
#   环境变量（.env，见仓库根 .env.example）——海康/minio/上报平台
#   /app/config/cameras/      摄像头标定配置（挂载）
#   /app/data                 SQLite 与归档帧（持久卷）
VOLUME ["/app/data"]

CMD ["uv", "run", "--no-sync", "python", "main.py"]
