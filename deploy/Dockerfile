# 河道排污/漂浮物检测系统镜像
# 构建上下文为仓库根目录：docker build -f deploy/Dockerfile .
FROM python:3.12-slim

# uv 包管理器（虚拟环境为 /app/.venv）
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

WORKDIR /app

# 先装依赖（锁文件层缓存）
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-dev

# 再拷代码
COPY outlet_detect/ outlet_detect/
COPY main.py ./
COPY config/deploy.example.json config/

# 运行时挂载：
#   /app/config/deploy.json   部署配置（密钥）
#   /app/config/cameras/      摄像头标定配置
#   /app/data                 SQLite 与归档帧（持久卷）
VOLUME ["/app/data"]

CMD ["uv", "run", "--no-sync", "python", "main.py"]
