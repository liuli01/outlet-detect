#!/bin/sh
# 容器冒烟测试：程序在"所有外部接口不可达"时也应正常降级而非崩溃。
# 用法: deploy/smoke_test.sh <镜像名>
# 步骤：容器内写入 dummy 配置（全部指向不可达地址）→ build_pipelines 组装 →
#       run_once 必须内部捕获接口异常并返回 sample_error，不得抛 Traceback。
set -u

IMAGE="${1:?用法: smoke_test.sh <镜像名>}"

OUT=$(docker run --rm --entrypoint sh "$IMAGE" -c '
  mkdir -p config/cameras
  cat > config/deploy.json << EOF
{"hik_base_url":"https://127.0.0.1:1","app_key":"k","app_secret":"s",
 "minio_endpoint":"127.0.0.1:1","minio_ak":"a","minio_sk":"s",
 "report_url":"http://127.0.0.1:1/api","external_host":""}
EOF
  cat > config/cameras/smoke01.json << EOF
{"camera_id":"smoke01","outfall_point":[640,300],"reference_frame":"",
 "hik_body":{"cameraIndexCode":"smoke01"},
 "float_engine_id":"floating/test","float_confidence":0.3}
EOF
  uv run --no-sync python - << PYEOF
import logging
logging.basicConfig(level=logging.INFO)
from main import build_pipelines

pipes = build_pipelines()
assert list(pipes) == ["smoke01"], f"组装结果异常: {list(pipes)}"
status = pipes["smoke01"].run_once()   # 接口全不可达：必须降级而非崩溃
assert status == "sample_error", f"接口故障未被容错: {status}"
print("PIPELINE SMOKE OK")
PYEOF
' 2>&1)
CODE=$?

echo "$OUT"

if [ "$CODE" -ne 0 ]; then
  echo "SMOKE FAIL: 退出码 $CODE"; exit 1
fi
# 注意：log.exception 会打印已捕获异常的 Traceback（属正常容错日志），
# 不能以 Traceback 字样判失败；Python 断言与退出码才是判据。
if ! echo "$OUT" | grep -q "PIPELINE SMOKE OK"; then
  echo "SMOKE FAIL: 未见成功标记"; exit 1
fi
echo "SMOKE PASS"
