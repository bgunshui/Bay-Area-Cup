#!/bin/bash
# build_on_server.sh — 在 GPU 服务器上一键构建并导出 Docker 镜像
# 用法:bash build_on_server.sh
set -e

WORK=${WORK:-/root/docker_build}
MODEL_SRC=${MODEL_SRC:-/root/sj-tmp/Qwen2.5-14B-Instruct}
TAG=${TAG:-ai-text-detector:0.981069}

echo "=== [1/5] 准备目录 ==="
mkdir -p "$WORK"
cd "$WORK"

if [ ! -d "$WORK/model/Qwen2.5-14B-Instruct" ]; then
  echo "复制基座模型 $MODEL_SRC -> $WORK/model/"
  mkdir -p "$WORK/model"
  cp -r "$MODEL_SRC" "$WORK/model/"
else
  echo "基座模型已存在,跳过复制"
fi

echo "=== [2/5] 检查 adapters ==="
N=$(ls -d "$WORK"/adapters/lora14b_p4_fold*_adapter 2>/dev/null | wc -l)
if [ "$N" -lt 5 ]; then
  echo "[!] adapters 不完整(找到 $N 个,需要 5 个)"
  echo "    请先把 docker_submit/adapters/ 上传到 $WORK/adapters/"
  exit 1
fi
echo "adapters: $N 个 ✓"

echo "=== [3/5] 检查 Docker ==="
if ! command -v docker >/dev/null 2>&1; then
  echo "[!] 服务器无 docker,正在安装..."
  curl -fsSL https://get.docker.com | sh || {
    echo "[!] docker 安装失败,请手动安装"; exit 1; }
fi
docker --version

echo "=== [4/5] 构建镜像(约 20-40 分钟)==="
docker build -t "$TAG" "$WORK"

echo "=== [5/5] 导出镜像 tar ==="
OUT=/root/ai-text-detector-0.981069.tar
docker save "$TAG" -o "$OUT"
ls -la "$OUT"
echo ""
echo "✅ 构建完成:$OUT"
echo "提示:用 sha256sum $OUT 生成 digest 用于提交"
sha256sum "$OUT" | head -1
