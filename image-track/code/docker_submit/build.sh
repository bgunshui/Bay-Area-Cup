#!/bin/bash
# =============================================================================
# build.sh —— 一键产出可提交的 Docker 镜像 tar
#
#   bash build.sh                     # 完整流程（备资材 → 构建 → 冒烟 → docker save）
#   bash build.sh --skip-download     # 骨干已在 app/backbones 时跳过下载
#   bash build.sh --no-save           # 只构建，不导出 tar
#   TAG=xxx bash build.sh             # 自定义镜像 tag
#
# 前置：装了 Docker 的 linux/amd64 机器 + 能访问 hf-mirror（下载骨干，约 12 GB）
#       磁盘至少留 40 GB（构建上下文 13 GB + 镜像 20 GB + tar 20 GB）
# =============================================================================
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-baycup-image-aidet:0.918887}"
TAR="${TAR:-baycup_image_aidet_0.918887.tar}"
SKIP_DL=""
NO_SAVE=""
for a in "$@"; do
  case "$a" in
    --skip-download) SKIP_DL="--skip-download" ;;
    --no-save)       NO_SAVE=1 ;;
    *) echo "未知参数 $a"; exit 2 ;;
  esac
done

echo "=============================================================="
echo " ① 归拢资产（8 个 LoRA 权重 + 查表）"
echo "=============================================================="
python3 prepare_assets.py $SKIP_DL

echo
echo "=============================================================="
echo " ② 离线自检：逐图管线必须复现 LB 0.918887 的提交文件"
echo "=============================================================="
python3 verify_reproduce.py --from-csv | tail -8

echo
echo "=============================================================="
echo " ③ docker build"
echo "=============================================================="
DOCKER_BUILDKIT=1 docker build -t "$TAG" .
docker images "$TAG"

echo
echo "=============================================================="
echo " ④ 冒烟测试（按《交付标准》第 6 节的调用流程走一遍）"
echo "=============================================================="
CID=$(docker run -d --gpus all -p 18080:8080 "$TAG")
trap 'docker rm -f "$CID" >/dev/null 2>&1 || true' EXIT

echo -n "  等 /health/startup …"
for i in $(seq 1 100); do
  code=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:18080/health/startup || true)
  [ "$code" = "200" ] && { echo " 200（用时约 $((i*3))s）"; break; }
  sleep 3
  [ "$i" = "100" ] && { echo " 超时（>300s）"; docker logs "$CID" | tail -30; exit 1; }
done
echo "  /health/liveness   -> $(curl -s -o /dev/null -w '%{http_code}' http://localhost:18080/health/liveness)"
echo "  /health/readiness  -> $(curl -s -o /dev/null -w '%{http_code}' http://localhost:18080/health/readiness)"
echo "  /debug/info        -> $(curl -s http://localhost:18080/debug/info)"

python3 - <<'PY'
import base64, glob, io, json, time, urllib.request
from PIL import Image
# 用一张真实测试图做一次 /predict
cand = sorted(glob.glob("../data/official_test/image_test/*.jpg")
              + glob.glob("../_official_test_images/*.jpg")
              + glob.glob("../**/_official_test_images/*.jpg", recursive=True))
if not cand:
    # 造一张
    buf = io.BytesIO(); Image.new("RGB", (1024, 768), (120, 130, 140)).save(buf, "JPEG")
    raw = buf.getvalue()
else:
    raw = open(cand[0], "rb").read()
req = {"req_id": "smoke-0001", "data": base64.b64encode(raw).decode()}
t0 = time.time()
r = urllib.request.urlopen(urllib.request.Request(
    "http://localhost:18080/predict", data=json.dumps(req).encode(),
    headers={"Content-Type": "application/json"}))
j = json.loads(r.read())
dt = time.time() - t0
print(f"  POST /predict -> {j}   用时 {dt:.2f}s")
assert j["code"] == 0 and j["req_id"] == "smoke-0001" and 0.0 <= j["score"] <= 1.0
# 错误路径
for bad in ({"req_id": "e1", "data": "!!!notbase64!!!"}, {"data": "x"}, {}):
    try:
        urllib.request.urlopen(urllib.request.Request(
            "http://localhost:18080/predict", data=json.dumps(bad).encode(),
            headers={"Content-Type": "application/json"}))
        print(f"  ⚠️ {bad} 竟然返回 200")
    except urllib.error.HTTPError as e:
        print(f"  错误路径 {str(bad)[:34]:36s} -> HTTP {e.code}  {e.read().decode()[:80]}")
print("  冒烟通过 ✅")
PY

docker rm -f "$CID" >/dev/null 2>&1 || true
trap - EXIT

if [ -z "$NO_SAVE" ]; then
  echo
  echo "=============================================================="
  echo " ⑤ docker save -> $TAR"
  echo "=============================================================="
  docker save "$TAG" -o "$TAR"
  ls -lh "$TAR"
  echo "  sha256 = $(sha256sum "$TAR" | cut -d' ' -f1)"
  echo "  digest = $(docker inspect --format='{{index .RepoDigests 0}}' "$TAG" 2>/dev/null || echo '(本地 build 无 digest，push 后才有)')"
fi

echo
echo "=============================================================="
echo " ✅ 完成：$TAG"
echo "    镜像 tar : $TAR"
echo "    提交材料 : 本 tar + TECHNICAL_REPORT.md，按《附件提交要求》打包"
echo "=============================================================="
