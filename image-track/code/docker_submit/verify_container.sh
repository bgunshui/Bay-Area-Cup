#!/bin/bash
# =============================================================================
# verify_container.sh —— 不需要 Docker / 不需要 GPU 的**成品镜像复检**
#
#   bash verify_container.sh <镜像 tar> [工作目录，默认 /tmp/imgverify]
#
# 为什么要有它：
#   构建机上拿不到 Docker 特权（`docker build` → `unshare: operation not permitted`），
#   所以"`docker run` 一下看看"这条最自然的验证路走不通。但 **root 的 `chroot` 只需要
#   CAP_SYS_CHROOT**，于是可以把镜像的每一层真的摊开成一个 rootfs，真的在里面跑
#   `server.py` —— 这是在没有 Docker 的环境里能做到的最强验证。
#
# 它复刻的正是 2026-09-18 那次「镜像权重崩溃」的失败路径：
#   镜像 config 里 `User=appuser`，而 3 个骨干 `.safetensors` 是 `-rw------- root:root`
#   ⇒ 容器里加载骨干时 PermissionError。所以本脚本**所有文件访问一律以 uid 1000 进行**。
#
# 依赖：bash / tar / python3（宿主）/ mknod / chroot（GNU coreutils，带 --userspec）
# 实测：Ubuntu 22.04 宿主，18.98 GB 镜像解包 122 秒。
# =============================================================================
set -u
IMG="${1:?用法: bash verify_container.sh <镜像 tar> [工作目录]}"
WORK="${2:-/tmp/imgverify}"
SELF="$(cd "$(dirname "$0")" && pwd)"
ROOT="$WORK/rootfs"

echo "############ 0. 摊开镜像（后写的层覆盖先写的，语义与 overlay 一致）############"
rm -rf "$WORK"; mkdir -p "$ROOT"
python3 "$SELF/verify_container_extract.py" "$IMG" "$ROOT" || exit 1

echo
echo "############ 1. 容器里那个用户到底是谁 ############"
grep -E '^(appuser|root)' "$ROOT/etc/passwd"
python3 - "$IMG" <<'PY'
import json, sys, tarfile
t = tarfile.open(sys.argv[1], "r:")
m = t.getmember("manifest.json")
o = json.loads(t.extractfile(m).read())
o = o[0] if isinstance(o, list) else o
c = json.loads(t.extractfile(t.getmember(o["Config"])).read())
print("  User      :", c["config"].get("User"))
print("  WorkingDir:", c["config"].get("WorkingDir"))
print("  Cmd       :", c["config"].get("Cmd"))
print("  Env       :")
for e in c["config"]["Env"]:
    print("     ", e)
print("  Healthcheck:", c["config"].get("Healthcheck", {}).get("Test"))
print("  层数      :", len(o["Layers"]))
PY

echo
echo "############ 2. 补齐容器运行期由 docker 注入的设备节点 ############"
for spec in "null c 1 3" "zero c 1 5" "random c 1 8" "urandom c 1 9" "tty c 5 0"; do
  set -- $spec
  [ -e "$ROOT/dev/$1" ] || mknod -m 666 "$ROOT/dev/$1" $2 $3 $4 2>/dev/null
done
ls -la "$ROOT/dev/" | tail -6
echo "  （/proc 通常无法挂载：无 CAP_SYS_ADMIN。python 不需要它，只有 torch 的 cpuinfo 会打一行警告）"

# 镜像内的解释器位置取决于基座：pytorch 官方镜像在 /opt/conda/bin，
# python:*-slim 在 /usr/local/bin。**不要写死**（这里写死过一次，白跑一轮）。
PY=$(chroot "$ROOT" /bin/sh -c 'for p in /opt/conda/bin/python /usr/local/bin/python /usr/bin/python3 /usr/bin/python; do [ -x "$p" ] && { echo "$p"; break; }; done')
echo "  镜像内解释器：${PY:-<找不到！>}"
[ -n "${PY:-}" ] || exit 1

echo
echo "############ 3. 以最终运行身份 appuser(uid 1000) 审计 /app 每一个文件 ############"
cp -f "$SELF/verify_container_scan.py" "$ROOT/tmp/_vscan.py"
chroot --userspec=1000:1000 "$ROOT" /bin/sh -c \
  "cd /app && PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/app/.hf $PY /tmp/_vscan.py"

echo
echo "############ 4. 以 appuser 启动 server.py，看三探针与 HEALTHCHECK ###########"
PORT=8080
chroot --userspec=1000:1000 "$ROOT" /bin/sh -c \
  "cd /app && PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/app/.hf PORT=$PORT timeout 90 $PY server.py" \
  > "$WORK/server.log" 2>&1 &
BG=$!
sleep 25
for p in liveness startup readiness; do
  printf "  /health/%-10s HTTP %s\n" "$p" \
    "$(curl -s -m 5 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health/$p)"
done
echo -n "  /debug/info     "; curl -s -m 5 http://127.0.0.1:$PORT/debug/info; echo
echo -n "  POST /predict   "; curl -s -m 5 -X POST -H 'Content-Type: application/json' \
  -d '{"req_id":"t1","data":"aGVsbG8="}' http://127.0.0.1:$PORT/predict; echo
echo -n "  HEALTHCHECK 那条 curl（应 rc=0）: "
chroot "$ROOT" /bin/sh -c "curl -fsS http://127.0.0.1:$PORT/health/liveness >/dev/null && echo rc=0 || echo rc=\$?"
wait $BG 2>/dev/null
echo "  --- server 日志 ---"; sed 's/^/    /' "$WORK/server.log" | tail -30

echo
echo "############ 结论怎么读 ############"
cat <<'TXT'
  · 第 3 步必须 `UNREADABLE 0` —— 否则就是事故 1（PermissionError）复发，**绝不能交付**。
  · 无 GPU 时第 4 步的 startup/readiness 应为 503，且 server 日志里**只有一条**
        RuntimeError: 要求 GPU 但 torch.cuda.is_available()=False
    若出现 PermissionError / ModuleNotFoundError，说明还有别的问题。
  · 有 GPU 时 startup/readiness 应转为 200，`/predict` 返回 {"code":0,"score":…}。
TXT
