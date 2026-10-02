#!/bin/bash
# pull_pt.sh —— 用 skopeo 拉 **pytorch 官方基座**到 OCI 布局（`base_oci_pt/`）。
#
# 为什么单独存在这个文件：`README.md` 的「路径 B（没有 Docker daemon）」一节写的是
#   `bash pull_pt.sh`
# 而实际落地时那份脚本是照着某台机器的路径写的、名字叫 `pull_pt_4090b.sh`。
# 这里补一个同名的薄封装，让 README 里的命令**照抄就能跑**（文件名不一致是很容易踩的坑）。
#
# 默认基座：pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
#   · 支持多源重试（docker.1panel.live / docker.1ms.run / docker.xuanyuan.me / docker.m.daocloud.io）
#   · 输出 `$R/base_oci_pt` 与 `$R/pull_pt.log`
#   · 若你的机器没有 skopeo：`apt-get install -y skopeo`
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

if ! command -v skopeo >/dev/null 2>&1; then
  echo "❌ 没有 skopeo。安装：apt-get update && apt-get install -y skopeo"
  exit 1
fi

if [ -f "$HERE/pull_pt_4090b.sh" ]; then
  exec bash "$HERE/pull_pt_4090b.sh" "$@"
fi

# 兜底：没有那份脚本时，用同样的默认值直接拉
IMG="${IMG:-pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime}"
R="${R:-/hy-tmp/build}"
mkdir -p "$R"
for M in docker.1panel.live docker.1ms.run docker.xuanyuan.me docker.m.daocloud.io; do
  echo "--- 试 $M ---"
  if skopeo inspect --override-os linux --override-arch amd64 "docker://$M/$IMG" >/dev/null 2>&1; then
    echo "  ✅ 源可用，开始拉"
    rm -rf "$R/base_oci_pt"
    if skopeo copy --override-os linux --override-arch amd64 \
         "docker://$M/$IMG" "oci:$R/base_oci_pt:base"; then
      echo "PT_BASE_OK mirror=$M"; exit 0
    fi
  fi
  echo "  ❌ 失败，换下一个源"
done
echo "❌ 所有源都失败"; exit 1
