#!/bin/bash
# pull_pt_4090b.sh —— 在 gpu4090b 上拉 pytorch 官方基座（skopeo，多源重试）
set -u
R=/hy-tmp/build
LOG=$R/pull_pt.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=========== PULL PT BASE $(date -u +%H:%M:%S) ==========="
IMG=pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

for M in docker.1panel.live docker.1ms.run docker.xuanyuan.me docker.m.daocloud.io; do
  echo
  echo "--- 试 $M ---"
  if ! timeout 120 skopeo inspect --override-os linux --override-arch amd64 "docker://$M/$IMG" >/dev/null 2>&1; then
    echo "  ❌ inspect 失败"; continue
  fi
  echo "  ✅ 源可用，开始拉（约 7GB 压缩）"
  rm -rf "$R/base_oci_pt"
  t0=$(date +%s)
  if timeout 10800 skopeo copy --override-os linux --override-arch amd64 \
       "docker://$M/$IMG" "oci:$R/base_oci_pt:base"; then
    t1=$(date +%s); sz=$(du -sb "$R/base_oci_pt" 2>/dev/null | cut -f1)
    echo "  ✅ 完成 $(($t1-$t0))s  $((sz/1000000)) MB  平均 $((sz/1024/(($t1-$t0)+1))) KB/s"
    echo "PT_BASE_OK mirror=$M"
    break
  fi
  echo "  ❌ copy 失败：$(tail -1 $LOG)"
done

echo
du -sh "$R/base_oci_pt" 2>/dev/null || echo "  无"
df -h /hy-tmp | tail -1
python3 - <<'PY'
import json, pathlib
p = pathlib.Path("/hy-tmp/build/base_oci_pt/index.json")
if not p.exists():
    print("  无 index.json"); raise SystemExit
idx = json.loads(p.read_text()); m = idx["manifests"][0]
mf = json.loads((p.parent/"blobs"/"sha256"/m["digest"].split(":")[1]).read_text())
cfg = json.loads((p.parent/"blobs"/"sha256"/mf["config"]["digest"].split(":")[1]).read_text())
print(f"  Arch={cfg.get('architecture')} OS={cfg.get('os')} 层数={len(mf['layers'])}")
tot = sum((p.parent/'blobs'/'sha256'/L['digest'].split(':')[1]).stat().st_size for L in mf['layers'])
print(f"  压缩层合计 {tot/1e9:.2f} GB")
print(f"  Cmd={cfg.get('config',{}).get('Cmd')}")
PY
echo "=========== PULL PT BASE END $(date -u +%H:%M:%S) ==========="
