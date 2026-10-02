#!/bin/bash
# run_final.sh —— gpu4090b 上的最终流水线：等资材 → fp16 → safetensors 修 → OCI 组装 → 导出
set -u
R=/hy-tmp/build
LOG=$R/run_final.log
: > "$LOG"
exec > >(tee -a "$LOG") 2>&1
echo "=========== RUN FINAL $(date -u +%H:%M:%S) ==========="
echo "内存上限 $(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo 无限制)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1
df -h /hy-tmp | tail -1

echo
echo "--- 0) 等骨干齐 ---"
for i in $(seq 1 180); do
  ok=0
  [ -s "$R/backbones/clip_h14_cache/open_clip_pytorch_model.bin" ] && ok=$((ok+1))
  [ -s "$R/backbones/clip_bigg/pytorch_model-00002-of-00002.bin" ] && ok=$((ok+1))
  [ -s "$R/backbones/siglip2_384/model.safetensors" ] && ok=$((ok+1))
  [ "$ok" -ge 3 ] && { echo "  3/3 齐"; break; }
  echo "  齐 $ok/3 …"; sleep 20
done
du -sh $R/backbones/* 2>/dev/null

echo
echo "--- 1) 等 buildkit 上传完成 ---"
for i in $(seq 1 180); do
  sz=$(stat -c %s "$R/buildkit.tar.gz" 2>/dev/null || echo 0)
  [ "$sz" -ge 900000000 ] && break
  echo "  buildkit $((sz/1000000)) MB …"; sleep 20
done
ls -la $R/buildkit.tar.gz

echo
echo "--- 2) 等 base 镜像拉完 ---"
for i in $(seq 1 240); do
  [ -f "$R/base_oci_pt/index.json" ] && { echo "  base 就绪"; break; }
  echo "  base $(du -sh $R/base_oci_pt 2>/dev/null | cut -f1) …"; sleep 30
done
du -sh $R/base_oci_pt 2>/dev/null

echo
echo "--- 3) fp32 → fp16（18.4 → 9.3 GB）---"
python3 $R/bb_fp16.py

echo
echo "--- 4) 补 open_clip 的 safetensors（离线加载必需）---"
python3 $R/fix_openclip.py

echo
echo "--- 5) bigG 分片 .bin → model.safetensors ---"
python3 $R/fix_bigg_st.py

echo
echo "--- 6) 解包 buildkit + 用修好的 models.py 覆盖 ---"
rm -rf $R/buildkit; mkdir -p $R/buildkit
tar xzf $R/buildkit.tar.gz -C $R/buildkit
cp $R/models_fixed.py $R/buildkit/docker_submit/app/models.py
grep -c "img_cliph_rerun" $R/buildkit/docker_submit/app/models.py
df -h /hy-tmp | tail -1

echo
echo "--- 7) OCI 组装 + 导出 ---"
python3 $R/oci_build3.py

echo
echo "--- 8) 结果 ---"
ls -la $R/*.tar 2>/dev/null
df -h /hy-tmp | tail -1
echo "=========== RUN FINAL END $(date -u +%H:%M:%S) ==========="
