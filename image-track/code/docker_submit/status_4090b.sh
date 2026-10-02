echo "===== gpu4090b BUILD STATUS $(date -u +%H:%M:%S) UTC ====="
echo "[内存] max=$(cat /sys/fs/cgroup/memory.max 2>/dev/null) current=$(cat /sys/fs/cgroup/memory.current 2>/dev/null)"
echo "[磁盘 /hy-tmp] $(df -h /hy-tmp | tail -1 | awk '{print $4" 可用 ("$5" 已用)"}')   [磁盘 /] $(df -h / | tail -1 | awk '{print $4" 可用"}')"
echo "[GPU] $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null | head -1)"
echo
echo "[流水线进程]"
ps -eo pid,etimes,rss,cmd 2>/dev/null | grep -E "run_final|oci_build3|bb_fp16|fix_openclip|fix_bigg|skopeo|pip " | grep -v grep | head -5 | cut -c1-110 || echo "  无"
echo
echo "[run_final.log 尾部]"; tail -14 /hy-tmp/build/run_final.log 2>/dev/null | sed 's/^/    /' || echo "    (还没生成)"
echo
echo "[基座拉取]"; tail -3 /hy-tmp/build/pull_pt.log 2>/dev/null | sed 's/^/    /'
echo "    base_oci_pt: $(du -sh /hy-tmp/build/base_oci_pt 2>/dev/null | cut -f1)"
echo
echo "[资产]"
for d in backbones/clip_h14_cache backbones/clip_bigg backbones/siglip2_384 base_oci_pt buildkit work/piproot work/layer2 final_oci; do
  printf "    %-28s %s\n" "$d" "$(du -sh /hy-tmp/build/$d 2>/dev/null | cut -f1)"
done
echo "    buildkit.tar.gz            $(du -sh /hy-tmp/build/buildkit.tar.gz 2>/dev/null | cut -f1)"
echo
echo "[成果物]"; ls -la /hy-tmp/build/*.tar 2>/dev/null || echo "    image tar：还没有"
echo
echo "[buildkit 上传]"; ls -la /hy-tmp/build/buildkit.tar.gz 2>/dev/null
echo "===== END ====="
