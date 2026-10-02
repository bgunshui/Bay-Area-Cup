#!/bin/bash
# final_go.sh —— 删掉没用的 dinov3_7b，然后点火最终流水线
set -u
echo "===== FINAL GO $(date -u +%H:%M:%S) ====="
echo "--- 删除前的磁盘 ---"; df -h /hy-tmp | tail -1

echo "--- 删 dinov3_7b（DINO 线已关闭，26 GB）---"
du -sh /hy-tmp/sjmodels/dinov3_7b 2>/dev/null
rm -rf /hy-tmp/sjmodels/dinov3_7b
echo "  已删"
du -sh /hy-tmp/sjmodels/* 2>/dev/null

echo "--- 顺带清掉确定没用的 ---"
rm -f /hy-tmp/_speedtest.pt /hy-tmp/pull_data.sh /hy-tmp/pip1.log 2>/dev/null
rm -rf /hy-tmp/archive_dead 2>/dev/null && echo "  已删 archive_dead"
echo "--- 删除后的磁盘 ---"; df -h /hy-tmp | tail -1

echo "--- 点火最终流水线 ---"
cd /hy-tmp/build
setsid nohup bash /hy-tmp/build/run_final.sh > /dev/null 2>&1 < /dev/null &
sleep 8
tail -6 /hy-tmp/build/run_final.log 2>/dev/null || echo "  (日志还没生成)"
echo "===== END ====="
