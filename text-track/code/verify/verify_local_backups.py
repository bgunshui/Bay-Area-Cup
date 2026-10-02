"""核对本地备份是否覆盖服务器上的资产(删之前必须确认)。"""
import os
import sys
import tarfile

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
W = r"D:\湾区杯\文本赛道"
TR = os.path.join(W, "transfer")
DS = os.path.join(W, "docker_submit")


def list_tar(p, filt=None, limit=40):
    if not os.path.exists(p):
        print("  [缺失] " + p)
        return []
    names = []
    try:
        with tarfile.open(p, "r:*") as t:
            for m in t:
                n = m.name
                if filt and not filt(n):
                    continue
                names.append((n, m.size))
    except Exception as e:
        print("  [读取失败 %s] %s" % (p, e))
        return []
    print("  %s  成员命中 %d 个" % (os.path.basename(p), len(names)))
    for n, s in names[:limit]:
        print("      %-70s %10d" % (n, s))
    if len(names) > limit:
        print("      ...(共 %d)" % len(names))
    return names


print("=" * 78)
print("1) 14B p4 五折适配器 —— 本地备份")
print("=" * 78)
n1 = list_tar(os.path.join(DS, "adapters_p4.tar.gz"),
              lambda n: n.endswith("adapter_model.safetensors") or n.endswith("adapter_config.json"))
n1b = list_tar(os.path.join(TR, "adapters_p4.tar.gz"),
               lambda n: n.endswith("adapter_model.safetensors"))

print()
print("=" * 78)
print("2) 7B r3a/r3b 十折适配器 —— 本地备份")
print("=" * 78)
n2 = list_tar(os.path.join(TR, "7b_r3_adapters.tar.gz"),
              lambda n: n.endswith("adapter_model.safetensors") or n.endswith("adapter_config.json"))

print()
print("=" * 78)
print("3) 其它适配器归档")
print("=" * 78)
for f in ("v2_adapters.tar.gz", "p2_adapters.tar.gz"):
    list_tar(os.path.join(TR, f), lambda n: n.endswith("adapter_model.safetensors"), limit=25)

print()
print("=" * 78)
print("4) 本地各 backup 目录内容")
print("=" * 78)
for d in ("14b_p3_backup", "14b_p4_backup", "14b_p5_backup", "r3_backup", "p76_backup", "runs_full_backup"):
    p = os.path.join(TR, d)
    if not os.path.isdir(p):
        print("  [缺失] " + d)
        continue
    fs = sorted(os.listdir(p))
    print("  %-20s %3d 个: %s" % (d, len(fs), ", ".join(fs[:8]) + (" ..." if len(fs) > 8 else "")))

print()
print("=" * 78)
print("5) 关键单文件")
print("=" * 78)
for rel in ("transfer/text_test.csv", "transfer/text_sample_data.csv", "transfer/labels.npy",
            "transfer/folds.npy", "transfer/pseudo_best.npy",
            "submissions/submit_p4_90_7b2_10.csv", "submissions/submit_14b_p4_only.csv",
            "p97_p4_7b_sweep.py", "p101_verify_7b2.py"):
    p = os.path.join(W, rel)
    print("  %-46s %s" % (rel, ("%d bytes" % os.path.getsize(p)) if os.path.exists(p) else "[缺失]"))

print()
print("=" * 78)
print("6) 服务器上有、本地是否也有:wanqu 曾含内容对照")
print("=" * 78)
runs = os.path.join(TR, "runs_full_backup")
if os.path.isdir(runs):
    fs = set(os.listdir(runs))
    want = []
    for tag in ("lora14b_p3", "lora14b_p4", "lora14b_p5"):
        for k in ("_oof.npy", "_replay.npy", "_replay_raw_mean.npy"):
            want.append(tag + k)
        for f in range(5):
            for k in ("_oof.npy", "_raw.npy", "_replay.npy"):
                want.append("%s_fold%d%s" % (tag, f, k))
    hit = [w for w in want if w in fs]
    miss = [w for w in want if w not in fs]
    print("  期望 %d 项,本地 runs_full_backup 命中 %d,缺 %d" % (len(want), len(hit), len(miss)))
    if miss:
        print("  缺:", miss)
