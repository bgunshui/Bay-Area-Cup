#!/usr/bin/env python3
"""生成第 4 轮伪标签 pseudo_best_v4(来自 triple 14B 集成 = p3 + p4 + p5)。
同时导出 triple 14B 的提交文件。
运行位置:新服务器 /root/sj-tmp/wanqu
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit

RUNS = "/root/sj-tmp/wanqu/runs"
DATA = "/root/sj-tmp/wanqu/data"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

seeds = []
for tag in ["lora14b_p3", "lora14b_p4", "lora14b_p5"]:
    p = os.path.join(RUNS, f"{tag}_replay.npy")
    if not os.path.exists(p):
        print(f"MISSING {p}"); sys.exit(1)
    r = rank01(np.load(p))
    seeds.append(r)
    print(f"  loaded {tag}: unique {np.unique(r).size}")

triple = np.mean(seeds, axis=0)
print(f"triple ensemble: {len(seeds)} seeds, unique {np.unique(triple).size}")

# submission file (rank01 of the mean)
final = rank01(triple)
test = pd.read_csv(os.path.join(DATA, "text_test.csv"), encoding="utf-8-sig")
out = os.path.join(RUNS, "..", "submissions_triple14b.csv")
pd.DataFrame({"id": test.id.astype(str), "score": final}).to_csv(out, index=False)
print(f"saved triple submission -> {out}")

# pseudo labels v4
soft = expit(4.0 * (final - 0.5))
np.save(os.path.join(RUNS, "pseudo_best_v4.npy"), soft)
print(f"pseudo_best_v4: mean {soft.mean():.4f} min {soft.min():.4f} max {soft.max():.4f}")
print("MK_PSEUDO_V4 DONE")
