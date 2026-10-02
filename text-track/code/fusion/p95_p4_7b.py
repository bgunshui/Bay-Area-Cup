#!/usr/bin/env python3
"""分析:7B 三个种子(r3a/r3b/p3s*) 与 14B p4 的相关性 + 生成混合候选。"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"
P4DIR = RUNS / "backup_14b_p4"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def load(tag, dirs):
    for d in dirs:
        p = Path(d) / f"{tag}_replay.npy"
        if p.exists():
            return rank01(np.load(p))
    return None

p4 = load("lora14b_p4", [P4DIR])
print(f"p4 loaded: {p4 is not None}")

# 7B seeds
b7 = {}
for t in ["lora7b_distill_r3a", "lora7b_distill_r3b", "lora7b_distill_p3s1",
          "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2", "lora7b_distill_r2"]:
    r = load(t, [RUNS, P4DIR, RUNS / "backup_14b_p4"])
    if r is not None:
        b7[t] = r

print("\n=== 7B 分支与 p4 的相关性 ===")
for t, r in sorted(b7.items(), key=lambda kv: spearmanr(kv[1], p4)[0]):
    print(f"  {t:28s} corr={spearmanr(r,p4)[0]:.4f}  unique={np.unique(r).size}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

# 7B ensemble (3 seeds: r3a, r3b, p3s1)
three = [k for k in ["lora7b_distill_r3a", "lora7b_distill_r3b", "lora7b_distill_p3s1"] if k in b7]
if len(three) >= 2:
    ens7 = np.mean([b7[k] for k in three], axis=0)
    print(f"\n=== 7B 三 seed 集成({len(three)} 个: {three}) ===")
    print(f"  corr(7b_ens, p4) = {spearmanr(ens7, p4)[0]:.4f}")
    print("\n=== 生成 p4 + 7B集成 候选 ===")
    for w in (0.05, 0.10, 0.15, 0.20):
        lab = f"submit_p4_{int((1-w)*100):02d}_7b3_{int(w*100):02d}"
        final = rank01((1-w)*p4 + w*ens7)
        pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{lab}.csv", index=False)
        print(f"  {lab}: unique={np.unique(final).size}")
    # single best 7B (r3a)
    if "lora7b_distill_r3a" in b7:
        for w in (0.05, 0.10):
            lab = f"submit_p4_{int((1-w)*100):02d}_7br3a_{int(w*100):02d}"
            final = rank01((1-w)*p4 + w*b7["lora7b_distill_r3a"])
            pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{lab}.csv", index=False)
            print(f"  {lab}: unique={np.unique(final).size}")
print("DONE")
