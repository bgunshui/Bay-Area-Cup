#!/usr/bin/env python3
"""本地分析:p4 vs 7B 种子 相关性 + 生成混合候选(不依赖服务器)。"""
import sys, os, glob
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TR = r"D:\湾区杯\文本赛道\transfer"
SUB = r"D:\湾区杯\文本赛道\submissions"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def load_replay(dirs, tag):
    for d in dirs:
        p = os.path.join(d, f"{tag}_replay.npy")
        if os.path.exists(p):
            return rank01(np.load(p))
    return None

p4 = load_replay([os.path.join(TR, "14b_p4_backup")], "lora14b_p4")
r3a = load_replay([os.path.join(TR, "r3_backup")], "lora7b_distill_r3a")
r3b = load_replay([os.path.join(TR, "p76_backup"), os.path.join(TR, "r3_backup")], "lora7b_distill_r3b")
print("loaded:", {"p4": p4 is not None, "r3a": r3a is not None, "r3b": r3b is not None})
if p4 is None:
    print("p4 missing"); sys.exit(1)

print("\n=== 相关性(与 p4)===")
for name, r in [("7b r3a", r3a), ("7b r3b", r3b)]:
    if r is not None:
        print(f"  {name}: corr={spearmanr(r, p4)[0]:.4f}  unique={np.unique(r).size}")

# 7B ensemble
ens7 = None
parts = [x for x in [r3a, r3b] if x is not None]
if len(parts) >= 1:
    ens7 = np.mean(parts, axis=0)
    print(f"\n7B ensemble({len(parts)} seeds): corr with p4 = {spearmanr(ens7, p4)[0]:.4f}")

# get id column from an existing submission
sample = os.path.join(SUB, "submit_14b_p4_only.csv")
df_id = pd.read_csv(sample, encoding="utf-8-sig")
ids = df_id["id"].astype(str).tolist()
print(f"\nid column from {os.path.basename(sample)}: {len(ids)} rows")

def emit(label, score):
    final = rank01(score)
    out = os.path.join(SUB, f"{label}.csv")
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique={np.unique(final).size}")

print("\n=== 生成混合候选 ===")
if ens7 is not None:
    for w in (0.05, 0.10, 0.15, 0.20):
        emit(f"submit_p4_{int((1-w)*100):02d}_7b2_{int(w*100):02d}", (1-w)*p4 + w*ens7)
if r3a is not None:
    for w in (0.05, 0.10):
        emit(f"submit_p4_{int((1-w)*100):02d}_7br3a_{int(w*100):02d}", (1-w)*p4 + w*r3a)
print("DONE")
