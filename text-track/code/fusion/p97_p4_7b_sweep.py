#!/usr/bin/env python3
"""p4 + 7B 权重精细扫描(基于 10% 有效的发现)。"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

TR = r"D:\湾区杯\文本赛道\transfer"
SUB = r"D:\湾区杯\文本赛道\submissions"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def load(d, tag):
    p = os.path.join(d, f"{tag}_replay.npy")
    return rank01(np.load(p)) if os.path.exists(p) else None

p4 = load(os.path.join(TR, "14b_p4_backup"), "lora14b_p4")
r3a = load(os.path.join(TR, "r3_backup"), "lora7b_distill_r3a")
r3b = load(os.path.join(TR, "p76_backup"), "lora7b_distill_r3b")
if r3b is None:
    r3b = load(os.path.join(TR, "r3_backup"), "lora7b_distill_r3b")
ens7 = np.mean([x for x in [r3a, r3b] if x is not None], axis=0)
print(f"7B ensemble: {sum(x is not None for x in [r3a,r3b])} seeds, corr(p4)={spearmanr(ens7,p4)[0]:.4f}")

ids = pd.read_csv(os.path.join(SUB, "submit_14b_p4_only.csv"), encoding="utf-8-sig")["id"].astype(str).tolist()

def emit(label, w):
    final = rank01((1-w)*p4 + w*ens7)
    pd.DataFrame({"id": ids, "score": final}).to_csv(os.path.join(SUB, f"{label}.csv"), index=False)
    print(f"  {label:34s} 7B={w:.3f} unique={np.unique(final).size}")

print("\n=== p4 + 7B集成 权重扫描 ===")
emit("submit_p4_975_7b2_025", 0.025)
emit("submit_p4_95_7b2_05", 0.05)
emit("submit_p4_925_7b2_075", 0.075)
emit("submit_p4_90_7b2_10", 0.10)
emit("submit_p4_875_7b2_125", 0.125)
emit("submit_p4_85_7b2_15", 0.15)
emit("submit_p4_825_7b2_175", 0.175)
emit("submit_p4_80_7b2_20", 0.20)
emit("submit_p4_75_7b2_25", 0.25)
emit("submit_p4_70_7b2_30", 0.30)
print("DONE")
