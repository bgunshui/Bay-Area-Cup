#!/usr/bin/env python3
"""P81: p3 版 v9 候选(用 p3 系替代 p2 系作为微量分支)。
同时给出 p2/p3/混合 三套对照。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"
P4 = RUNS / "backup_14b_p4"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def o_raw(t):
    p = RUNS / f"{t}_oof.npy"; return np.load(p).astype(np.float64) if p.exists() else None
def r_raw(t):
    p = RUNS / f"{t}_replay.npy"
    if p.exists():
        return np.load(p).astype(np.float64)
    p2 = P4 / f"{t}_replay.npy"
    return np.load(p2).astype(np.float64) if p2.exists() else None

# dual 14B
p3o, p3r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))
p4o, p4r = rank01(np.load(P4 / "lora14b_p4_oof.npy")), rank01(np.load(P4 / "lora14b_p4_replay.npy"))
dual_o, dual_r = 0.5*(p3o+p4o), 0.5*(p3r+p4r)

# p2 components
p7b2o, p7b2r = rank01(o_raw("lora7b_p2")), rank01(r_raw("lora7b_p2"))
p15b2o, p15b2r = rank01(o_raw("lora1.5b_p2")), rank01(r_raw("lora1.5b_p2"))
# p3 components (mean of seeds)
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3"]
p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2"]
p7p3o = np.mean([rank01(o_raw(t)) for t in p7n], axis=0); p7p3r = np.mean([rank01(r_raw(t)) for t in p7n], axis=0)
p15p3o = np.mean([rank01(o_raw(t)) for t in p15n], axis=0); p15p3r = np.mean([rank01(r_raw(t)) for t in p15n], axis=0)

print("=== 相关性(与 dual)===")
for nm, oo in [("7b_p2", p7b2o), ("1.5b_p2", p15b2o), ("7b_p3mean", p7p3o), ("1.5b_p3mean", p15p3o)]:
    print(f"  {nm:12s} {spearmanr(dual_o, oo)[0]:.4f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

def emit(label, so, sr):
    final = rank01(sr)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{label}.csv", index=False)
    print(f"  {label}: unique {np.unique(final).size}")

print("\n=== 生成 p3 版 + 混合版 ===")
# p3 versions at various dilution
for lab, w in [("v9_p3_98_1_1", 0.01), ("v9_p3_95_25_25", 0.025), ("v9_p3_90_5_5", 0.05), ("v9_p3_85_75_75", 0.075)]:
    so = (1-2*w)*dual_o + w*p7p3o + w*p15p3o
    sr = (1-2*w)*dual_r + w*p7p3r + w*p15p3r
    emit("submit_" + lab, so, sr)
# p2 versions for direct comparison
for lab, w in [("v9_p2_98_1_1", 0.01), ("v9_p2_90_5_5", 0.05)]:
    so = (1-2*w)*dual_o + w*p7b2o + w*p15b2o
    sr = (1-2*w)*dual_r + w*p7b2r + w*p15b2r
    emit("submit_" + lab, so, sr)
# mixed p2+p3
for lab, w in [("v9_mixP_90_5_5", 0.05), ("v9_mixP_95_25_25", 0.025)]:
    so = (1-2*w)*dual_o + w*0.5*(p7b2o+p7p3o) + w*0.5*(p15b2o+p15p3o)
    sr = (1-2*w)*dual_r + w*0.5*(p7b2r+p7p3r) + w*0.5*(p15b2r+p15p3r)
    emit("submit_" + lab, so, sr)
print("P81 DONE")
