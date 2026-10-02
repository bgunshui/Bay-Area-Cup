#!/usr/bin/env python3
"""v8: 双 14B seed 集成候选(在旧服务器生成;p4 的 npy 已同步到 backup_14b_p4/)。
14B_dual = mean(rank01(p3), rank01(p4))
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

y = np.load(RUNS / "labels.npy").astype(np.int64)
NREP = 20000
def auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1, dtype=np.float64)
    t = np.sort(s); u, c = np.unique(t, return_counts=True)
    if (c > 1).any():
        st = 0
        for uu, cc in zip(u, c, strict=True):
            if cc > 1: r[(s == uu)] = (st + 1 + st + cc) / 2.0
            st += cc
    p = np.flatnonzero(y == 1); ss = r[p].sum(); n1 = len(p); n0 = len(y) - n1
    return (ss - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def o_raw(t):
    p = RUNS / f"{t}_oof.npy"; return np.load(p).astype(np.float64) if p.exists() else None
def r_raw(t):
    p = RUNS / f"{t}_replay.npy"; return np.load(p).astype(np.float64) if p.exists() else None

# 14B seeds
p3o, p3r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))
o4 = P4 / "lora14b_p4_oof.npy"; r4 = P4 / "lora14b_p4_replay.npy"
if not o4.exists() or not r4.exists():
    print("P4 FILES MISSING:", o4.exists(), r4.exists()); raise SystemExit
p4o, p4r = rank01(np.load(o4)), rank01(np.load(r4))
print(f"p3 OOF {auc(y, p3o):.6f} | p4 OOF {auc(y, p4o):.6f}")
print(f"corr(p3,p4) oof={spearmanr(p3o, p4o)[0]:.4f} replay={spearmanr(p3r, p4r)[0]:.4f}")
dual_o = 0.5 * (p3o + p4o); dual_r = 0.5 * (p3r + p4r)
print(f"dual OOF {auc(y, dual_o):.6f}")

# other families (same as v6 structure)
ft_o = np.zeros(len(y)); ft_r = np.zeros(NREP)
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20,
             "lora7b_official_rt": .10}.items():
    ft_o += w * rank01(o_raw(t)); ft_r += w * rank01(r_raw(t))
p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15o = np.mean([rank01(o_raw(t)) for t in p15n], axis=0); m15r = np.mean([rank01(r_raw(t)) for t in p15n], axis=0)
m7o = np.mean([rank01(o_raw(t)) for t in p7n], axis=0); m7r = np.mean([rank01(r_raw(t)) for t in p7n], axis=0)

base_o = 0.15*ft_o + 0.33*m15o + 0.27*m7o
base_r = 0.15*ft_r + 0.33*m15r + 0.27*m7r

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

print("\n=== 双 seed 集成(v8)权重扫描 ===")
plan = {"v8_dual_w45": 0.45, "v8_dual_w55": 0.55, "v8_dual_w65": 0.65, "v8_dual_w80": 0.80}
for lab, w14 in plan.items():
    s = (1 - w14) / 0.75
    o = s*base_o + w14*dual_o; r_ = s*base_r + w14*dual_r
    final = rank01(r_)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{lab}.csv", index=False)
    print(f"  {lab} (dual={w14:.2f}): OOF {auc(y, o):.6f} unique {np.unique(final).size}")

# pure dual
final = rank01(dual_r)
pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / "submit_v8_dual_only.csv", index=False)
print(f"  v8_dual_only: OOF {auc(y, dual_o):.6f} unique {np.unique(final).size}")

# three-model: p3 + p4 + 7B best
for lab, w in {"v8_dual7b_w60": 0.60, "v8_dual7b_w70": 0.70}.items():
    o = w*dual_o + (1-w)*m7o; r_ = w*dual_r + (1-w)*m7r
    final = rank01(r_)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{lab}.csv", index=False)
    print(f"  {lab}: OOF {auc(y, o):.6f} unique {np.unique(final).size}")
print("V8 DONE")
