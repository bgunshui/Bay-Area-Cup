#!/usr/bin/env python3
"""v7: 14B_p3 极限权重扫描(无 llm),含纯 14B 与 14B+微量的组合。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"

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

ft_o = np.zeros(len(y)); ft_r = np.zeros(NREP)
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20,
             "lora7b_official_rt": .10}.items():
    ft_o += w * rank01(o_raw(t)); ft_r += w * rank01(r_raw(t))
p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15o = np.mean([rank01(o_raw(t)) for t in p15n], axis=0); m15r = np.mean([rank01(r_raw(t)) for t in p15n], axis=0)
m7o = np.mean([rank01(o_raw(t)) for t in p7n], axis=0); m7r = np.mean([rank01(r_raw(t)) for t in p7n], axis=0)
p14o, p14r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))

base_o = 0.15*ft_o + 0.33*m15o + 0.27*m7o
base_r = 0.15*ft_r + 0.33*m15r + 0.27*m7r

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

plan = {"v7_w70": 0.70, "v7_w80": 0.80, "v7_w90": 0.90, "v7_w100": 1.00}
print("=== 14B_p3 极限权重(无 llm)===")
for lab, w14 in plan.items():
    if w14 >= 1.0:
        o, r_ = p14o, p14r
    else:
        s = (1 - w14) / 0.75
        o, r_ = s*base_o + w14*p14o, s*base_r + w14*p14r
    final = rank01(r_)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{lab}.csv", index=False)
    print(f"  {lab} (14B={w14:.2f}): OOF {auc(y, o):.6f} unique {np.unique(final).size}")

# 14B_p3 + 仅 7B(最强搭档)
for lab, w14 in {"v7_14b_7b_w60": 0.60, "v7_14b_7b_w70": 0.70}.items():
    s = 1 - w14
    o = w14*p14o + s*m7o; r_ = w14*p14r + s*m7r
    final = rank01(r_)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{lab}.csv", index=False)
    print(f"  {lab} (14B={w14:.2f}+7B): OOF {auc(y, o):.6f} unique {np.unique(final).size}")
print("V7 DONE")
