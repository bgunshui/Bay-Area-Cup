#!/usr/bin/env python3
"""v6: 无 llm(部署友好)+ 14B_p3 高权重 组合扫描。
依据:v5_w40(14B .40) 0.979419 > v4A(.25) 0.979091;v4B(nollm) 0.979265 > v4A。
"""
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

# 无 llm 基础部分:v4B = ft .15 / 1.5B .33 / 7B .27  (sum .75, 14B .25)
base_o = 0.15*ft_o + 0.33*m15o + 0.27*m7o
base_r = 0.15*ft_r + 0.33*m15r + 0.27*m7r

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

print("=== 无 llm + 14B 权重扫描 (OOF) ===")
plan = {"v6_nollm_w35": 0.35, "v6_nollm_w45": 0.45, "v6_nollm_w55": 0.55, "v6_nollm_w65": 0.65}
results = {}
for lab, w14 in plan.items():
    s = (1 - w14) / 0.75
    o = s*base_o + w14*p14o
    r_ = s*base_r + w14*p14r
    results[lab] = (o, r_)
    print(f"  14B={w14:.2f}  OOF {auc(y, o):.6f}")

print("\n=== 生成候选 ===")
for lab, (o, r_) in results.items():
    final = rank01(r_)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{lab}.csv", index=False)
    print(f"  submit_{lab}: unique {np.unique(final).size}")

# extra: 14B extremely heavy, no llm
for lab, w14 in {"v6_nollm_w75": 0.75, "v6_nollm_w85": 0.85}.items():
    s = (1 - w14) / 0.75
    r_ = s*base_r + w14*p14r
    final = rank01(r_)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{lab}.csv", index=False)
    print(f"  submit_{lab}: unique {np.unique(final).size}")
print("V6 DONE")
