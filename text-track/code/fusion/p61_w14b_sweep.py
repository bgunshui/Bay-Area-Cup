#!/usr/bin/env python3
"""14B_p3 权重扫描:0.25(当前) → 0.30/0.35/0.40/0.50,其余按比例缩减。"""
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

# components (all-rerun versions)
ft_o = np.zeros(len(y)); ft_r = np.zeros(NREP)
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20,
             "lora7b_official_rt": .10}.items():
    ft_o += w * rank01(o_raw(t)); ft_r += w * rank01(r_raw(t))
llm_o = np.zeros(len(y)); llm_r = np.zeros(NREP)
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    llm_o += w * rank01(o_raw(t)); llm_r += w * rank01(r_raw(t))
p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15o = np.mean([rank01(o_raw(t)) for t in p15n], axis=0); m15r = np.mean([rank01(r_raw(t)) for t in p15n], axis=0)
m7o = np.mean([rank01(o_raw(t)) for t in p7n], axis=0); m7r = np.mean([rank01(r_raw(t)) for t in p7n], axis=0)
p14o, p14r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, sc):
    final = rank01(sc); out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique {int(np.unique(final).size)}")

# base weights (excluding 14B): ft .12, llm .08, 1.5B .30, 7B .25  => sum .75
base_o = 0.12*ft_o + 0.08*llm_o + 0.30*m15o + 0.25*m7o
base_r = 0.12*ft_r + 0.08*llm_r + 0.30*m15r + 0.25*m7r

print("=== 14B_p3 权重扫描 OOF ===")
plan = {"v5_w30": 0.30, "v5_w35": 0.35, "v5_w40": 0.40, "v5_w50": 0.50, "v5_w60": 0.60}
for lab, w14 in plan.items():
    scale = (1 - w14) / 0.75
    o = scale * base_o + w14 * p14o
    print(f"  14B={w14:.2f} (others x{scale:.3f})  OOF {auc(y, o):.6f}")

print("\n=== 生成候选 ===")
for lab, w14 in plan.items():
    scale = (1 - w14) / 0.75
    emit(f"submit_{lab}", scale * base_r + w14 * p14r)

# also: pure 14B + minimal others (deployable)
emit("submit_v5_14b_only", p14r)
emit("submit_v5_14b_7b", 0.5*p14r + 0.5*m7r)
print("WEIGHT SWEEP DONE")
