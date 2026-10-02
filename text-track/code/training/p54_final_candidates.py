#!/usr/bin/env python3
"""全量 p3 融合预演 + 候选生成(7B: p3s1/s2/s3 + p2; 1.5B: p3s1/s2 + p2; ft + llm)"""
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

z = np.load(RUNS / "fusion_mat.npz")
names = list(z["names"]); ro = np.asarray(z["oof_mat"], dtype=np.float64); rp_ = np.asarray(z["rep_mat"], dtype=np.float64)
ix = {n: i for i, n in enumerate(names)}
def o(t):
    p = RUNS / f"{t}_oof.npy"; return np.load(p).astype(np.float64) if p.exists() else None
def r(t):
    p = RUNS / f"{t}_replay.npy"; return rank01(np.load(p)) if p.exists() else None

# families
llm = np.zeros(len(y)); llm_r = np.zeros(rp_.shape[0])
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    if t in ix:
        llm += w * ro[:, ix[t]]; llm_r += w * rp_[:, ix[t]]
ft = np.zeros(len(y)); ft_r = np.zeros(rp_.shape[0])
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20,
             "lora7b_official_rt": .10}.items():
    if t in ix:
        ft += w * ro[:, ix[t]]; ft_r += w * rp_[:, ix[t]]
ft_slim = np.zeros(len(y)); ft_slim_r = np.zeros(rp_.shape[0])
for t, w in {"lora1.5b_distill_r2": .65, "lora14b_official_v2": .15, "lora7b_distill_r2": .20}.items():
    if t in ix:
        ft_slim += w * ro[:, ix[t]]; ft_slim_r += w * rp_[:, ix[t]]

p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
p15 = [o(t) for t in p15n if o(t) is not None]; p7 = [o(t) for t in p7n if o(t) is not None]
p15r = [r(t) for t in p15n if r(t) is not None]; p7r = [r(t) for t in p7n if r(t) is not None]
m15 = np.mean(p15, axis=0); m7 = np.mean(p7, axis=0)
m15r = np.mean(p15r, axis=0); m7r = np.mean(p7r, axis=0)
print(f"1.5B models={len(p15)} | 7B models={len(p7)} | official_rt present={'lora7b_official_rt' in ix}")

print("\n=== OOF 对比 ===")
combos = {
    "全量(ft+llm+15b+7b)": 0.15*ft + 0.10*llm + 0.40*m15 + 0.35*m7,
    "去llm(ft+15b+7b)": 0.17*ft + 0.45*m15 + 0.38*m7,
    "slim ft(去llm去official)": 0.17*ft_slim + 0.45*m15 + 0.38*m7,
    "仅15b系": m15,
    "仅7b系": m7,
    "15b+7b均权": 0.5*m15 + 0.5*m7,
}
for k, v in combos.items():
    print(f"  {k:28s} OOF {auc(y, v):.6f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, sc):
    final = rank01(sc); out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique {int(np.unique(final).size)}")
print("\n=== 生成候选 ===")
emit("submit_final_full", 0.15*ft_r + 0.10*llm_r + 0.40*m15r + 0.35*m7r)
emit("submit_final_nollm", 0.17*ft_r + 0.45*m15r + 0.38*m7r)
emit("submit_final_slim", 0.17*ft_slim_r + 0.45*m15r + 0.38*m7r)
emit("submit_final_only7b", m7r)
emit("submit_final_only15b", m15r)
print("FINAL CANDIDATES DONE")
