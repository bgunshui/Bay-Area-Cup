#!/usr/bin/env python3
"""终极融合:14B_p3 + 7B系(s1/s2/s3) + 1.5B系(s1/s2) + ft + llm。
生成多套候选,含部署友好版本。
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

# components
def fam_o(d, fallback_ix=True):
    v = np.zeros(len(y))
    for t, w in d.items():
        if t in ix: v += w * ro[:, ix[t]]
        else:
            x = o(t)
            if x is not None: v += w * x
    return v
def fam_r(d):
    v = np.zeros(rp_.shape[0])
    for t, w in d.items():
        if t in ix: v += w * rp_[:, ix[t]]
        else:
            x = r(t)
            if x is not None: v += w * x
    return v

ft_d = {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20, "lora7b_official": .10}
llm_d = {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}
ft_o, ft_r = fam_o(ft_d), fam_r(ft_d)
llm_o, llm_r = fam_o(llm_d), fam_r(llm_d)

p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15o = np.mean([o(t) for t in p15n if o(t) is not None], axis=0)
m7o = np.mean([o(t) for t in p7n if o(t) is not None], axis=0)
m15r = np.mean([r(t) for t in p15n if r(t) is not None], axis=0)
m7r = np.mean([r(t) for t in p7n if r(t) is not None], axis=0)
p14o, p14r = o("lora14b_p3"), r("lora14b_p3")
print("14B_p3 OOF:", round(auc(y, p14o), 6))
print("corr(14B_p3, 7B mean):", round(spearmanr(p14o, m7o)[0], 3))
print("corr(14B_p3, 1.5B mean):", round(spearmanr(p14o, m15o)[0], 3))

schemes = {
    "A 全量": 0.12*ft_o + 0.08*llm_o + 0.30*m15o + 0.25*m7o + 0.25*p14o,
    "B 无llm": 0.15*ft_o + 0.33*m15o + 0.27*m7o + 0.25*p14o,
    "C 仅自训练系": 0.35*m15o + 0.30*m7o + 0.35*p14o,
    "D 14B_p3+7B": 0.55*p14o + 0.45*m7o,
    "E 14B_p3 alone": p14o,
    "F 14B_p3 heavy": 0.20*ft_o + 0.10*llm_o + 0.22*m15o + 0.18*m7o + 0.30*p14o,
}
print("\n=== OOF 对比 ===")
for k, v in schemes.items():
    print(f"  {k:16s} {auc(y, v):.6f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, sc):
    final = rank01(sc); out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique {int(np.unique(final).size)}")
print("\n=== 生成候选 ===")
emit("submit_v3A_full", 0.12*ft_r + 0.08*llm_r + 0.30*m15r + 0.25*m7r + 0.25*p14r)
emit("submit_v3B_nollm", 0.15*ft_r + 0.33*m15r + 0.27*m7r + 0.25*p14r)
emit("submit_v3C_selfonly", 0.35*m15r + 0.30*m7r + 0.35*p14r)
emit("submit_v3D_14b7b", 0.55*p14r + 0.45*m7r)
emit("submit_v3F_14bheavy", 0.20*ft_r + 0.10*llm_r + 0.22*m15r + 0.18*m7r + 0.30*p14r)
print("ULTIMATE CANDIDATES DONE")
