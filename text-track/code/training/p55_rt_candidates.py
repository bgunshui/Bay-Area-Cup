#!/usr/bin/env python3
"""用重训版 lora7b_official_rt(更强)替换原版,生成升级候选。"""
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

rt_o, rt_r = o("lora7b_official_rt"), r("lora7b_official_rt")
print("rt oof AUC:", round(auc(y, rt_o), 6) if rt_o is not None else None)

# ft with rt (replacing original official)
ft_o = np.zeros(len(y)); ft_r = np.zeros(rp_.shape[0])
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20}.items():
    if t in ix:
        ft_o += w * ro[:, ix[t]]; ft_r += w * rp_[:, ix[t]]
ft_o += 0.10 * rt_o; ft_r += 0.10 * rt_r

llm_o = np.zeros(len(y)); llm_r = np.zeros(rp_.shape[0])
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    if t in ix:
        llm_o += w * ro[:, ix[t]]; llm_r += w * rp_[:, ix[t]]

p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15 = np.mean([o(t) for t in p15n if o(t) is not None], axis=0)
m7 = np.mean([o(t) for t in p7n if o(t) is not None], axis=0)
m15r = np.mean([r(t) for t in p15n if r(t) is not None], axis=0)
m7r = np.mean([r(t) for t in p7n if r(t) is not None], axis=0)

combos = {
    "全量+rt": 0.15*ft_o + 0.10*llm_o + 0.40*m15 + 0.35*m7,
    "去llm+rt": 0.17*ft_o + 0.45*m15 + 0.38*m7,
}
print("\n=== 含 rt 的 OOF ===")
for k, v in combos.items():
    print(f"  {k:16s} {auc(y, v):.6f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, sc):
    final = rank01(sc); out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique {int(np.unique(final).size)}")
print("\n=== 生成 ===")
emit("submit_rt_full", 0.15*ft_r + 0.10*llm_r + 0.40*m15r + 0.35*m7r)
emit("submit_rt_nollm", 0.17*ft_r + 0.45*m15r + 0.38*m7r)
print("RT CANDIDATES DONE")
