#!/usr/bin/env python3
"""对比 v3A 结构下:原版 lora7b_official vs 重训版 lora7b_official_rt。"""
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

off_o, off_r = o("lora7b_official"), r("lora7b_official")
rt_o, rt_r = o("lora7b_official_rt"), r("lora7b_official_rt")
print("lora7b_official OOF:", round(auc(y, off_o), 6))
print("lora7b_official_rt OOF:", round(auc(y, rt_o), 6))
print("corr(off, rt):", round(spearmanr(off_o, rt_o)[0], 4))

def build_ft(use_rt):
    v_o = np.zeros(len(y)); v_r = np.zeros(rp_.shape[0])
    for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20}.items():
        v_o += w * ro[:, ix[t]]; v_r += w * rp_[:, ix[t]]
    if use_rt:
        v_o += 0.10 * rt_o; v_r += 0.10 * rt_r
    else:
        v_o += 0.10 * off_o; v_r += 0.10 * off_r
    return v_o, v_r

llm_o = np.zeros(len(y)); llm_r = np.zeros(rp_.shape[0])
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    llm_o += w * ro[:, ix[t]]; llm_r += w * rp_[:, ix[t]]

p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15o = np.mean([o(t) for t in p15n], axis=0); m7o = np.mean([o(t) for t in p7n], axis=0)
m15r = np.mean([r(t) for t in p15n], axis=0); m7r = np.mean([r(t) for t in p7n], axis=0)
p14o, p14r = o("lora14b_p3"), r("lora14b_p3")

for label, use_rt in (("原版 official", False), ("重训版 official_rt", True)):
    fo, fr = build_ft(use_rt)
    v = 0.12*fo + 0.08*llm_o + 0.30*m15o + 0.25*m7o + 0.25*p14o
    print(f"  v3A with {label:22s} OOF {auc(y, v):.6f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, sc):
    final = rank01(sc); out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique {int(np.unique(final).size)}")

fo_rt, fr_rt = build_ft(True)
print("\n=== 生成重训版候选 ===")
emit("submit_v3A_rt", 0.12*fr_rt + 0.08*llm_r + 0.30*m15r + 0.25*m7r + 0.25*p14r)
emit("submit_v3B_rt", 0.15*fr_rt + 0.33*m15r + 0.27*m7r + 0.25*p14r)
print("RT VARIANTS DONE")
