#!/usr/bin/env python3
"""全重跑版候选:ft 用 official_rt,llm 用新版 q4b_hidden + 导出的 qwen_pair,其余用 runs/ 最新。"""
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

z = np.load(RUNS / "fusion_mat.npz")
names = list(z["names"]); ro = np.asarray(z["oof_mat"], dtype=np.float64); rp_ = np.asarray(z["rep_mat"], dtype=np.float64)
ix = {n: i for i, n in enumerate(names)}

def o_raw(t):
    p = RUNS / f"{t}_oof.npy"; return np.load(p).astype(np.float64) if p.exists() else None
def r_raw(t):
    p = RUNS / f"{t}_replay.npy"; return np.load(p).astype(np.float64) if p.exists() else None

# ALL-RERUN components: use runs/ direct + rank01 (recomputed versions)
def comp(t, from_mat=False):
    """return (oof_rank, replay_rank)"""
    if from_mat and t in ix:
        return ro[:, ix[t]], rp_[:, ix[t]]
    a, b = o_raw(t), r_raw(t)
    if a is None or b is None:
        return None, None
    return rank01(a), rank01(b)

ft_o = np.zeros(len(y)); ft_r = np.zeros(NREP)
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20,
             "lora7b_official_rt": .10}.items():
    a, b = comp(t, from_mat=False)
    if a is None:
        a, b = comp(t, from_mat=True)
    ft_o += w * a; ft_r += w * b

llm_o = np.zeros(len(y)); llm_r = np.zeros(NREP)
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    a, b = comp(t, from_mat=False)
    if a is None:
        a, b = comp(t, from_mat=True)
    llm_o += w * a; llm_r += w * b

p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora1.5b_p2"]
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3", "lora7b_p2"]
m15o = np.mean([rank01(o_raw(t)) for t in p15n], axis=0)
m15r = np.mean([rank01(r_raw(t)) for t in p15n], axis=0)
m7o = np.mean([rank01(o_raw(t)) for t in p7n], axis=0)
m7r = np.mean([rank01(r_raw(t)) for t in p7n], axis=0)
p14o, p14r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))

print("=== 全重跑版 单分支 OOF ===")
for t in ["lora7b_official_rt", "q4b_hidden_mean", "q4b_hidden_last", "qwen_pair", "lora14b_p3"]:
    a = o_raw(t)
    print(f"  {t}: {auc(y, a):.6f}" if a is not None else f"  {t}: n/a")

schemes = {
    "A 全量": 0.12*ft_o + 0.08*llm_o + 0.30*m15o + 0.25*m7o + 0.25*p14o,
    "B 无llm": 0.15*ft_o + 0.33*m15o + 0.27*m7o + 0.25*p14o,
    "C 仅自训练": 0.35*m15o + 0.30*m7o + 0.35*p14o,
    "F 14B重权": 0.20*ft_o + 0.10*llm_o + 0.22*m15o + 0.18*m7o + 0.30*p14o,
}
print("\n=== 全重跑版 OOF ===")
for k, v in schemes.items():
    print(f"  {k:12s} {auc(y, v):.6f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, sc):
    final = rank01(sc); out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label}: unique {int(np.unique(final).size)}")
print("\n=== 生成(全重跑版) ===")
emit("submit_v4A_allrerun", 0.12*ft_r + 0.08*llm_r + 0.30*m15r + 0.25*m7r + 0.25*p14r)
emit("submit_v4B_allrerun_nollm", 0.15*ft_r + 0.33*m15r + 0.27*m7r + 0.25*p14r)
emit("submit_v4F_allrerun_14b", 0.20*ft_r + 0.10*llm_r + 0.22*m15r + 0.18*m7r + 0.30*p14r)
print("ALL-RERUN CANDIDATES DONE")
