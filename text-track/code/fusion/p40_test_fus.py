#!/usr/bin/env python3
"""Test fus_lgb as new branch: correlation with existing families + additive OOF gain.

Quick checks:
 - OOF AUC 0.9946 vs families
 - spearman with ft/llm family scores
 - add fus_lgb as a 4th component: ft*.70 + llm*.25 + fus*.05 (small), and ft*.65+llm*.25+fus*.10
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

y = np.load(RUNS / "labels.npy").astype(np.int64)
def roc_auc_fast(y, scores):
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    tied = np.sort(scores)
    uniq, counts = np.unique(tied, return_counts=True)
    if (counts > 1).any():
        start = 0
        for u, c in zip(uniq, counts, strict=True):
            if c > 1:
                ranks[(scores == u)] = (start + 1 + start + c) / 2.0
            start += c
    pos = np.flatnonzero(y == 1)
    s = ranks[pos].sum()
    n1 = len(pos); n0 = len(y) - n1
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)

z = np.load(RUNS / "fusion_mat.npz")
names = list(z["names"])
rep = np.asarray(z["rep_mat"], dtype=np.float64)
idx = {n: i for i, n in enumerate(names)}

ft_fixed = {"lora1.5b_distill_r2": .25, "lora14b_official_v2": .25, "lora7b_distill_r2": .25, "lora7b_official": .25}
ft = np.zeros(rep.shape[0])
for t, w in ft_fixed.items():
    ft += w * rep[:, idx[t]]
llm = np.zeros(rep.shape[0])
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    llm += w * rep[:, idx[t]]

fus_oof = np.load(RUNS / "fus_lgb_oof.npy")
fus_rep = rank01(np.load(RUNS / "fus_lgb_replay.npy"))
print("fus_lgb OOF AUC", round(float(roc_auc_fast(y, fus_oof)), 6))
# oof correlations
ft_o = np.zeros(len(y)); llm_o = np.zeros(len(y))
for t, w in ft_fixed.items():
    ft_o += w * np.asarray(z["oof_mat"])[:, idx[t]]
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    if t in idx:
        llm_o += w * np.asarray(z["oof_mat"])[:, idx[t]]
print("oof spearman ft vs fus", round(float(spearmanr(ft_o, fus_oof)[0]), 3))
print("oof spearman llm vs fus", round(float(spearmanr(llm_o, fus_oof)[0]), 3))

# additive test on OOF: ft*.75+llm*.25 baseline vs with fus
base_o = 0.75 * ft_o + 0.25 * llm_o
print("base ft.75/llm.25 OOF", round(float(roc_auc_fast(y, base_o)), 6))
for wf, wl, wf2 in [(0.70, 0.25, 0.05), (0.65, 0.25, 0.10), (0.60, 0.25, 0.15), (0.75, 0.20, 0.05)]:
    o = wf * ft_o + wl * llm_o + wf2 * fus_oof
    print(f"ft {wf} llm {wl} fus {wf2} OOF", round(float(roc_auc_fast(y, o)), 6))
