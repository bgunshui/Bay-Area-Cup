#!/usr/bin/env python3
"""p3 fusion preview: how much do the new self-trained p3 branches add on top of p2 structure?
Compares OOF of: p2-only structure vs p2+p3 variants.
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr

RUNS = Path("/data/湾区杯/文本赛道/runs")
y = np.load(RUNS / "labels.npy").astype(np.int64)

def roc_auc_fast(y, s):
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1, dtype=np.float64)
    t = np.sort(s); u, c = np.unique(t, return_counts=True)
    if (c > 1).any():
        st = 0
        for uu, cc in zip(u, c, strict=True):
            if cc > 1: r[(s == uu)] = (st + 1 + st + cc) / 2.0
            st += cc
    pos = np.flatnonzero(y == 1); ss = r[pos].sum(); n1 = len(pos); n0 = len(y) - n1
    return (ss - n1 * (n1 + 1) / 2.0) / (n1 * n0)

z = np.load(RUNS / "fusion_mat.npz")
names = list(z["names"]); ro = np.asarray(z["oof_mat"], dtype=np.float64)
idx = {n: i for i, n in enumerate(names)}

def oof_of(tag):
    p = RUNS / f"{tag}_oof.npy"
    return np.load(p).astype(np.float64) if p.exists() else None

# components
ft_comp = None
ftw = {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20, "lora7b_official": .10}
ft_comp = np.zeros(len(y))
for t, w in ftw.items():
    if t in idx: ft_comp += w * ro[:, idx[t]]
llm = np.zeros(len(y))
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    if t in idx: llm += w * ro[:, idx[t]]
p15 = oof_of("lora1.5b_p2"); p7 = oof_of("lora7b_p2")
available = {}
for t in ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2", "lora7b_distill_p3s1", "lora7b_distill_p3s2"]:
    o = oof_of(t)
    if o is not None:
        available[t] = o
        print(f"{t}: OOF AUC {roc_auc_fast(y, o):.6f}")

base = 0.15 * ft_comp + 0.10 * llm + 0.45 * p15 + 0.30 * p7
print(f"\nbaseline (=p2AD structure): {roc_auc_fast(y, base):.6f}")

# correlations of p3 vs baseline
for t, o in available.items():
    print(f"  corr(p3 {t}, baseline) = {spearmanr(o, base)[0]:.3f}")

# p3 ensemble (mean of available p3)
p3mean = np.mean(list(available.values()), axis=0)
print(f"\np3 mean ensemble OOF: {roc_auc_fast(y, p3mean):.6f}")
print(f"corr(p3mean, baseline) = {spearmanr(p3mean, base)[0]:.3f}")

print("\n=== add p3 ensemble to baseline ===")
for w in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
    o = (1 - w) * base + w * p3mean
    print(f"  +{w:.2f} p3mean -> {roc_auc_fast(y, o):.6f}")

print("\n=== replace p2 components with p3 (p2->p3 swap) ===")
# swap the p15/p7 part with p3 ensemble
for wp3 in [0.30, 0.45, 0.60, 0.75]:
    rest = 1 - wp3
    o = rest * (0.15/0.25 * ft_comp + 0.10/0.25 * llm) + wp3 * p3mean
    print(f"  p3={wp3:.2f} (ft/llm scaled) -> {roc_auc_fast(y, o):.6f}")
