#!/usr/bin/env python3
"""本地对比 p4 vs p5 的 fold 级特征(解释为何 p5 更差)。"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import rankdata, spearmanr

B4 = r"D:\湾区杯\文本赛道\transfer\14b_p4_backup"
B5 = r"D:\湾区杯\文本赛道\transfer\14b_p5_backup"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

for label, d in [("p4(seed2, 0.981069)", B4), ("p5(seed3, 0.979225)", B5)]:
    print(f"=== {label} ===")
    if not os.path.isdir(d):
        print("  (dir missing)"); continue
    folds = []
    for f in range(5):
        p = os.path.join(d, f"lora14b_p{4 if 'p4' in d else 5}_fold{f}_replay.npy")
        if not os.path.exists(p):
            # try alt naming
            import glob
            g = glob.glob(os.path.join(d, f"*fold{f}_replay.npy"))
            p = g[0] if g else None
        if not p or not os.path.exists(p):
            print(f"  fold{f}: MISSING"); continue
        r = np.load(p)
        folds.append(rank01(r))
        print(f"  fold{f}: unique={np.unique(r).size:5d}  frac==max={100*float(np.mean(np.isclose(r,r.max(),atol=1e-6))):5.2f}%")
    if len(folds) >= 2:
        cors = []
        for i in range(len(folds)):
            for j in range(i+1, len(folds)):
                cors.append(spearmanr(folds[i], folds[j])[0])
        print(f"  fold 间平均相关性: {np.mean(cors):.4f}  (最低 {np.min(cors):.4f})")
        # ensemble of folds (like the top-level)
        ens = np.mean(folds, axis=0)
        print(f"  5折平均 unique: {np.unique(rank01(ens)).size}")
    print()
