#!/usr/bin/env python3
"""Build pseudo_best.npy from the current best submission (submit_v2_ftintF.csv, 0.973656).
pseudo = sigmoid(4 * (rank01(score) - 0.5))  (soft preserving ranking; confidence weights computed in p42).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit

SUB = "/data/湾区杯/文本赛道/submissions"
RUNS = "/data/湾区杯/文本赛道/runs"
df = pd.read_csv(SUB + "/submit_v2_ftintF.csv")
s = rankdata(df.score.to_numpy(np.float64)) / (len(df) + 1.0)
p = expit(4.0 * (s - 0.5))
np.save(RUNS + "/pseudo_best.npy", p)
print("pseudo_best: mean", round(float(p.mean()), 4), "min", round(float(p.min()), 4),
      "max", round(float(p.max()), 4))
print("PSEUDO BEST DONE")
