#!/usr/bin/env python3
"""Build pseudo_best_v2 from the CURRENT best submission (submit_v2_p2AD.csv, 0.978284).
pseudo = sigmoid(4*(rank01(score)-0.5)) (soft preserving ranking; conf in p43).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy.special import expit

SUB = "/data/湾区杯/文本赛道/submissions"
RUNS = "/data/湾区杯/文本赛道/runs"
df = pd.read_csv(SUB + "/submit_v2_p2AD.csv")
assert len(df) == 20000
s = rankdata(df.score.to_numpy(np.float64)) / (len(df) + 1.0)
p = expit(4.0 * (s - 0.5))
np.save(RUNS + "/pseudo_best_v2.npy", p)
print("pseudo_best_v2: mean", round(float(p.mean()), 4), "min", round(float(p.min()), 4),
      "max", round(float(p.max()), 4))
print("PSEUDO V2 DONE")
