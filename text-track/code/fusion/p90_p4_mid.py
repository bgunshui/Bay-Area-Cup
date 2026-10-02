#!/usr/bin/env python3
"""P90: p4 + 中等相关分支(最后的"不重训"尝试)。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"
P4DIR = RUNS / "backup_14b_p4"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

p4 = rank01(np.load(P4DIR / "lora14b_p4_replay.npy"))
test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

# 中等相关且有一定强度的分支
picks = ["q7b_lik", "q4b_lik", "nll_fp", "qwen_pair", "lora7b_distill_r2", "lora1.5b_distill_r2"]
print("=== p4 + 中等相关/强分支 ===")
for n in picks:
    p = RUNS / f"{n}_replay.npy"
    if not p.exists():
        print(f"  {n}: MISSING"); continue
    r = rank01(np.load(p))
    c = spearmanr(r, p4)[0]
    for w in (0.03, 0.05, 0.08):
        lab = f"submit_p4_{int((1-w)*100):02d}_{n}_{int(w*100):02d}"
        final = rank01((1-w)*p4 + w*r)
        pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{lab}.csv", index=False)
        print(f"  {lab:44s} corr={c:.3f} unique={np.unique(final).size}")
print("DONE")
