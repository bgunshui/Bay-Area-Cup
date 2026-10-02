#!/usr/bin/env python3
"""P88: seed2(p4)高权重变体。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"
P4DIR = RUNS / "backup_14b_p4"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

p3 = rank01(np.load(RUNS / "lora14b_p3_replay.npy"))
p4 = rank01(np.load(P4DIR / "lora14b_p4_replay.npy"))
p5 = rank01(np.load(P4DIR / "lora14b_p5_replay.npy"))

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

def emit(label, w3, w4, w5):
    final = rank01(w3*p3 + w4*p4 + w5*p5)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{label}.csv", index=False)
    print(f"  {label:32s} p3={w3:.2f} p4={w4:.2f} p5={w5:.2f}  unique={np.unique(final).size}")

print("=== p4(seed2)高权重变体 ===")
# p4 vs p3
emit("submit_14b_p4_60_p3_40", 0.40, 0.60, 0.00)
emit("submit_14b_p4_70_p3_30", 0.30, 0.70, 0.00)
emit("submit_14b_p4_80_p3_20", 0.20, 0.80, 0.00)
emit("submit_14b_p4_90_p3_10", 0.10, 0.90, 0.00)
# p4 + p5(排除 p3)
emit("submit_14b_p4_60_p5_40", 0.00, 0.60, 0.40)
emit("submit_14b_p4_70_p5_30", 0.00, 0.70, 0.30)
emit("submit_14b_p4_80_p5_20", 0.00, 0.80, 0.20)
print("P88 DONE")
