#!/usr/bin/env python3
"""生成纯 p4 候选(从未提交过的组合)。"""
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

p4 = rank01(np.load(RUNS / "backup_14b_p4" / "lora14b_p4_replay.npy"))
print(f"p4 replay: unique {np.unique(p4).size}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
out = OUT_SUB / "submit_14b_p4_only.csv"
pd.DataFrame({"id": test.id.astype(str), "score": rank01(p4)}).to_csv(out, index=False)
print(f"saved -> {out}")

# 对照:同时给出 p5-only(便于比较)
p5 = rank01(np.load(RUNS / "backup_14b_p4" / "lora14b_p5_replay.npy"))
out5 = OUT_SUB / "submit_14b_p5_only.csv"
pd.DataFrame({"id": test.id.astype(str), "score": rank01(p5)}).to_csv(out5, index=False)
print(f"saved -> {out5} (对照)")
print("DONE")
