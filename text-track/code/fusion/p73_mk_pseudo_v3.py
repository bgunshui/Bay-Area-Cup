#!/usr/bin/env python3
"""生成第三轮伪标签 pseudo_best_v3(来自双 14B 的 0.980896 方案)。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
SUB = BASE / "文本赛道" / "submissions"

# 从提交文件读取 dual-only 的分数(rank01 后),或从 npy 重算
sub_file = SUB / "submit_v8_dual_only.csv"
if sub_file.exists():
    df = pd.read_csv(sub_file, encoding="utf-8-sig")
    scores = df["score"].to_numpy(dtype=np.float64)
    print(f"read {len(scores)} scores from {sub_file.name}")
else:
    # fallback: recompute dual from npy
    P4 = RUNS / "backup_14b_p4"
    def rank01(x):
        return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)
    a = rank01(np.load(RUNS / "lora14b_p3_replay.npy"))
    b = rank01(np.load(P4 / "lora14b_p4_replay.npy"))
    scores = rank01(0.5 * (a + b))
    print("recomputed dual from npy")

# sanity: already 0~1 ranks
if scores.min() < 0 or scores.max() > 1:
    scores = rankdata(scores, method="average") / (len(scores) + 1.0)

soft = 1.0 / (1.0 + np.exp(-4.0 * (scores - 0.5)))
conf = np.clip(4.0 * np.abs(soft - 0.5), 0.0, 1.0)
print(f"soft: mean={soft.mean():.4f} min={soft.min():.4f} max={soft.max():.4f}")
print(f"conf: mean={conf.mean():.4f}  frac(conf>0.5)={float((conf>0.5).mean()):.3f}")
np.save(RUNS / "pseudo_best_v3.npy", soft)
print(f"saved -> runs/pseudo_best_v3.npy  (shape {soft.shape})")
# also save raw scores for reference
np.save(RUNS / "pseudo_best_v3_scores.npy", scores)
print("P73 DONE")
