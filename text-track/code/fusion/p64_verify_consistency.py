#!/usr/bin/env python3
"""验证:p62 重训 q4b 头后生成的 replay 与 p11 版(/旧版)是否一致 ->
间接确认重新生成的 v4A 与提交过 0.979091 的版本是否等价。
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
from scipy.stats import rankdata, spearmanr

RUNS = Path("/data/湾区杯/文本赛道/runs")
def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

print("=== q4b_hidden replay: 备份(p11 前/旧版) vs 现在(p62 重训版) ===")
for n in ("mean", "last", "first"):
    new = RUNS / f"q4b_hidden_{n}_replay.npy"
    old = RUNS / "bak_q4b" / f"q4b_hidden_{n}_replay.npy"
    if new.exists() and old.exists():
        a, b = np.load(new), np.load(old)
        print(f"  {n}: spearman(rank)={spearmanr(a, b)[0]:.6f} maxdiff={np.abs(a-b).max():.6f} "
              f"unique new={np.unique(a).size} old={np.unique(b).size}")
    else:
        print(f"  {n}: missing (new={new.exists()} old={old.exists()})")

print("\n=== q4b_hidden oof: 备份 vs 现在 ===")
for n in ("mean", "last", "first"):
    new = RUNS / f"q4b_hidden_{n}_oof.npy"
    old = RUNS / "bak_q4b" / f"q4b_hidden_{n}_oof.npy"
    if new.exists() and old.exists():
        a, b = np.load(new), np.load(old)
        print(f"  {n}: spearman={spearmanr(a, b)[0]:.6f} maxdiff={np.abs(rank01(a)-rank01(b)).max():.6f}")
