#!/usr/bin/env python3
"""P86: 14B 三 seed 权重变体 —— 基于 triple 掉分的发现(p5 拖累)。
生成多组权重供提交验证。
"""
import sys, os
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

def load_replay(tag, alt=None):
    p = RUNS / f"{tag}_replay.npy"
    if not p.exists() and alt:
        p = Path(alt) / f"{tag}_replay.npy"
    return rank01(np.load(p)) if p.exists() else None

p3 = load_replay("lora14b_p3")
p4 = load_replay("lora14b_p4", str(P4DIR))
p5 = load_replay("lora14b_p5", str(P4DIR))
print("loaded:", {k: (v is not None) for k, v in [("p3", p3), ("p4", p4), ("p5", p5)]})

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

def emit(label, w3, w4, w5):
    sc = w3*p3 + w4*p4 + w5*p5
    final = rank01(sc)
    out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"  {label:28s} p3={w3:.2f} p4={w4:.2f} p5={w5:.2f}  unique={np.unique(final).size}")

print("\n=== 生成权重变体 ===")
# 参考:已知最优(双 seed)
emit("submit_14b_p3p4_5050", 0.5, 0.5, 0.0)
# p3 高权重(p3 OOF 最高)
emit("submit_14b_p3_60_p4_40", 0.6, 0.4, 0.0)
emit("submit_14b_p3_70_p4_30", 0.7, 0.3, 0.0)
# 引入少量 p5(降低其影响)
emit("submit_14b_p3_50_p4_40_p5_10", 0.50, 0.40, 0.10)
emit("submit_14b_p3_55_p4_35_p5_10", 0.55, 0.35, 0.10)
# p4 高权重(p4 提供多样性)
emit("submit_14b_p3_40_p4_60", 0.4, 0.6, 0.0)
# p4+p5(排除 p3,验证假设)
emit("submit_14b_p4p5_5050", 0.0, 0.5, 0.5)
print("P86 DONE")
