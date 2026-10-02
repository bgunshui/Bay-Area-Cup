#!/usr/bin/env python3
"""分析:哪些分支与 p4 相关性最低(潜在的互补信号),并生成 p4+低相关分支候选。"""
import sys, glob, os
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
p3 = rank01(np.load(RUNS / "lora14b_p3_replay.npy"))
p5 = rank01(np.load(P4DIR / "lora14b_p5_replay.npy"))

print("=== 各分支与 p4 的相关性(从低到高)===")
cands = {}
for f in sorted(glob.glob(str(RUNS / "*_replay.npy"))):
    name = os.path.basename(f).replace("_replay.npy", "")
    if name.startswith(("lora14b_p3", "lora14b_p4")):
        continue
    try:
        r = rank01(np.load(f))
        if len(r) != 20000:
            continue
        c = spearmanr(r, p4)[0]
        cands[name] = c
    except Exception:
        pass
for name, c in sorted(cands.items(), key=lambda kv: kv[1])[:12]:
    print(f"  {name:34s} corr={c:.4f}")

print(f"\n  [参考] p3: corr={spearmanr(p3,p4)[0]:.4f}   p5: corr={spearmanr(p5,p4)[0]:.4f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

# best low-correlation candidates
low = [n for n, c in sorted(cands.items(), key=lambda kv: kv[1])[:6]]
print(f"\n=== 生成 p4 + 低相关分支候选 ===")
for n in low:
    try:
        r = rank01(np.load(RUNS / f"{n}_replay.npy"))
    except Exception:
        continue
    for w in (0.03, 0.05):
        lab = f"submit_p4_95_{n[:20]}_{int(w*100):02d}"
        final = rank01((1-w)*p4 + w*r)
        pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{lab}.csv", index=False)
        print(f"  {lab}: corr={spearmanr(r,p4)[0]:.3f} unique={np.unique(final).size}")

# p4 + p5 (same training config)
for w in (0.10, 0.20):
    lab = f"submit_p4_{int((1-w)*100)}_p5_{int(w*100)}"
    final = rank01((1-w)*p4 + w*p5)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{lab}.csv", index=False)
    print(f"  {lab}: unique={np.unique(final).size}")
print("DONE")
