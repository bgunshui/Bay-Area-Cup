#!/usr/bin/env python3
"""Generate p3-based candidates (p3 = round-2 self-training with better pseudo labels).
Structure mirrors p2AD (0.978284): ft .15 / llm .10 / 1.5b-part .45 / 7b-part .30
Variants:
  p3A: fully p3   -> 1.5b: mean(p3s1,p3s2) ; 7b: p3s1
  p3B: half-half  -> 1.5b: mean(p2_1.5b, p3s1, p3s2) ; 7b: mean(p2_7b, p3s1)
"""
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

z = np.load(RUNS / "fusion_mat.npz")
names = list(z["names"]); rep = np.asarray(z["rep_mat"], dtype=np.float64); idx = {n: i for i, n in enumerate(names)}

ft = np.zeros(rep.shape[0])
for t, w in {"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10, "lora7b_distill_r2": .20, "lora7b_official": .10}.items():
    ft += w * rep[:, idx[t]]
llm = np.zeros(rep.shape[0])
for t, w in {"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50}.items():
    llm += w * rep[:, idx[t]]

def rl(tag):
    p = RUNS / f"{tag}_replay.npy"
    return rank01(np.load(p)) if p.exists() else None

p15_p2 = rl("lora1.5b_p2"); p7_p2 = rl("lora7b_p2")
p15s1 = rl("lora1.5b_distill_p3s1"); p15s2 = rl("lora1.5b_distill_p3s2")
p7s1 = rl("lora7b_distill_p3s1"); p7s2 = rl("lora7b_distill_p3s2")
print("available:", [n for n, v in [("p15_p2", p15_p2), ("p7_p2", p7_p2), ("p15s1", p15s1), ("p15s2", p15s2), ("p7s1", p7s1), ("p7s2", p7s2)] if v is not None])

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()
def emit(label, a15, a7, wft=0.15, wllm=0.10, w15=0.45, w7=0.30):
    final = rank01(wft * ft + wllm * llm + w15 * a15 + w7 * a7)
    out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(label, "unique", int(np.unique(final).size), "->", out)

# A: fully p3
a15 = np.mean([x for x in [p15s1, p15s2] if x is not None], axis=0)
a7 = p7s1 if p7s2 is None else np.mean([p7s1, p7s2], axis=0)
emit("submit_p3A_full", a15, a7)

# B: p2+p3 mix
b15 = np.mean([x for x in [p15_p2, p15s1, p15s2] if x is not None], axis=0)
b7 = np.mean([x for x in [p7_p2, p7s1, p7s2] if x is not None], axis=0)
emit("submit_p3B_mix", b15, b7)
print("P3 CANDIDATES DONE")
