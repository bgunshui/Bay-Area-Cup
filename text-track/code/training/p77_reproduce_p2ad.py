#!/usr/bin/env python3
"""P77: 用严格重训版 lora7b_official_sr 复现 pseudo_best_v2(p2AD = 0.978284)。

p2AD 原始结构(v2_p2peak.py docstring):
  p2AD = 0.15*ft + 0.10*llm + 0.45*p15 + 0.30*p7
  ft  = 0.60*1.5b_distill_r2 + 0.10*14b_official_v2 + 0.20*7b_distill_r2 + 0.10*7b_official
  llm = 0.25*qwen_pair + 0.25*q4b_hidden_last + 0.50*q4b_hidden_mean
  p15 = rank01(lora1.5b_p2_replay)
  p7  = rank01(lora7b_p2_replay)

复现版:把 7b_official -> 7b_official_sr(严格重训)
输出:submit_v2_p2AD_REPRO.csv + 与原版 pseudo_best_v2 的相关性报告
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def r_raw(t):
    p = RUNS / f"{t}_replay.npy"
    if p.exists():
        return np.load(p).astype(np.float64)
    p2 = RUNS / "backup_14b_p4" / f"{t}_replay.npy"
    return np.load(p2).astype(np.float64) if p2.exists() else None

z = np.load(RUNS / "fusion_mat.npz")
names = list(z["names"]); rep = np.asarray(z["rep_mat"], dtype=np.float64)
idx = {n: i for i, n in enumerate(names)}

def fam(d, override=None):
    v = np.zeros(rep.shape[0])
    for t, w in d.items():
        if override and t in override:
            x = r_raw(override[t])
            v = v + w * rank01(x)
        elif t in idx:
            v = v + w * rep[:, idx[t]]
        else:
            x = r_raw(t)
            if x is not None:
                v = v + w * rank01(x)
    return v

# ft with sr replacing official
ft_orig = fam({"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10,
               "lora7b_distill_r2": .20, "lora7b_official": .10})
ft_sr = fam({"lora1.5b_distill_r2": .60, "lora14b_official_v2": .10,
             "lora7b_distill_r2": .20, "lora7b_official_sr": .10},
            override={"lora7b_official_sr": "lora7b_official_sr"})
llm = fam({"qwen_pair": .25, "q4b_hidden_last": .25, "q4b_hidden_mean": .50})
p15 = rank01(r_raw("lora1.5b_p2"))
p7 = rank01(r_raw("lora7b_p2"))

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

def emit(label, sc):
    final = rank01(sc)
    out = OUT_SUB / f"{label}.csv"
    pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
    print(f"{label}: unique {np.unique(final).size} -> {out}")
    return final

print("=== 复现 p2AD(0.978284)===")
orig = emit("submit_v2_p2AD_ORIGCHK", 0.15*ft_orig + 0.10*llm + 0.45*p15 + 0.30*p7)
repro = emit("submit_v2_p2AD_REPRO", 0.15*ft_sr + 0.10*llm + 0.45*p15 + 0.30*p7)

# compare with the pseudo_best_v2 (derived from the original 0.978284 submission)
pv2 = np.load(RUNS / "pseudo_best_v2.npy")
print(f"\n=== 与原版 pseudo_best_v2 的相关性 ===")
print(f"  原结构重建 vs pseudo_best_v2 : spearman {spearmanr(orig, pv2)[0]:.6f}")
print(f"  复现版(sr) vs pseudo_best_v2 : spearman {spearmanr(repro, pv2)[0]:.6f}")
print(f"  复现版 vs 原结构重建          : spearman {spearmanr(repro, orig)[0]:.6f}")

# also directly compare with the original submission file if present
orig_sub = OUT_SUB / "submit_v2_p2AD.csv"
if orig_sub.exists():
    d = pd.read_csv(orig_sub, encoding="utf-8-sig")
    o = rank01(d.score.to_numpy(np.float64))
    print(f"\n  原始提交文件 submit_v2_p2AD.csv 存在:")
    print(f"    复现版 vs 原始提交 : spearman {spearmanr(repro, o)[0]:.6f}")
    print(f"    重建版 vs 原始提交 : spearman {spearmanr(orig, o)[0]:.6f}")
else:
    print("\n  (submit_v2_p2AD.csv 不在 submissions/)")

# sr branch quality
sro = RUNS / "lora7b_official_sr_replay.npy"
if sro.exists():
    print(f"\n  lora7b_official_sr replay unique = {np.unique(np.load(sro)).size}")
print("P77 DONE")
