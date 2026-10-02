#!/usr/bin/env python3
"""p2 重跑完成 -> 对比 OOF + 生成 v9 重跑版候选。"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"
P4 = RUNS / "backup_14b_p4"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

y = np.load(RUNS / "labels.npy").astype(np.int64)
NREP = 20000
def auc(y, s):
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1, dtype=np.float64)
    t = np.sort(s); u, c = np.unique(t, return_counts=True)
    if (c > 1).any():
        st = 0
        for uu, cc in zip(u, c, strict=True):
            if cc > 1: r[(s == uu)] = (st + 1 + st + cc) / 2.0
            st += cc
    p = np.flatnonzero(y == 1); ss = r[p].sum(); n1 = len(p); n0 = len(y) - n1
    return (ss - n1 * (n1 + 1) / 2.0) / (n1 * n0)

def o_raw(t):
    p = RUNS / f"{t}_oof.npy"; return np.load(p).astype(np.float64) if p.exists() else None
def r_raw(t):
    p = RUNS / f"{t}_replay.npy"; return np.load(p).astype(np.float64) if p.exists() else None

print("=== p2 原版 vs 重跑版 OOF ===")
for orig, rerun in [("lora7b_p2", "lora7b_distill_p2r"), ("lora1.5b_p2", "lora1.5b_distill_p2r")]:
    a, b = o_raw(orig), o_raw(rerun)
    if a is not None and b is not None:
        print(f"  {orig}: {auc(y,a):.6f}  |  {rerun}: {auc(y,b):.6f}  |  corr={spearmanr(a,b)[0]:.4f}")
    else:
        print(f"  {orig}: {a is not None} | {rerun}: {b is not None}")

# dual 14B
p3o, p3r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))
p4o, p4r = rank01(np.load(P4 / "lora14b_p4_oof.npy")), rank01(np.load(P4 / "lora14b_p4_replay.npy"))
dual_o, dual_r = 0.5*(p3o+p4o), 0.5*(p3r+p4r)

# p2 versions
o7_orig, r7_orig = rank01(o_raw("lora7b_p2")), rank01(r_raw("lora7b_p2"))
o15_orig, r15_orig = rank01(o_raw("lora1.5b_p2")), rank01(r_raw("lora1.5b_p2"))
o7r, r7r = rank01(o_raw("lora7b_distill_p2r")), rank01(r_raw("lora7b_distill_p2r"))
o15r, r15r = rank01(o_raw("lora1.5b_distill_p2r")), rank01(r_raw("lora1.5b_distill_p2r"))

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

schemes = {
    # 用户要求的重跑版
    "submit_v9_user90_5_5_RERUN": (0.90*dual_o + 0.05*o7r + 0.05*o15r,
                                   0.90*dual_r + 0.05*r7r + 0.05*r15r),
    # 原版对照(与已提交 0.980896 的结构相同)
    "submit_v9_user90_5_5_ORIG":  (0.90*dual_o + 0.05*o7_orig + 0.05*o15_orig,
                                   0.90*dual_r + 0.05*r7_orig + 0.05*r15_orig),
    # 重跑版 + 原版混合(两个版本都放进去 = 更稳)
    "submit_v9_mix_both":         (0.90*dual_o + 0.025*o7r + 0.025*o15r + 0.025*o7_orig + 0.025*o15_orig,
                                   0.90*dual_r + 0.025*r7r + 0.025*r15r + 0.025*r7_orig + 0.025*r15_orig),
    # 纯 dual 参考
    "submit_v9_dual_ref":         (dual_o, dual_r),
}
print("\n=== 生成候选 ===")
for k, (oo, rr) in schemes.items():
    final = rank01(rr)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"{k}.csv", index=False)
    print(f"  {k}: OOF {auc(y, oo):.6f} unique {np.unique(final).size}")
print("P72 DONE")
