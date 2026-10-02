#!/usr/bin/env python3
"""v9: dual 14B + 微量其他分支(验证"少量多样性"是否加分)。
用户方案:dual .90 + 7b_p2 .05 + 1.5b_p2 .05
变体:不同稀释比例 + 用 p3 系替代 p2
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

# dual 14B
p3o, p3r = rank01(o_raw("lora14b_p3")), rank01(r_raw("lora14b_p3"))
p4o, p4r = rank01(np.load(P4 / "lora14b_p4_oof.npy")), rank01(np.load(P4 / "lora14b_p4_replay.npy"))
dual_o, dual_r = 0.5*(p3o+p4o), 0.5*(p3r+p4r)
print(f"dual OOF {auc(y, dual_o):.6f}")

# candidates
p7b2o, p7b2r = rank01(o_raw("lora7b_p2")), rank01(r_raw("lora7b_p2"))
p15b2o, p15b2r = rank01(o_raw("lora1.5b_p2")), rank01(r_raw("lora1.5b_p2"))
# p3 families
p7n = ["lora7b_distill_p3s1", "lora7b_distill_p3s2", "lora7b_distill_p3s3"]
p15n = ["lora1.5b_distill_p3s1", "lora1.5b_distill_p3s2"]
p7p3o = np.mean([rank01(o_raw(t)) for t in p7n], axis=0); p7p3r = np.mean([rank01(r_raw(t)) for t in p7n], axis=0)
p15p3o = np.mean([rank01(o_raw(t)) for t in p15n], axis=0); p15p3r = np.mean([rank01(r_raw(t)) for t in p15n], axis=0)

print("\n=== 相关性(与 dual)===")
for nm, oo in [("7b_p2", p7b2o), ("1.5b_p2", p15b2o), ("7b_p3mean", p7p3o), ("1.5b_p3mean", p15p3o)]:
    print(f"  corr(dual, {nm}) = {spearmanr(dual_o, oo)[0]:.4f}")

test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
ids = test.id.astype(str).tolist()

schemes = {
    # 用户方案
    "v9_user90_5_5":      (0.90*dual_o + 0.05*p7b2o + 0.05*p15b2o,  0.90*dual_r + 0.05*p7b2r + 0.05*p15b2r),
    # 更保守(更接近纯 dual)
    "v9_95_25_25":        (0.95*dual_o + 0.025*p7b2o + 0.025*p15b2o, 0.95*dual_r + 0.025*p7b2r + 0.025*p15b2r),
    "v9_98_1_1":          (0.98*dual_o + 0.01*p7b2o + 0.01*p15b2o,  0.98*dual_r + 0.01*p7b2r + 0.01*p15b2r),
    # 用 p3 系(p2 的升级版)替代 p2
    "v9_p3_90_5_5":       (0.90*dual_o + 0.05*p7p3o + 0.05*p15p3o,  0.90*dual_r + 0.05*p7p3r + 0.05*p15p3r),
    "v9_p3_95_25_25":     (0.95*dual_o + 0.025*p7p3o + 0.025*p15p3o, 0.95*dual_r + 0.025*p7p3r + 0.025*p15p3r),
    # 纯 dual 参考
    "v9_dual_pure":       (dual_o, dual_r),
}
print("\n=== OOF 对比 ===")
for k, (oo, rr) in schemes.items():
    print(f"  {k:18s} {auc(y, oo):.6f}")

print("\n=== 生成 ===")
for k, (oo, rr) in schemes.items():
    final = rank01(rr)
    pd.DataFrame({"id": ids, "score": final}).to_csv(OUT_SUB / f"submit_{k}.csv", index=False)
    print(f"  submit_{k}: unique {np.unique(final).size}")
print("V9 DONE")
