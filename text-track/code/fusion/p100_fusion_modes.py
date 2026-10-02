#!/usr/bin/env python3
"""验证:排序级融合(rank01,线上提交用) vs 分数级融合(Docker 用)的差异。
Docker 逐条推理 → 只能用 sigmoid(logit) + 加权,无法 rank01。
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

TR = r"D:\湾区杯\文本赛道\transfer"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def load_raw(d, tag):
    p = os.path.join(d, f"{tag}_replay_raw_mean.npy")
    if not os.path.exists(p):
        p = os.path.join(d, f"{tag}_raw.npy")
    return np.load(p).astype(np.float64) if os.path.exists(p) else None

# p4 raw (top-level raw mean)
p4_raw = load_raw(os.path.join(TR, "14b_p4_backup"), "lora14b_p4")
r3a_raw = load_raw(os.path.join(TR, "r3_backup"), "lora7b_distill_r3a")
r3b_raw = load_raw(os.path.join(TR, "p76_backup"), "lora7b_distill_r3b")
print("loaded:", {"p4_raw": p4_raw is not None, "r3a_raw": r3a_raw is not None, "r3b_raw": r3b_raw is not None})

# 如果有 fold 级 raw,可以算 fold 平均
def load_fold_raws(d, tag):
    raws = []
    for f in range(5):
        p = os.path.join(d, f"{tag}_fold{f}_raw.npy")
        if os.path.exists(p):
            raws.append(np.load(p).astype(np.float64))
    return raws

p4_folds = load_fold_raws(os.path.join(TR, "14b_p4_backup"), "lora14b_p4")
r3a_folds = load_fold_raws(os.path.join(TR, "r3_backup"), "lora7b_distill_r3a")
r3b_folds = load_fold_raws(os.path.join(TR, "p76_backup"), "lora7b_distill_r3b")
print(f"fold raws: p4={len(p4_folds)} r3a={len(r3a_folds)} r3b={len(r3b_folds)}")

def sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))

# ---- 方法A: rank01 融合(线上提交用) ----
if p4_raw is not None and r3a_raw is not None and r3b_raw is not None:
    p4r = rank01(p4_raw)
    e7 = 0.5 * (rank01(r3a_raw) + rank01(r3b_raw))
    rank_way = rank01(0.90 * p4r + 0.10 * e7)

    # ---- 方法B: 分数级融合(Docker 用) ----
    # 用 sigmoid 把 logit 转概率,再加权平均
    p4p = sig(p4_raw)
    e7p = 0.5 * (sig(r3a_raw) + sig(r3b_raw))
    prob_way = 0.90 * p4p + 0.10 * e7p

    # ---- 方法C: 只用 p4(简化 Docker) ----
    p4_only_prob = sig(p4_raw)

    # 用 OOF 无法评估复赛;这里只能比较**排序一致性**
    from scipy.stats import spearmanr
    print("\n=== 排序一致性(复赛集)===")
    print(f"  corr(rank_way, prob_way)  = {spearmanr(rank_way, prob_way)[0]:.6f}")
    print(f"  corr(rank_way, p4_only)   = {spearmanr(rank_way, p4_only_prob)[0]:.6f}")
    # 关键:prob_way 与 rank_way 的排序差异有多大?
    n = len(rank_way)
    diff = np.abs(rank01(prob_way) - rank01(rank_way))
    print(f"  rank 差异: mean {diff.mean():.5f}  max {diff.max():.5f}  >0.01 的比例 {100*float((diff>0.01).mean()):.3f}%")
    print(f"  prob_way unique = {np.unique(prob_way).size}")
