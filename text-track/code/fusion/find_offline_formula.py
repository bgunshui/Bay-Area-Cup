"""查清离线的精确聚合/融合公式,再与线上算法对比。"""
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TR = r"D:\湾区杯\文本赛道\transfer"
SUB = r"D:\湾区杯\文本赛道\submissions"


def rank01(x):
    return rankdata(np.asarray(x, np.float64), method="average") / (len(x) + 1.0)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(x, np.float64), -30, 30)))


def same(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    if np.array_equal(rankdata(a), rankdata(b)):
        return "排序一致"
    return "不同 (spearman=%.6f)" % spearmanr(a, b)[0]


CASES = {
    "14B p4": (os.path.join(TR, "14b_p4_backup"), "lora14b_p4", 5),
    "7B r3a": (os.path.join(TR, "r3_backup"), "lora7b_distill_r3a", 5),
    "7B r3b": (os.path.join(TR, "p76_backup"), "lora7b_distill_r3b", 5),
}

for tag, (d, pre, K) in CASES.items():
    print("=" * 78)
    print(tag, " ", d)
    print("=" * 78)
    saved_replay = np.load(os.path.join(d, pre + "_replay.npy"))
    saved_rawn = np.load(os.path.join(d, pre + "_replay_raw_mean.npy"))
    folds = [np.load(os.path.join(d, "%s_fold%d_raw.npy" % (pre, f))) for f in range(K)]
    print("  已保存 replay.npy     : min=%.5f max=%.5f mean=%.5f std=%.5f"
          % (saved_replay.min(), saved_replay.max(), saved_replay.mean(), saved_replay.std()))
    print("  已保存 raw_mean.npy   : min=%.4f max=%.4f mean=%.4f std=%.4f"
          % (saved_rawn.min(), saved_rawn.max(), saved_rawn.mean(), saved_rawn.std()))
    print("  单折 raw 是否等于 raw_mean 的组成:")
    for f in range(K):
        print("      fold%d: min=%.4f max=%.4f mean=%.4f 与 raw_mean spearman=%.6f"
              % (f, folds[f].min(), folds[f].max(), folds[f].mean(),
                 spearmanr(folds[f], saved_rawn)[0]))
    cands = {
        "rank01(mean_f logits)": rank01(np.mean(folds, axis=0)),
        "mean_f rank01(logits)": np.mean([rank01(x) for x in folds], axis=0),
        "mean_f sigmoid(logits)": np.mean([sigmoid(x) for x in folds], axis=0),
        "rank01(mean_f sigmoid)": rank01(np.mean([sigmoid(x) for x in folds], axis=0)),
        "sigmoid(mean_f logits)": sigmoid(np.mean(folds, axis=0)),
    }
    print("  候选公式与'已保存 replay.npy'的关系:")
    for k, v in cands.items():
        print("      %-26s -> %s" % (k, same(v, saved_replay)))
    print("  候选公式与'已保存 raw_mean.npy'的关系:")
    for k, v in cands.items():
        print("      %-26s -> %s" % (k, same(v, saved_rawn)))
    # raw_mean 是否就是 mean_f logits
    print("  raw_mean == mean_f logits ?", same(saved_rawn, np.mean(folds, axis=0)))
    print()
