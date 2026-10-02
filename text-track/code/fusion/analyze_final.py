"""定稿分析:离线(提交)与线上(镜像)两条链路的精确差异与偏差量化。"""
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


def flips(a, b):
    """a、b 两个分数向量的排序翻转对数。"""
    ra, rb = rank01(a), rank01(b)
    order = np.argsort(ra, kind="mergesort")
    vals = rankdata(rb[order], method="ordinal").astype(int)
    n = len(vals)
    bit = np.zeros(n + 2)

    def add(i):
        i += 1
        while i <= n:
            bit[i] += 1
            i += i & (-i)

    def ssum(i):
        s = 0
        while i > 0:
            s += bit[i]
            i -= i & (-i)
        return s

    inv = 0
    seen = 0
    for v in vals:
        inv += seen - ssum(v)
        add(v)
        seen += 1
    return inv, n * (n - 1) // 2


SPEC = {
    "14B p4": (os.path.join(TR, "14b_p4_backup"), "lora14b_p4", 5),
    "7B r3a": (os.path.join(TR, "r3_backup"), "lora7b_distill_r3a", 5),
    "7B r3b": (os.path.join(TR, "p76_backup"), "lora7b_distill_r3b", 5),
}

off = {}
onl = {}
print("=" * 78)
print("单模型层面:离线(逐折秩平均) vs 线上(logit 平均后 sigmoid)")
print("=" * 78)
for tag, (d, pre, K) in SPEC.items():
    folds = [np.load(os.path.join(d, "%s_fold%d_raw.npy" % (pre, f))) for f in range(K)]
    a = np.mean([rank01(x) for x in folds], axis=0)          # 离线
    b = sigmoid(np.mean(folds, axis=0))                      # 线上
    off[tag] = a
    onl[tag] = b
    inv, tot = flips(a, b)
    print("  %-8s  Spearman=%.9f   翻转对数=%7d / %d (%.4f%%)"
          % (tag, spearmanr(a, b)[0], inv, tot, 100.0 * inv / tot))

print()
print("=" * 78)
print("① 离线提交(rank 链路)能否精确重建 submit_p4_90_7b2_10.csv")
print("=" * 78)
ens7_off = (off["7B r3a"] + off["7B r3b"]) / 2.0
submitted = rank01(0.90 * off["14B p4"] + 0.10 * ens7_off)
ref = pd.read_csv(os.path.join(SUB, "submit_p4_90_7b2_10.csv"), encoding="utf-8-sig")
print("  Spearman(重建, 提交) = %.9f" % spearmanr(submitted, ref["score"].values)[0])
print("  排序完全一致:", bool(np.array_equal(rankdata(submitted), rankdata(ref["score"].values))))
print("  数值最大绝对差 = %.3e" % np.abs(submitted - ref["score"].values).max())

print()
print("=" * 78)
print("② 线上镜像链路(server.py 实际算法)")
print("=" * 78)
ens7_on = sigmoid((np.load(os.path.join(TR, "r3_backup", "lora7b_distill_r3a_replay_raw_mean.npy"))
                   + np.load(os.path.join(TR, "p76_backup", "lora7b_distill_r3b_replay_raw_mean.npy"))) / 2.0)
online = 0.90 * onl["14B p4"] + 0.10 * ens7_on
print("  score = 0.90*sigmoid(mean5[14B logits]) + 0.10*sigmoid(mean10[7B logits])")
print("  取值范围 [%.6f, %.6f]" % (online.min(), online.max()))

print()
print("=" * 78)
print("③ 两条链路的最终偏差")
print("=" * 78)
print("  Spearman(离线提交, 线上镜像) = %.9f" % spearmanr(submitted, online)[0])
inv, tot = flips(submitted, online)
print("  排序翻转对数 = %d / %d (%.4f%%)" % (inv, tot, 100.0 * inv / tot))
print("  AUC 严格上界(20000 条,n1=n0=10000): |ΔAUC| ≤ %d/1e8 = %.6f" % (inv, inv / 1e8))
print("  翻转方向随机时的典型量级: ~%.1e" % (np.sqrt(inv) / 1e8))
d = np.abs(rank01(submitted) - rank01(online))
print("  秩位移: 均值=%.6f 中位=%.6f 99分位=%.6f 最大=%.6f" %
      (d.mean(), np.median(d), np.percentile(d, 99), d.max()))
for th in (0.01, 0.005, 0.001):
    c = int((d > th).sum())
    print("     秩位移 > %.3f : %5d / %d (%.3f%%)" % (th, c, len(d), 100.0 * c / len(d)))

print()
print("=" * 78)
print("④ 两个作用分别贡献多少")
print("=" * 78)
invA, _ = flips(off["14B p4"], onl["14B p4"])
print("  作用 A(折间聚合:秩平均 vs logit平均)——仅 14B 侧就翻转 %d 对 (%.4f%%)"
      % (invA, 100.0 * invA / tot))
p4_rank_equiv = sigmoid(np.load(os.path.join(TR, "14b_p4_backup", "lora14b_p4_replay_raw_mean.npy")))
invB, _ = flips(0.90 * off["14B p4"] + 0.10 * ens7_off, 0.90 * onl["14B p4"] + 0.10 * ens7_on)
print("  作用 B(跨模型融合:秩 vs sigmoid)——合计翻转 %d 对 (%.4f%%)" % (invB, 100.0 * invB / tot))

np.save(r"D:\湾区杯\文本赛道\_offline_submitted.npy", submitted)
np.save(r"D:\湾区杯\文本赛道\_online_image.npy", online)
print("\n已保存 _offline_submitted.npy / _online_image.npy")
