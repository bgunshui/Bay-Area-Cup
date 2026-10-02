"""精确重建离线提交,并量化"离线 rank 融合"与"线上 sigmoid 融合"的差异。

关键:复赛 20000 条没有公开标签,所以 AUC 差值无法本地计算;
但 AUC 只取决于排序,因此可以用"翻转对数"给出严格上界。
"""
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


raw14 = np.load(os.path.join(TR, "14b_p4_backup", "lora14b_p4_replay_raw_mean.npy"))
raw7a = np.load(os.path.join(TR, "r3_backup", "lora7b_distill_r3a_replay_raw_mean.npy"))
raw7b = np.load(os.path.join(TR, "p76_backup", "lora7b_distill_r3b_replay_raw_mean.npy"))
print("原始 logits(五折均值)规模:", raw14.shape, raw7a.shape, raw7b.shape)
for n, a in (("14B p4", raw14), ("7B r3a", raw7a), ("7B r3b", raw7b)):
    print("  %-8s min=%8.4f max=%8.4f mean=%8.4f std=%7.4f" % (n, a.min(), a.max(), a.mean(), a.std()))

sub = pd.read_csv(os.path.join(SUB, "submit_p4_90_7b2_10.csv"), encoding="utf-8-sig")
print("\n提交 CSV 行数:", len(sub))

print("\n" + "=" * 78)
print("1) 对齐性验证:14B-only 提交应等于 rank01(14B 原始分)")
print("=" * 78)
p4_only = pd.read_csv(os.path.join(SUB, "submit_14b_p4_only.csv"), encoding="utf-8-sig")
print("  行数:", len(p4_only), " id 是否与最终提交同序:",
      bool((p4_only["id"].values == sub["id"].values).all()))
cand = rank01(raw14)
print("  rank01(14B raw) 与 14B-only 提交 Spearman = %.9f" % spearmanr(cand, p4_only["score"].values)[0])
print("  两者排序完全一致:", bool(np.array_equal(rankdata(cand), rankdata(p4_only["score"].values))))

print("\n" + "=" * 78)
print("2) 离线 rank 融合能否精确重建提交文件")
print("=" * 78)
ens7_rank = (rank01(raw7a) + rank01(raw7b)) / 2.0
offline = rank01(0.90 * rank01(raw14) + 0.10 * ens7_rank)
print("  Spearman(重建, 提交) = %.9f" % spearmanr(offline, sub["score"].values)[0])
print("  排序完全相同:", bool(np.array_equal(rankdata(offline), rankdata(sub["score"].values))))
print("  逐元素完全相同:", bool(np.allclose(offline, sub["score"].values, atol=1e-12)))

print("\n" + "=" * 78)
print("3) 线上 sigmoid 融合(镜像实际使用的算法)")
print("=" * 78)
online = 0.90 * sigmoid(raw14) + 0.10 * ((sigmoid(raw7a) + sigmoid(raw7b)) / 2.0)
sp = spearmanr(offline, online)[0]
print("  Spearman(离线 rank 融合, 线上 sigmoid 融合) = %.9f" % sp)

ro = rank01(offline)
rn = rank01(online)
disp = np.abs(ro - rn)
print("  秩位移: 均值=%.6f 中位=%.6f 90分位=%.6f 99分位=%.6f 最大=%.6f"
      % (disp.mean(), np.median(disp), np.percentile(disp, 90),
         np.percentile(disp, 99), disp.max()))
for th in (0.10, 0.05, 0.02, 0.01, 0.005, 0.001):
    c = int((disp > th).sum())
    print("     秩位移 > %.3f : %5d / %d (%.3f%%)" % (th, c, len(disp), 100.0 * c / len(disp)))

print("\n" + "=" * 78)
print("4) 排序翻转量(AUC 只取决于排序,这是偏差的直接来源)")
print("=" * 78)
# 按 offline 排序,统计 online 次序中的逆序对数
order = np.argsort(ro, kind="mergesort")
seq = rn[order]
n = len(seq)
bit = np.zeros(n + 2)


def bit_add(i):
    i += 1
    while i <= n:
        bit[i] += 1
        i += i & (-i)


def bit_sum(i):
    s = 0
    while i > 0:
        s += bit[i]
        i -= i & (-i)
    return s


# 用序号值(1..n)做逆序对计数
vals = rankdata(seq, method="ordinal").astype(int)
inv = 0
seen = 0
for v in vals:
    inv += seen - bit_sum(v)
    bit_add(v)
    seen += 1
total_pairs = n * (n - 1) // 2
print("  总对数 C(20000,2) = %d" % total_pairs)
print("  次序翻转对数      = %d (%.4f%%)" % (inv, 100.0 * inv / total_pairs))
print("  AUC 上界(设 n1=n0=10000,翻转对全为跨类): |dAUC| <= %d / 1e8 = %.6f"
      % (inv, inv / 1e8))
print("  若翻转方向随机,期望 |dAUC| 约 %.2e(= sqrt(翻转数)/1e8)"
      % (np.sqrt(inv) / 1e8))

# 翻转对的两个样本在离线分数上有多接近
print("\n  翻转对附近的分数密集程度:")
srt = np.sort(ro)
gaps = np.diff(srt)
print("     20000 个离线分的相邻间隔: 中位=%.3e  90分位=%.3e  最大=%.3e"
      % (np.median(gaps), np.percentile(gaps, 90), gaps.max()))

print("\n" + "=" * 78)
print("5) 两种融合最终分数的分布形态(解释为什么加权后会不等效)")
print("=" * 78)
for tag, a in (("14B p4", raw14), ("7B r3a", raw7a), ("7B r3b", raw7b)):
    r = rank01(a)
    s = sigmoid(a)
    print("  %-8s rank01: 标准差=%.4f   sigmoid: 标准差=%.4f   sigmoid均值=%.4f"
          % (tag, r.std(), s.std(), s.mean()))
print("\n  说明:rank01 恒为均匀分布(标准差≈0.2887),而 sigmoid 保留各模型的置信度尺度;")
print("        两个模型尺度不同 -> 同一个 0.9/0.1 权重在两种表示下对应的相对影响不同。")
np.save(r"D:\湾区杯\文本赛道\_offline_fusion.npy", offline)
np.save(r"D:\湾区杯\文本赛道\_online_fusion.npy", online)
print("\n已保存 _offline_fusion.npy / _online_fusion.npy 供后续对照")
