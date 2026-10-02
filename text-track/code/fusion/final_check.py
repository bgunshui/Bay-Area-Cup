"""收尾核对:精确重建 + 文本数据对齐 + 逐样本对照素材准备。"""
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


p4 = np.load(os.path.join(TR, "14b_p4_backup", "lora14b_p4_replay.npy"))
r3a = np.load(os.path.join(TR, "r3_backup", "lora7b_distill_r3a_replay.npy"))
r3b = np.load(os.path.join(TR, "p76_backup", "lora7b_distill_r3b_replay.npy"))
ref = pd.read_csv(os.path.join(SUB, "submit_p4_90_7b2_10.csv"), encoding="utf-8-sig")

print("=" * 78)
print("1) 用 p97 完全相同的输入精确重建")
print("=" * 78)
reb = rank01(0.90 * rank01(p4) + 0.10 * ((rank01(r3a) + rank01(r3b)) / 2.0))
print("  Spearman = %.12f" % spearmanr(reb, ref["score"].values)[0])
print("  排序完全一致:", bool(np.array_equal(rankdata(reb), rankdata(ref["score"].values))))
print("  数值最大绝对差 = %.3e" % np.abs(reb - ref["score"].values).max())
print("  数值完全相同:", bool(np.allclose(reb, ref["score"].values, atol=0, rtol=0)))

print()
print("=" * 78)
print("2) text_test.csv 是否与 npy / 提交 id 同序")
print("=" * 78)
tt = pd.read_csv(os.path.join(TR, "text_test.csv"))
print("  行数:", len(tt), " 列:", list(tt.columns)[:6])
idc = [c for c in tt.columns if c.lower() in ("id", "req_id", "uuid", "index")]
print("  候选 id 列:", idc)
if idc:
    a = tt[idc[0]].astype(str).values
    b = ref["id"].astype(str).values
    print("  与提交 CSV 的 id 完全同序:", bool(len(a) == len(b) and (a == b).all()))
    if len(a) == len(b) and not (a == b).all():
        print("  前 3 个: text_test =", list(a[:3]), " submit =", list(b[:3]))
    txtcol = [c for c in tt.columns if c not in idc]
    if txtcol:
        L = tt[txtcol[0]].astype(str).str.len()
        print("  文本列 '%s' 长度: 中位=%d 最大=%d 最小=%d" % (txtcol[0], L.median(), L.max(), L.min()))

print()
print("=" * 78)
print("3) 准备真机逐样本对照素材(前 300 条,固定抽样)")
print("=" * 78)
if idc and len(tt) == len(ref):
    rng = np.random.RandomState(0)
    pick = np.sort(rng.choice(len(tt), 300, replace=False))
    txtcol = [c for c in tt.columns if c not in idc][0]
    out = pd.DataFrame({
        "req_id": tt[idc[0]].astype(str).values[pick],
        "data": tt[txtcol].astype(str).values[pick],
        "offline_score": ref["score"].values[pick],
        "online_pred": np.load(r"D:\湾区杯\文本赛道\_online_image.npy")[pick],
    })
    dst = r"D:\湾区杯\文本赛道\_gpu_compare.csv"
    out.to_csv(dst, index=False, encoding="utf-8")
    print("  已写出 %s  行数 %d  字节 %d" % (dst, len(out), os.path.getsize(dst)))
    print(out.head(3).to_string(index=False)[:400])
else:
    print("  !! 无法对齐,跳过")
