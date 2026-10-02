#!/usr/bin/env python3
"""根据最新排行榜计算综合分与晋级形势。Score = e^(4*AUC_text) + e^(4*AUC_image)"""
import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

teams = {
    "LLMs":            (0.984748, 0.928954),
    "好易评-OPC":       (0.981605, 0.963792),
    "南枝大学(我们)":    (0.980896, 0.914498),
    "打假打假":         (0.978947, 0.958583),
    "gogogo":          (0.975518, 0.906452),
    "初浙沥以萧飒":      (0.972666, 0.940412),
    "AIGCD":           (0.968151, 0.926360),
    "陪跑队不对":        (0.965303, 0.923199),
    "kawhi00":         (0.964960, 0.944285),
    "火眼+金睛":         (0.962227, 0.0),
    "你们给我好好检测啊":  (0.956524, 0.882167),
    "zxg":             (0.948162, 0.887177),
    "汪汪队立大功":       (0.940197, 0.936461),
    "小马过河队":        (0.933736, 0.934680),
    "yndky":           (0.895206, 0.874660),
    "务必休息半天":       (0.885489, 0.893818),
}

def score(t, i):
    return math.exp(4*t) + math.exp(4*i)

rows = sorted(teams.items(), key=lambda kv: -score(*kv[1]))
print(f"{'排名':<4}{'队伍':<18}{'文本':<10}{'图片':<10}{'文本分':<10}{'图片分':<10}{'综合':<10}")
print("-" * 78)
for r, (name, (t, i)) in enumerate(rows, 1):
    st, si = math.exp(4*t), math.exp(4*i)
    mark = "  <<< US" if "南枝" in name else ""
    print(f"{r:<4}{name:<18}{t:<10.6f}{i:<10.6f}{st:<10.2f}{si:<10.2f}{st+si:<10.2f}{mark}")

us = score(*teams["南枝大学(我们)"])
fifth = score(*rows[4][1])
print(f"\n我们综合分: {us:.2f}")
print(f"第5名({rows[4][0]}): {fifth:.2f}")
print(f"距晋级(前5)差距: {fifth - us:+.2f}")

# sensitivity: needed text / image improvement to reach 5th
import numpy as np
t0, i0 = teams["南枝大学(我们)"]
# needed text gain with image fixed
lo, hi = 0.0, 0.02
for _ in range(60):
    mid = (lo+hi)/2
    if score(t0+mid, i0) < fifth: lo = mid
    else: hi = mid
print(f"\n仅靠文本追平第5名需要文本 +{hi:.6f} (达到 {t0+hi:.6f})")
lo, hi = 0.0, 0.05
for _ in range(60):
    mid = (lo+hi)/2
    if score(t0, i0+mid) < fifth: lo = mid
    else: hi = mid
print(f"仅靠图片追平第5名需要图片 +{hi:.6f} (达到 {i0+hi:.6f})")
print(f"\n当前文本排名: 第3 (全国) | 图片排名: 第10")
print(f"文本距第1名(LLMs 0.984748)还差 {0.984748 - t0:.6f}")
