#!/usr/bin/env python3
"""最新排行榜综合分计算 + 晋级差距分析。"""
import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# 最新榜单(2026-09-14 01:0x 截图)
teams = {
    "LLMs":            (0.984794, 0.930874),
    "好易评-OPC":       (0.981932, 0.964054),
    "南枝大学(我们)":    (0.980896, 0.916175),
    "打假打假":         (0.978947, 0.960880),
    "gogogo":          (0.975518, 0.906452),
    "初浙沥以萧飒":      (0.972666, 0.940412),
    "AIGCD":           (0.968151, 0.926360),
    "陪跑队不对":        (0.965303, 0.923199),
    "kawhi00":         (0.964960, 0.944746),
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
print(f"{'名':<4}{'队伍':<18}{'文本':<10}{'图片':<10}{'文本分':<9}{'图片分':<9}{'综合':<9}")
print("-" * 76)
for r, (name, (t, i)) in enumerate(rows, 1):
    st, si = math.exp(4*t), math.exp(4*i)
    mark = "  <<< US" if "南枝" in name else ""
    print(f"{r:<4}{name:<18}{t:<10.6f}{i:<10.6f}{st:<9.2f}{si:<9.2f}{st+si:<9.2f}{mark}")

us = score(*teams["南枝大学(我们)"])
fifth = score(*rows[4][1])
print(f"\n我们综合分: {us:.3f}  ({rows.index([x for x in rows if '南枝' in x[0]][0])+1} 名)")
print(f"第5名({rows[4][0]}): {fifth:.3f}")
print(f"距晋级(前5): {fifth - us:+.3f}")

t0, i0 = teams["南枝大学(我们)"]
# needed improvements
for lo, hi, label in [(0.0, 0.03, "文本"), (0.0, 0.05, "图片")]:
    a, b = lo, hi
    for _ in range(70):
        mid = (a+b)/2
        v = score(t0+mid, i0) if label == "文本" else score(t0, i0+mid)
        if v < fifth: a = mid
        else: b = mid
    base = t0 if label == "文本" else i0
    print(f"仅靠{label}追平第5名: +{b:.6f} → 需达 {base+b:.6f}")

print(f"\n我们排名: 文本第 3 | 图片第 10")
print(f"文本距第1(LLMs {teams['LLMs'][0]:.6f}): {teams['LLMs'][0]-t0:+.6f}")
print(f"图片距第1(好易评 {teams['好易评-OPC'][1]:.6f}): {teams['好易评-OPC'][1]-i0:+.6f}")
print(f"\n对比上一版榜单:")
print(f"  文本: 0.980896 (不变)")
print(f"  图片: 0.914498 → 0.916175 (+0.001677) ✓ 有进步!")
