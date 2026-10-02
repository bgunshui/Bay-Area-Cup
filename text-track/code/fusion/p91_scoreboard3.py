#!/usr/bin/env python3
"""最新排行榜(9/14 18:xx)综合分计算。"""
import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

teams = {
    "LLMs":            (0.984794, 0.930964),
    "好易评-OPC":       (0.981932, 0.964239),
    "南枝大学(我们)":    (0.981069, 0.916175),
    "打假打假":         (0.979018, 0.960880),
    "gogogo":          (0.975518, 0.906452),
    "初浙沥以萧飒":      (0.972666, 0.940412),
    "AIGCD":           (0.968151, 0.926360),
    "陪跑队不对":        (0.965303, 0.923884),
    "kawhi00":         (0.964960, 0.944746),
    "火眼+金睛":         (0.962227, 0.0),
    "你们给我好好检测啊":  (0.956524, 0.882167),
    "zxg":             (0.948162, 0.887177),
    "汪汪队立大功":       (0.940197, 0.936604),
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
print(f"\n我们: 文本 {teams['南枝大学(我们)'][0]:.6f} | 图片 {teams['南枝大学(我们)'][1]:.6f} | 综合 {us:.3f}")
print(f"第5名({rows[4][0]}): {fifth:.3f}  →  差距 {fifth-us:+.3f}")

t0, i0 = teams["南枝大学(我们)"]
for label in ("文本", "图片"):
    a, b = 0.0, 0.05
    for _ in range(70):
        mid = (a+b)/2
        v = score(t0+mid, i0) if label == "文本" else score(t0, i0+mid)
        if v < fifth: a = mid
        else: b = mid
    base = t0 if label == "文本" else i0
    print(f"  仅靠{label}追平第5名: +{b:.6f} → 需达 {base+b:.6f}")

print(f"\n【对比上一版】文本 0.980896 → 0.981069 (+0.000173)")
print(f"  综合分: {score(0.980896, i0):.3f} → {us:.3f}  (+{us-score(0.980896,i0):.3f})")
print(f"\n我们排名: 文本第 3 | 图片第 10")
