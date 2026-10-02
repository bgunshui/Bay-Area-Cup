#!/usr/bin/env python3
"""完整 16 队综合分排名(用最新已知数据 + 我们更新后的文本)。"""
import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

teams = {
    "好易评-OPC":       (0.982232, 0.964239),
    "打假打假":         (0.979018, 0.961885),
    "初浙沥以萧飒":      (0.986704, 0.940412),
    "LLMs":            (0.984794, 0.930964),
    "kawhi00":         (0.964960, 0.944746),
    "南枝大学(我们)":    (0.981301, 0.918887),
    "AIGCD":           (0.968151, 0.926360),
    "陪跑队不对":        (0.965303, 0.924947),
    "gogogo":          (0.975518, 0.906452),
    "汪汪队立大功":       (0.940197, 0.936604),
    "小马过河队":        (0.933736, 0.934680),
    "你们给我好好检测啊":  (0.956524, 0.882167),
    "zxg":             (0.948162, 0.887177),
    "务必休息半天":       (0.885489, 0.893818),
    "yndky":           (0.895206, 0.874660),
    "火眼+金睛":         (0.962227, 0.0),
}

def score(t, i): return math.exp(4*t) + math.exp(4*i)

rows = sorted(teams.items(), key=lambda kv: -score(*kv[1]))
print("=" * 92)
print(f"{'名次':<5}{'队伍':<20}{'文本':<11}{'图片':<11}{'文本分':<10}{'图片分':<10}{'综合分':<10}{'晋级'}")
print("=" * 92)
for r, (name, (t, i)) in enumerate(rows, 1):
    st, si = math.exp(4*t), math.exp(4*i)
    adv = "✅ 晋级区" if r <= 5 else "  —"
    mark = " ← US" if "南枝" in name else ""
    print(f"{r:<5}{name:<20}{t:<11.6f}{i:<11.6f}{st:<10.2f}{si:<10.2f}{st+si:<10.2f}{adv}{mark}")
print("=" * 92)
us = score(*teams["南枝大学(我们)"])
fifth = score(*rows[4][1])
sixth = score(*rows[5][1])
print(f"\n我们: 第 {[i for i,x in enumerate(rows,1) if '南枝' in x[0]][0]} 名 | 综合 {us:.3f}")
print(f"晋级线(第5名 {rows[4][0]}): {fifth:.3f}  →  差距 {fifth-us:+.3f}")
print(f"第7名({rows[6][0]}): {score(*rows[6][1]):.3f}  →  我们领先 {us-score(*rows[6][1]):+.3f}")
print(f"\n若晋级,需超过: {rows[4][0]}({fifth:.3f})")
