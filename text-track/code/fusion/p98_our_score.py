#!/usr/bin/env python3
"""当前我们的分数(含最新提交 0.981301)。"""
import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

t_new = 0.981301   # 最新提交(p4 .90 + 7B .10)
i_now = 0.918887   # 图片最新
t_old = 0.981069   # 上一版

def score(t, i): return math.exp(4*t) + math.exp(4*i)

print("=== 我们的当前分数 ===")
print(f"  文本: {t_new:.6f}  ({t_old:.6f} → {t_new:.6f}, +{t_new-t_old:.6f})")
print(f"  图片: {i_now:.6f}")
print(f"  文本分: {math.exp(4*t_new):.3f}")
print(f"  图片分: {math.exp(4*i_now):.3f}")
print(f"  ─────────────────────")
print(f"  综合分: {score(t_new,i_now):.3f}")

# 晋级线(最新榜单 kawhi00)
fifth = 91.229
print(f"\n  晋级线(第5名 kawhi00): {fifth:.3f}")
print(f"  差距: {fifth - score(t_new,i_now):+.3f}")

# 追平路径
for label in ("文本", "图片"):
    a, b = 0.0, 0.05
    for _ in range(70):
        mid = (a+b)/2
        v = score(t_new+mid, i_now) if label == "文本" else score(t_new, i_now+mid)
        if v < fifth: a = mid
        else: b = mid
    base = t_new if label == "文本" else i_now
    print(f"  仅靠{label}: +{b:.6f} → {base+b:.6f}")

print(f"\n=== 排名(最新可查)===")
print(f"  文本: 第 4 名(0.981301,若榜单已更新)")
print(f"  图片: 第 10 名(0.918887)")
print(f"  综合: 第 6 名(约 {score(t_new,i_now):.2f})")
print(f"\n=== 分数演进 ===")
for t, i, note in [(0.980896, 0.916175, "p3+p4 各半 + 旧图片"),
                   (0.981069, 0.916175, "纯 p4"),
                   (0.981069, 0.918887, "纯 p4 + 新图片"),
                   (0.981301, 0.918887, "p4+7B ⭐ 当前")]:
    print(f"  文本 {t:.6f} | 图片 {i:.6f} → 综合 {score(t,i):.3f}   ({note})")
