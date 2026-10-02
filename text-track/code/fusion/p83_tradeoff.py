#!/usr/bin/env python3
"""给定文本提升量,计算追平晋级线所需的图片提升量。"""
import math, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

cur_t, cur_i = 0.980896, 0.916175
target = 91.229          # 第5名 kawhi00 的综合分(当前)
def score(t, i):
    return math.exp(4*t) + math.exp(4*i)

def need_image(t_new, tgt=target):
    """求最小 i 使得 score(t_new, i) >= tgt"""
    lo, hi = 0.0, 0.999
    for _ in range(80):
        mid = (lo+hi)/2
        if score(t_new, mid) < tgt:
            lo = mid
        else:
            hi = mid
    return hi

print(f"当前:文本 {cur_t:.6f} | 图片 {cur_i:.6f} | 综合 {score(cur_t,cur_i):.3f}")
print(f"晋级线(第5名 kawhi00): {target:.3f}  → 差距 {target-score(cur_t,cur_i):+.3f}")
print()
print(f"{'文本提升':<12}{'新文本':<12}{'文本分增益':<12}{'所需图片':<12}{'图片需提升':<12}{'可行性'}")
print("-" * 82)
base_st = math.exp(4*cur_t)
for dt in [0.0, 0.0005, 0.0009, 0.0015, 0.0018, 0.0025, 0.0028, 0.0040]:
    t_new = cur_t + dt
    i_need = need_image(t_new)
    di = i_need - cur_i
    st_gain = math.exp(4*t_new) - base_st
    if di < 0:
        feas = "已晋级 ✓"
    elif di <= 0.006:
        feas = "较易"
    elif di <= 0.012:
        feas = "可行"
    elif di <= 0.020:
        feas = "困难"
    else:
        feas = "极难"
    print(f"+{dt:<11.4f}{t_new:<12.6f}{st_gain:<+12.3f}{i_need:<12.6f}+{di:<11.6f}{feas}")

print()
print("【对照】历史上我们的图片提升:")
print("  09-13: 0.914498 → 0.916175 (+0.001677) —— 一次提交做到")
print()
print("【换算参考】")
print(f"  图片每 +0.001 ≈ 综合 +{math.exp(4*(cur_i+0.001))-math.exp(4*cur_i):.3f} 分")
print(f"  文本每 +0.001 ≈ 综合 +{math.exp(4*(cur_t+0.001))-math.exp(4*cur_t):.3f} 分")
