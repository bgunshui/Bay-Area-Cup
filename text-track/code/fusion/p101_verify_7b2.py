#!/usr/bin/env python3
"""核实 submit_p4_90_7b2_10 的组成(来自 p96_local_p4_7b.py)。"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import numpy as np
from scipy.stats import rankdata, spearmanr

TR = r"D:\湾区杯\文本赛道\transfer"
SUB = r"D:\湾区杯\文本赛道\submissions"

def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)

def load(d, tag):
    p = os.path.join(d, f"{tag}_replay.npy")
    return rank01(np.load(p)) if os.path.exists(p) else None

p4 = load(os.path.join(TR, "14b_p4_backup"), "lora14b_p4")
r3a = load(os.path.join(TR, "r3_backup"), "lora7b_distill_r3a")
r3b = load(os.path.join(TR, "p76_backup"), "lora7b_distill_r3b")
if r3b is None:
    r3b = load(os.path.join(TR, "r3_backup"), "lora7b_distill_r3b")

print("=== 7b2 的定义(p96 脚本)===")
print('  ens7 = mean([r3a, r3b])            ← 7B 两个第三轮 seed 的集成')
print('  emit("submit_p4_90_7b2_10", 0.90*p4 + 0.10*ens7)')
print()
print(f"  组件状态: p4={p4 is not None}  r3a={r3a is not None}  r3b={r3b is not None}")
if r3a is not None and r3b is not None:
    print(f"  corr(r3a, r3b) = {spearmanr(r3a, r3b)[0]:.4f}")
    ens7 = np.mean([r3a, r3b], axis=0)
    print(f"  corr(ens7, p4)  = {spearmanr(ens7, p4)[0]:.4f}")

# 重建候选并与已提交文件比对
submitted = os.path.join(SUB, "submit_p4_90_7b2_10.csv")
if os.path.exists(submitted) and r3a is not None and r3b is not None:
    import pandas as pd
    d = pd.read_csv(submitted, encoding="utf-8-sig")
    sc = rank01(d["score"].to_numpy(np.float64))
    rebuilt = rank01(0.90*p4 + 0.10*np.mean([r3a, r3b], axis=0))
    print(f"\n=== 与已提交文件比对 ===")
    print(f"  corr(已提交, 重建) = {spearmanr(sc, rebuilt)[0]:.8f}")
    print(f"  完全一致: {np.allclose(sc, rebuilt, atol=1e-9)}")

print("\n=== 结论 ===")
print("  7b2 = 7B r3a + r3b(两个第三轮 seed)的等权集成")
print("  → Docker 镜像需要: 14B基座 + p4(5折) + 7B基座 + r3a/r3b(10折)")
