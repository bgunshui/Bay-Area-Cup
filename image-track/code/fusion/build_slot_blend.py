#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_slot_blend.py —— 在基底某个**槽位内部**按比例混入新分量，再叠加尺寸先验出提交候选。

与 `build_new_component.py` 的区别：
  那个是「整槽替换」（alpha 隐含 = 1）；本脚本支持 **部分混入**（alpha ∈ [0,1]）：
      槽内容 = (1-α)·rank(原分量) + α·rank(新分量)
  为什么需要：2026-09-13 实测 `cliph300k` 与 `cliph_rerun` 的 rank 相关只有 **0.7785**，
  看起来是难得的独立强分量；但它**分数极度饱和**（42.6% > 0.99、28.2% < 0.01，中间段只剩 17.9%），
  而 `cliph_rerun` 是罕见的"饱满分布"（99.5% 落在 0.05~0.95）。
  → 饱和分量放在 **65% 主槽**会把 70% 的图打成并列、把排序权交给只占 35% 的弱分量；
    放在 **小比例** 则只贡献它真正能分辨的那部分信息。所以要用 α 控制。

基底（已逐图验证与 v49 系列一致，corr=1.000000）：
    0.65·rank(cliph_rerun) + 0.20·rank(bigg_rerun) + 0.15·rank(s384r)
先验最优权重 w* ≈ 0.125（由 0.914498 / 0.909226 两个实测点反解，ρ 从 0 到 0.4 稳健）

用法：
    python build_slot_blend.py <slot> <新探针.csv> <alpha> <输出名> [prior_w]
      slot ∈ {cliph, bigg, s384}
例：
    python build_slot_blend.py cliph img_cliph300k.csv 0.20 v58_cliph300k20
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from scipy.stats import norm

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent

SLOTS = {
    "cliph": ("img_cliph_rerun.csv", 0.65),
    "bigg": ("img_bigg_rerun.csv", 0.20),
    "s384": ("img_bigg_s384_retrain.csv", 0.15),
}
REF = "v49_rt_prior20.csv"        # 已知 LB 0.914498，用于自检
BASE_AUC = 0.9133                 # 反解出的重跑基底真实 AUC
BASE_D = norm.ppf(BASE_AUC)       # 其正态分位（混合模型的 d 值）


def L(fn):
    d = pd.read_csv(HERE / fn, dtype={"id": str})
    return d.set_index("id")["score"]


def rk(s, ids):
    return pd.Series(rankdata(s.loc[ids], method="average") / (len(ids) + 1.0), index=ids)


def blend_auc(d1, d2, rho, a):
    """两个 d 值（正态分位）分量按 a 混合后的 AUC（AUC 混合的标准正态模型）。"""
    num = (1 - a) * d1 + a * d2
    den = np.sqrt((1 - a) ** 2 + a ** 2 + 2 * a * (1 - a) * rho)
    return norm.cdf(num / den)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return
    slot, new_csv, alpha, outname = sys.argv[1], sys.argv[2], float(sys.argv[3]), sys.argv[4]
    pw = float(sys.argv[5]) if len(sys.argv) > 5 else 0.125
    if slot not in SLOTS:
        print(f"slot 必须是 {list(SLOTS)} 之一")
        return

    comp = {k: L(v[0]) for k, v in SLOTS.items()}
    pr = L("size_prior2.csv")
    ids = list(comp["cliph"].index)

    # ---- 自检：α=0 + w=0.20 必须复现 v49_rt_prior20 ----
    base0 = sum(SLOTS[k][1] * rk(comp[k], ids) for k in SLOTS)
    chk = (1 - 0.20) * base0 + 0.20 * rk(pr, ids)
    print(f"[自检] α=0 重建 w=0.20 vs {REF}  corr = "
          f"{spearmanr(chk, L(REF).loc[ids]).statistic:.6f}  （应 ≈1.000000）")

    new = L(new_csv)
    rho = spearmanr(rk(new, ids), rk(comp[slot], ids)).statistic
    base = sum(SLOTS[k][1] * ((1 - alpha) * rk(comp[k], ids) + alpha * rk(new, ids) if k == slot
                              else rk(comp[k], ids)) for k in SLOTS)

    print(f"\n[混入] {SLOTS[slot][0]} 槽（权重 {SLOTS[slot][1]:.2f}）"
          f"内 = {(1-alpha)*100:.0f}% 原 + {alpha*100:.0f}% {new_csv}")
    print(f"       新分量 vs 该槽原分量 rank corr = {rho:+.4f}")
    print(f"       新基底 vs 原基底 rank corr      = {spearmanr(base, base0).statistic:+.4f}")

    # ---- 预期：只算**槽级**（定义清晰），基底级不要硬套公式 ----
    # ⚠️ 不要试图用两分量混合公式去算"新基底"：基底是三个分量的加权和，
    #    而三个分量各自单独的真实 AUC 我们**并不知道**，硬凑出来的数字是假的。
    #    这里只给可辩护的两件事：① 槽内混合后的 AUC 随 c 的变化；② 盈亏平衡的 c。
    a_slot = float(os.environ.get("A_SLOT", "0.910"))   # 假定原 cliph 槽单独 AUC（≈ 基底 0.9133，corr 0.9758）
    print(f"\n[预期] 假定原 cliph 槽单独 AUC = {a_slot:.3f}（= 基底的近似，实测两者 corr 0.9758）")
    print(f"       槽内 = {(1-alpha)*100:.0f}% 原 + {alpha*100:.0f}% 新（ρ = {rho:.4f}）")
    print(f"       {'新分量c':>8} {'槽混合AUC':>10} {'Δ槽':>8} {'Δ基底×0.65(保守)':>16}")
    d_old = norm.ppf(a_slot)
    for c in (0.80, 0.825, 0.85, 0.875, 0.90, 0.91, 0.93, 0.95):
        sn = blend_auc(d_old, norm.ppf(c), rho, alpha)
        print(f"       {c:>8.3f} {sn:>10.4f} {sn-a_slot:>+8.4f} {(sn-a_slot)*0.65:>+16.4f}")
    # 盈亏平衡：槽混合 == 原槽
    from scipy.optimize import brentq
    try:
        be = brentq(lambda c: blend_auc(d_old, norm.ppf(c), rho, alpha) - a_slot, 0.50, 0.999)
        print(f"       → **盈亏平衡点 c* = {be:.4f}**：新分量单独真实 AUC 高于它，混入才有正收益")
    except Exception as e:
        print(f"       （盈亏平衡求解失败：{e}）")

    z = (1 - pw) * base + pw * rk(pr, ids)
    out = pd.DataFrame({"id": ids,
                        "score": rankdata(z.to_numpy(), method="average") / (len(z) + 1.0)})
    dst = HERE / f"{outname}.csv"
    out.to_csv(dst, index=False, float_format="%.10f")
    print(f"\n已生成 {dst.name}   基底×{1-pw:.3f} + prior2×{pw:.3f}")
    print(f"请跑： python check_submission.py {dst.name}")


if __name__ == "__main__":
    main()
