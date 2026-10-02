#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_slot_multi.py —— 在基底某个**槽位内部**按任意多分量混入，再叠加尺寸先验出提交候选。

与 `build_slot_blend.py`（只支持单分量）的关系：这是它的推广。
    槽内容 = (1 − Σwᵢ)·rank(原分量) + Σ wᵢ·rank(新分量ᵢ)

为什么需要多分量（2026-09-13 实测）：
    主槽（65% 权重）现在 100% 是 `cliph_rerun`，而**基底与 cliph_rerun 单独的 rank 相关高达 0.9758**
    —— 所以能改变分数的几乎只有"往主槽里注入独立信号"。
    在手候选注入源的实测特性：

    | 分量 | vs cliph_rerun | vs 重跑基底 | 单独实测 LB |
    |---|---|---|---|
    | `cliph300k`   | 0.7785 | 0.8023 | 未知（探针 val 0.9931，全场最高） |
    | `gpp`         | 0.7914 | **0.7848** | **0.9054（实测）** |
    | `orig`        | 0.8545 | 0.8455 | 0.9047（实测） |

    而 **`gpp` 与 `cliph300k` 之间只有 0.7403 相关** → 两个一起混进槽，理论上优于任一单独混入
    （模型算过：α=0.20 单 cliph300k → 槽 0.9169；α=0.20 单 gpp → 0.9165；
      两路各 0.15+0.15 → **槽 0.9201**；各 0.30 → 0.9246）。
    ⚠️ 但 `cliph300k` 的单独实力是**猜的**、`gpp` 是**实测的** —— 所以配比要给 gpp 留足权重，
       这样即使 cliph300k 只有 0.85，下行也仍有保护。

用法：
    python build_slot_multi.py <slot> <输出名> <prior_w> <w1> <csv1> [<w2> <csv2> ...]
例：
    python build_slot_multi.py cliph v59_c3gpp 0.125 0.15 img_cliph300k.csv 0.20 img_gpp_siglip2.csv
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent

SLOTS = {
    "cliph": ("img_cliph_rerun.csv", 0.65),
    "bigg": ("img_bigg_rerun.csv", 0.20),
    "s384": ("img_bigg_s384_retrain.csv", 0.15),
}
REF = "v49_rt_prior20.csv"     # 已知 LB 0.914498，用于自检
SLOT_AUC = 0.910               # 原 cliph 槽单独 AUC 的估计（基底 0.9133，两者 corr 0.9758）


def L(fn):
    d = pd.read_csv(HERE / fn, dtype={"id": str})
    return d.set_index("id")["score"]


def rk(s, ids):
    return pd.Series(rankdata(s.loc[ids], method="average") / (len(ids) + 1.0), index=ids)


def blend_auc(d_list, w_list, R):
    """多分量 rank 混合的 AUC（正态模型）：d = Σwᵢdᵢ / sqrt(Σᵢⱼ wᵢwⱼρᵢⱼ)。"""
    num = float(np.dot(w_list, d_list))
    den = float(np.sqrt(np.array(w_list) @ np.array(R) @ np.array(w_list)))
    return norm.cdf(num / den)


def main():
    if len(sys.argv) < 6:      # 脚本名 + slot + outname + prior_w + 至少一对(权重,csv)
        print(__doc__)
        return
    slot, outname, pw = sys.argv[1], sys.argv[2], float(sys.argv[3])
    rest = sys.argv[4:]
    if len(rest) % 2:
        print("错误：新分量必须成对给出 <权重> <csv>")
        return
    pairs = [(float(rest[i]), rest[i + 1]) for i in range(0, len(rest), 2)]
    if slot not in SLOTS:
        print(f"slot 必须是 {list(SLOTS)} 之一")
        return
    tot = sum(w for w, _ in pairs)
    if tot > 1.0:
        print(f"错误：槽内新分量权重之和 {tot} > 1")
        return

    comp = {k: L(v[0]) for k, v in SLOTS.items()}
    pr = L("size_prior2.csv")
    ids = list(comp["cliph"].index)

    base0 = sum(SLOTS[k][1] * rk(comp[k], ids) for k in SLOTS)
    chk = (1 - 0.20) * base0 + 0.20 * rk(pr, ids)
    print(f"[自检] α=0 重建 w=0.20 vs {REF}  corr = "
          f"{spearmanr(chk, L(REF).loc[ids]).statistic:.6f}  （应 ≈1.000000）")
    print()

    news = [(w, Path(f).name, L(f)) for w, f in pairs]
    old = comp[slot]
    # 相关：新分量 vs 槽原分量 / 新分量之间
    print(f"[槽] {SLOTS[slot][0]}（基底权重 {SLOTS[slot][1]:.2f}）× "
          f"{1 - tot:.2f} + " + " + ".join(f"{w:.2f}·{n}" for w, n, _ in news))
    for w, n, s in news:
        print(f"     ρ({n:22s}, 槽原) = {spearmanr(s.loc[ids], old.loc[ids]).statistic:+.4f}")
    for i in range(len(news)):
        for j in range(i + 1, len(news)):
            print(f"     ρ({news[i][1]:22s}, {news[j][1]:22s}) = "
                  f"{spearmanr(news[i][2].loc[ids], news[j][2].loc[ids]).statistic:+.4f}")

    parts = [(1 - tot, old)] + [(w, s) for w, _, s in news]
    slotmix = sum(w * rk(s, ids) for w, s in parts)
    base = sum(SLOTS[k][1] * (slotmix if k == slot else rk(comp[k], ids)) for k in SLOTS)
    print(f"     新基底 vs 旧基底 rank corr = {spearmanr(base, base0).statistic:+.4f}")

    # ---- 预期（多分量正态模型，只算槽级，别硬套基底级）----
    d_old = norm.ppf(SLOT_AUC)
    names = ["槽原"] + [n for _, n, _ in news]
    D = [d_old] + [None] * len(news)
    R = [[1.0] * len(names) for _ in names]
    for i in range(len(news)):
        R[0][i + 1] = R[i + 1][0] = spearmanr(news[i][2].loc[ids], old.loc[ids]).statistic
        for j in range(len(news)):
            if i != j:
                R[i + 1][j + 1] = spearmanr(news[i][2].loc[ids], news[j][2].loc[ids]).statistic
    print(f"\n[预期] 假设原槽单独 AUC = {SLOT_AUC:.3f}；扫新分量各自的真实 AUC：")
    for c in (0.85, 0.875, 0.90, 0.9054, 0.92, 0.94):
        Dl = [d_old] + [norm.ppf(c)] * len(news)
        W = [1 - tot] + [w for w, _, _ in news]
        a = blend_auc(Dl, W, R)
        print(f"       新分量 AUC={c:.4f} → 槽 AUC={a:.4f}  Δ槽={a-SLOT_AUC:+.4f}  "
              f"Δ基底×0.65={((a-SLOT_AUC)*0.65):+.4f}")

    z = (1 - pw) * base + pw * rk(pr, ids)
    out = pd.DataFrame({"id": ids,
                        "score": rankdata(z.to_numpy(), method="average") / (len(z) + 1.0)})
    dst = HERE / f"{outname}.csv"
    out.to_csv(dst, index=False, float_format="%.10f")
    print(f"\n已生成 {dst.name}   基底×{1-pw:.3f} + prior2×{pw:.3f}")
    print(f"请跑： python check_submission.py {dst.name}")


if __name__ == "__main__":
    main()
