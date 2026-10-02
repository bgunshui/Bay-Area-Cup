#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_matrix.py —— 打印候选分量之间的 rank 相关矩阵，用于找"还值得一试的独立分量"。

背景：由两个实测点反解出重跑基底 ≈0.9133、先验 ≈0.652、最优先验权重 ≈0.13。
所以现在的杠杆在**基底**上；本脚本检查哪些在手分量与重跑基底足够独立、有资格进基底。
"""
import sys
from pathlib import Path

import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent

NAMES = [
    ("cliph_rerun", "img_cliph_rerun.csv"),
    ("bigg_rerun", "img_bigg_rerun.csv"),
    ("s384r", "img_bigg_s384_retrain.csv"),
    ("orig", "img_genimage_orig.csv"),
    ("gpp", "img_gpp_siglip2.csv"),
    ("align", "img_align.csv"),
    ("bigG300k25", "img_bigG_crop300k_ckpt_18924.csv"),
    ("bigG300k50", "img_bigG_crop300k_ckpt_37849.csv"),
    ("bigGpilot75", "img_bigG_crop_pilot_ckpt_35676.csv"),
    ("prior2", "size_prior2.csv"),
]


def load(fn):
    p = HERE / fn
    if not p.exists():
        return None
    d = pd.read_csv(p, dtype={"id": str})
    return d.set_index("id")["score"]


def main():
    probes = {}
    for alias, fn in NAMES:
        s = load(fn)
        if s is not None:
            probes[alias] = s
    ids = list(next(iter(probes.values())).index)

    # 重跑基底（已验证构造）
    rt = (0.65 * pd.Series(rankdata(probes["cliph_rerun"].loc[ids]) / (len(ids) + 1.0), index=ids)
          + 0.20 * pd.Series(rankdata(probes["bigg_rerun"].loc[ids]) / (len(ids) + 1.0), index=ids)
          + 0.15 * pd.Series(rankdata(probes["s384r"].loc[ids]) / (len(ids) + 1.0), index=ids))

    cols = ["rt_base"] + [a for a in probes]
    print(f"分量 {len(cols)} 个，n={len(ids)}")
    print()
    w = max(len(c) for c in cols) + 1
    print(" " * w + "".join(f"{c:>12}" for c in cols))
    for a in cols:
        sa = rt if a == "rt_base" else probes[a]
        row = f"{a:<{w}}"
        for b in cols:
            sb = rt if b == "rt_base" else probes[b]
            j = pd.concat([sa.rename("x"), sb.rename("y")], axis=1).dropna()
            row += f"{spearmanr(j.x, j.y).statistic:>12.4f}"
        print(row)

    print()
    print("=== 与重跑基底的相关（越低越可能与基底互补）===")
    for a in probes:
        j = pd.concat([probes[a].rename("x"), rt.rename("y")], axis=1).dropna()
        c = spearmanr(j.x, j.y).statistic
        tag = "  ← 已含在基底里" if a in ("cliph_rerun", "bigg_rerun", "s384r") else ""
        print(f"  {a:<14} {c:+.4f}{tag}")


if __name__ == "__main__":
    main()
