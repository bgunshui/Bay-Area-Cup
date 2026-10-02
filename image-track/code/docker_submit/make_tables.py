#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_tables.py —— 把"批 rank 融合"改写成"逐图可计算"的形式，并**逐图验证**。

═══ 为什么必须改写 ═══
提交文件 v100_gpp20_seed4_c75_prior200.csv（LB **0.918887**）是
`build_final.py` 用 **20000 张测试图的 rank 加权** 算出来的：
    rk(x) = rankdata(x)/(n+1)          ← **批操作**，单张图算不出来
而 Docker 服务必须对**单张图片**给出 score。两者不可能直接一致。

═══ 改写方法 ═══
把 `rk` 换成 **ECDF（经验分布函数）**：
    rk(x)        →  ECDF(x) = 该分量在 20000 张测试图上的分位
    rank(z)/(n+1) →  ECDF_z(z)
在**那 20000 张上两者恒等**（ECDF 就是 rank），所以能逐图复现提交文件；
对**新图** ECDF 是良定义的单调映射，等价于"用测试集分布做分位数标定"。

═══ 验证目标 ═══
用 ECDF 路径重算 20000 行，必须与 `v100_gpp20_seed4_c75_prior200.csv`
**排序完全一致**（spearman = 1.0，即 AUC 完全相同）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

HERE = Path(__file__).resolve().parent          # docker_submit/
WS = HERE.parent                                 # 工程根
OUT = HERE / "app" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ── v100 的配方（与 build_final.py 的 gpp20_seed4_c75 完全对应） ─────────────
SLOTS = [
    ("cliph", 0.75, [("img_cliph_rerun.csv", 0.20),
                     ("img_cliph_s20261101.csv", 0.20),
                     ("img_cliph_s20261102.csv", 0.20),
                     ("img_cliph_s20261103.csv", 0.20),
                     ("img_gpp_siglip2.csv", 0.20)]),
    ("bigg", 0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50),
                                  ("img_bigg_rerun3.csv", 0.50)]),
    ("s384", 0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
]
PRIOR = "size_prior2.csv"
PRIOR_W = 0.20
REF = "v100_gpp20_seed4_c75_prior200.csv"       # LB 0.918887


def load(fn):
    d = pd.read_csv(WS / fn, encoding="utf-8-sig", dtype={"id": str})
    assert len(d) == 20000, f"{fn} 行数 {len(d)}"
    assert not d.id.duplicated().any(), f"{fn} 有重复 id"
    return d.set_index("id")["score"]


ids = None
comp = {}
for _slot, _sw, comps in SLOTS:
    for f, _cw in comps:
        s = load(f)
        if ids is None:
            ids = list(s.index)
        else:
            assert list(s.index) == ids, f"{f} 的 id 顺序不一致"
        comp[f] = s.to_numpy()
prior_raw = load(PRIOR).to_numpy()
if list(load(PRIOR).index) != ids:
    raise SystemExit("prior 的 id 顺序与分量不一致")
print(f"载入 {len(comp)} 个模型分量 + prior，各 {len(ids)} 行")

# ── ① 原路径（批 rank）—— 应与参考文件逐位一致 ─────────────────────────────
def rk(a):
    return rankdata(a, method="average") / (len(a) + 1.0)

base = np.zeros(len(ids))
for _slot, sw, comps in SLOTS:
    for f, cw in comps:
        base += sw * cw * rk(comp[f])
z = (1 - PRIOR_W) * base + PRIOR_W * rk(prior_raw)
score_batch = rankdata(z, method="average") / (len(z) + 1.0)

ref = load(REF)
print(f"\n[自检 1] 批 rank 路径 vs {REF}")
print(f"    spearman = {spearmanr(score_batch, ref.to_numpy()).statistic:.10f}   （应 = 1.0000000000）")
assert abs(spearmanr(score_batch, ref.to_numpy()).statistic - 1.0) < 1e-9, "批路径都对不上，先查配方"

# ── ② 逐图路径（ECDF）★ 这是要走 Docker 的路径 ──────────────────────────────
class Ecdf:
    """把 20000 个参考值存成一张**有序表**，新值 x 用 searchsorted 取平均秩。

    ⚠️ 必须是 `(lo+hi+1)/2/(n+1)`（**带那个 +1**）才能与 `rankdata(method='average')/(n+1)`
    在并列时**完全一致**；少 +1 会恒小 0.5/(n+1)，在并列组大的量（如尺寸先验，只有 43 档）上
    会让查询值落到并列组另一侧、凭空产生 ~3.5e-3 的跳变。详见 `app/fusion.py` 的同名函数。
    """

    def __init__(self, ref_vals):
        self.sorted = np.sort(np.asarray(ref_vals, dtype=np.float64))
        self.n = len(self.sorted)

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float64)
        lo = np.searchsorted(self.sorted, x, side="left")
        hi = np.searchsorted(self.sorted, x, side="right")
        return (lo + hi + 1) / 2.0 / (self.n + 1.0)


ecdfs = {f: Ecdf(comp[f]) for f in comp}
# 逐图重算
base2 = np.zeros(len(ids))
for _slot, sw, comps in SLOTS:
    for f, cw in comps:
        base2 += sw * cw * ecdfs[f](comp[f])
prior_pct = rk(prior_raw)                       # prior 列本身已是分位，直接用
z2 = (1 - PRIOR_W) * base2 + PRIOR_W * prior_pct
score_ecdf_before_final = z2
ecdf_z = Ecdf(z2)

# ── ③ 两组对比 ─────────────────────────────────────────────────────────────
print("\n[自检 2] 中间量逐位对比（批 rank vs ECDF）")
for _slot, sw, comps in SLOTS:
    for f, cw in comps:
        d = np.abs(rk(comp[f]) - ecdfs[f](comp[f])).max()
        print(f"    {f:34s} max|Δrk| = {d:.3e}")
print(f"    {'基础融合 z':34s} max|Δ| = {np.abs(z - z2).max():.3e}")

print("\n[自检 3] 最终分数（逐图 ECDF 路径）vs 参考文件")
s_ecdf = ecdf_z(z2)
sp = spearmanr(s_ecdf, ref.to_numpy()).statistic
print(f"    spearman = {sp:.10f}   max|Δscore| = {np.abs(s_ecdf - ref.to_numpy()).max():.3e}")
assert abs(sp - 1.0) < 1e-9, "ECDF 路径与参考不一致"
print("    ✅ 逐图路径与原提交文件**排序完全一致** ⇒ AUC 完全相同")

# ── ④ 落盘所有查表 ─────────────────────────────────────────────────────────
np.savez_compressed(OUT / "ecdf_models.npz",
                    **{f.replace(".csv", ""): np.sort(comp[f]) for f in comp})
np.savez_compressed(OUT / "ecdf_final.npz", sorted_z=np.sort(z2))
np.save(OUT / "prior_pct.npy", prior_pct)      # 兼容/核对用（服务端实际用 ECDF_z 即可）
meta = {
    "recipe": "gpp20_seed4_c75_prior200",
    "lb_score": 0.918887,
    "reference_file": REF,
    "n_reference": int(len(ids)),
    "prior_weight": PRIOR_W,
    "slots": [{"name": s, "slot_weight": float(sw),
               "components": [{"csv": f, "in_slot_weight": float(cw),
                               "ecdf_key": f.replace(".csv", "")} for f, cw in comps]}
              for s, sw, comps in SLOTS],
    "table_note": "每个 ecdf_key 存的是该分量在 20000 张测试图上的**升序分数表**；"
                  "score = (searchsorted(left)+searchsorted(right))/2/(n+1)",
}
(OUT / "fusion_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
print(f"\n已写出查表 -> {OUT}")
for p in sorted(OUT.iterdir()):
    print(f"    {p.name:24s} {p.stat().st_size/1024:.1f} KB")
