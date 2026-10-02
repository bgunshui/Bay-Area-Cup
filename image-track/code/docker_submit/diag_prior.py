#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_prior.py —— 定位逐图管线与提交文件之间那 1.85e-3 的残差来自哪一步。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
WS = HERE.parent
sys.path.insert(0, str(HERE / "app"))
from fusion import Fusion, ecdf            # noqa: E402

F = Fusion(HERE / "app" / "tables")
spec = pd.read_csv(WS / "test_specs.csv", dtype={"id": str})
ref_prior = pd.read_csv(WS / "size_prior2.csv", encoding="utf-8-sig", dtype={"id": str})
assert list(spec.id) == list(ref_prior.id)

W = spec.w.to_numpy(); H = spec.h.to_numpy(); NB = spec.bytes.to_numpy()
print(f"test_specs: {len(spec)} 行  列={list(spec.columns)}")
print(f"  bytes: min={NB.min()} max={NB.max()} mean={NB.mean():.0f}")

# ---- 用 fusion 里的逐图函数重算 prior，与 size_prior2.csv 比 ----
got = np.array([F.prior_pct(int(W[i]), int(H[i]), int(NB[i])) for i in range(len(spec))])
r = ref_prior.score.to_numpy()
print(f"\n[prior 逐图重算 vs size_prior2.csv]")
print(f"  spearman = {spearmanr(got, r).statistic:.10f}")
print(f"  max|Δ|   = {np.abs(got - r).max():.3e}   平均|Δ| = {np.abs(got - r).mean():.3e}")
bad = np.where(np.abs(got - r) > 1e-6)[0]
print(f"  |Δ|>1e-6 的个数 = {len(bad)}  ({len(bad)/len(r)*100:.3f}%)")

# 参考路径：直接对 size_prior2 的列做 rank/(n+1)
ref_rank = rankdata(r, method="average") / (len(r) + 1.0)
print(f"\n[对照] 若直接用 size_prior2.csv 的列做 rank，残差应为 0")
print(f"  ecdf_pm 对 mix 的映射 vs rank/(n+1): max|Δ| = "
      f"{np.abs(np.array([ecdf(F.ecdf_pm, v) for v in r[:2000]]) - ref_rank[:2000]).max():.3e}")

# ---- 看 prior_A / prior_D 两步各自的复刻情况 ----
import json
import pickle
b = json.loads((HERE / "app" / "tables" / "bucket.json").read_text(encoding="utf-8"))
with (HERE / "app" / "tables" / "gbm.pkl").open("rb") as f:
    g = pickle.load(f)
gbm, feat = g["model"], g["feat"]
# 用 test_specs 整表复算 A / D
pa = np.array([b["buckets"].get(f"{int(W[i])}x{int(H[i])}", b["global_mean"]) for i in range(len(spec))])
X = np.column_stack([W.astype(float), H.astype(float), W / H, (W * H).astype(float), NB.astype(float),
                     NB / (W * H), (W == H).astype(int),
                     np.minimum(W, H).astype(float), np.maximum(W, H).astype(float)])
Xf = X[:, [feat.index(c) for c in feat]]
Xf[:, 0] = np.log1p(Xf[:, 0]); Xf[:, 1] = np.log1p(Xf[:, 1])
Xf[:, 3] = np.log1p(Xf[:, 3]); Xf[:, 4] = np.log1p(Xf[:, 4])
pd_ = gbm.predict_proba(Xf)[:, 1]
wg = float(json.loads((HERE / "app" / "tables" / "prior_meta.json").read_text(encoding="utf-8"))["w_gbm"])
mix = (1 - wg) * rankdata(pa, method="average") / (len(pa) + 1.0) + \
      wg * rankdata(pd_, method="average") / (len(pd_) + 1.0)
print(f"\n[整表复算 A/D 混合 vs size_prior2.csv]")
print(f"  spearman = {spearmanr(mix, r).statistic:.10f}   max|Δ| = {np.abs(mix - r).max():.3e}")
print(f"  ⇒ 残差 {'来自逐图查表' if np.abs(mix-r).max() < 1e-9 else '来自 A/D 复算本身'}")
