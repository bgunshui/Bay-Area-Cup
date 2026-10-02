#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""diag_prior2.py —— 逐图查表 vs 整表复算，逐步比对 A / D / mix 三处。"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
H = Path(__file__).resolve().parent
WS = H.parent
sys.path.insert(0, str(H / "app"))
from fusion import Fusion, ecdf            # noqa: E402

T = H / "app" / "tables"
F = Fusion(T)
spec = pd.read_csv(WS / "test_specs.csv", dtype={"id": str})
W = spec.w.to_numpy().astype(int); Hh = spec.h.to_numpy().astype(int); NB = spec.bytes.to_numpy().astype(int)
b = json.loads((T / "bucket.json").read_text(encoding="utf-8"))
with (T / "gbm.pkl").open("rb") as f:
    g = pickle.load(f)
gbm, feat = g["model"], g["feat"]

# ---- 整表 A / D（与 make_prior 同构）----
pa_tab = np.array([b["buckets"].get(f"{w}x{h}", b["global_mean"]) for w, h in zip(W, Hh)])
print(f"尺寸档命中率 = {np.mean([f'{w}x{h}' in b['buckets'] for w, h in zip(W, Hh)])*100:.1f}%"
      f"   档数={len(b['buckets'])}")

X = np.column_stack([W, Hh, W / Hh, W * Hh, NB, NB / (W * Hh), (W == Hh).astype(int),
                     np.minimum(W, Hh), np.maximum(W, Hh)]).astype(float)
X[:, 0] = np.log1p(X[:, 0]); X[:, 1] = np.log1p(X[:, 1])
X[:, 3] = np.log1p(X[:, 3]); X[:, 4] = np.log1p(X[:, 4])
pd_tab = gbm.predict_proba(X)[:, 1]

# ---- 逐图（走 Fusion 的真实代码路径）----
pa_one = np.array([F.bucket["buckets"].get(f"{w}x{h}", F.bucket["global_mean"])
                   for w, h in zip(W, Hh)])
pd_one = np.empty(len(spec))
for i in range(len(spec)):
    f_ = F._prior_features(int(W[i]), int(Hh[i]), int(NB[i]))
    x = np.array([[f_[k] for k in feat]], dtype=float)
    x[:, 0] = np.log1p(x[:, 0]); x[:, 1] = np.log1p(x[:, 1])
    x[:, 3] = np.log1p(x[:, 3]); x[:, 4] = np.log1p(x[:, 4])
    pd_one[i] = float(F.gbm.predict_proba(x)[0, 1])

print(f"\nA：max|Δ| = {np.abs(pa_tab - pa_one).max():.3e}")
print(f"D：max|Δ| = {np.abs(pd_tab - pd_one).max():.3e}")
print(f"   pd_tab[:5] = {pd_tab[:5]}")
print(f"   pd_one[:5] = {pd_one[:5]}")

rA_tab = rankdata(pa_tab, method="average") / (len(pa_tab) + 1.0)
rD_tab = rankdata(pd_tab, method="average") / (len(pd_tab) + 1.0)
rA_one = np.array([ecdf(F.ecdf_pa, v) for v in pa_tab])
rD_one = np.array([ecdf(F.ecdf_pd, v) for v in pd_tab])
print(f"\nrA：max|Δ| = {np.abs(rA_tab - rA_one).max():.3e}")
print(f"rD：max|Δ| = {np.abs(rD_tab - rD_one).max():.3e}")

wg = float(json.loads((T / "prior_meta.json").read_text(encoding="utf-8"))["w_gbm"])
mix_tab = (1 - wg) * rA_tab + wg * rD_tab
mix_one = (1 - wg) * rA_one + wg * rD_one
print(f"mix：max|Δ| = {np.abs(mix_tab - mix_one).max():.3e}")

ref = pd.read_csv(WS / "size_prior2.csv", encoding="utf-8-sig").score.to_numpy()
print(f"\nmix_tab vs size_prior2.csv  spearman={spearmanr(mix_tab, ref).statistic:.10f} max|Δ|={np.abs(mix_tab-ref).max():.3e}")
srt = np.sort(mix_tab)
out_tab = np.array([ecdf(srt, v) for v in mix_tab])
out_one = np.array([ecdf(F.ecdf_pm, v) for v in mix_one])
print(f"最终 prior：max|Δ| = {np.abs(out_tab - out_one).max():.3e}")
print(f"  ecdf_pm 与 np.sort(mix_tab) 一致？ max|Δ| = {np.abs(F.ecdf_pm - srt).max():.3e}")
print(f"  ecdf_pa 与 np.sort(pa_tab) 一致？ max|Δ| = {np.abs(F.ecdf_pa - np.sort(pa_tab)).max():.3e}")
print(f"  ecdf_pd 与 np.sort(pd_tab) 一致？ max|Δ| = {np.abs(F.ecdf_pd - np.sort(pd_tab)).max():.3e}")
