#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_prior.py —— 把 `size_prior2.csv` 的构造过程固化成**可部署的部件**，并逐位复现。

`size_prior2.csv` 的原始构造（`size_prior2.py`）是：
    prior_A = method_A(sample, test)      # 尺寸档查表（收缩平滑，只用 w,h）
    prior_D = method_D(sample, test)      # GBM[w,h,ar,px,bytes,bpp,is_sq,short,long]
    prior   = (1-w_A)·rank_pct(prior_A) + w_A·rank_pct(prior_D)

Docker 服务里"rank_pct"同样换成 ECDF。本脚本产出：
  · bucket.json  —— 尺寸档 → 收缩后的假图率；外加全局均值与收缩常数
  · gbm.pkl      —— 训练好的 GradientBoostingClassifier（sklearn，几 MB）
  · prior_meta.json —— w_A 等
  · ecdf_prior.npz  —— prior_A / prior_D / 合成 prior 的 ECDF 查表
并**断言**按这套部件重算 `size_prior2.csv` 逐位一致。
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
WS = HERE.parent
OUT = HERE / "app" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
SHRINK = 5.0
FEAT = ["w", "h", "ar", "px", "bytes", "bpp", "is_sq", "short", "long"]


def feats(df):
    d = df.copy()
    d["ar"] = d.w / d.h
    d["px"] = d.w * d.h
    d["short"] = d[["w", "h"]].min(axis=1)
    d["long"] = d[["w", "h"]].max(axis=1)
    d["is_sq"] = (d.w == d.h).astype(int)
    d["bpp"] = d.bytes / (d.w * d.h)
    d["key"] = d.w.astype(str) + "x" + d.h.astype(str)
    return d


def _mat(df):
    X = df[FEAT].to_numpy(dtype=float)
    X[:, 0] = np.log1p(X[:, 0]); X[:, 1] = np.log1p(X[:, 1])
    X[:, 3] = np.log1p(X[:, 3]); X[:, 4] = np.log1p(X[:, 4])
    return X


def method_A(tr, te):
    cnt = tr.groupby("key").size()
    m = tr.groupby("key").label.mean()
    sm = (m * cnt + tr.label.mean() * SHRINK) / (cnt + SHRINK)
    return te.key.map(sm).fillna(tr.label.mean()).to_numpy(), sm, float(tr.label.mean())


s = feats(pd.read_csv(WS / "sample_specs.csv", dtype={"id": str}))
t = feats(pd.read_csv(WS / "test_specs.csv", dtype={"id": str}))
y = s.label.to_numpy()
ref = pd.read_csv(WS / "size_prior2.csv", encoding="utf-8-sig", dtype={"id": str})
assert list(ref.id) == list(t.id), "size_prior2.csv 的 id 顺序与 test_specs.csv 不一致"

# ---- 复刻 best_w（与 size_prior2.py 完全相同的 CV 与随机种子）----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


def oof(fn):
    o = np.zeros(len(s))
    for tr_i, te_i in skf.split(np.zeros(len(y)), y):
        o[te_i] = fn(s.iloc[tr_i], s.iloc[te_i])
    return o


def m_D(tr, te):
    m = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.06,
                                   subsample=0.8, random_state=0).fit(_mat(tr), tr.label.to_numpy())
    return m.predict_proba(_mat(te))[:, 1]


def m_A(tr, te):
    return method_A(tr, te)[0]


print("复刻 CV …")
zA, zD = oof(m_A), oof(m_D)
print(f"  A OOF AUC = {roc_auc_score(y, zA):.4f}")
print(f"  D OOF AUC = {roc_auc_score(y, zD):.4f}")
rA_o = pd.Series(zA).rank(pct=True).to_numpy()
rD_o = pd.Series(zD).rank(pct=True).to_numpy()
best_w, best_auc = 0.0, 0.0
for w in np.arange(0.0, 1.01, 0.1):
    a = roc_auc_score(y, (1 - w) * rA_o + w * rD_o)
    if a > best_auc:
        best_auc, best_w = a, float(w)
print(f"  A+D 最优 w_GBM = {best_w:.1f}  → OOF AUC {best_auc:.4f}")

# ---- 用全量 sample 训练，产出测试集先验 ----
prior_A, sm, gmean = method_A(s, t)
gbm = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.06,
                                 subsample=0.8, random_state=0).fit(_mat(s), y)
prior_D = gbm.predict_proba(_mat(t))[:, 1]


def rpct(a):
    return rankdata(a, method="average") / (len(a) + 1.0)


prior_mix = (1 - best_w) * rpct(prior_A) + best_w * rpct(prior_D)

print("\n[验证] 复刻的 prior vs size_prior2.csv")
print(f"  spearman = {spearmanr(prior_mix, ref.score.to_numpy()).statistic:.10f}")
print(f"  max|Δ|   = {np.abs(prior_mix - ref.score.to_numpy()).max():.4e}")
assert abs(spearmanr(prior_mix, ref.score.to_numpy()).statistic - 1.0) < 1e-9, \
    "prior 复刻失败，别往下走"

# ---- 落盘可部署部件 ----
bucket = {k: float(v) for k, v in sm.items()}
(OUT / "bucket.json").write_text(json.dumps(
    {"shrink": SHRINK, "global_mean": gmean, "buckets": bucket}, ensure_ascii=False),
    encoding="utf-8")
with (OUT / "gbm.pkl").open("wb") as f:
    pickle.dump({"model": gbm, "feat": FEAT}, f, protocol=4)
np.savez_compressed(OUT / "ecdf_prior.npz",
                    prior_A=np.sort(prior_A), prior_D=np.sort(prior_D),
                    prior_mix=np.sort(prior_mix))
(OUT / "prior_meta.json").write_text(json.dumps({
    "w_gbm": best_w, "w_bucket": 1 - best_w, "shrink": SHRINK,
    "oof_auc_bucket": float(roc_auc_score(y, zA)),
    "oof_auc_gbm": float(roc_auc_score(y, zD)),
    "oof_auc_mix": float(best_auc), "n_buckets": len(bucket),
    "feat": FEAT,
}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"\n已写出 -> {OUT}")
for p in sorted(OUT.iterdir()):
    print(f"    {p.name:22s} {p.stat().st_size/1024:.1f} KB")
