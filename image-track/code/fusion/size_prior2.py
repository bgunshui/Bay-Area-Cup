# -*- coding: utf-8 -*-
"""size_prior2.py: 强化版「规格先验」—— 比较多种方案，严格 5-fold CV 选最优。

现状: 纯"尺寸档查表"先验 OOF AUC 0.8172，但测试集 3626 种尺寸里只有 80.5% 命中 sample 的 43 档，
      剩余 19.5% 只能 fallback 全局均值。本脚本比较:
  A) 尺寸档查表(收缩平滑)          —— 基线
  B) 尺寸档 + 长宽比近似档(容差匹配)
  C) KNN(在归一化尺寸空间找最近邻居)
  D) GBM/LogReg 多特征(w,h,ar,px,bytes,bpp,is_square,short,long)
选 CV 最优者产出 size_prior2.csv。
"""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
WS = r"D:\湾区杯\图像赛道"


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


s = feats(pd.read_csv(f"{WS}\\sample_specs.csv", dtype={"id": str}))
t = feats(pd.read_csv(f"{WS}\\test_specs.csv", dtype={"id": str}))
y = s.label.to_numpy()
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ---------- A) 尺寸档查表 ----------
def method_A(tr, te):
    cnt = tr.groupby("key").size()
    m = tr.groupby("key").label.mean()
    sm = (m * cnt + tr.label.mean() * 5.0) / (cnt + 5.0)
    return te.key.map(sm).fillna(tr.label.mean()).to_numpy()

# ---------- B) 尺寸档 + 容差匹配(±2px 内的同比例尺寸) ----------
def method_B(tr, te):
    base = method_A(tr, te)
    tr_keys = tr[["w", "h", "label"]].copy()
    out = base.copy()
    miss = ~te.key.isin(set(tr.key)).to_numpy()
    if miss.sum():
        for i in np.where(miss)[0]:
            w0, h0 = te.w.iloc[i], te.h.iloc[i]
            ar0 = w0 / h0
            cand = tr_keys[(tr_keys.w.sub(w0).abs() <= 3) & (tr_keys.h.sub(h0).abs() <= 3)]
            if len(cand) == 0:
                cand = tr_keys[(tr_keys.w / tr_keys.h).sub(ar0).abs() <= 0.02]
            if len(cand) >= 3:
                out[i] = cand.label.mean()
    return out

# ---------- C) KNN 尺寸空间 ----------
def method_C(tr, te, k=25):
    X = np.column_stack([np.log(tr.w), np.log(tr.h)])
    X = (X - X.mean(0)) / X.std(0)
    Q = np.column_stack([np.log(te.w), np.log(te.h)])
    Q = (Q - np.column_stack([np.log(tr.w), np.log(tr.h)]).mean(0)) / np.column_stack([np.log(tr.w), np.log(tr.h)]).std(0)
    lab = tr.label.to_numpy()
    d2 = ((Q[:, None, :] - X[None, :, :]) ** 2).sum(-1)
    idx = np.argsort(d2, axis=1)[:, :k]
    return lab[idx].mean(1)

# ---------- D) 逻辑回归 / GBM 多特征 ----------
FEAT = ["w", "h", "ar", "px", "bytes", "bpp", "is_sq", "short", "long"]

def _mat(df):
    X = df[FEAT].to_numpy(dtype=float)
    X[:, 0] = np.log1p(X[:, 0]); X[:, 1] = np.log1p(X[:, 1])
    X[:, 3] = np.log1p(X[:, 3]); X[:, 4] = np.log1p(X[:, 4])
    return X

def method_D(tr, te):
    from sklearn.ensemble import GradientBoostingClassifier
    m = GradientBoostingClassifier(n_estimators=120, max_depth=3, learning_rate=0.06,
                                   subsample=0.8, random_state=0)
    m.fit(_mat(tr), tr.label.to_numpy())
    return m.predict_proba(_mat(te))[:, 1]


print("=== 5-fold CV 比较（尺寸先验各方案）===")
methods = {"A 尺寸档查表": method_A, "B 档+容差匹配": method_B, "C KNN(k=25)": method_C, "D GBM 多特征": method_D}
oofs = {}
for nm, fn in methods.items():
    oof = np.zeros(len(s))
    for tr_i, te_i in skf.split(np.zeros(len(y)), y):
        tr, te = s.iloc[tr_i], s.iloc[te_i]
        oof[te_i] = fn(tr, te)
    auc = roc_auc_score(y, oof)
    oofs[nm] = (auc, oof)
    print(f"  {nm:16s} OOF AUC = {auc:.4f}")

best_nm = max(oofs, key=lambda k: oofs[k][0])
print(f"\n最优: {best_nm}  (AUC {oofs[best_nm][0]:.4f})")

# 组合 A+D（低相关方法互补）
za = oofs["A 尺寸档查表"][1]; zd = oofs["D GBM 多特征"][1]
ra = pd.Series(za).rank(pct=True).to_numpy(); rd = pd.Series(zd).rank(pct=True).to_numpy()
best_w, best_auc = 0, 0
for w in np.arange(0.0, 1.01, 0.1):
    a = roc_auc_score(y, (1 - w) * ra + w * rd)
    if a > best_auc:
        best_auc, best_w = a, w
print(f"  A+D 最优混合 w_GBM={best_w:.1f} -> OOF AUC {best_auc:.4f}")

# ---- 用全量 sample 训练，产出测试集先验 ----
prior_A = method_A(s, t)
prior_D = method_D(s, t)
rA = pd.Series(prior_A).rank(pct=True).to_numpy()
rD = pd.Series(prior_D).rank(pct=True).to_numpy()
final = (1 - best_w) * rA + best_w * rD
out = pd.DataFrame({"id": t.id, "score": final})
out.to_csv(f"{WS}\\size_prior2.csv", index=False)
print(f"\nwrote {WS}\\size_prior2.csv   (A weight={1-best_w:.1f}, GBM weight={best_w:.1f})")
print(f"测试集先验分布: mean={final.mean():.3f} std={final.std():.3f}")
