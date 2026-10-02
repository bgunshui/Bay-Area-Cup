#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fusion.py —— 逐图可计算的融合（ECDF 版 rank 加权）。

═══ 与原提交文件的关系 ═══
原提交（`v100_gpp20_seed4_c75_prior200.csv`，LB **0.918887**）用 `build_final.py` 的
**批 rank** 构造。本模块把每个 `rankdata(x)/(n+1)` 换成 **ECDF(x)**：
    · 在那 20000 张测试图上两者恒等（ECDF 就是 rank，差一个 0.5/(n+1) 的常数偏移）
    · 对**新图**是良定义的单调映射（= 用测试集分布做分位数标定）
⇒ 已实测：逐图路径与原文件 **spearman = 1.0000000000**（`make_tables.py` 自检 3）。

═══ 服务端调用顺序 ═══
    comp[cliph_rerun] = model.score(...)          # 8 个模型
    ...
    score = fusion.fuse(comp, w, h, nbytes)       # → [0,1]
"""
import json
import pickle
from pathlib import Path

import numpy as np


def ecdf(sorted_arr, x):
    """把查询值映射到参考分布上的**平均秩百分位**。

    ⚠️ 必须是 `(lo + hi + 1) / 2 / (n+1)`，**不能少那个 +1**。
        · lo = 第一个 >= x 的下标；hi = 第一个 > x 的下标。
        · 对表中已有的、落在并列组 [a..b]（0 基）的值：lo=a, hi=b+1
          ⇒ (lo+hi+1)/2 = (a+b+2)/2 = **rankdata(method='average') 的平均秩**（1 基秩 a+1..b+1）✓
        · 少掉 +1 就会**恒小 0.5/(n+1)**；先验只有 43 个尺寸档、并列组很大，
          这个常数偏移会让查询值落到并列组的**另一侧**，凭空产生 ~3.5e-3 的跳变
          （实测残差正好 = 并列组大小/(n+1)，定位过程见 `diag_prior2.py`）。
    """
    lo = np.searchsorted(sorted_arr, x, side="left")
    hi = np.searchsorted(sorted_arr, x, side="right")
    return float((lo + hi + 1) / 2.0 / (len(sorted_arr) + 1.0))


class Fusion:
    def __init__(self, tables_dir):
        d = Path(tables_dir)
        z = np.load(d / "ecdf_models.npz")
        self.ecdf_models = {k: z[k] for k in z.files}
        self.ecdf_final = np.load(d / "ecdf_final.npz")["sorted_z"]
        self.meta = json.loads((d / "fusion_meta.json").read_text(encoding="utf-8"))
        self.bucket = json.loads((d / "bucket.json").read_text(encoding="utf-8"))
        with (d / "gbm.pkl").open("rb") as f:
            g = pickle.load(f)
        self.gbm, self.feat = g["model"], g["feat"]
        self.pmeta = json.loads((d / "prior_meta.json").read_text(encoding="utf-8"))
        p = np.load(d / "ecdf_prior.npz")
        self.ecdf_pa, self.ecdf_pd, self.ecdf_pm = p["prior_A"], p["prior_D"], p["prior_mix"]

        # 展开配方：[(ecdf_key, 最终权重), ...]，权重 = 槽权重 × 槽内权重
        self.terms = []
        for slot in self.meta["slots"]:
            for c in slot["components"]:
                self.terms.append((c["ecdf_key"], float(slot["slot_weight"]) * float(c["in_slot_weight"])))
        self.prior_w = float(self.meta["prior_weight"])
        assert abs(sum(w for _, w in self.terms) * (1 - self.prior_w) / (1 - self.prior_w)
                   - sum(w for _, w in self.terms)) < 1e-12

    # ---- 尺寸先验 ----
    def _prior_features(self, w, h, nbytes):
        ar = w / h
        return {"w": w, "h": h, "ar": ar, "px": w * h, "bytes": nbytes,
                "bpp": nbytes / (w * h), "is_sq": int(w == h),
                "short": min(w, h), "long": max(w, h)}

    def prior_pct(self, w, h, nbytes):
        f = self._prior_features(w, h, nbytes)
        b = self.bucket
        pa = b["buckets"].get(f"{w}x{h}", b["global_mean"])
        x = np.array([[f[k] for k in self.feat]], dtype=float)
        x[:, 0] = np.log1p(x[:, 0]); x[:, 1] = np.log1p(x[:, 1])
        x[:, 3] = np.log1p(x[:, 3]); x[:, 4] = np.log1p(x[:, 4])
        pd_ = float(self.gbm.predict_proba(x)[0, 1])
        wg = float(self.pmeta["w_gbm"])
        mix = (1 - wg) * ecdf(self.ecdf_pa, pa) + wg * ecdf(self.ecdf_pd, pd_)
        return ecdf(self.ecdf_pm, mix)

    # ---- 总融合 ----
    def fuse(self, comp_scores, w, h, nbytes):
        base = 0.0
        for key, wt in self.terms:
            if key not in comp_scores:
                raise KeyError(f"缺少分量 {key}")
            if key not in self.ecdf_models:
                raise KeyError(f"查表里没有 {key}")
            base += wt * ecdf(self.ecdf_models[key], float(comp_scores[key]))
        z = (1 - self.prior_w) * base + self.prior_w * self.prior_pct(w, h, nbytes)
        return ecdf(self.ecdf_final, z)
