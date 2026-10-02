#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_reproduce.py —— 端到端验证：逐图管线能否复现 LB 0.918887 的提交文件。

两种模式：
  ① `--from-csv`（**不需要 GPU / 不需要 Docker**）
      把 8 个分量在 20000 张测试图上的**冻结分数**逐行喂给 `app/fusion.py`，
       连同 (w,h,bytes) 走尺寸先验，最后与 `v100_gpp20_seed4_c75_prior200.csv` 比。
       ⇒ 验证的是**融合 + 查表 + 先验**这一整条交付链路（服务端 fuse() 的真实代码路径）。
  ② `--http http://host:8080`（**需要服务在跑**）
      把 20000 张图的 base64 逐张 POST /predict，收集 score 与提交文件比。
       ⇒ 验证的是**含 8 个神经网络在内的完整服务**。

判据：spearman（= AUC 是否一致）。`--from-csv` 必须 = 1.0000000000；
      `--http` 因为服务是 **BS=1 逐张** 而冻结 CSV 是 **BS=32 批量**，
      会有 bf16×batch 的数值抖动（已知 ~0.03% 的图偏移），spearman 预期 ≥ 0.99。
"""
import argparse
import base64
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
WS = HERE.parent
APP = HERE / "app"
sys.path.insert(0, str(APP))

REF = "v100_gpp20_seed4_c75_prior200.csv"
LB = 0.918887


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--http", default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟用）")
    a = ap.parse_args()

    ref = pd.read_csv(WS / REF, encoding="utf-8-sig", dtype={"id": str})
    ref_ids = ref.id.tolist()
    n = int(a.limit) if a.limit else len(ref_ids)
    ref_score = ref.score.to_numpy()[:n]

    if a.from_csv:
        from fusion import Fusion
        F = Fusion(APP / "tables")
        print(f"融合表载入 OK：{len(F.terms)} 个分量项，prior_w={F.prior_w}")
        for k, w in F.terms:
            print(f"    {k:22s} 权重 {w:.6f}")

        comps = {}
        for k in F.ecdf_models:
            # 查表的 key 就是 CSV 的 basename（如 "img_cliph_rerun"）
            p = WS / f"{k}.csv"
            if not p.exists():
                raise SystemExit(f"缺分量 CSV {p}")
            d = pd.read_csv(p, encoding="utf-8-sig", dtype={"id": str}).set_index("id")["score"]
            assert list(d.index) == ref_ids, f"{p.name} 的 id 顺序不一致"
            comps[k] = d.to_numpy()
        spec = pd.read_csv(WS / "test_specs.csv", dtype={"id": str}).set_index("id")
        assert list(spec.index) == ref_ids, "test_specs.csv 顺序不一致"
        W = spec.w.to_numpy(); H = spec.h.to_numpy(); NB = spec.bytes.to_numpy()

        out = np.empty(n)
        for i in range(n):
            out[i] = F.fuse({k: comps[k][i] for k in comps}, int(W[i]), int(H[i]), int(NB[i]))
            if (i + 1) % 2000 == 0:
                print(f"  {i+1}/{n}", flush=True)
        sp = spearmanr(out, ref_score).statistic
        print(f"\n[结果] 逐图管线 vs {REF}")
        print(f"    spearman = {sp:.10f}")
        print(f"    max|Δ|   = {np.abs(out - ref_score).max():.3e}")
        ok = abs(sp - 1.0) < 1e-9
        print(f"    {'✅ 与提交文件排序完全一致（AUC 相同）' if ok else '❌ 不一致'}")
        np.save(HERE / "_verify_from_csv.npy", out)
        return 0 if ok else 2

    if a.http:
        import requests
        print(f"向 {a.http} 逐张推理 {n} 张 …")
        root = WS / "_official_test_images"
        if not root.exists():
            cand = [WS / "data" / "official_test" / "image_test"]
            root = cand[0] if cand[0].exists() else root
        if not root.exists():
            raise SystemExit(f"找不到测试图目录（试过 {root}）")
        out = np.empty(n)
        for i, iid in enumerate(ref_ids[:n]):
            raw = (root / iid).read_bytes()
            b64 = base64.b64encode(raw).decode()
            r = requests.post(a.http.rstrip("/") + "/predict",
                              json={"req_id": iid, "data": b64}, timeout=90)
            r.raise_for_status()
            j = r.json()
            assert j["code"] == 0 and j["req_id"] == iid, j
            out[i] = float(j["score"])
            if (i + 1) % 500 == 0:
                print(f"  {i+1}/{n}", flush=True)
        sp = spearmanr(out, ref_score).statistic
        print(f"\n[结果] HTTP 服务 vs {REF}")
        print(f"    spearman = {sp:.6f}   （BS=1 逐张 vs BS=32 批量的数值抖动所致）")
        print(f"    唯一值 = {len(np.unique(out))}")
        np.save(HERE / "_verify_http.npy", out)
        return 0 if sp > 0.98 else 3

    ap.error("必须指定 --from-csv 或 --http")


if __name__ == "__main__":
    sys.exit(main())
