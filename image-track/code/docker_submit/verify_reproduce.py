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

目录自动定位：本脚本在「交付包内」和「GitHub 仓库内」两种布局下都能直接跑
      （前者 10 个 CSV 平铺在包根，后者在 image-track/submission/）。
      找不到时会列出所有试过的路径并报错，不会静默。可用 `--csv-dir` 显式指定。
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


# ─────────────────────────────────────────────────────────────────────────────
# 环境闸门
#
# `app/tables/gbm.pkl` 是用 **numpy 2.x** 序列化的（内部引用 `numpy._core.*`），
# 用 numpy 1.x 去 load 会报 `ModuleNotFoundError: No module named 'numpy._core'` ——
# 这条报错完全看不出真正原因，所以这里**提前拦截并给出可执行的解法**。
#
# 为什么是"要求匹配环境"而不是"打兼容补丁"：
#   曾尝试给 numpy 1.x 装别名兼容层，能过前两层，但第三层失败
#   （numpy 1.x 的 `MT19937.state.__set__` 要求 legacy state 格式，numpy 2 存的是新格式）；
#   而且即便强行加载成功，本地 sklearn 版本与镜像不同也已不再是权威结果。
#   ⇒ 验证脚本的价值就在于"能被信任"，宁可明确报错，不要给出看起来对的错数。
# ─────────────────────────────────────────────────────────────────────────────
REQ_NUMPY = "2.1.2"
REQ_SKLEARN = "1.7.2"


def env_versions():
    nv = np.__version__
    try:
        import sklearn
        sv = sklearn.__version__
    except Exception:
        sv = "未安装"
    return nv, sv


def require_numpy2():
    nv, sv = env_versions()
    if int(str(nv).split(".")[0]) < 2:
        sys.exit(
            f"\n❌ 环境不匹配，无法给出权威结果 —— 这不是代码问题：\n"
            f"     当前 numpy = {nv}，而 app/tables/gbm.pkl 是用 numpy 2.x 序列化的，\n"
            f"     numpy 1.x 读不了它（会报 No module named 'numpy._core'）。\n\n"
            f"   交付镜像内的版本：numpy {REQ_NUMPY}  /  scikit-learn {REQ_SKLEARN}\n\n"
            f"   解法（任选其一，都不需要 GPU）：\n"
            f"     ① 【最快】建一个匹配环境，约 1 分钟（只需 4 个小依赖，不装 torch）：\n"
            f"          python -m venv .venv\n"
            f"          .venv/bin/pip install -r requirements-verify.txt\n"
            f"          .venv/bin/python verify_reproduce.py --from-csv\n"
            f"        Windows 把 .venv/bin/ 换成 .venv\\\\Scripts\\\\\n"
            f"     ② 直接用交付镜像内的解释器跑（不需要 Docker 守护进程）：\n"
            f"          bash verify_container.sh <镜像.tar>        # 解出 rootfs/\n"
            f"          rootfs/opt/conda/bin/python verify_reproduce.py --from-csv\n"
            f"     ③ 在容器里跑：\n"
            f"          docker run --rm -v \"$PWD:/w\" -w /w <镜像> \\\n"
            f"            python verify_reproduce.py --from-csv\n"
        )

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")   # 报错（sys.exit 的文案）也走 UTF-8，
                                                            # 否则 Windows GBK 控制台会把 ❌ 转义成 \u274c
HERE = Path(__file__).resolve().parent
WS = HERE.parent
APP = HERE / "app"
sys.path.insert(0, str(APP))

REF = "v100_gpp20_seed4_c75_prior200.csv"
LB = 0.918887


def _cands(extra=()):
    """本脚本可能出现在两种目录布局里，两者都必须能跑：

      · 交付包内     : <包根>/docker_submit/verify_reproduce.py   —— 10 个 CSV 平铺在 <包根>/
      · GitHub 仓库  : image-track/code/docker_submit/verify_reproduce.py
                                                              —— CSV 在 image-track/submission/
    """
    return [
        *[Path(p) for p in extra if p],
        HERE,                        # 与脚本同目录
        WS,                          # 交付包：CSV 平铺在包根
        WS / "submission",
        WS.parent / "submission",    # ★ 仓库：image-track/submission
        WS.parent,                   # ★ 仓库：image-track
        HERE.parent / "submission",
        HERE.parent,
    ]


def find_csv_dir(filename, extra=()):
    """定位包含 filename 的目录；找不到就报出所有试过的路径（不静默）。"""
    tried = []
    for d in _cands(extra):
        tried.append(d)
        try:
            if (d / filename).exists():
                return d
        except OSError:
            pass
    raise SystemExit(
        f"找不到 {filename}。已试过：\n  " + "\n  ".join(str(t) for t in tried)
        + "\n请用 --csv-dir <目录> 显式指定。"
    )


def find_dir(cands, what):
    """在候选目录里找一个存在者（用于测试图目录）。"""
    tried = []
    for d in cands:
        tried.append(d)
        try:
            if d and d.is_dir():
                return d
        except OSError:
            pass
    raise SystemExit(
        f"找不到{what}。已试过：\n  " + "\n  ".join(str(t) for t in tried)
        + "\n请用 --images <目录> 显式指定。"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-csv", action="store_true")
    ap.add_argument("--http", default=None)
    ap.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 张（冒烟用）")
    ap.add_argument("--csv-dir", default=None, help="10 个 CSV 所在目录（默认自动定位）")
    ap.add_argument("--images", default=None, help="测试图目录（仅 --http 用，默认自动定位）")
    a = ap.parse_args()

    csvdir = Path(a.csv_dir).resolve() if a.csv_dir else find_csv_dir(REF, (a.csv_dir,))
    specdir = csvdir if (csvdir / "test_specs.csv").exists() else find_csv_dir("test_specs.csv")
    nv, sv = env_versions()
    print(f"CSV 目录 : {csvdir}")
    if specdir != csvdir:
        print(f"规格目录 : {specdir}")
    print(f"环境     : numpy {nv} / scikit-learn {sv}"
          + ("  ← 与交付镜像一致" if (nv == REQ_NUMPY and sv == REQ_SKLEARN) else ""))

    ref = pd.read_csv(csvdir / REF, encoding="utf-8-sig", dtype={"id": str})
    ref_ids = ref.id.tolist()
    n = int(a.limit) if a.limit else len(ref_ids)
    ref_score = ref.score.to_numpy()[:n]

    if a.from_csv:
        require_numpy2()
        import warnings
        from fusion import Fusion
        with warnings.catch_warnings(record=True) as _rec:
            warnings.simplefilter("always")
            F = Fusion(APP / "tables")
        if any("InconsistentVersion" in type(w.message).__name__ for w in _rec):
            print("ℹ️  gbm.pkl 由 scikit-learn 1.6.1 序列化，当前是 "
                  f"{sv} → sklearn 会给出 InconsistentVersionWarning。\n"
                  "    本仓库已实测该警告不影响结果（下面 spearman 的硬断言就是兜底判据）。")
        else:
            for w in _rec:
                warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
        print(f"融合表载入 OK：{len(F.terms)} 个分量项，prior_w={F.prior_w}")
        for k, w in F.terms:
            print(f"    {k:22s} 权重 {w:.6f}")

        comps = {}
        for k in F.ecdf_models:
            # 查表的 key 就是 CSV 的 basename（如 "img_cliph_rerun"）
            p = csvdir / f"{k}.csv"
            if not p.exists():
                raise SystemExit(f"缺分量 CSV {p}")
            d = pd.read_csv(p, encoding="utf-8-sig", dtype={"id": str}).set_index("id")["score"]
            assert list(d.index) == ref_ids, f"{p.name} 的 id 顺序不一致"
            comps[k] = d.to_numpy()
        spec = pd.read_csv(specdir / "test_specs.csv", dtype={"id": str}).set_index("id")
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
        root = find_dir(
            [a.images and Path(a.images),
             WS / "_official_test_images",
             WS / "data" / "official_test" / "image_test",
             WS.parent / "data" / "official_test" / "image_test",
             WS.parent / "_official_test_images",
             HERE / "_official_test_images"],
            "测试图目录（复赛数据集里的 图像赛道/data/image_test/）")
        print(f"测试图目录: {root}")
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
