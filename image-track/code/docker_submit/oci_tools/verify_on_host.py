#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_on_host.py（gpu4090b 版）—— 用镜像里的同一份代码+权重，在真实测试图上真跑一遍。

环境：gpu4090b 的 torch 是 **2.5.1+cu124**，与镜像里的**完全一致** ⇒ 这个验证最有代表性。

判据：
    · 8 个分量 vs 冻结 CSV 的 spearman ≥ 0.98
      （不是 1.0：冻结 CSV 是 BS=32 批量 + bf16 autocast 产出，服务是 BS=1）
    · 最终 fusion vs 提交文件 spearman ≥ 0.98
"""
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import spearmanr

APP = Path("/hy-tmp/build/buildkit/docker_submit/app")
REF = Path("/hy-tmp/verify/ref")
sys.path.insert(0, str(APP))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=200)
ap.add_argument("--device", default="cuda")
ap.add_argument("--img-dir", default=None)
ap.add_argument("--out", default="/hy-tmp/verify/host_scores.csv")
a = ap.parse_args()

print("=" * 74)
print("镜像推理链路 · gpu4090b 端到端验证（torch 与镜像一致）")
print("=" * 74)
print(f"app   = {APP}")
print(f"torch = {torch.__version__}  cuda={torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"GPU   = {torch.cuda.get_device_name(0)}  {p.total_memory/2**30:.1f} GiB")

img_root = a.img_dir
if not img_root:
    # 不靠 shell 传中文路径（会被转义坏），直接在远端 glob
    cands = [c for c in Path("/hy-tmp/dl/fusai").rglob("image_test")
             if c.is_dir() and next(c.glob("*.jpg"), None)]
    if not cands:
        cands = [c for c in Path("/hy-tmp").rglob("image_test")
                 if c.is_dir() and next(c.glob("*.jpg"), None)]
    if not cands:
        sys.exit("❌ 找不到含 *.jpg 的 image_test 目录")
    img_root = max(cands, key=lambda c: len(list(c.glob("*.jpg"))))
img_root = Path(img_root)
n_img = len(list(img_root.glob("*.jpg")))
print(f"测试图 = {img_root}  （{n_img} 张）")

sub = pd.read_csv(REF / "v100_gpp20_seed4_c75_prior200.csv", encoding="utf-8-sig",
                  dtype={"id": str})
ref_ids = sub.id.tolist()
pick = [i for i in ref_ids[:a.n] if (img_root / i).exists()][:a.n]
print(f"实际验证 {len(pick)} 张")

print("\n--- 加载 8 个分量 ---")
t0 = time.time()
from models import ComponentEnsemble          # noqa: E402
# ⚠️ 骨干目录允许用环境变量覆盖：**镜像里**它在 `/app/backbones`，
#    而这台构建机上它留在 `/hy-tmp/build/backbones`（没进 buildkit 源目录）。
#    加这个开关是为了让验证脚本不必依赖源码目录的布局。
BB = os.environ.get("BACKBONES_DIR", str(APP / "backbones"))
print(f"  骨干目录 = {BB}")
ens = ComponentEnsemble(str(APP / "weights"), BB,
                        device=a.device, dtype=torch.float16)
print(f"  加载用时 {time.time()-t0:.1f}s")
from fusion import Fusion                     # noqa: E402
F = Fusion(APP / "tables")
print(f"  融合表：{len(F.terms)} 项，prior_w={F.prior_w}")
print(f"  分量键：{list(ens.heads)}")

print(f"\n--- 推理 ---")
recs = []
t0 = time.time()
for i, iid in enumerate(pick):
    p = img_root / iid
    raw = p.read_bytes()
    im = Image.open(p); im.load(); im = im.convert("RGB")
    comp = {k: ens.score(k, im) for k in ens.heads}
    sc = F.fuse(comp, im.size[0], im.size[1], len(raw))
    recs.append({"id": iid, **comp, "fused": sc})
    if (i + 1) % 25 == 0:
        el = time.time() - t0
        print(f"  {i+1}/{len(pick)}  {el:.0f}s  {el/(i+1):.2f} s/张", flush=True)

df = pd.DataFrame(recs)
df.to_csv(a.out, index=False)
el = time.time() - t0
print(f"\n  写出 {a.out}  用时 {el:.0f}s（{el/len(pick):.2f} s/张）")

print("\n" + "=" * 74)
print("对照冻结产物")
print("=" * 74)
print(f"  {'分量':26s} {'spearman':>10s} {'pearson':>10s} {'max|Δ|':>9s}   冻结 mean / 本次 mean")
bad = 0
for k in ens.heads:
    p = REF / f"{k}.csv"
    if not p.exists():
        print(f"  {k:26s}  (缺 {p.name})"); continue
    r = pd.read_csv(p, encoding="utf-8-sig", dtype={"id": str}).set_index("id")["score"]
    x = df.set_index("id")[k]
    x, r = x.loc[pick], r.loc[pick]
    sp = spearmanr(x, r).statistic
    pe = np.corrcoef(x, r)[0, 1]
    md = np.abs(x.to_numpy() - r.to_numpy()).max()
    if sp < 0.98:
        bad += 1
    print(f"  {k:26s} {sp:>10.6f} {pe:>10.6f} {md:>9.4f}   {r.mean():.4f} / {x.mean():.4f}"
          + ("" if sp >= 0.98 else "  ⚠️"))

r = sub.set_index("id")["score"].loc[pick]
sp = spearmanr(df.set_index("id")["fused"].loc[pick], r).statistic
print(f"\n  【最终 fusion】spearman vs 提交文件 = {sp:.6f}"
      f"   {'✅ 通过' if sp >= 0.98 else '❌ 不达标'}")
print(f"  分量不达标：{bad}/{len(ens.heads)}")
print("\n结论：" + ("✅ **镜像推理链路验证通过**" if (sp >= 0.98 and bad == 0)
                 else "⚠️ 有偏差，需排查"))
