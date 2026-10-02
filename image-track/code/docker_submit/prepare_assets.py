#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""prepare_assets.py —— 把构建镜像所需的**全部资产**归拢到 `app/` 下。

产出：
  app/weights/            8 个 LoRA+head 权重（约 0.95 GB）—— 从本地工程目录拷贝
  app/backbones/
      clip_h14_cache/     open_clip CLIP ViT-H-14 的 HF 缓存（≈3.7 GB）
      clip_bigg/          HF CLIPVisionModel（ViT-bigG-14，≈7.4 GB fp32）
      siglip2_384/        HF SiglipVisionModel（so400m-patch14-384，≈1.8 GB）
  （app/tables/ 由 make_tables.py / make_prior.py 产出，本脚本只校验存在）

用法：
  python prepare_assets.py                 # 拷权重 + 下载三个骨干（需要网络，走 hf-mirror）
  python prepare_assets.py --skip-download # 只拷权重（骨干已在别处备好时用）
  python prepare_assets.py --endpoint https://hf-mirror.com
  python prepare_assets.py --copy-backbones-hint /hy-tmp/sjmodels   # 提示从实例盘拷贝
"""
import argparse
import hashlib
import os
import shutil
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
WS = HERE.parent
APP = HERE / "app"
W = APP / "weights"
B = APP / "backbones"
T = APP / "tables"

# 权重来源：本地工程目录（相对 WS）
SRC = {
    "cliph_rerun.pt":       "_gpu4090_final/cliph_rerun_last.pt",
    "cliph_s20261101.pt":   "_seed_weights/cliph_s20261101_last.pt",
    "cliph_s20261102.pt":   "_seed_weights/cliph_s20261102_last.pt",
    "cliph_s20261103.pt":   "_seed_weights/cliph_s20261103_last.pt",
    "gpp.pt":               "lora_ckpt/gpp/best_lora_only.pt",
    "bigg_rerun2.pt":       "_backup_bigg_rerun2/best.pt",
    "bigg_rerun3.pt":       "_backup_bigg_rerun3/best.pt",
    "bigg_s384_retrain.pt": "_ctrl_s384r/best_lora_only.pt",
}
BACKBONES = [
    ("clip_h14_cache", "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
     ["open_clip_pytorch_model.bin", "config.json"]),
    ("clip_bigg", "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
     ["config.json", "model.safetensors", "preprocessor_config.json"]),
    ("siglip2_384", "google/siglip2-so400m-patch14-384",
     ["config.json", "model.safetensors", "preprocessor_config.json"]),
]


def md5(p, blk=1 << 22):
    h = hashlib.md5()
    with open(p, "rb") as f:
        while True:
            b = f.read(blk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def copy_weights():
    W.mkdir(parents=True, exist_ok=True)
    import torch
    print("=== ① 归拢 8 个 LoRA+head 权重 ===")
    total = 0
    for dst, rel in SRC.items():
        s = WS / rel
        if not s.exists():
            raise SystemExit(f"❌ 缺权重 {s}")
        d = W / dst
        if not d.exists() or d.stat().st_size != s.stat().st_size:
            shutil.copy2(s, d)
        # ---- 校验：必须是 LoRA+head 的 checkpoint，张量数要够 ----
        st = torch.load(d, map_location="cpu", weights_only=False)
        lora = st.get("lora", st.get("vision"))
        if lora is None:
            raise SystemExit(f"❌ {d.name} 顶层键异常 {list(st.keys())[:6]}")
        n_lora = sum(1 for k in lora if "lora_" in k)
        r = next((int(v.shape[0]) for k, v in lora.items()
                  if "lora_A" in k and getattr(v, "dim", lambda: 0)() == 2), None)
        has_head = "head" in st
        if not (n_lora and r and has_head):
            raise SystemExit(f"❌ {d.name}: LoRA={n_lora} r={r} head={has_head}")
        total += d.stat().st_size
        print(f"  ✅ {dst:24s} {d.stat().st_size/1e6:7.1f} MB  LoRA={n_lora} r={r} head={has_head}")
    print(f"  权重合计 {total/1e6:.0f} MB")


def download_backbones(endpoint):
    os.environ["HF_ENDPOINT"] = endpoint
    from huggingface_hub import snapshot_download
    print(f"\n=== ② 下载 3 个骨干（endpoint={endpoint}）===")
    for name, repo, must in BACKBONES:
        out = B / name
        out.mkdir(parents=True, exist_ok=True)
        print(f"  --- {name}  <- {repo}")
        try:
            if name == "clip_h14_cache":
                # open_clip 走 HF 缓存目录结构，直接下到 cache_dir
                snapshot_download(repo_id=repo, cache_dir=str(out),
                                  allow_patterns=["*.bin", "*.json", "*.txt"])
                got = list(out.rglob("open_clip_pytorch_model.bin"))
                if not got:
                    raise RuntimeError("缓存里没有 open_clip_pytorch_model.bin")
            else:
                snapshot_download(repo_id=repo, local_dir=str(out),
                                  allow_patterns=must + ["*.json", "*.txt"],
                                  local_dir_use_symlinks=False)
                miss = [m for m in must if not (out / m).exists()]
                if miss:
                    raise RuntimeError(f"缺文件 {miss}")
        except Exception as e:
            print(f"  ❌ {name} 下载失败：{type(e).__name__}: {e}")
            print(f"     → 可从训练实例盘直接拷贝：scp -P <port> -r root@<host>:/hy-tmp/sjmodels/{name} {out}")
            raise
        sz = sum(f.stat().st_size for f in out.rglob("*") if f.is_file())
        print(f"  ✅ {name}  {sz/1e9:.2f} GB")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--endpoint", default="https://hf-mirror.com")
    ap.add_argument("--weights-only", action="store_true")
    a = ap.parse_args()

    copy_weights()
    if not a.skip_download and not a.weights_only:
        download_backbones(a.endpoint)

    print("\n=== ③ 校验查表 ===")
    need = ["ecdf_models.npz", "ecdf_final.npz", "fusion_meta.json",
            "bucket.json", "gbm.pkl", "prior_meta.json", "ecdf_prior.npz"]
    miss = [n for n in need if not (T / n).exists()]
    if miss:
        raise SystemExit(f"❌ 缺查表 {miss} —— 先跑 make_tables.py 与 make_prior.py")
    for n in need:
        print(f"  ✅ {n:22s} {(T/n).stat().st_size/1024:7.1f} KB")

    print("\n=== ④ 体积预估 ===")
    def du(d):
        return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0
    base_w = sum(f.stat().st_size for f in B.rglob("*") if f.is_file()) if B.exists() else 0
    print(f"  权重        {du(W)/1e9:6.2f} GB")
    print(f"  骨干        {base_w/1e9:6.2f} GB")
    print(f"  查表+代码   {(du(T)+du(APP/'__pycache__'))/1e6:6.1f} MB")
    print(f"  ⇒ 构建上下文合计 ≈ {(du(W)+base_w+du(T))/1e9:.1f} GB（镜像在此基础上加 torch 等运行时约 6-7 GB）")
    print("\n下一步： bash build.sh")


if __name__ == "__main__":
    main()
