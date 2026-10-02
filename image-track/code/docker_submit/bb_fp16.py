#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""bb_fp16.py —— 把 3 个骨干转成 **fp16**，镜像体积从 ~18.4 GB 降到 ~9.2 GB。

为什么可以这么做而**不损失任何信息**：
    服务的推理路径本来就是 `model.half()` + `x.to(fp16)`
    （见 `app/models.py` 的 `_to_half()` / `score()`），
    所以"以 fp16 存储、以 fp16 计算"与"以 fp32 存储、加载后立刻转 fp16"**结果完全一致**。
    ⚠️ 唯一要小心的是 **head 与 LoRA 权重**：它们不转（保持原样），因为 LoRA 的 B 矩阵极小、
        转 fp16 会放大相对误差，而我们本来就用 fp32 存它们。

逐个处理：
    clip_h14_cache/open_clip_pytorch_model.bin          fp32 → fp16（保持 open_clip 的键结构）
    clip_bigg/pytorch_model-0000{1,2}-of-00002.bin      fp32 → fp16（保持分片名 + index.json 不变）
    siglip2_384/model.safetensors                       fp32 → fp16
"""
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
B = Path("/hy-tmp/build/backbones")


def sz(p):
    return p.stat().st_size if p.exists() else 0


def dirsz(d):
    return sum(f.stat().st_size for f in Path(d).rglob("*") if f.is_file())


def conv_torch_file(src, dst, min_bytes=1 << 20):
    """fp32 → fp16 的通用 .bin 转换（保持键与结构，只改浮点 dtype）。"""
    if dst.exists() and dst.stat().st_size > min_bytes:
        print(f"    skip {dst.name}（已存在 {dst.stat().st_size/1e6:.1f} MB）", flush=True)
        return
    t0 = time.time()
    obj = torch.load(src, map_location="cpu", weights_only=False)
    print(f"    载入 {src.name} ({sz(src)/1e9:.2f} GB, {time.time()-t0:.0f}s)", flush=True)

    def cast(o):
        if isinstance(o, torch.Tensor):
            return o.half() if o.is_floating_point() else o
        if isinstance(o, dict):
            return {k: cast(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return type(o)(cast(v) for v in o)
        return o

    had_sd = isinstance(obj, dict) and "state_dict" in obj
    new = cast(obj)
    torch.save(new, dst)
    del obj, new
    print(f"    写出 {dst.name} ({sz(dst)/1e9:.2f} GB, {time.time()-t0:.0f}s)", flush=True)


def conv_safetensors(src, dst):
    from safetensors.torch import load_file, save_file
    if dst.exists() and dst.stat().st_size > (1 << 20):
        print(f"    skip {dst.name}", flush=True)
        return
    t0 = time.time()
    sd = load_file(str(src))
    sd16 = {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}
    save_file(sd16, str(dst), metadata={"format": "pt"})
    print(f"    {dst.name}  {sz(dst)/1e9:.2f} GB  {time.time()-t0:.0f}s", flush=True)


print("=" * 70)
print("fp16 转换（原地：转完替换原文件，省磁盘）")
print("=" * 70)
print(f"转换前合计 {dirsz(B)/1e9:.2f} GB")

# ---- 1) CLIP ViT-H-14 (open_clip) ----
d = B / "clip_h14_cache"
src = d / "open_clip_pytorch_model.bin"
if src.exists() and sz(src) > 1e9:
    conv_torch_file(src, d / "open_clip_pytorch_model.fp16.bin")
    src.unlink()
    os.replace(d / "open_clip_pytorch_model.fp16.bin", src)
    print(f"  ✅ clip_h14_cache：{sz(src)/1e9:.2f} GB", flush=True)
else:
    print(f"  ⚠️ clip_h14_cache 权重缺失或已转换（{sz(src)/1e6:.1f} MB）", flush=True)

# ---- 2) CLIP ViT-bigG-14（分片）----
d = B / "clip_bigg"
shards = sorted(d.glob("pytorch_model-*-of-*.bin"))
print(f"\n  clip_bigg：找到 {len(shards)} 个分片", flush=True)
for s in shards:
    dst16 = s.with_suffix(".fp16.bin")
    conv_torch_file(s, dst16)
    s.unlink()
    os.replace(dst16, s)
print(f"  ✅ clip_bigg：{dirsz(d)/1e9:.2f} GB", flush=True)

# ---- 3) SigLIP2 ----
d = B / "siglip2_384"
src = d / "model.safetensors"
if src.exists() and sz(src) > 1e9:
    conv_safetensors(src, d / "model.fp16.safetensors")
    src.unlink()
    os.replace(d / "model.fp16.safetensors", src)
    print(f"  ✅ siglip2_384：{sz(src)/1e9:.2f} GB", flush=True)
else:
    print(f"  ⚠️ siglip2 权重缺失（{sz(src)/1e6:.1f} MB）", flush=True)

print()
print("=" * 70)
print(f"转换后合计 {dirsz(B)/1e9:.2f} GB")
for sub in ("clip_h14_cache", "clip_bigg", "siglip2_384"):
    print(f"  {sub:18s} {dirsz(B/sub)/1e9:6.2f} GB")
print("=" * 70)
