#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_bigg_safetensors.py —— 把 bigG 的分片 `.bin` 合并成单个 `model.safetensors`。

为什么必须做（验证时实测抓到）：
    HF 的分片 `.bin` 是 **torch pickle** 格式，`torch.load` 在 **torch ≥ 2.6** 里默认
    `weights_only=True`，直接拒绝加载：
        `_pickle.UnpicklingError: Weights only load failed. In PyTorch 2.6, we changed the
         default value of the weights_only argument in torch.load from False to True`
    镜像里是 torch 2.5.1（默认 False）**不会触发**，但：
      · 评测平台若换了 torch 版本就会崩；
      · pickle 格式本身也不如 safetensors 安全/可移植。
    ⇒ 转成 `model.safetensors`（HF 会**优先**用它），彻底消除这个不确定性。

⚠️ 只改**存储格式**，张量一个不动（含文本塔，虽然推理不用它 —— 不做任何裁剪）。
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("/hy-tmp/build/buildkit/docker_submit/app/backbones/clip_bigg")
OUT = D / "model.safetensors"

shards = sorted(D.glob("pytorch_model-*-of-*.bin"))
print(f"bigG 分片 {len(shards)} 个：")
for s in shards:
    print(f"  {s.name}  {s.stat().st_size/1e9:.2f} GB")

if OUT.exists() and OUT.stat().st_size > 1e9:
    print(f"✅ {OUT.name} 已存在（{OUT.stat().st_size/1e9:.2f} GB），跳过")
else:
    merged = {}
    for s in shards:
        print(f"  载入 {s.name} …", flush=True)
        sd = torch.load(s, map_location="cpu", weights_only=False)
        if isinstance(sd, dict) and isinstance(sd.get("state_dict"), dict):
            sd = sd["state_dict"]
        merged.update({k: v.contiguous() for k, v in sd.items() if torch.is_tensor(v)})
        del sd
    print(f"  合并后 {len(merged)} 个张量，dtype 样例 {list(merged.values())[0].dtype}")
    save_file(merged, str(OUT), metadata={"format": "pt"})
    print(f"✅ 写出 {OUT}  {OUT.stat().st_size/1e9:.2f} GB")
    del merged

# 回读校验
chk = load_file(str(OUT), device="cpu")
print(f"  回读校验：{len(chk)} 张量  dtype={list(chk.values())[0].dtype}")
del chk

# 删掉 .bin 分片与索引，避免 HF 选错格式
for s in shards:
    s.unlink()
    print(f"  已删 {s.name}")
idx = D / "pytorch_model.bin.index.json"
if idx.exists():
    idx.unlink()
    print(f"  已删 {idx.name}")
tiny = D / "pytorch_model.bin"
if tiny.exists() and tiny.stat().st_size < 1024:
    tiny.unlink()
    print("  已删 残留的 pytorch_model.bin（15B 空文件）")

print("\n--- clip_bigg 最终内容 ---")
for f in sorted(D.iterdir()):
    print(f"  {f.name:36s} {f.stat().st_size/1e6:9.2f} MB")
