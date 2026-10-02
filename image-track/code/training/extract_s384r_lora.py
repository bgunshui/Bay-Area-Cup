#!/usr/bin/env python3
"""extract_s384r_lora.py: 从 s384r 的 best.pt(7.57GB) 提取 LoRA-only(约100MB) 便于备份。

原因: best.pt 是 peft 包装后的完整 state_dict，其中 7.4GB 是公开的 CLIP-bigG-14 底模，
      真正不可再生的只有 LoRA 适配器 + 分类头。
"""
import os

import torch

BASE = os.environ.get("BASE_DIR", "/root/sj-tmp/baycup")
RUN = os.path.join(BASE, "runs", "clean_s384")
src = os.path.join(RUN, "best.pt")
dst = os.path.join(RUN, "best_lora_only.pt")

st = torch.load(src, map_location="cpu")
small = {k: (v.half() if v.dtype == torch.float32 else v)
         for k, v in st["lora"].items() if "lora_" in k}
torch.save({"lora": small, "head": st["head"]}, dst)
n = sum(v.numel() for v in small.values()) + sum(v.numel() for v in st["head"].values())
print(f"saved {dst}  {os.path.getsize(dst)/1048576:.1f}MB  params={n/1e6:.2f}M  "
      f"lora_keys={len(small)}/{len(st['lora'])}", flush=True)
