#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""fix_openclip_safetensors.py —— 补上 open_clip 需要的 `open_clip_model.safetensors`。

背景（**验证时实测抓到的 bug**）：
    open_clip 3.3.0 的 `load_checkpoint()` 对 HF 仓库**优先找 `open_clip_model.safetensors`**，
    找不到才退回 `open_clip_pytorch_model.bin`。我们的缓存里只有 `.bin`，
    于是它去 `huggingface.co` 做 HEAD 请求 → 超时/`Network is unreachable`。
    **镜像里设了 `HF_HUB_OFFLINE=1`（规范要求禁止运行期联网），所以它在 Docker 里会直接加载失败。**
    ⇒ 必须把 safetensors 也放进镜像。

做法：**不重新下载**（那是 3.9 GB），直接用本地已转成 fp16 的 `.bin` 生成 `.safetensors`。
      safetensors 要求扁平张量字典，所以先剥掉 `state_dict` 外壳。
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
D = Path("/hy-tmp/build/buildkit/docker_submit/app/backbones/clip_h14_cache")
BIN = D / "open_clip_pytorch_model.bin"
OUT = D / "open_clip_model.safetensors"

print(f"源 {BIN}  {BIN.stat().st_size/1e9:.2f} GB")
obj = torch.load(BIN, map_location="cpu", weights_only=False)
print(f"  顶层类型 {type(obj).__name__}，键数 {len(obj)}")
if isinstance(obj, dict) and "state_dict" in obj and isinstance(obj["state_dict"], dict):
    sd = obj["state_dict"]
    print("  剥掉 state_dict 外壳")
else:
    sd = obj
print(f"  张量数 {len(sd)}；样例键 {list(sd)[:3]}")

# 全部转成连续张量（safetensors 要求 contiguity）
out = {}
for k, v in sd.items():
    if not torch.is_tensor(v):
        continue
    out[k] = v.contiguous()
print(f"  待写 {len(out)} 个张量")

save_file(out, str(OUT), metadata={"format": "pt"})
print(f"✅ 写出 {OUT}  {OUT.stat().st_size/1e9:.2f} GB")

# 校验能读回
from safetensors.torch import load_file
chk = load_file(str(OUT), device="cpu")
print(f"  回读校验：{len(chk)} 个张量，dtype 样例 {list(chk.values())[0].dtype}")
