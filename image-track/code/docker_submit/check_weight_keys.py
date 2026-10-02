#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_weight_keys.py —— 静态校验 8 个权重的键名结构与 models.py 的 target_modules 是否一致。

为什么必须查：`models.py` 用 `LoraConfig(target_modules=...)` 建模型，再把 checkpoint 的键
按归一化名字映射进去。**如果 target_modules 与 checkpoint 的键不匹配，LoRA 会被静默丢弃**
（peft 只打一行 missing/unexpected 警告）—— AGENTS 里记载过这个坑导致"探针分数退化成随机"。
这里在**不加载骨干**的前提下，纯靠键名把这件事验死。
"""
import re
import sys
from collections import Counter
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
W = Path(__file__).resolve().parent / "app" / "weights"

EXPECT = {
    # 文件                    期望 target_modules（**checkpoint 里真实存在的**）   期望 r
    #
    # ⚠️ cliph 家族只有 3 个模块：open_clip 的 OpenAI 式 ViT 用**融合的 in_proj_weight**，
    #    根本没有 q_proj/k_proj/v_proj 子模块，peft 匹配不到就跳过。
    #    `models.py` 里仍写 6 个名字，是为了**与训练脚本保持完全相同的模块结构**；
    #    多出来的 q/k/v LoRA 因 `B=0` 初始化而贡献恒为 0，不影响结果。
    #    （实测：32 层 × 3 模块 × 2(A/B) = 192，与 checkpoint 张量数吻合）
    "cliph_rerun.pt":        ({"out_proj", "c_fc", "c_proj"}, 32),
    "cliph_s20261101.pt":    ({"out_proj", "c_fc", "c_proj"}, 32),
    "cliph_s20261102.pt":    ({"out_proj", "c_fc", "c_proj"}, 32),
    "cliph_s20261103.pt":    ({"out_proj", "c_fc", "c_proj"}, 32),
    "gpp.pt":                ({"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}, 16),
    "bigg_rerun2.pt":        ({"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}, 16),
    "bigg_rerun3.pt":        ({"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}, 16),
    "bigg_s384_retrain.pt":  ({"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"}, 32),
}
# models.py 里实际声明的 target_modules —— 必须是 checkpoint 目标集的**超集**（否则会丢权重）
MODEL_TARGETS = {
    "cliph": {"q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"},
    "hf":    {"q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"},
}

ok_all = True
for fn, (targets, r_exp) in EXPECT.items():
    p = W / fn
    if not p.exists():
        print(f"❌ 缺 {fn}")
        ok_all = False
        continue
    st = torch.load(p, map_location="cpu", weights_only=False)
    sd = st.get("lora", st.get("vision"))
    keys = [k for k in sd if "lora_" in k]
    # 归一化后按段取 target 名：...<模块名>.lora_A.default.weight
    # ⚠️ 不能用正则字符类 `[.\b]`（那是"点或退格"，不是词边界）—— 我第一版就是这么写错的。
    mods = set()
    layers = set()
    for k in keys:
        seg = k.replace("base_model.model.", "").replace(".default", "").split(".")
        if "lora_A" in seg or "lora_B" in seg:
            i = seg.index("lora_A") if "lora_A" in seg else seg.index("lora_B")
            mods.add(seg[i - 1])
            for s in seg:
                if s.isdigit():
                    layers.add(int(s))
                    break
    rs = {int(v.shape[0]) for k, v in sd.items()
          if "lora_A" in k and getattr(v, "dim", lambda: 0)() == 2}
    head_ok = "head" in st
    head_shape = tuple(st["head"]["weight"].shape) if head_ok else None
    # 模型声明的 target 必须是 checkpoint 目标集的**超集**，否则映射时会丢权重
    model_t = MODEL_TARGETS["cliph"] if fn.startswith("cliph") else MODEL_TARGETS["hf"]
    superset_ok = mods <= model_t
    ok = (mods == targets) and (rs == {r_exp}) and head_ok and superset_ok
    ok_all &= ok
    print(f"{'✅' if ok else '❌'} {fn:24s} r={sorted(rs)} (期望 {r_exp})"
          f"  超集校验 {'OK' if superset_ok else 'FAIL'}")
    print(f"     命中模块 {sorted(mods)}")
    if mods != targets:
        print(f"     ⚠️ 与期望不符！期望 {sorted(targets)}  差集 {sorted(mods ^ targets)}")
    print(f"     LoRA 张量 {len(keys)}  head={head_shape}")

print()
print("全部一致 ✅" if ok_all else "存在不一致 ❌ —— 必须修 models.py 的 target_modules 或换权重")
sys.exit(0 if ok_all else 1)
