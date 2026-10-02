#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_cliph.py —— 给 **open_clip 系（CLIP ViT-H-14）** 的 LoRA checkpoint 出探针。

为什么需要单独一个（`probe_ckpt.py` 覆盖不了）：
  1. `train_cliph300k.py` / `cliph_rerun.py` 存的是 `{"vision": lora_sd, "head": sd}`，
     而 `probe_ckpt.py` 读的是 `st["lora"]` → KeyError
  2. 基座是 **open_clip** 的 ViT-H-14（`model.visual`），`probe_ckpt.py` 只会 transformers
  3. LoRA target 名也不同：open_clip 是 `q_proj/k_proj/v_proj/out_proj/c_fc/c_proj`

⚠️ 加载 LoRA 的两个坑（都是踩过的）：
  · peft 版本差异会让键名多/少一层 `.default` → 必须归一化后按名映射
  · 映射数必须 == LoRA 张量数，否则**静默丢弃全部权重**（只打一行警告）→ 这里直接抛错

用法：
  CKPT=/root/sj-tmp/baycup/ds300k/runs/cliph300k/ckpt_2365.pt TAG=cliph300k_p25 \
  CLIP_CACHE=/root/sj-tmp/baycup/models/clip_h14_cache \
  OUTDIR=/root/sj-tmp/baycup/ds300k/runs/cliph300k \
  python3 probe_cliph.py
"""
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CKPT = Path(os.environ["CKPT"])
TAG = os.environ.get("TAG", CKPT.stem)
CLIP_CACHE = os.environ.get("CLIP_CACHE", "/root/sj-tmp/baycup/models/clip_h14_cache")
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", "/root/sj-tmp/baycup/ds300k/baycup/data/sample"))
TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/root/sj-tmp/baycup/ds300k/baycup/data/official_test"))
OUTDIR = Path(os.environ.get("OUTDIR", str(CKPT.parent)))
DEV = torch.device("cuda")
TFM = os.environ.get("TFM", "crop").lower()          # crop（cliph 系一直用这个）| stretch
SHORT = int(os.environ.get("SHORT", "256"))
CROP = int(os.environ.get("CROP", "224"))
BS = int(os.environ.get("BS", "32"))
WORKERS = int(os.environ.get("WORKERS", "8"))
R = int(os.environ.get("R", "32"))
VAL_ONLY = os.environ.get("VAL_ONLY", "0") == "1"
# ⚠️ 2026-09-15 新增：`AMP=0` 关掉 bf16 autocast，用 fp32 推理（判"饱和是不是精度造成的"）。
#   默认 1 = 与历史所有探针完全一致的行为（**改默认值会让历史探针不可比，所以不改**）。
AMP = os.environ.get("AMP", "1") == "1"


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


class DS(Dataset):
    def __init__(self, records):
        self.records = records
        if TFM == "stretch":
            t = [transforms.Resize((CROP, CROP), interpolation=InterpolationMode.BICUBIC)]
        else:
            t = [transforms.Resize(SHORT, interpolation=InterpolationMode.BICUBIC),
                 transforms.CenterCrop(CROP)]
        self.tfm = transforms.Compose(t + [transforms.ToTensor(),
                                           transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        p, lb = self.records[i]
        im, err = None, None
        for _ in range(3):                     # 绝不返回黑图（那会静默污染 val）
            try:
                im = Image.open(p).convert("RGB")
                im.load()
                break
            except Exception as e:
                err = e
        if im is None:
            raise RuntimeError(f"读图连续失败: {p} ({err})")
        return self.tfm(im), np.float32(lb)


@torch.inference_mode()
def run(vision, head, records):
    dl = DataLoader(DS(records), batch_size=BS, shuffle=False, num_workers=WORKERS)
    out = []
    for x, _ in dl:
        # ⚠️ 2026-09-15 01:15 新增 `AMP` 开关（默认 1 = 保持原行为，向后兼容）：
        #   起因：`cliph300k` 的分数**极度饱和**（42.6% > 0.99、28.2% < 0.01），
        #   而它在 v64 里当主分量时 LB 掉 −0.0444。要判断这是"模型确实弱"还是
        #   "**bf16 把 logit 的细微差别抹平、导致排序分辨率被毁**"（logit≈15 处 bf16 分辨率只有 0.0625），
        #   就必须用 fp32 重跑一次对比。⇒ `AMP=0` 关掉 autocast。
        #   ⚠️ 关掉后显存会涨（CLIP-H 全 fp32），所以调用方要相应调小 BS。
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=AMP):
            f = vision(x.to(DEV, non_blocking=True)).float()
            out.append(torch.sigmoid(head(f).squeeze(-1).float()).cpu().numpy())
    return np.concatenate(out)


def build():
    import open_clip
    from peft import LoraConfig, get_peft_model
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=CLIP_CACHE)
    vision = model.visual
    embed = getattr(vision, "output_dim", 1280)
    # ⚠️ 必须与 cliph_rerun.py / train_cliph300k.py 完全一致，否则键名对不上
    vision = get_peft_model(vision, LoraConfig(
        r=R, lora_alpha=R, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"])).to(DEV)
    head = torch.nn.Linear(embed, 1).to(DEV)
    return vision, head


def main():
    print(f"[{TAG}] CKPT={CKPT} TFM={TFM} SHORT={SHORT} CROP={CROP} R={R}", flush=True)
    vision, head = build()

    st = torch.load(CKPT, map_location="cpu", weights_only=False)
    if not isinstance(st, dict) or not ({"vision", "lora"} & set(st.keys())):
        raise RuntimeError(f"顶层键异常：{list(st.keys())[:8] if isinstance(st, dict) else type(st)}")
    lora_sd = st["vision"] if "vision" in st else st["lora"]
    if any("lora_" not in k for k in list(lora_sd.keys())[:5]):
        print("  [warn] 输入像全量 state_dict，仍按 LoRA 处理", flush=True)

    def _norm(k):                              # 抹平 peft 版本/包装差异
        for seg in ("base_model.model.", "vision_model.", ".default"):
            k = k.replace(seg, "")
        return k

    tgt = list(vision.state_dict().keys())
    tgt_set, tgt_by_norm = set(tgt), {_norm(k): k for k in tgt}
    mapped, miss = {}, []
    for k, v in lora_sd.items():
        if k in tgt_set:
            mapped[k] = v
        elif _norm(k) in tgt_by_norm:
            mapped[tgt_by_norm[_norm(k)]] = v
        else:
            miss.append(k)
    res = vision.load_state_dict(mapped, strict=False)
    n_lora = sum(1 for k in lora_sd if "lora_" in k)
    print(f"  [LoRA] 输入 {len(lora_sd)}（LoRA {n_lora}）→ 映射 {len(mapped)}，"
          f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
    if res.unexpected_keys:
        raise RuntimeError(f"键名不匹配，样本: {res.unexpected_keys[:3]}")
    if n_lora and len(mapped) < n_lora:
        raise RuntimeError(f"LoRA 只映射 {len(mapped)}/{n_lora}，权重不完整，拒绝出探针")
    if "head" not in st:
        raise RuntimeError("checkpoint 里没有 head —— 只存 LoRA 无法复现分数")
    head.load_state_dict(st["head"])
    vision.eval(); head.eval()

    csvs = glob.glob(str(SAMPLE_DIR / "**" / "image_sample_data.csv"), recursive=True)
    if csvs:
        smp = pd.read_csv(csvs[0], encoding="utf-8-sig", dtype={"id": str})
        y = smp.label.astype(int).to_numpy()
        root = Path(csvs[0]).parent / "image_sample_data"
        # 与 probe_ckpt.py 用同一个划分种子，保证不同工具出的 val 可比
        sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=20260925)
        _, va = next(sp.split(np.zeros(len(y)), y))
        vr = [(str(root / str(smp.iloc[i].id)), int(y[i])) for i in va]
        print(f"[{TAG}] val_auc = {roc_auc_score(y[va], run(vision, head, vr)):.6f}  (n={len(vr)})",
              flush=True)
    if VAL_ONLY:
        print(f"[{TAG}] VAL_ONLY=1，跳过测试集推理", flush=True)
        return

    ids = pd.read_csv(TEST_ROOT / "image_test.csv", encoding="utf-8-sig", dtype={"id": str}).id.tolist()
    sc = run(vision, head, [(str(TEST_ROOT / "image_test" / i), 0) for i in ids])
    OUTDIR.mkdir(parents=True, exist_ok=True)
    dst = OUTDIR / f"img_{TAG}.csv"
    pd.DataFrame({"id": ids, "score": sc}).to_csv(dst, index=False)
    print(f"[{TAG}] probe -> {dst}  n={len(sc)}  mean={sc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
