#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_ckpt.py —— 对任意 checkpoint 单独出探针 CSV（20000 测试图）+ 在官方样例 1000 上算 val。
用于：训练中途的 25%/50%/75% checkpoint 提前评估，不必等整个 epoch 跑完。

用法：
  CKPT=/hy-tmp/baycup/runs/bigG_crop_pilot/ckpt_11892.pt TAG=bigG_p25 \
  BACKBONE=bigg MODEL_DIR=/hy-tmp/sjmodels/clip_bigg SHORT=299 CROP=224 \
  python3 probe_ckpt.py
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PImage
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CKPT = Path(os.environ["CKPT"])
TAG = os.environ.get("TAG", CKPT.stem)
BACKBONE = os.environ.get("BACKBONE", "bigg")
MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/hy-tmp/sjmodels/clip_bigg"))
TEST_ROOT = Path(os.environ.get("TEST_ROOT", "/hy-tmp/baycup/data/official_test"))
SAMPLE_DIR = Path(os.environ.get("SAMPLE_DIR", "/hy-tmp/baycup/data/sample"))
OUTDIR = Path(os.environ.get("OUTDIR", str(CKPT.parent)))
DEVICE = torch.device("cuda")
SHORT = int(os.environ.get("SHORT", "299"))
CROP = int(os.environ.get("CROP", "224"))
# TFM: stretch = 方形直接拉伸 Resize((CROP,CROP))（历史所有拿分模型用的就是这个）
#      crop    = 短边缩放到 SHORT 再 CenterCrop(CROP)（train_crop.py 的新策略）
# 用于"固定权重、只变推理变换"的单变量消融。
TFM = os.environ.get("TFM", "crop").lower()
BS = int(os.environ.get("BS", "32"))
WORKERS = int(os.environ.get("WORKERS", "10"))
NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) if BACKBONE == "siglip2" else \
       ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])

# ⚠️ 2026-09-13 补的第二个 bug（与 train_stretch.py 的 fwd() 必须一致，§0.9 第 9 条）：
#   `clip_bigg` 实测是 **CLIP ViT-bigG-14 的 vision tower**（model_type=clip, hidden=1664, 48 层, patch14,
#   **原生 image_size=224**）；`siglip2_384` 是 SigLIP so400m（hidden=1152, 27 层, 原生 384）。
#   本脚本此前直接 `vision(x)` —— 于是用 bigG 权重在 **384** 输入下出探针必然抛
#   `ValueError: Input image size (384*384) doesn't match model (224*224)`
#   （**同一个 bug 已经让 gpu4090 空转过 ~10 小时**，那次是 train_stretch.py 漏了插值）。
#   SigLIP 系（原生 384）自己按输入尺寸插值，不需要、也不能传这个参数。
_ip = os.environ.get("INTERP", "auto").lower()
INTERP = (BACKBONE != "siglip2") if _ip == "auto" else (_ip in ("1", "true", "yes"))


def fwd(v, x):
    """统一前向：CLIP 系换分辨率必须插值位置编码。**所有前向只能走这里。**"""
    return v(x, interpolate_pos_encoding=True) if INTERP else v(x)


class DS(Dataset):
    def __init__(self, records):
        self.records = records
        if TFM == "stretch":
            self.tfm = transforms.Compose([
                transforms.Resize((CROP, CROP), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(), transforms.Normalize(*NORM)])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize(SHORT, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(CROP),
                transforms.ToTensor(), transforms.Normalize(*NORM)])

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        p = self.records[i][0]
        # ⚠️ 绝不返回"黑图+标签"：那会静默污染 val AUC（在 train_crop.py 里曾导致梯度爆炸）
        im, err = None, None
        for _ in range(3):
            try:
                im = PImage.open(p).convert("RGB")
                im.load()
                break
            except Exception as e:
                err = e
        if im is None:
            raise RuntimeError(f"读图连续失败: {p} ({err})")
        return self.tfm(im), np.float32(self.records[i][1])


@torch.inference_mode()
def run(vision, head, records):
    dl = DataLoader(DS(records), batch_size=BS, shuffle=False, num_workers=WORKERS)
    out = []
    for x, _ in dl:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            f = fwd(vision, x.to(DEVICE, non_blocking=True)).last_hidden_state[:, 0].float()
            out.append(torch.sigmoid(head(f).squeeze(-1)).float().cpu().numpy())
    return np.concatenate(out)


def detect_r(lora_sd):
    """从 checkpoint 自身推断 LoRA rank —— 不许硬编码（§0.9 第 10 条：静默/硬崩都算 bug）。"""
    for k, v in lora_sd.items():
        if "lora_A" in k and hasattr(v, "shape") and v.dim() == 2:
            return int(v.shape[0])
    return None


def main():
    from peft import LoraConfig, get_peft_model
    print(f"[{TAG}] BACKBONE={BACKBONE} TFM={TFM} SHORT={SHORT} CROP={CROP} INTERP={INTERP} CKPT={CKPT}",
          flush=True)

    # ⚠️ 必须先读 checkpoint、按它自己的 r 建模型。
    #    2026-09-13 实测 bug：本脚本曾硬编码 r=16，而 s384 系（train_crop/train_stretch/train_clean_s384）
    #    用的是 R=32 → `load_state_dict` 直接 shape mismatch 崩掉（`lora_A` 形状 (32,1664) vs (16,1664)）。
    #    自训脚本一律 `lora_alpha = 2*R`（scaling 恒为 2.0），所以按 ckpt 的 r 建、alpha 取 2r，
    #    形状与缩放**同时**对上；r=16 的老 checkpoint 行为完全不变。
    _st = torch.load(CKPT, map_location="cpu", weights_only=False)
    _lora = _st["lora"] if "lora" in _st else _st.get("vision")
    if _lora is None:
        raise RuntimeError(f"checkpoint 里找不到 lora/vision 键：{list(_st.keys())[:6]}")
    r_ck = int(os.environ.get("R", "0")) or detect_r(_lora) or 16
    alpha = float(os.environ.get("LORA_ALPHA", 2 * r_ck))
    print(f"  [LoRA 配置] r={r_ck} lora_alpha={alpha:g} → scaling={alpha / r_ck:.3f}", flush=True)

    M = None
    if BACKBONE == "siglip2":
        from transformers import SiglipVisionModel as M
    else:
        from transformers import CLIPVisionModel as M
    vision = M.from_pretrained(str(MODEL_DIR), local_files_only=True)
    vision = get_peft_model(vision, LoraConfig(
        r=r_ck, lora_alpha=alpha, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])).to(DEVICE)
    head = torch.nn.Linear(vision.config.hidden_size, 1).to(DEVICE)
    st, lora_sd = _st, _lora
    # 兼容两种 checkpoint：全量 state_dict（7.7GB）与 LoRA-only（~100MB，extract_lora_ckpt.py 产出）
    # ⚠️ LoRA-only 必须用 load_state_dict(strict=False) 直接载 —— 键名本身就是 peft 的 state_dict 格式；
    #    若改用 peft.set_peft_model_state_dict()，它会再补一层 ".default" 变成 xxx.lora_A.default.default.weight，
    #    结果 LoRA 权重全部进 unexpected_keys 被丢弃（实测 val 0.772 vs 正确 0.8019）。
    # ⚠️ 跨机/跨脚本加载还要兼容 peft 包装差异：不同脚本把 peft 包在不同层级上，键名会多/少一段
    #    train_orig.py  : get_peft_model(SiglipModel)      -> base_model.model.vision_model.encoder...
    #    train_crop.py  : get_peft_model(SiglipVisionModel) -> base_model.model.encoder...
    #    实测直接 load 会 0/330 命中、LoRA 全部被丢弃（val 会退化成随机附近）
    full = any("lora_" not in k for k in list(lora_sd.keys())[:5])
    if full:
        vision.load_state_dict(lora_sd)
        print("  [全量 ckpt] load_state_dict OK", flush=True)
    else:
        def _norm(k):                       # 抹平包装前缀与 peft 版本的 ".default" 差异
            for seg in ("base_model.model.", "vision_model.", ".default"):
                k = k.replace(seg, "")
            return k

        tgt_keys = list(vision.state_dict().keys())
        tgt_set = set(tgt_keys)
        tgt_by_norm = {_norm(k): k for k in tgt_keys}
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
        print(f"  [LoRA-only ckpt] 输入 {len(lora_sd)} 张量（其中 LoRA {n_lora}）→ 映射 {len(mapped)}，"
              f"missing={len(res.missing_keys)} unexpected={len(res.unexpected_keys)}", flush=True)
        if res.unexpected_keys:
            raise RuntimeError(f"LoRA 键名不匹配，样本: {res.unexpected_keys[:3]}")
        if miss:
            print(f"  [warn] 未映射 {len(miss)} 个，样本 {miss[:2]}", flush=True)
        if n_lora and len(mapped) < n_lora:
            raise RuntimeError(f"LoRA 张量只映射了 {len(mapped)}/{n_lora}，权重不完整，拒绝出探针")
    head.load_state_dict(st["head"])
    vision.eval(); head.eval()

    # val：官方样例 1000 的 20%
    csvs = glob.glob(str(SAMPLE_DIR / "**" / "image_sample_data.csv"), recursive=True)
    if csvs:
        smp = pd.read_csv(csvs[0], encoding="utf-8-sig", dtype={"id": str})
        y = smp.label.astype(int).to_numpy()
        root = Path(csvs[0]).parent / "image_sample_data"
        sp = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=20260925)
        _, va = next(sp.split(np.zeros(len(y)), y))
        vr = [(str(root / str(smp.iloc[i].id)), int(y[i])) for i in va]
        pr = run(vision, head, vr)
        print(f"[{TAG}] val_auc = {roc_auc_score(y[va], pr):.6f}  (n={len(vr)})", flush=True)

    # VAL_ONLY=1：只出 val 就退出。用于单变量消融（20000 张测试图推理耗时长且易被抢占打断，
    # 而 val 在它之前就已打印，跳过推理能保证答案必达）
    if os.environ.get("VAL_ONLY", "0") == "1":
        print(f"[{TAG}] VAL_ONLY=1，跳过 20000 张测试图推理", flush=True)
        return

    ids = pd.read_csv(TEST_ROOT / "image_test.csv", encoding="utf-8-sig", dtype={"id": str}).id.tolist()
    tr = [(str(TEST_ROOT / "image_test" / i), 0) for i in ids]
    sc = run(vision, head, tr)
    dst = OUTDIR / f"img_{TAG}.csv"
    pd.DataFrame({"id": ids, "score": sc}).to_csv(dst, index=False)
    print(f"[{TAG}] probe -> {dst}  n={len(sc)}  mean={sc.mean():.4f}", flush=True)


if __name__ == "__main__":
    main()
