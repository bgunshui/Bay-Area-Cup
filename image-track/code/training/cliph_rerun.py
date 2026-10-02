#!/usr/bin/env python3
"""cliph_rerun.py: 重跑 anchor 的 CLIP-H 侧（自蒸馏），使 anchor 这条线彻底可复现。

背景:
  anchor = baycup_v28_cliph50_siglip030 = CLIP-H 侧(cliph50) + SigLIP 侧(siglip030) + SPAI，
  其中 CLIP-H 侧 = img_clip_rebuilt 家族，由方法「自蒸馏」得到
  (历史 train_selfdistill_cliph.py: teacher=上轮学生分数, hard 1000 w=4 + soft 20000 w=1,
   LoRA r32, 全图方形 resize 224, 4 epochs)。
  该脚本与训练权重均已丢失 -> anchor 只能复现到 corr 0.96，无法彻底复现。

本脚本用同一方法重跑 CLIP-H 侧:
  * 基座: open_clip ViT-H-14 (laion2b_s32b_b79k, 632M)
  * teacher: 当前 anchor 自身的分数(自蒸馏)，soft = sigmoid(4*(rank-0.5))
  * 数据: sample 1000(真标签 w=4) + 测试集 20000(软标签 w=1) + GenImage 对齐缓存(真标签 w=1)
  * 预处理改进(对齐原则): 短边缩放到 SHORT 保持长宽比 -> 裁剪 CROP，而非全图方形拉伸
  * 断点续训: 每 epoch 保存 lora+head+optimizer+step 到 last.pt，重跑同命令自动恢复
输出: RUN_DIR/img_cliph_rerun.csv
"""
import glob
import io
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

BASE = Path(os.environ.get("BASE_DIR", "/hy-tmp/baycup"))
RUN = Path(os.environ.get("RUN_DIR", str(BASE / "runs" / "cliph_rerun")))
RUN.mkdir(parents=True, exist_ok=True)
CACHE = Path(os.environ.get("CACHE_DIR", "/hy-tmp/align_cache"))
TEACHER = os.environ.get("TEACHER", str(BASE / "probes" / "baycup_v28_cliph50_siglip030.csv"))
CLIP_CACHE = os.environ.get("CLIP_CACHE", str(BASE / "models" / "clip_h14_cache"))
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"
SHORT = int(os.environ.get("SHORT", "256"))
CROP = int(os.environ.get("CROP", "224"))
BS = int(os.environ.get("BS", "32"))
LR = float(os.environ.get("LR", "1e-4"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
N_CACHE = int(os.environ.get("N_CACHE", "30000"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "500"))
RESUME = int(os.environ.get("RESUME", "1"))
SMOKE = int(os.environ.get("SMOKE", "0"))
SEED = int(os.environ.get("SEED", "20261003"))
HARD_W = float(os.environ.get("HARD_W", "4.0"))
R = int(os.environ.get("R", "32"))
TAG = os.environ.get("TAG", "cliph_rerun")

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


def jpeg_aug(im, rng):
    q = rng.randint(50, 95)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class DS(Dataset):
    def __init__(self, items, train=True, seed=0):
        self.items, self.train = items, train
        self.rng = random.Random(seed)
        if train:
            self.tfm = transforms.Compose([
                transforms.Resize(SHORT, interpolation=InterpolationMode.BICUBIC),
                transforms.RandomCrop(CROP),
                transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize(SHORT, interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(CROP),
                transforms.ToTensor(), transforms.Normalize(CLIP_MEAN, CLIP_STD)])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, tgt, w = self.items[i]
        try:
            im = Image.open(path).convert("RGB")
        except Exception:
            im = Image.new("RGB", (SHORT, SHORT))
        if self.train:
            im = jpeg_aug(im, self.rng)
            if self.rng.random() < 0.5:
                im = im.transpose(Image.FLIP_LEFT_RIGHT)
        return self.tfm(im), np.float32(tgt), np.float32(w)


def trainable_state(vision, head):
    return {"vision": {k: v.detach().cpu() for k, v in vision.state_dict().items() if "lora_" in k},
            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}}


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

    import open_clip
    from peft import LoraConfig, get_peft_model
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    model, _, _ = open_clip.create_model_and_transforms(
        "ViT-H-14", pretrained="laion2b_s32b_b79k", cache_dir=CLIP_CACHE)
    vision = model.visual
    embed = vision.output_dim if hasattr(vision, "output_dim") else 1280
    lcfg = LoraConfig(r=R, lora_alpha=R, lora_dropout=0.05, bias="none",
                      target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "c_fc", "c_proj"])
    vision = get_peft_model(vision, lcfg).to(DEV)
    head = torch.nn.Linear(embed, 1).to(DEV)
    print(f"基座 ViT-H-14 embed={embed} 可训参数="
          f"{sum(p.numel() for p in vision.parameters() if p.requires_grad)/1e6:.2f}M", flush=True)

    # ---- 数据 ----
    rcsv = BASE / "data" / "replay" / "image_test.csv"
    rep = pd.read_csv(rcsv, encoding="utf-8-sig", dtype={"id": str})
    rroot = rcsv.parent / "data" / "image_test"
    RIDS = rep.id.astype(str).tolist()
    t = pd.read_csv(TEACHER, dtype={"id": str})
    t["id"] = t.id.astype(str)
    if t.id.tolist() != RIDS:
        t = t.set_index("id").loc[RIDS].reset_index()
    t_rank = rankdata(t.score.to_numpy(), method="average") / (len(t) + 1.0)
    soft = 1.0 / (1.0 + np.exp(-4.0 * (t_rank - 0.5)))
    print(f"teacher={Path(TEACHER).name} soft mean={soft.mean():.3f} std={soft.std():.3f}", flush=True)

    scsv = list((BASE / "data").glob("initial/**/image_sample_data.csv"))[0]
    smp = pd.read_csv(scsv, encoding="utf-8-sig", dtype={"id": str})
    sroot = scsv.parent / "image_sample_data"
    y = smp.label.astype(int).to_numpy()

    items = [(str(sroot / str(i)), float(lb), HARD_W) for i, lb in zip(smp.id, y)]
    items += [(str(rroot / i), float(s), 1.0) for i, s in zip(RIDS, soft)]
    if N_CACHE:
        lines = [l for l in (CACHE / "index.tsv").read_text().splitlines() if l.strip()]
        random.shuffle(lines)
        for line in lines[:N_CACHE]:
            p, lab = line.split("\t")[:2]
            items.append((p, float(lab), 1.0))
    print(f"训练样本 {len(items)} (hard={len(smp)} w={HARD_W}, soft={len(RIDS)}, cache={N_CACHE})", flush=True)

    params = [p for p in vision.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in vision.parameters() if p.requires_grad], "lr": LR, "weight_decay": 0.01},
                             {"params": head.parameters(), "lr": LR, "weight_decay": 0.01}])

    # ---- 断点续训 ----
    last = RUN / "last.pt"
    start_ep = 0
    if RESUME and last.exists():
        st = torch.load(last, map_location="cpu")
        vision.load_state_dict(st["vision"], strict=False)
        head.load_state_dict(st["head"])
        try:
            opt.load_state_dict(st["opt"])
        except Exception as e:
            print("optimizer 恢复失败(忽略):", repr(e)[:100], flush=True)
        start_ep = int(st.get("epoch", 0))
        print(f"** 断点续训: 从 epoch {start_ep} 恢复", flush=True)

    for ep in range(start_ep, EPOCHS):
        vision.train(); head.train()
        dl = DataLoader(DS(items, train=True, seed=SEED + ep), batch_size=BS, shuffle=True,
                        num_workers=12, pin_memory=True, drop_last=True)
        t0 = time.time(); tot = 0.0
        lim = min(len(dl), 30) if SMOKE else len(dl)
        for step, (x, tg, w) in enumerate(dl):
            if step >= lim:
                break
            x = x.to(DEV, non_blocking=True); tg = tg.to(DEV); w = w.to(DEV)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = vision(x).float()          # open_clip ViT 返回 pooled [B, embed]
                logit = head(f).squeeze(-1).float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, tg, weight=w)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tot += float(loss.detach())
            if step % 200 == 0 or step == lim - 1:
                print(f"ep={ep} step={step}/{lim} loss={tot/(step+1):.4f} sec={time.time()-t0:.0f} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            if CKPT_EVERY and step > 0 and step % CKPT_EVERY == 0:
                torch.save({"vision": trainable_state(vision, head)["vision"],
                            "head": trainable_state(vision, head)["head"],
                            "opt": opt.state_dict(), "epoch": ep}, last)
                print(f"ckpt saved @ ep{ep} step{step}", flush=True)
        if SMOKE:
            print("SMOKE DONE", flush=True)
            return
        torch.save({"vision": trainable_state(vision, head)["vision"],
                    "head": trainable_state(vision, head)["head"],
                    "opt": opt.state_dict(), "epoch": ep + 1}, last)
        print(f"** epoch {ep} 完成并保存 ({last})", flush=True)

    # ---- 推理测试集 ----
    vision.eval(); head.eval()
    rdl = DataLoader(DS([(str(rroot / i), 0.0, 1.0) for i in RIDS], train=False),
                     batch_size=BS, shuffle=False, num_workers=10)
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _, _ in rdl:
            f = vision(x.to(DEV)).float()
            out.append(torch.sigmoid(head(f).squeeze(-1).float()).cpu().numpy())
    sc = np.concatenate(out)
    dst = RUN / f"img_{TAG}.csv"
    pd.DataFrame({"id": RIDS, "score": sc}).to_csv(dst, index=False)
    print(f"DONE {TAG} output={dst}", flush=True)


if __name__ == "__main__":
    main()
