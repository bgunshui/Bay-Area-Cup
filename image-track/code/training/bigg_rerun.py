#!/usr/bin/env python3
"""bigg_rerun.py: 重跑 bigG 分量（CLIP ViT-bigG-14 + LoRA），使候选不再依赖历史资产。

历史 img_bigg_clean_nb.csv 由 train_clean_v2.py 产出，配方:
  CLIP ViT-bigG-14, LoRA r16/a32 (q/k/v/out_proj+fc1/fc2), SIZE 224,
  Tiny-GenImage 28k + 官方 sample(800 train/200 val), JPEG aug q55-95, 2 epochs。
本脚本照此重跑，并加入:
  * 断点续训(每 epoch 存 lora-only + optimizer + epoch 到 last.pt，重跑同命令自动恢复)
  * 每 K 步保存 checkpoint
输出: RUN_DIR/img_bigg_rerun.csv
"""
import glob
import io
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image as PImage
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

BASE = Path(os.environ.get("BASE_DIR", "/root/sj-tmp/baycup"))
RUN = Path(os.environ.get("RUN_DIR", str(BASE / "runs" / "bigg_rerun")))
RUN.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE / "data" / "Tiny-GenImage" / "data"
MODEL_DIR = Path(os.environ.get("MODEL_DIR", str(BASE / "models" / "clip_bigg")))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "20261004"))
SIZE = int(os.environ.get("S", "224"))
BS = int(os.environ.get("BS", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "7e-5"))
R = int(os.environ.get("R", "16"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "1000"))
RESUME = int(os.environ.get("RESUME", "1"))
SMOKE = int(os.environ.get("SMOKE", "0"))
TAG = os.environ.get("TAG", "bigg_rerun")
NORM = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
LORA = dict(r=R, lora_alpha=2 * R, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])


def jpeg_aug(im, rng):
    q = rng.randint(55, 95)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return PImage.open(buf).convert("RGB")


class DS(Dataset):
    def __init__(self, records, sample_root, train, seed=0):
        self.records, self.sample_root, self.train = records, sample_root, train
        self.rng = random.Random(seed)
        self.cache = {}
        if train:
            self.tfm = transforms.Compose([
                transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(), transforms.Normalize(*NORM)])
        else:
            self.tfm = transforms.Compose([
                transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(), transforms.Normalize(*NORM)])

    def __len__(self):
        return len(self.records)

    def _pq(self, key, name):
        t = self.cache.get(key)
        if t is None:
            df = pd.read_parquet(DATA_DIR / name, columns=["image", "label"])
            t = df[df.label == (0 if key.startswith("pqreal") else 1)].reset_index(drop=True)
            self.cache[key] = t
        return t

    def __getitem__(self, i):
        kind, loc, label = self.records[i]
        try:
            if kind == "file":
                im = PImage.open(self.sample_root / loc).convert("RGB")
            else:
                row = self._pq(loc[0], loc[1]).iloc[loc[2]]["image"]
                raw = row["bytes"] if isinstance(row, dict) else row
                im = PImage.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            im = PImage.new("RGB", (SIZE, SIZE))
        if self.train:
            im = jpeg_aug(im, self.rng)
        return self.tfm(im), np.float32(label)


def external_records():
    recs = []
    for path in sorted(DATA_DIR.glob("train-*.parquet")):
        df = pd.read_parquet(path, columns=["image", "label"])
        for pos in range(int((df.label == 0).sum())):
            recs.append(("tiny", ("pqreal", path.name, pos), 0))
        for pos in range(int((df.label == 1).sum())):
            recs.append(("tiny", ("pqfake", path.name, pos), 1))
    return recs


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    csv = Path(glob.glob(str(BASE / "data" / "initial" / "**" / "image_sample_data.csv"), recursive=True)[0])
    smp = pd.read_csv(csv, encoding="utf-8-sig", dtype={"id": str})
    y = smp.label.astype(int).to_numpy()
    sroot = csv.parent / "image_sample_data"
    if not sroot.exists():
        sroot = csv.parent / "data" / "image_sample_data"
    tr, va = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED).split(np.zeros(len(y)), y))
    sample_recs = [("file", str(smp.iloc[i].id), int(y[i])) for i in tr]
    val_recs = [("file", str(smp.iloc[i].id), int(y[i])) for i in va]
    ext = external_records()
    train_recs = ext + sample_recs
    n_real = sum(1 for r in train_recs if r[2] == 0)
    n_fake = sum(1 for r in train_recs if r[2] == 1)
    print(f"device={DEVICE} S={SIZE} BS={BS} R={R} EPOCHS={EPOCHS} ext={len(ext)} "
          f"real={n_real} fake={n_fake}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import CLIPModel
    clm = CLIPModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    vision = get_peft_model(clm.vision_model, LoraConfig(**LORA)).to(DEVICE)
    vision.config.use_cache = False
    head = torch.nn.Linear(vision.config.hidden_size, 1).to(DEVICE)
    ntr = sum(p.numel() for p in vision.parameters() if p.requires_grad)
    print(f"trainable={ntr/1e6:.2f}M embed={vision.config.hidden_size}", flush=True)

    params = [p for p in vision.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW([{"params": [p for p in vision.parameters() if p.requires_grad], "lr": LR, "weight_decay": 0.01},
                             {"params": head.parameters(), "lr": 2e-4, "weight_decay": 0.01}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, EPOCHS))

    last = RUN / "last.pt"
    start_ep = 0
    if RESUME and last.exists():
        st = torch.load(last, map_location="cpu")
        vision.load_state_dict(st["lora"], strict=False)
        head.load_state_dict(st["head"])
        try:
            opt.load_state_dict(st["opt"])
        except Exception as e:
            print("optimizer 恢复失败(忽略):", repr(e)[:100], flush=True)
        start_ep = int(st.get("epoch", 0))
        print(f"** 断点续训: 从 epoch {start_ep} 恢复", flush=True)

    best = -1.0
    for ep in range(start_ep, EPOCHS):
        vision.train(); head.train()
        dl = DataLoader(DS(train_recs, sroot, True, SEED + ep), batch_size=BS, shuffle=True,
                        num_workers=8, pin_memory=True, drop_last=True)
        tot = 0.0; t0 = time.time()
        lim = min(len(dl), 20) if SMOKE else len(dl)
        for step, (x, tgt) in enumerate(dl):
            if step >= lim:
                break
            x = x.to(DEVICE, non_blocking=True); tgt = tgt.to(DEVICE)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = vision(x, interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(f).squeeze(-1), tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); tot += float(loss.detach())
            if step % 100 == 0 or step == lim - 1:
                print(f"ep={ep} step={step}/{lim} loss={tot/(step+1):.4f} sec={time.time()-t0:.0f} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            if CKPT_EVERY and step > 0 and step % CKPT_EVERY == 0:
                torch.save({"lora": {k: v.detach().cpu() for k, v in vision.state_dict().items() if "lora_" in k},
                            "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                            "opt": opt.state_dict(), "epoch": ep}, last)
                print(f"ckpt saved @ ep{ep} step{step}", flush=True)
        if SMOKE:
            print("SMOKE DONE", flush=True)
            return
        sched.step()
        # 验证
        vdl = DataLoader(DS(val_recs, sroot, False), batch_size=BS, shuffle=False, num_workers=6)
        vision.eval(); head.eval(); out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, _ in vdl:
                f = vision(x.to(DEVICE), interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
                out.append(head(f).squeeze(-1).float().cpu().numpy())
        auc = roc_auc_score(y[va], np.concatenate(out))
        print(f"ep={ep} train_loss={tot/lim:.4f} val_auc={auc:.6f}", flush=True)
        torch.save({"lora": {k: v.detach().cpu() for k, v in vision.state_dict().items() if "lora_" in k},
                    "head": {k: v.detach().cpu() for k, v in head.state_dict().items()},
                    "opt": opt.state_dict(), "epoch": ep + 1}, last)
        if auc > best:
            best = auc
            torch.save({"lora": {k: v.detach().cpu() for k, v in vision.state_dict().items() if "lora_" in k},
                        "head": {k: v.detach().cpu() for k, v in head.state_dict().items()}}, RUN / "best.pt")
            print("best.pt saved", flush=True)

    st = torch.load(RUN / "best.pt", map_location="cpu")
    vision.load_state_dict(st["lora"], strict=False); head.load_state_dict(st["head"])
    rcsv = BASE / "data" / "replay" / "image_test.csv"
    rep = pd.read_csv(rcsv, encoding="utf-8-sig", dtype={"id": str})
    rroot = rcsv.parent / "data" / "image_test"
    rdl = DataLoader(DS([("file", str(i), 0) for i in rep.id], rroot, False),
                     batch_size=BS, shuffle=False, num_workers=8)
    vision.eval(); head.eval(); out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in rdl:
            f = vision(x.to(DEVICE), interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
            out.append(head(f).squeeze(-1).float().cpu().numpy())
    rs = np.concatenate(out)
    dst = RUN / f"img_{TAG}.csv"
    pd.DataFrame({"id": rep.id.astype(str), "score": 1.0 / (1.0 + np.exp(-rs))}).to_csv(dst, index=False)
    print(f"DONE {TAG} best_val_auc={best:.6f} output={dst}", flush=True)


if __name__ == "__main__":
    main()
