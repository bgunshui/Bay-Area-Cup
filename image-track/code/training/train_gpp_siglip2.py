#!/usr/bin/env python3
"""train_gpp_siglip2.py: GenImage++ 缓存(78k real/fake) 训练 SigLIP2-384 LoRA。
val = 官方 sample 1000（与测试集同源）。输出 runs/gpp/img_gpp_siglip2.csv"""
import glob
import io
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import torch
from PIL import Image as PImage
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

CACHE = Path("/root/sj-fs/gpp_cache")
BASE = Path("/root/sj-fs/baycup_data")          # replay/initial 在 NVMe
RUN = Path("/root/sj-fs/baycup/runs/gpp")
RUN.mkdir(parents=True, exist_ok=True)
MODEL_DIR = Path("/root/baycup_data/models/siglip2_384")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "20260921"))
SIZE = int(os.environ.get("S", "384"))
BS = int(os.environ.get("BS", "16"))
EPOCHS = int(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "7e-5"))
R = int(os.environ.get("R", "16"))
SMOKE = int(os.environ.get("SMOKE", "0"))
PER_SUBSET = int(os.environ.get("PER_SUBSET", "0"))   # >0 时每个子集只取前 N 张(试验用)
NORM = ([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
LORA = dict(r=R, lora_alpha=2 * R, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])

TRAIN_TFM = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(*NORM),
])
VAL_TFM = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(*NORM),
])


def jpeg_augment(im, rng):
    q = rng.randint(55, 95)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return PImage.open(buf).convert("RGB")


class GppDataset(Dataset):
    """records: (path, label) for gpp; ('sample', path, label) for official sample."""

    def __init__(self, records, sample_root, train, seed):
        self.records, self.sample_root, self.train = records, sample_root, train
        self.rng = random.Random(seed)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        r = self.records[index]
        try:
            if r[0] == "sample":
                im = PImage.open(self.sample_root / r[1]).convert("RGB")
                label = r[2]
            else:
                im = PImage.open(r[0]).convert("RGB")
                label = r[1]
        except Exception:
            im = PImage.new("RGB", (SIZE, SIZE))
            label = r[-1]
        if self.train:
            im = jpeg_augment(im, self.rng)
            x = TRAIN_TFM(im)
        else:
            x = VAL_TFM(im)
        return x, np.float32(label)


def gpp_records():
    idx = CACHE / "index.tsv"
    recs = []
    per_sub = {}
    for line in idx.read_text().splitlines():
        if not line.strip():
            continue
        p, lab, sub = line.split("\t")
        if PER_SUBSET > 0:
            c = per_sub.get(sub, 0)
            if c >= PER_SUBSET:
                continue
            per_sub[sub] = c + 1
        recs.append((p, int(lab)))
    return recs


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    sample_csv = Path(glob.glob(str(BASE / "initial" / "**" / "image_sample_data.csv"), recursive=True)[0])
    sample = pd.read_csv(sample_csv, encoding="utf-8-sig", dtype={"id": str})
    y = sample.label.astype(int).to_numpy()
    sample_root = sample_csv.parent / "data" / "image_sample_data"
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y))
    sample_records = [("sample", str(sample.iloc[i].id), int(y[i])) for i in tr_idx]
    val_records = [("sample", str(sample.iloc[i].id), int(y[i])) for i in va_idx]
    gpp = gpp_records()
    train_records = gpp + sample_records
    n_real = sum(1 for r in train_records if r[-1] == 0)
    n_fake = sum(1 for r in train_records if r[-1] == 1)
    print(f"device={DEVICE} S={SIZE} BS={BS} R={R} EPOCHS={EPOCHS} LR={LR} gpp={len(gpp)} real={n_real} fake={n_fake} sample_val={len(val_records)}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import SiglipVisionModel
    vision = SiglipVisionModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    vision = get_peft_model(vision, LoraConfig(**LORA)).to(DEVICE)
    vision.config.use_cache = False
    head = torch.nn.Linear(vision.config.hidden_size, 1).to(DEVICE)
    print("trainable", sum(p.numel() for p in vision.parameters() if p.requires_grad), flush=True)

    opt = torch.optim.AdamW([
        {"params": vision.parameters(), "lr": LR, "weight_decay": 0.01},
        {"params": head.parameters(), "lr": 2e-4, "weight_decay": 0.01},
    ])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    val_ds = GppDataset(val_records, sample_root, train=False, seed=SEED)
    best_auc = -1.0
    for ep in range(EPOCHS):
        vision.train(); head.train()
        ds = GppDataset(train_records, sample_root, train=True, seed=SEED + ep)
        dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=12, pin_memory=True, drop_last=True)
        total = 0.0; t0 = time.time()
        lim = 20 if SMOKE else len(dl)
        for step, (x, target) in enumerate(dl):
            if step >= lim:
                break
            x = x.to(DEVICE, non_blocking=True); target = target.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = vision(x).last_hidden_state[:, 0].float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(f).squeeze(-1), target)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vision.parameters()) + list(head.parameters()), 1.0)
            opt.step(); total += float(loss.detach())
            if step % 50 == 0 or step == lim - 1:
                print(f"ep={ep} step={step}/{lim} loss={total/(step+1):.4f} sec={time.time()-t0:.0f} mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
        if SMOKE:
            print("SMOKE DONE", flush=True)
            return
        sched.step()
        out = []
        va_dl = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=8)
        vision.eval(); head.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, _ in va_dl:
                x = x.to(DEVICE, non_blocking=True)
                f = vision(x).last_hidden_state[:, 0].float()
                out.append(head(f).squeeze(-1).float().cpu().numpy())
        va = np.concatenate(out)
        auc = roc_auc_score(y[va_idx], va)
        print(f"ep={ep} train_loss={total/len(dl):.4f} val_auc={auc:.6f}", flush=True)
        if auc > best_auc:
            best_auc = auc
            torch.save({"lora": vision.state_dict(), "head": head.state_dict()}, RUN / "best.pt")

    state = torch.load(RUN / "best.pt", map_location="cpu")
    vision.load_state_dict(state["lora"]); head.load_state_dict(state["head"])
    replay_csv = BASE / "replay" / "image_test.csv"
    replay = pd.read_csv(replay_csv, encoding="utf-8-sig", dtype={"id": str})
    replay_root = replay_csv.parent / "data" / "image_test"
    rr = [("sample", str(i), 0) for i in replay.id.tolist()]
    rd = DataLoader(GppDataset(rr, replay_root, train=False, seed=0), batch_size=BS, shuffle=False, num_workers=10)
    vision.eval(); head.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in rd:
            x = x.to(DEVICE, non_blocking=True)
            f = vision(x).last_hidden_state[:, 0].float()
            out.append(head(f).squeeze(-1).float().cpu().numpy())
    rs = np.concatenate(out)
    pd.DataFrame({"id": replay.id.astype(str), "score": 1.0 / (1.0 + np.exp(-rs))}).to_csv(RUN / "img_gpp_siglip2.csv", index=False)
    print(f"DONE gpp_siglip2 best_val_auc={best_auc:.6f} output={RUN}/img_gpp_siglip2.csv", flush=True)


if __name__ == "__main__":
    main()
