#!/usr/bin/env python3
"""train_s384_sj.py: 新机(suanjiayun 59820)版 s384 重训。
配置与原始 s384 一致: bigG@384, LoRA r32, Tiny-GenImage 28k(50/50), 2 epochs, BS8, LR7e-5
改进: 每 5000 步 checkpoint（防丢）"""
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

BASE = Path("/root/sj-tmp/baycup")
RUN = BASE / "runs" / "clean_s384"
RUN.mkdir(parents=True, exist_ok=True)
DATA_DIR = BASE / "data" / "Tiny-GenImage" / "data"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = int(os.environ.get("SEED", "20260914"))
SIZE = int(os.environ.get("S", "384"))
BS = int(os.environ.get("BS", "8"))
EPOCHS = int(os.environ.get("EPOCHS", "2"))
LR = float(os.environ.get("LR", "7e-5"))
R = int(os.environ.get("R", "32"))
SMOKE = int(os.environ.get("SMOKE", "0"))
CKPT_EVERY = int(os.environ.get("CKPT_EVERY", "2000"))
MODEL_DIR = BASE / "models" / "clip_bigg"
EMBED = 1664
NORM = ([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711])
LORA = dict(r=R, lora_alpha=2 * R, lora_dropout=0.05, bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"])

TRAIN_TFM = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)])
VAL_TFM = transforms.Compose([
    transforms.Resize((SIZE, SIZE), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(*NORM)])


def jpeg_augment(im, rng):
    q = rng.randint(55, 95)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return PImage.open(buf).convert("RGB")


class TinyDataset(Dataset):
    def __init__(self, records, sample_root, train, seed):
        self.records, self.sample_root, self.train = records, sample_root, train
        self.rng = random.Random(seed)
        self.cache = {}

    def __len__(self):
        return len(self.records)

    def _pq(self, key, parquet_name):
        table = self.cache.get(key)
        if table is None:
            df = pd.read_parquet(DATA_DIR / parquet_name, columns=["image", "label"])
            table = df[df.label == (0 if key.startswith("pqreal") else 1)].reset_index(drop=True)
            self.cache[key] = table
        return table

    def __getitem__(self, index):
        kind, loc, label = self.records[index]
        try:
            if kind == "sample":
                im = PImage.open(self.sample_root / loc).convert("RGB")
            else:
                table = self._pq(loc[0], loc[1])
                row = table.iloc[loc[2]]["image"]
                raw = row["bytes"] if isinstance(row, dict) else row
                im = PImage.open(io.BytesIO(raw)).convert("RGB")
        except Exception:
            im = PImage.new("RGB", (SIZE, SIZE))
        if self.train:
            im = jpeg_augment(im, self.rng)
            x = TRAIN_TFM(im)
        else:
            x = VAL_TFM(im)
        return x, np.float32(label)


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
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
    tr_idx, va_idx = next(splitter.split(np.zeros(len(y)), y))
    sample_recs = [("sample", str(smp.iloc[i].id), int(y[i])) for i in tr_idx]
    val_recs = [("sample", str(smp.iloc[i].id), int(y[i])) for i in va_idx]
    ext = external_records()
    train_recs = ext + sample_recs
    n_real = sum(1 for r in train_recs if r[2] == 0)
    n_fake = sum(1 for r in train_recs if r[2] == 1)
    print(f"device={DEVICE} S={SIZE} BS={BS} R={R} EPOCHS={EPOCHS} ext={len(ext)} real={n_real} fake={n_fake}", flush=True)

    from peft import LoraConfig, get_peft_model
    from transformers import CLIPModel
    clm = CLIPModel.from_pretrained(str(MODEL_DIR), local_files_only=True)
    vision = clm.vision_model
    vision = get_peft_model(vision, LoraConfig(**LORA)).to(DEVICE)
    vision.config.use_cache = False
    head = torch.nn.Linear(EMBED, 1).to(DEVICE)
    print("trainable", sum(p.numel() for p in vision.parameters() if p.requires_grad), flush=True)

    opt = torch.optim.AdamW([
        {"params": vision.parameters(), "lr": LR, "weight_decay": 0.01},
        {"params": head.parameters(), "lr": 2e-4, "weight_decay": 0.01}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    val_ds = TinyDataset(val_recs, sroot, train=False, seed=SEED)
    best = -1.0
    for ep in range(EPOCHS):
        vision.train(); head.train()
        ds = TinyDataset(train_recs, sroot, train=True, seed=SEED + ep)
        dl = DataLoader(ds, batch_size=BS, shuffle=True, num_workers=8, pin_memory=True, drop_last=True)
        tot = 0.0; t0 = time.time()
        lim = 20 if SMOKE else len(dl)
        for step, (x, tgt) in enumerate(dl):
            if step >= lim:
                break
            x = x.to(DEVICE, non_blocking=True); tgt = tgt.to(DEVICE, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                f = vision(x, interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(head(f).squeeze(-1), tgt)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(list(vision.parameters()) + list(head.parameters()), 1.0)
            opt.step(); tot += float(loss.detach())
            if step % 50 == 0 or step == lim - 1:
                print(f"ep={ep} step={step}/{lim} loss={tot/(step+1):.4f} sec={time.time()-t0:.0f} "
                      f"mem={torch.cuda.max_memory_allocated()/1e9:.1f}G", flush=True)
            if CKPT_EVERY > 0 and step > 0 and step % CKPT_EVERY == 0:
                torch.save({"lora": vision.state_dict(), "head": head.state_dict()}, RUN / f"ckpt_{step}.pt")
                print(f"ckpt @ {step}", flush=True)
        if SMOKE:
            print("SMOKE DONE", flush=True)
            return
        sched.step()
        out = []
        vdl = DataLoader(val_ds, batch_size=BS, shuffle=False, num_workers=6)
        vision.eval(); head.eval()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for x, _ in vdl:
                f = vision(x.to(DEVICE, non_blocking=True), interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
                out.append(head(f).squeeze(-1).float().cpu().numpy())
        preds = np.concatenate(out)
        auc = roc_auc_score(y[va_idx], preds)
        print(f"ep={ep} train_loss={tot/lim:.4f} val_auc={auc:.6f}", flush=True)
        if auc > best:
            best = auc
            torch.save({"lora": vision.state_dict(), "head": head.state_dict()}, RUN / "best.pt")
            print("best.pt saved", flush=True)

    state = torch.load(RUN / "best.pt", map_location="cpu")
    vision.load_state_dict(state["lora"]); head.load_state_dict(state["head"])
    rcsv = BASE / "data" / "replay" / "image_test.csv"
    rep = pd.read_csv(rcsv, encoding="utf-8-sig", dtype={"id": str})
    rroot = rcsv.parent / "data" / "image_test"
    rr = [("sample", str(i), 0) for i in rep.id.tolist()]
    rd = DataLoader(TinyDataset(rr, rroot, train=False, seed=0), batch_size=BS, shuffle=False, num_workers=8)
    vision.eval(); head.eval()
    out = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for x, _ in rd:
            f = vision(x.to(DEVICE, non_blocking=True), interpolate_pos_encoding=True).last_hidden_state[:, 0].float()
            out.append(head(f).squeeze(-1).float().cpu().numpy())
    rs = np.concatenate(out)
    pd.DataFrame({"id": rep.id.astype(str), "score": 1.0 / (1.0 + np.exp(-rs))}).to_csv(RUN / "img_bigg_s384_retrain.csv", index=False)
    print(f"DONE s384-retrain best_val_auc={best:.6f} output={RUN}/img_bigg_s384_retrain.csv", flush=True)


if __name__ == "__main__":
    main()
