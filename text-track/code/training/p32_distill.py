#!/usr/bin/env python3
"""Phase 32 (new server): distillation student (LoRA) — hard official x4 + soft replay x1.

Usage: /root/miniconda3/bin/python code/p32_distill.py --size 7b|1.5b
- 5-fold on official (folds.npy); official train-fold hard labels x4; all 20k replay soft labels x1
- early stop per fold on val AUC (official val fold)
- resumable fold checkpoints; Tag: lora{size}_distill
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

BASE = Path("/root/sj-tmp/wanqu")
OUT = BASE / "runs"
DEV = torch.device("cuda:0")
SEED = 20260901

tokenizer = None


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


class DistillDataset(Dataset):
    """rows: (text, target, weight) — target is soft (0..1) or hard (0/1)."""

    def __init__(self, rows, max_len):
        self.rows = [(tokenizer.encode(t, add_special_tokens=False, max_length=max_len - 2),
                      float(tgt), float(w)) for t, tgt, w in rows]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, tgt, w = self.rows[i]
        return {"input_ids": ids, "target": tgt, "weight": w}


def collate(rows):
    feats = [{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]),
              "target": r["target"], "weight": r["weight"]} for r in rows]
    out = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)(feats)
    out["target"] = torch.tensor([r["target"] for r in rows], dtype=torch.float32)
    out["weight"] = torch.tensor([r["weight"] for r in rows], dtype=torch.float32)
    return out


def predict(model, texts, batch=16, max_len=512):
    model.eval()
    vals = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tokenizer(texts[s:s + batch], padding=True, truncation=True, max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            log = model(**enc).logits.float()
            vals.append(torch.sigmoid(log[:, 0]).cpu().numpy())
    return np.concatenate(vals)


def main():
    global tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="7b", choices=["7b", "1.5b"])
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--suffix", type=str, default="")
    ap.add_argument("--soft-max", type=int, default=20000)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = args.size
    TAG = f"lora{size}_distill{args.suffix}"
    model_dir = Path("/root/sj-tmp/Qwen2.5-14B-Instruct" if size == "7b" else "/root/sj-tmp/models")
    # use 1.5B instruct for the small student (weights on new server? not present -> use 7B path trick)
    # fallback: instruct version of the same family if present
    cands = {
        "7b": ["/root/models/Qwen2.5-7B-Instruct", "/root/sj-tmp/Qwen2.5-14B-Instruct"],
        "1.5b": ["/root/models/Qwen2.5-1.5B-Instruct", "/root/sj-tmp/Qwen2.5-14B-Instruct"],
    }
    chosen = None
    for c in cands[size]:
        if Path(c).exists() and any(Path(c).glob("*.safetensors")):
            chosen = Path(c)
            break
    if chosen is None:
        print("NO MODEL FOUND for", size, flush=True)
        sys.exit(1)
    model_dir = chosen
    print("student", size, "model", model_dir, flush=True)

    train = pd.read_csv(str(BASE / "data" / "text_sample_data.csv"), encoding="utf-8-sig")
    test = pd.read_csv(str(BASE / "data" / "text_test.csv"), encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(str(OUT / "folds.npy"))
    soft = np.load(str(OUT / "soft_labels.npy"))
    print("rows", len(official_texts), len(test_texts), "soft mean", float(soft.mean()), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import LoraConfig, get_peft_model

    oof = np.zeros(len(y), dtype=np.float64)
    replay = []
    t0 = time.time()
    for f in range(5):
        tr_i = np.flatnonzero(folds != f)
        va_i = np.flatnonzero(folds == f)
        fp = OUT / f"{TAG}_fold{f}_replay.npy"
        if fp.exists():
            oof[va_i] = np.load(OUT / f"{TAG}_fold{f}_oof.npy")
            replay.append(np.load(fp))
            print("reuse fold", f, flush=True)
            continue
        rows = []
        # official hard x4
        for i in tr_i:
            rows.append((official_texts[i], float(y[i]), 4.0))
        # replay soft x1 (optionally subsample for speed)
        n_soft = args.soft_max if args.soft_max and args.soft_max < len(test_texts) else len(test_texts)
        sl = slice(0, n_soft)
        for i in range(*sl.indices(len(test_texts))):
            rows.append((test_texts[i], float(soft[i]), 1.0))
        ds = DistillDataset(rows[:] , 512)
        loader = DataLoader(ds, batch_size=(4 if size == "7b" else 8), shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="eager")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 bias="none", task_type="SEQ_CLS",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=(5e-5 if size == "7b" else 1e-4), weight_decay=0.01)
        total = len(loader) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)
        best_auc, best_state = -1.0, None
        lossf = torch.nn.BCEWithLogitsLoss(reduction="none")
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for b in loader:
                b = {k: v.to(DEV) for k, v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
                    l = lossf(logits[:, 0], b["target"])
                    loss = (l * b["weight"]).mean()
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            vp = predict(model, [official_texts[i] for i in va_i])
            auc = float(roc_auc_score(y[va_i], vp))
            if not np.isfinite(auc):
                print("  VAL AUC NaN — keep prev", flush=True)
                auc = best_auc
            print(f"  fold {f} ep {epoch} loss {np.mean(losses):.4f} val_auc {auc:.5f} "
                  f"t {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        vp = predict(model, [official_texts[i] for i in va_i])
        rp = predict(model, test_texts)
        oof[va_i] = vp
        replay.append(rp)
        np.save(OUT / f"{TAG}_fold{f}_oof.npy", vp)
        np.save(OUT / f"{TAG}_fold{f}_replay.npy", rp)
        print("fold", f, "final auc", round(float(roc_auc_score(y[va_i], vp)), 5),
              "elapsed", round(time.time() - t0, 0), flush=True)
        del model
        torch.cuda.empty_cache()

    replay = np.mean(replay, axis=0)
    np.save(OUT / f"{TAG}_oof.npy", oof)
    np.save(OUT / f"{TAG}_replay.npy", rank01(replay))
    auc = float(roc_auc_score(y, oof))
    print(TAG, "OOF AUC", round(auc, 6), flush=True)
    with open(OUT / f"{TAG}_report.json", "w", encoding="utf-8") as fh:
        json.dump({"oof_auc": auc, "size": size}, fh)
    print("P32 DONE", flush=True)


if __name__ == "__main__":
    main()
