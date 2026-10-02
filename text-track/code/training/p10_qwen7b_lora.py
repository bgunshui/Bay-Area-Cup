#!/usr/bin/env python3
"""Phase 10: Qwen2.5-7B-Instruct LoRA classifier (5-fold) - ft family upgrade.

Usage: python3 p10_qwen7b_lora.py [--data official|advmix] [--epochs 2] [--batch 2] [--max-len 768]
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

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
MODELS = BASE / "文本赛道" / "models"
DEV = torch.device("cuda:0")
SEED = 20260827

tokenizer = None


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


class PairDataset(Dataset):
    def __init__(self, texts, labels, max_len):
        self.rows = [(tokenizer.encode(t, add_special_tokens=False, max_length=max_len - 2), int(l))
                     for t, l in zip(texts, labels)]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, label = self.rows[i]
        return {"input_ids": ids, "labels": label}


def collate(rows):
    feats = [{"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]),
              "labels": r["labels"]} for r in rows]
    out = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)(feats)
    out["labels"] = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
    return out


def predict(model, texts, batch=8, max_len=768):
    model.eval()
    vals = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tokenizer(texts[s:s + batch], padding=True, truncation=True, max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                log = model(**enc).logits.float()
            vals.append(torch.softmax(log, -1)[:, 1].cpu().numpy())
    return np.concatenate(vals)


def main():
    global tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="official", choices=["official", "advmix"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--tag", type=str, default=None)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(OUT / "folds.npy")

    adv_texts, adv_labels = None, None
    if args.data == "advmix":
        adv_rows = []
        import glob as _g

        for p in _g.glob(str(OUT / "rewrite_Qwen3-1.7B_*_600.jsonl")):
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        item = json.loads(line)
                        adv_rows.append((item["text"], int(item["label"])))
        adv_texts = [r[0] for r in adv_rows]
        adv_labels = np.asarray([r[1] for r in adv_rows], dtype=np.int64)
        print("adv rows", len(adv_texts), flush=True)

    tag = args.tag or f"lora7b_{args.data}"
    model_path = MODELS / "Qwen2.5-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
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
        # resumable: reuse saved fold predictions
        fp = OUT / f"{tag}_fold{f}_replay.npy"
        if fp.exists():
            oof[va_i] = np.load(OUT / f"{tag}_fold{f}_oof.npy")
            replay.append(np.load(fp))
            print("reuse fold", f, flush=True)
            continue
        texts = [official_texts[i] for i in tr_i]
        labels = y[tr_i]
        if adv_texts is not None:
            rng = np.random.RandomState(args.seed + f)
            pick = rng.choice(len(adv_texts), min(6000, len(adv_texts)), replace=False)
            texts = texts + [adv_texts[i] for i in pick]
            labels = np.concatenate([labels, adv_labels[pick]])
        ds = PairDataset(texts, labels, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="sdpa")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 bias="none", task_type="SEQ_CLS",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=2e-4, weight_decay=0.01)
        total = len(loader) * args.epochs
        sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)
        best_auc, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for b in loader:
                b = {k: v.to(DEV) for k, v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**b).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            vp = predict(model, [official_texts[i] for i in va_i])
            auc = float(roc_auc_score(y[va_i], vp))
            print(f"  fold {f} ep {epoch} loss {np.mean(losses):.4f} auc {auc:.5f} "
                  f"t {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        vp = predict(model, [official_texts[i] for i in va_i])
        rp = predict(model, test_texts)
        oof[va_i] = vp
        replay.append(rp)
        np.save(OUT / f"{tag}_fold{f}_oof.npy", vp)
        np.save(OUT / f"{tag}_fold{f}_replay.npy", rp)
        print("fold", f, "final auc", round(float(roc_auc_score(y[va_i], vp)), 5),
              "elapsed", round(time.time() - t0, 0), flush=True)
        del model
        torch.cuda.empty_cache()

    replay = np.mean(replay, axis=0)
    np.save(OUT / f"{tag}_oof.npy", oof)
    np.save(OUT / f"{tag}_replay.npy", rank01(replay))
    auc = float(roc_auc_score(y, oof))
    print(tag, "OOF AUC", round(auc, 6), flush=True)
    with open(OUT / f"{tag}_report.json", "w", encoding="utf-8") as fh:
        json.dump({"oof_auc": auc, "data": args.data, "epochs": args.epochs,
                   "max_len": args.max_len}, fh)
    print("P10 DONE", flush=True)


if __name__ == "__main__":
    main()
