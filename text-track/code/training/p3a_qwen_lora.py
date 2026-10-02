#!/usr/bin/env python3
"""Phase 3a: LoRA fine-tune Qwen LLM as binary AIGC text classifier, 5-fold CV.

Usage:
  python3 p3a_qwen_lora.py --model Qwen2.5-1.5B-Instruct --data official --epochs 2 --max-len 768
  python3 p3a_qwen_lora.py --model Qwen2.5-3B-Instruct --data augmix --epochs 2 --max-len 768
Saves: runs/lora_{model}_{data}_oof.npy, ..._replay.npy, branch JSON.
"""
import argparse
import glob
import json
import os
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
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
MODELS = BASE / "文本赛道" / "models"
DEV = torch.device("cuda:0")
SEED = 20260827

sys.path.insert(0, "/data/湾区杯/文本赛道/repro")
from wanqu_first_submission.folds import make_folds, text_near_duplicate_groups  # noqa: E402
from wanqu_first_submission.io_utils import seed_everything  # noqa: E402


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


class PairDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.rows = []
        for t, l in zip(texts, labels):
            enc = tokenizer.encode(t, add_special_tokens=False, max_length=max_len - 2)
            self.rows.append((enc, int(l)))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        ids, label = self.rows[i]
        return {"input_ids": ids, "labels": label}


def collate(rows):
    from transformers import DataCollatorWithPadding

    feats = [
        {"input_ids": r["input_ids"], "attention_mask": [1] * len(r["input_ids"]), "labels": r["labels"]}
        for r in rows
    ]
    out = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)(feats)
    out["labels"] = torch.tensor([r["labels"] for r in rows], dtype=torch.long)
    return out


tokenizer = None


def load_external(kind):
    """Return (texts, labels) from external corpora."""
    from wanqu_first_submission.paths import ROOT

    rows = []
    counts = {0: 0, 1: 0}
    hc3 = ROOT / "external_data" / "hc3_chinese" / "all.jsonl"
    if hc3.exists():
        with hc3.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for v in item.get("human_answers", []):
                    if v and counts[0] < 4000:
                        rows.append((v, 0)); counts[0] += 1
                for v in item.get("chatgpt_answers", []):
                    if v and counts[1] < 4000:
                        rows.append((v, 1)); counts[1] += 1
                if min(counts.values()) >= 4000:
                    break
    m4 = ROOT / "external_data" / "m4"
    for fn, label in (("qazh_chatgpt.jsonl", 1), ("qazh_davinci.jsonl", 1)):
        p = m4 / fn
        if p.exists():
            with p.open(encoding="utf-8") as f:
                for line in f:
                    if counts[label] < 5500:
                        rows.append((json.loads(line).get("text", ""), label))
                        counts[label] += 1
    qs = ROOT / "external_data" / "qwen_synthetic.jsonl"
    if qs.exists():
        with qs.open(encoding="utf-8") as f:
            for line in f:
                if counts[1] < 6500:
                    rows.append((json.loads(line).get("text", ""), 1))
                    counts[1] += 1
    txts = [r[0] for r in rows if r[0]]
    labs = [r[1] for r in rows if r[0]]
    return txts, np.asarray(labs, dtype=np.int64)


def main():
    global tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen2.5-1.5B-Instruct")
    ap.add_argument("--data", default="official", choices=["official", "augmix"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--adversarial", nargs="*", default=[])
    args = ap.parse_args()
    seed_everything(SEED)

    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    folds = np.load(OUT / "folds.npy") if (OUT / "folds.npy").exists() else make_folds(
        y, text_near_duplicate_groups(official_texts), 5, 20260823)
    np.save(OUT / "folds.npy", folds)
    test_texts = test.text.fillna("").astype(str).tolist()

    adv_texts, adv_labels = None, None
    if args.adversarial:
        adv_rows = []
        import json as _json

        for p in args.adversarial:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        item = _json.loads(line)
                        adv_rows.append((item["text"], int(item["label"])))
        if adv_rows:
            adv_texts = [r[0] for r in adv_rows]
            adv_labels = np.asarray([r[1] for r in adv_rows], dtype=np.int64)
            print("adversarial rows", len(adv_texts), flush=True)
    tag = f"lora_{args.model}_{args.data}" + ("_adv" if adv_texts is not None else "")
    model_path = MODELS / args.model
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    print("load model", model_path, flush=True)

    oof = np.zeros(len(official_texts), dtype=np.float64)
    replay = []
    t0 = time.time()
    from peft import LoraConfig, get_peft_model
    from transformers import (  # noqa
        DataCollatorWithPadding, get_linear_schedule_with_warmup, Trainer, TrainingArguments,
    )

    def make_trainer(model, train_dataset, total_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        sched = get_linear_schedule_with_warmup(opt, max(1, total_steps // 10), total_steps)
        return opt, sched

    # external-augmented variant: append external texts to training fold, dedup + weight cap
    ext_texts = ext_labels = None
    if args.data == "augmix":
        ext_texts, ext_labels = load_external("augmix")
        # mix: keep official fold train + 12k external (balanced), dedup by text
        print("external rows", len(ext_texts), flush=True)

    from transformers import DataCollatorWithPadding

    def train_fold(train_idx, valid_idx):
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=2, trust_remote_code=True,
            torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True,
            attn_implementation="sdpa")
        model.config.pad_token_id = tokenizer.pad_token_id
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          task_type="SEQ_CLS", target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora)
        model.to(DEV)

        ttexts = [official_texts[i] for i in train_idx]
        tlabels = y[train_idx]
        if adv_texts is not None:
            rng2 = np.random.RandomState(SEED + f)
            pick = rng2.choice(len(adv_texts), min(6000, len(adv_texts)), replace=False)
            ttexts = ttexts + [adv_texts[i] for i in pick]
            tlabels = np.concatenate([tlabels, adv_labels[pick]])
        if ext_texts is not None:
            # balanced sample 6000 per class
            idx0 = np.flatnonzero(ext_labels == 0); idx1 = np.flatnonzero(ext_labels == 1)
            rng = np.random.RandomState(SEED)
            pick = np.concatenate([rng.choice(idx0, min(6000, len(idx0)), replace=False),
                                   rng.choice(idx1, min(6000, len(idx1)), replace=False)])
            ttexts = ttexts + [ext_texts[i] for i in pick]
            tlabels = np.concatenate([tlabels, ext_labels[pick]])
        ds = PairDataset(ttexts, tlabels, tokenizer, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=collate)
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
        total_steps = len(loader) * args.epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, max(1, total_steps // 10), total_steps)

        best_auc = -1
        best_state = None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for b in loader:
                b = {k: v.to(DEV) for k, v in b.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(**b)
                    loss = out.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            # eval on fold
            model.eval()
            ps = []
            for i in valid_idx:
                enc = tokenizer(official_texts[i], truncation=True, max_length=args.max_len,
                                return_tensors="pt")
                with torch.inference_mode():
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        log = model(**{k: v.to(DEV) for k, v in enc.items()}).logits.float()
                ps.append(torch.softmax(log, -1)[0, 1].item())
            auc = roc_auc_score(y[valid_idx], ps)
            print(f"  epoch {epoch} loss {np.mean(losses):.4f} fold_auc {auc:.5f} "
                  f"elapsed {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        # OOF + replay preds with best state
        model.eval()
        with torch.no_grad():
            vp = []
            for i in valid_idx:
                enc = tokenizer(official_texts[i], truncation=True, max_length=args.max_len, return_tensors="pt")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    log = model(**{k: v.to(DEV) for k, v in enc.items()}).logits.float()
                vp.append(torch.softmax(log, -1)[0, 1].item())
            rp = []
            for s in range(0, len(test_texts), 32):
                chunk = test_texts[s:s + 32]
                enc = tokenizer(chunk, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    log = model(**{k: v.to(DEV) for k, v in enc.items()}).logits.float()
                rp.extend(torch.softmax(log, -1)[:, 1].tolist())
        del model
        torch.cuda.empty_cache()
        return np.asarray(vp), np.asarray(rp), best_auc

    for f in range(5):
        tr = np.flatnonzero(folds != f)
        va = np.flatnonzero(folds == f)
        vp, rp, best_auc = train_fold(tr, va)
        oof[va] = vp
        replay.append(rp)
        print("fold", f, "auc", round(best_auc, 6), "elapsed", round(time.time() - t0, 0), flush=True)

    replay = np.mean(replay, axis=0)
    np.save(OUT / f"{tag}_oof.npy", oof)
    np.save(OUT / f"{tag}_replay.npy", rank01(replay))
    print(tag, "OOF AUC", round(float(roc_auc_score(y, oof)), 6), flush=True)
    pd.DataFrame({"id": test.id.astype(str), "score": rank01(replay)}).to_csv(
        OUT / f"{tag}_candidate.csv", index=False)
    with open(OUT / f"{tag}_report.json", "w", encoding="utf-8") as fh:
        json.dump({"model": args.model, "data": args.data, "epochs": args.epochs,
                   "max_len": args.max_len, "lr": args.lr, "oof_auc": float(roc_auc_score(y, oof))},
                  fh, ensure_ascii=False, indent=2)
    print("P3a DONE", flush=True)


if __name__ == "__main__":
    main()
