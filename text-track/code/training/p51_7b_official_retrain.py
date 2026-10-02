#!/usr/bin/env python3
"""Phase 51: RE-TRAIN lora7b_official (the lost 'pure 7B on official 1000' branch).

Original: p10_qwen7b_lora.py --data official --epochs 2 --batch 2 --max-len 768 --seed 20260827
Improvements (avoid past mistakes):
  - predict uses logit diff (logits[:,1]-logits[:,0]) instead of softmax -> no saturation
  - batch 4 (24GB card is fine for 7B bf16 + grad-ckpt at max_len 768)
  - NaN guards, adapter saved per fold, raw scores saved, fold resume, direction guard
  - config preserved: official 1000 only (no pseudo labels) -> keeps information independence
Usage: python3 p51_7b_official_retrain.py [--batch 4] [--seed 20260827] [--tag lora7b_official_rt]
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
tokenizer = None
DEV = None


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
    """CE training => positive class is logits[:,1]. Use logit diff (no saturation)."""
    model.eval()
    vals = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tokenizer(texts[s:s + batch], padding=True, truncation=True, max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            log = model(**enc).logits.float()
            vals.append((log[:, 1] - log[:, 0]).cpu().numpy())
    return np.concatenate(vals)


def main():
    global tokenizer, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=768)
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--tag", type=str, default="lora7b_official_rt")
    ap.add_argument("--only-fold", type=int, default=-1)
    args = ap.parse_args()
    DEV = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    TAG = args.tag

    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(OUT / "folds.npy")

    model_path = MODELS / "Qwen2.5-7B-Instruct"
    assert model_path.exists(), model_path
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import LoraConfig, get_peft_model

    t0 = time.time()
    fold_list = [args.only_fold] if args.only_fold >= 0 else list(range(5))
    for f in fold_list:
        tr_i = np.flatnonzero(folds != f)
        va_i = np.flatnonzero(folds == f)
        fp = OUT / f"{TAG}_fold{f}_replay.npy"
        if fp.exists():
            print("reuse fold", f, flush=True)
            continue
        texts = [official_texts[i] for i in tr_i]
        labels = y[tr_i]
        ds = PairDataset(texts, labels, args.max_len)
        loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="eager")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 bias="none", task_type="SEQ_CLS",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=1e-4, weight_decay=0.01)
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
                if not torch.isfinite(loss):
                    opt.zero_grad(set_to_none=True)
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            vp = predict(model, [official_texts[i] for i in va_i], batch=8, max_len=args.max_len)
            auc = float(roc_auc_score(y[va_i], vp))
            if not np.isfinite(auc):
                print("  VAL AUC NaN — keep prev", flush=True)
                auc = best_auc
            print(f"  fold {f} ep {epoch} loss {np.mean(losses):.4f} auc {auc:.5f} "
                  f"t {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        vp = predict(model, [official_texts[i] for i in va_i], batch=8, max_len=args.max_len)
        rp_raw = predict(model, test_texts, batch=8, max_len=args.max_len)
        a = float(roc_auc_score(y[va_i], vp))
        if a < 0.5:
            print(f"  fold {f} INVERTED ({a:.4f}) — flipping", flush=True)
            vp = -vp; rp_raw = -rp_raw
        rp = rank01(rp_raw)
        np.save(OUT / f"{TAG}_fold{f}_oof.npy", vp)
        np.save(OUT / f"{TAG}_fold{f}_raw.npy", rp_raw)
        np.save(OUT / f"{TAG}_fold{f}_replay.npy", rp)
        print("fold", f, "final auc", round(float(roc_auc_score(y[va_i], vp)), 5),
              "raw unique", int(np.unique(rp_raw).size),
              "raw frac==max", round(100 * float(np.mean(np.isclose(rp_raw, rp_raw.max(), atol=1e-6))), 2),
              "elapsed", round(time.time() - t0, 0), flush=True)
        adir = OUT / f"{TAG}_fold{f}_adapter"
        try:
            model.save_pretrained(str(adir))
            with open(adir / "meta.json", "w", encoding="utf-8") as fh:
                json.dump({"fold": f, "best_auc": best_auc, "tag": TAG, "config": "official-only",
                           "pred": "logit_diff_CE", "base": str(model_path)}, fh)
            print("adapter saved", adir, flush=True)
        except Exception as ex:
            print("ADAPTER SAVE FAILED:", repr(ex), flush=True)
        del model
        torch.cuda.empty_cache()

    have_all = all((OUT / f"{TAG}_fold{i}_replay.npy").exists() for i in range(5))
    if have_all:
        oofs = [np.load(OUT / f"{TAG}_fold{i}_oof.npy") for i in range(5)]
        reps = [np.load(OUT / f"{TAG}_fold{i}_replay.npy") for i in range(5)]
        raws = [np.load(OUT / f"{TAG}_fold{i}_raw.npy") for i in range(5)]
        oof = np.zeros(len(y), dtype=np.float64)
        for i in range(5):
            oof[np.flatnonzero(folds == i)] = oofs[i]
        np.save(OUT / f"{TAG}_oof.npy", oof)
        np.save(OUT / f"{TAG}_replay.npy", rank01(np.mean(reps, axis=0)))
        np.save(OUT / f"{TAG}_replay_raw_mean.npy", np.mean(raws, axis=0))
        auc = float(roc_auc_score(y, oof))
        print(TAG, "OOF AUC", round(auc, 6), flush=True)
        with open(OUT / f"{TAG}_report.json", "w", encoding="utf-8") as fh:
            json.dump({"oof_auc": auc, "config": "official-only", "pred": "logit_diff_CE"}, fh)
        print("P51 DONE", flush=True)
    else:
        missing = [i for i in range(5) if not (OUT / f"{TAG}_fold{i}_replay.npy").exists()]
        print(f"FOLD DONE (missing {missing})", flush=True)


if __name__ == "__main__":
    main()
