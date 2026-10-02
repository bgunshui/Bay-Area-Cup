#!/usr/bin/env python3
"""Phase 7: transductive pseudo-label self-training (Qwen2.5-1.5B LoRA).

Pseudo-labels = top/bottom 7.5% of ensemble scores on the replay set.
5-fold CV over official folds (pseudo augments each training fold), OOF + replay predictions.
Output: runs/pseudo15_oof.npy, runs/pseudo15_replay.npy, submissions/probe_pseudo.csv (blend w/ main).
"""
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
SUB = BASE / "文本赛道" / "submissions"
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


def predict(model, texts, batch=32, max_len=768):
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
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(OUT / "folds.npy")

    # pseudo labels from the ft/llm/robust signal (public detectors proven weak on replay)
    # submit_main (0.9394 config) is ft-heavy already; use it alone.
    s_main = rank01(pd.read_csv(SUB / "submit_main.csv").score.to_numpy(np.float64))
    s = s_main
    lo, hi = np.quantile(s, [0.075, 0.925])
    pseudo_idx = np.flatnonzero((s <= lo) | (s >= hi))
    pseudo_label = (s[pseudo_idx] >= hi).astype(np.int64)
    print("pseudo rows", len(pseudo_idx), "pos", int(pseudo_label.sum()), flush=True)

    model_path = MODELS / "Qwen2.5-1.5B-Instruct"
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
        texts = [official_texts[i] for i in tr_i] + [test_texts[i] for i in pseudo_idx]
        labels = np.concatenate([y[tr_i], pseudo_label])
        ds = PairDataset(texts, labels, 768)
        loader = DataLoader(ds, batch_size=4, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_path), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="sdpa")
        model.config.pad_token_id = tokenizer.pad_token_id
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 bias="none", task_type="SEQ_CLS",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=3e-4, weight_decay=0.01)
        total = len(loader)
        sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)
        model.train()
        for b in loader:
            b = {k: v.to(DEV) for k, v in b.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        # OOF + replay
        vp = predict(model, [official_texts[i] for i in va_i])
        rp = predict(model, test_texts)
        oof[va_i] = vp
        replay.append(rp)
        print(f"fold {f} auc {roc_auc_score(y[va_i], vp):.5f} elapsed {time.time()-t0:.0f}s", flush=True)
        del model
        torch.cuda.empty_cache()

    replay = np.mean(replay, axis=0)
    np.save(OUT / "pseudo15_oof.npy", oof)
    np.save(OUT / "pseudo15_replay.npy", rank01(replay))
    auc = float(roc_auc_score(y, oof))
    print("pseudo15 OOF AUC", round(auc, 6), flush=True)
    # probe blend: 0.85 main + 0.15 pseudo
    blend = 0.85 * s_main + 0.15 * rank01(replay)
    blend = rank01(blend)
    pd.DataFrame({"id": test.id.astype(str), "score": blend}).to_csv(
        SUB / "probe_pseudo.csv", index=False)
    pd.DataFrame({"id": test.id.astype(str), "score": rank01(replay)}).to_csv(
        SUB / "probe_pseudo_pure.csv", index=False)
    with open(OUT / "pseudo15_report.json", "w", encoding="utf-8") as fh:
        json.dump({"oof_auc": auc, "pseudo_rows": int(len(pseudo_idx))}, fh)
    print("P7 DONE", flush=True)


if __name__ == "__main__":
    main()
