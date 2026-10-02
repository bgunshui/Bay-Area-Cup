#!/usr/bin/env python3
"""Phase 20: Qwen2.5-7B LoRA adversarial training v2 (fixed config).

Fixes vs the failed advmix: oversample official 4x, cap adversarial rewrites at 3000,
batch 2, lr 1e-4, 3 epochs. Resumable fold checkpointing. Tag: lora7b_advmix_v2.
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
MODELS = BASE / "文本赛道" / "models"
DEV = torch.device("cuda:0")
SEED = 20260829
TAG = "lora7b_advmix_v2"

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


def predict(model, texts, batch=8, max_len=512):
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

    # load adversarial rewrites
    adv_rows = []
    for p in glob.glob(str(OUT / "rewrite_Qwen3-1.7B_*_600.jsonl")):
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    item = json.loads(line)
                    adv_rows.append((item["text"], int(item["label"])))
    adv_texts = [r[0] for r in adv_rows]
    adv_labels = np.asarray([r[1] for r in adv_rows], dtype=np.int64)
    print("adv rows", len(adv_texts), flush=True)

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
        fp = OUT / f"{TAG}_fold{f}_replay.npy"
        if fp.exists():
            oof[va_i] = np.load(OUT / f"{TAG}_fold{f}_oof.npy")
            replay.append(np.load(fp))
            print("reuse fold", f, flush=True)
            continue
        # balanced: official oversampled 4x + 3000 adv (balanced classes)
        base_texts = [official_texts[i] for i in tr_i]
        base_labels = y[tr_i]
        texts = list(base_texts) * 4
        labels = np.concatenate([base_labels] * 4)
        rng = np.random.RandomState(SEED + f)
        # balanced adv: 1500 each class
        a0 = np.flatnonzero(adv_labels == 0)
        a1 = np.flatnonzero(adv_labels == 1)
        pick0 = rng.choice(a0, min(1500, len(a0)), replace=False)
        pick1 = rng.choice(a1, min(1500, len(a1)), replace=False)
        pick = np.concatenate([pick0, pick1])
        texts = texts + [adv_texts[i] for i in pick]
        labels = np.concatenate([labels, adv_labels[pick]])
        print(f"fold {f}: train {len(texts)} (official {len(base_texts)}x4 + adv {len(pick)})", flush=True)

        ds = PairDataset(texts, labels, 512)
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=4,
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
                                lr=1e-4, weight_decay=0.01)
        total = len(loader) * 3
        sched = get_linear_schedule_with_warmup(opt, max(1, total // 10), total)
        best_auc, best_state = -1.0, None
        for epoch in range(3):
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
        json.dump({"oof_auc": auc}, fh)
    print("P20 DONE", flush=True)


if __name__ == "__main__":
    main()
