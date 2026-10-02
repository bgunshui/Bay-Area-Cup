#!/usr/bin/env python3
"""Phase 42 (NEW SERVER): pseudo-label self-training with the CURRENT best fusion (0.97366).

Differs from p21: labels from best submission, but:
 - uses SOFT labels (sigmoid(4*(rank-.5))) for pseudo rows -> keeps ranking info (no hard top10%)
 - high-confidence weighting: w = clamp(4*|rank-.5|, 0, 1) so mid-confidence gets low weight
 - official hard x2 (as p21), plus pseudo soft rows
 - logit-diff predict (logits[:,0], student convention: BCE on logits0 => AI high)
 - NaN guards, adapter save, raw save, resume, direction guard
 - batch: 7b->4, 1.5b->8 (48GB card, both models share GPU)
Usage: python3 p42_self_train.py --size 7b|1.5b --suffix _p2
"""
import argparse
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
MODELS = Path("/root/models")
tokenizer = None


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


class DistillDataset(Dataset):
    """rows: (text, target(s0..1), weight)."""

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
    """logits[:,0] high => AI (BCE trains logit0). Monotone with sigmoid, never saturates."""
    model.eval()
    vals = []
    with torch.inference_mode():
        for s in range(0, len(texts), batch):
            enc = tokenizer(texts[s:s + batch], padding=True, truncation=True, max_length=max_len,
                            return_tensors="pt")
            enc = {k: v.to(DEV) for k, v in enc.items()}
            log = model(**enc).logits.float()
            vals.append(log[:, 0].cpu().numpy())
    return np.concatenate(vals)


def main():
    global tokenizer, DEV
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="7b", choices=["7b", "1.5b"])
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--suffix", default="_p2")
    ap.add_argument("--labels", default="pseudo_best", help="tag of pseudo npy under runs/")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260906)
    args = ap.parse_args()
    DEV = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = args.size
    TAG = f"lora{size}{args.suffix}"
    model_dir = MODELS / ("Qwen2.5-7B-Instruct" if size == "7b" else "Qwen2.5-1.5B-Instruct")
    assert model_dir.exists(), model_dir
    print("self-train", size, "model", model_dir, "labels", args.labels, flush=True)

    train = pd.read_csv(BASE / "data" / "text_sample_data.csv", encoding="utf-8-sig")
    test = pd.read_csv(BASE / "data" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(OUT / "folds.npy")

    # pseudo soft labels + confidence weights (from best submission score)
    ps = np.load(OUT / f"{args.labels}.npy")           # (20000,) in [0,1]
    soft = ps.copy()
    conf = np.clip(4.0 * np.abs(soft - 0.5), 0.0, 1.0)  # high-confidence weight
    print("pseudo soft mean", float(soft.mean()), "conf mean", float(conf.mean()), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import LoraConfig, get_peft_model

    oof = np.zeros(len(y), dtype=np.float64)
    replay = []
    replay_raw = []
    t0 = time.time()
    for f in range(5):
        tr_i = np.flatnonzero(folds != f)
        va_i = np.flatnonzero(folds == f)
        fp = OUT / f"{TAG}_fold{f}_replay.npy"
        if fp.exists():
            oof[va_i] = np.load(OUT / f"{TAG}_fold{f}_oof.npy")
            replay.append(np.load(fp))
            replay_raw.append(np.load(OUT / f"{TAG}_fold{f}_raw.npy"))
            print("reuse fold", f, flush=True)
            continue
        rows = []
        # official hard x2 (like p21)
        for i in tr_i:
            rows.append((official_texts[i], float(y[i]), 2.0))
        # pseudo soft rows (all 20k, weight = confidence)
        for i in range(len(test_texts)):
            rows.append((test_texts[i], float(soft[i]), float(conf[i])))
        ds = DistillDataset(rows, 512)
        bs = 4 if size == "7b" else 8
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="eager")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
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
        rp_raw = predict(model, test_texts)
        a = float(roc_auc_score(y[va_i], vp))
        if a < 0.5:
            print(f"  fold {f} INVERTED ({a:.4f}) — flipping", flush=True)
            vp = -vp; rp_raw = -rp_raw; a = 1.0 - a
        rp = rank01(rp_raw)
        oof[va_i] = vp
        replay.append(rp)
        replay_raw.append(rp_raw)
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
                json.dump({"fold": f, "best_auc": best_auc, "tag": TAG,
                           "pred": "logit0", "labels": args.labels, "base": str(model_dir)}, fh)
            sz = sum(p.stat().st_size for p in adir.rglob("*"))
            print("adapter saved", adir, "MB", round(sz / 1e6, 1), flush=True)
        except Exception as ex:
            print("ADAPTER SAVE FAILED:", repr(ex), flush=True)
        del model
        torch.cuda.empty_cache()

    replay = np.mean(replay, axis=0)
    np.save(OUT / f"{TAG}_oof.npy", oof)
    np.save(OUT / f"{TAG}_replay.npy", rank01(replay))
    np.save(OUT / f"{TAG}_replay_raw_mean.npy", np.mean(replay_raw, axis=0))
    auc = float(roc_auc_score(y, oof))
    print(TAG, "OOF AUC", round(auc, 6), flush=True)
    o = np.load(OUT / f"{TAG}_oof.npy")
    print(TAG, "OOF frac==max", round(100 * float(np.mean(np.isclose(o, o.max(), atol=1e-6))), 2),
          "unique", int(np.unique(o).size), flush=True)
    with open(OUT / f"{TAG}_report.json", "w", encoding="utf-8") as fh:
        json.dump({"oof_auc": auc, "size": size, "labels": args.labels, "pred": "logit0"}, fh)
    print("P42 DONE", flush=True)


if __name__ == "__main__":
    main()
