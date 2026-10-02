#!/usr/bin/env python3
"""Phase 43: 14B _p2 self-training (14B as student, domain-aligned; THE big lever).

Uses CURRENT best fusion (p2AD 0.9783) as higher-quality pseudo source.
- Student = 14B (28G bf16, batch 4 + grad-ckpt, 48GB card)
- Labels: pseudo_best_v2.npy (soft, sigmoid(4*(rank-.5))) + confidence weights
- Official hard x2 + pseudo soft xConf
- logit0 predict (higher = AI), NaN guards, adapter save, raw save, resume, direction guard
- batch 4 (14B bf16 ~28G + grad-ckpt act ~2G + opt = ~32G; 48G OK)
Usage: python3 p43_14b_p2.py
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
MODELS = Path("/root/sj-tmp")
tokenizer = None
DEV = None


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


class DistillDataset(Dataset):
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
    """logits[:,0] high = AI (BCE trains logit0). Monotone, never saturates."""
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
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--suffix", default="_p3")
    ap.add_argument("--labels", default="pseudo_best_v2")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260909)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--only-fold", type=int, default=-1,
                    help="run ONLY this fold index then exit (fresh process per fold -> no HAMI mem leak)")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="disable gradient checkpointing (saves ~25-30%% compute; needs spare VRAM)")
    ap.add_argument("--workers", type=int, default=4, help="DataLoader num_workers")
    args = ap.parse_args()
    DEV = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    TAG = f"lora14b{args.suffix}"
    model_dir = MODELS / "Qwen2.5-14B-Instruct"
    assert model_dir.exists(), model_dir
    print("14B self-train, model", model_dir, "labels", args.labels, flush=True)

    train = pd.read_csv(BASE / "data" / "text_sample_data.csv", encoding="utf-8-sig")
    test = pd.read_csv(BASE / "data" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(OUT / "folds.npy")
    ps = np.load(OUT / f"{args.labels}.npy")
    soft = ps.copy()
    conf = np.clip(4.0 * np.abs(soft - 0.5), 0.0, 1.0)
    print("pseudo soft mean", float(soft.mean()), "conf mean", float(conf.mean()), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import LoraConfig, get_peft_model

    oof = np.zeros(len(y), dtype=np.float64)
    replay, replay_raw = [], []
    t0 = time.time()
    fold_list = [args.only_fold] if args.only_fold >= 0 else list(range(5))
    for f in fold_list:
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
        for i in tr_i:
            rows.append((official_texts[i], float(y[i]), 2.0))
        for i in range(len(test_texts)):
            rows.append((test_texts[i], float(soft[i]), float(conf[i])))
        ds = DistillDataset(rows, 512)
        bs = args.batch
        loader = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=args.workers,
                            pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="eager")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        if args.no_grad_ckpt:
            print("gradient checkpointing DISABLED (faster, more VRAM)", flush=True)
            model.gradient_checkpointing_disable()
        else:
            model.gradient_checkpointing_enable()
        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                                 bias="none", task_type="SEQ_CLS",
                                                 target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
        model.to(DEV)
        opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                lr=5e-5, weight_decay=0.01)
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
            print("adapter saved", adir, flush=True)
        except Exception as ex:
            print("ADAPTER SAVE FAILED:", repr(ex), flush=True)
        del model
        torch.cuda.empty_cache()

    replay = np.mean(replay, axis=0)
    # aggregate only when ALL 5 folds are available (single-fold runs must not overwrite tops)
    have_all = all((OUT / f"{TAG}_fold{i}_replay.npy").exists() for i in range(5))
    if have_all:
        oofs = [np.load(OUT / f"{TAG}_fold{i}_oof.npy") for i in range(5)]
        reps = [np.load(OUT / f"{TAG}_fold{i}_replay.npy") for i in range(5)]
        raws = [np.load(OUT / f"{TAG}_fold{i}_raw.npy") for i in range(5)]
        oof = np.concatenate(oofs)  # folds are contiguous masks in this dataset layout
        # rebuild oof in original index order
        oof = np.zeros(len(y), dtype=np.float64)
        for f in range(5):
            oof[np.flatnonzero(folds == f)] = oofs[f]
        rep_mean = np.mean(reps, axis=0)
        np.save(OUT / f"{TAG}_oof.npy", oof)
        np.save(OUT / f"{TAG}_replay.npy", rank01(rep_mean))
        np.save(OUT / f"{TAG}_replay_raw_mean.npy", np.mean(raws, axis=0))
        auc = float(roc_auc_score(y, oof))
        print(TAG, "OOF AUC", round(auc, 6), flush=True)
        o = np.load(OUT / f"{TAG}_oof.npy")
        print(TAG, "OOF frac==max", round(100 * float(np.mean(np.isclose(o, o.max(), atol=1e-6))), 2),
              "unique", int(np.unique(o).size), flush=True)
        with open(OUT / f"{TAG}_report.json", "w", encoding="utf-8") as fh:
            json.dump({"oof_auc": auc, "labels": args.labels, "pred": "logit0"}, fh)
        print("P43 DONE", flush=True)
    else:
        missing = [i for i in range(5) if not (OUT / f"{TAG}_fold{i}_replay.npy").exists()]
        print(f"FOLD DONE (missing folds {missing}) — rerun to continue", flush=True)


if __name__ == "__main__":
    main()
