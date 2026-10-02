#!/usr/bin/env python3
"""Phase 32 OLD-SERVER distilled students, retrained with NEW 14B teacher soft labels.

Usages (old server): /usr/bin/python3.10 code/p32_old_retrain.py --size 7b|1.5b --labels <tag>
REVIEW vs new-server p32 (fixes to avoid past mistakes):
1. soft-max: default 10000? NO - keep FULL 20000 (user never approved subsampling; keep parity)
2. inference predict: logit-diff logits[:,1]-logits[:,0]? Student train used sigmoid(logits[:,0]) whose
   order == -logit-diff order... wait: sigmoid(logit0) is monotone DECREASING in (logit1-logit0).
   So order is preserved under strict monotone transform; auc/rank identical. BUT to keep the
   EXACT same convention as teacher (higher = AI), use logit-diff logits[:,1]-logits[:,0].
   (Score sign: original sigmoid(log,0) high => AI; logit1-logit0 high => AI. Consistent.)
3. Save LoRA adapter per fold (asset).
4. Save raw scores (*_raw.npy) and mean; rank01 replay.
5. batch: 7b->2, 1.5b->4 on 24GB contended card (7B bf16 15GB + grad-ckpt act ~1-2GB + opt);
   kept modest since GPU shared with tenant (GPU0 busy 35GB). Use GPU1 (free).
6. use_cache=False with grad checkpointing (transformers warns) - add model.config.use_cache=False.
7. folds.npy + labels.npy + text csv from /data/湾区杯/文本赛道/runs + /data/湾区杯/initial_text/...
8. val AUC NaN guard (keep past fix).
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

BASE = Path("/data/湾区杯/文本赛道")
OUT = BASE / "runs"
DATA = BASE / "data"
INIT = Path("/data/湾区杯/initial_text")
import glob
DEV = torch.device("cuda:1")  # GPU1 free (GPU0 busy with tenant)
tokenizer = None


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
    """Student convention: BCE trains logits[:,0] high => AI. sigmoid(logit0) is monotone,
    so use logit0 directly (no saturation, order preserved). DO NOT use logit1-logit0
    (that would invert: AI samples would score LOW)."""
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
    global tokenizer
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="7b", choices=["7b", "1.5b"])
    ap.add_argument("--labels", default="soft_labels_v2",
                    help="soft label file tag under runs/ (e.g. soft_labels_v2)")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--suffix", type=str, default="_r2",
                    help="DO NOT use empty: would overwrite new-server student outputs")
    ap.add_argument("--soft-max", type=int, default=20000)
    ap.add_argument("--device", type=str, default="cuda:1")
    ap.add_argument("--only-fold", type=int, default=-1,
                    help="run ONLY this fold then exit (fresh process per fold -> no HAM I mem leak)")
    ap.add_argument("--batch", type=int, default=-1, help="override batch size (default: 4 for 7b, 8 for 1.5b)")
    args = ap.parse_args()
    global DEV
    DEV = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    size = args.size
    TAG = f"lora{size}_distill{args.suffix}"
    # 旧服务器路径：模型权重在 /data/湾区杯/文本赛道/models/
    model_dir = Path("/data/湾区杯/文本赛道/models/Qwen2.5-7B-Instruct" if size == "7b"
                     else "/data/湾区杯/文本赛道/models/Qwen2.5-1.5B-Instruct")
    assert model_dir.exists(), f"model missing {model_dir}"
    print("student", size, "model", model_dir, "labels", args.labels, flush=True)

    # data paths (old server)
    sample_csv = glob.glob(str(INIT / "**" / "text_sample_data.csv"), recursive=True)
    assert sample_csv, "sample csv not found"
    train = pd.read_csv(sample_csv[0], encoding="utf-8-sig")
    test = pd.read_csv(str(BASE / "repro" / "text_test.csv"), encoding="utf-8-sig") \
        if (BASE / "repro" / "text_test.csv").exists() else \
        pd.read_csv(str(Path("/data/湾区杯/replay_data/文本赛道/text_test.csv")), encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(str(OUT / "folds.npy"))
    soft = np.load(str(OUT / f"{args.labels}.npy"))
    assert soft.shape[0] == len(test_texts), f"soft label len {soft.shape[0]} vs {len(test_texts)}"
    print("rows", len(official_texts), len(test_texts), "soft mean", float(soft.mean()), flush=True)

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    from peft import LoraConfig, get_peft_model

    oof = np.zeros(len(y), dtype=np.float64)
    replay = []
    replay_raw = []
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
            rows.append((official_texts[i], float(y[i]), 4.0))
        n_soft = args.soft_max if args.soft_max and args.soft_max < len(test_texts) else len(test_texts)
        sl = slice(0, n_soft)
        for i in range(*sl.indices(len(test_texts))):
            rows.append((test_texts[i], float(soft[i]), 1.0))
        ds = DistillDataset(rows[:], 512)
        # 24GB card: 7b bf16 ~15GB -> batch4-6; 1.5b ~3GB -> batch can go much higher
        bs = args.batch if args.batch > 0 else (4 if size == "7b" else 8)
        loader = DataLoader(ds, batch_size=bs, shuffle=True,
                            num_workers=4, pin_memory=True, collate_fn=collate)
        model = AutoModelForSequenceClassification.from_pretrained(
            str(model_dir), num_labels=2, trust_remote_code=True, torch_dtype=torch.bfloat16,
            ignore_mismatched_sizes=True, attn_implementation="eager")
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False  # avoid SDPA/grad-ckpt warning & memory blowup
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
        # direction guard: student score must be AI-high on official val fold
        a = float(roc_auc_score(y[va_i], vp))
        if a < 0.5:
            print(f"  fold {f} INVERTED (auc {a:.4f}) — flipping", flush=True)
            vp = -vp
            rp_raw = -rp_raw
            a = 1.0 - a
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
                           "pred": "logit_diff", "base": str(model_dir),
                           "labels": args.labels}, fh)
            sz = sum(p.stat().st_size for p in adir.rglob("*"))
            print("adapter saved", adir, "size MB", round(sz / 1e6, 1), flush=True)
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
        print(TAG, "OOF frac==max", round(100 * float(np.mean(np.isclose(oof, oof.max(), atol=1e-6))), 2),
              "unique", int(np.unique(oof).size), flush=True)
        with open(OUT / f"{TAG}_report.json", "w", encoding="utf-8") as fh:
            json.dump({"oof_auc": auc, "size": size, "labels": args.labels, "pred": "logit_diff"}, fh)
        print("P32 OLD RETRAIN DONE", flush=True)
    else:
        missing = [i for i in range(5) if not (OUT / f"{TAG}_fold{i}_replay.npy").exists()]
        print(f"FOLD DONE (missing {missing}) — rerun to continue", flush=True)


if __name__ == "__main__":
    main()
