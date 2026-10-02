#!/usr/bin/env python3
"""Phase 6: Qwen3-4B-Base likelihood features (base model NLL is a strong AIGC signal).

Usage: python3 p6_4b_lik.py
Saves: runs/q4b_lik_oof.npy, runs/q4b_lik_replay.npy (+ raw feature npz)
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
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
MODEL_DIR = BASE / "文本赛道" / "models" / "Qwen3-4B-Base"
DEV = torch.device("cuda:0")


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def _row_features(logits, ids, mask):
    valid = mask[1:].bool()
    if not valid.any():
        return np.asarray([0.0] * 9 + [1.0], dtype=np.float32)
    prediction = logits[:-1][valid].float()
    target = ids[1:][valid]
    log_norm = torch.logsumexp(prediction, dim=-1)
    target_logits = prediction.gather(1, target[:, None]).squeeze(1)
    nll = log_norm - target_logits
    sampled = prediction[::16]
    sampled_targets = target[::16]
    sampled_norm = torch.logsumexp(sampled, dim=-1)
    probabilities = torch.softmax(sampled, dim=-1)
    entropy = sampled_norm - (probabilities * sampled).sum(dim=-1)
    chosen = sampled.gather(1, sampled_targets[:, None])
    rank = (sampled > chosen).sum(dim=-1).float() + 1.0
    nll_np = nll.detach().cpu().numpy()
    entropy_np = entropy.detach().cpu().numpy()
    rank_np = torch.log1p(rank).detach().cpu().numpy()
    unique_ratio = float(torch.unique(target).numel() / max(target.numel(), 1))
    return np.asarray(
        [nll_np.mean(), nll_np.std(), *np.quantile(nll_np, [0.1, 0.5, 0.9]),
         entropy_np.mean(), entropy_np.std(), rank_np.mean(), rank_np.std(), unique_ratio],
        dtype=np.float32,
    )


def _windows(ids, win=256):
    if not ids:
        return [[0]]
    if len(ids) <= win:
        return [ids]
    return [ids[0:win], ids[len(ids) - win:]]


def main():
    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    train_text = train.text.fillna("").astype(str).tolist()
    test_text = test.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    folds = np.load(OUT / "folds.npy")
    print("rows", len(train_text), len(test_text), flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), local_files_only=True, trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(DEV).eval()
    print("loaded 4B base", flush=True)

    def extract_lik(texts, cache):
        if cache.exists() and np.load(cache).shape[0] == len(texts):
            print("reuse", cache, flush=True)
            return np.load(cache)
        rows = []
        for i, s in enumerate(texts):
            ids = tok.encode(str(s or ""), add_special_tokens=False)[:2048]
            for wd in _windows(ids):
                rows.append((i, wd))
        feats, di = [], []
        t0 = time.time()
        for start in range(0, len(rows), 4):
            batch = rows[start:start + 4]
            width = max(len(r[1]) for r in batch)
            pad = tok.pad_token_id
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, (_, wd) in enumerate(batch):
                ids[j, :len(wd)] = torch.tensor(wd, dtype=torch.long)
                mask[j, :len(wd)] = 1
            inp = {"input_ids": ids.to(DEV), "attention_mask": mask.to(DEV)}
            with torch.inference_mode():
                logits = model(**inp).logits
            for row in range(logits.shape[0]):
                feats.append(_row_features(logits[row], inp["input_ids"][row], inp["attention_mask"][row]))
            di.append(np.asarray([r[0] for r in batch]))
            del logits
            if start % 120 == 0:
                print(f"  lik {start + len(batch)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        allf = np.stack(feats)
        alld = np.concatenate(di)
        agg = np.zeros((len(texts), allf.shape[1] * 3), dtype=np.float32)
        for doc_idx in range(len(texts)):
            v = allf[alld == doc_idx]
            agg[doc_idx] = np.concatenate([v.mean(0), v.min(0), v.max(0)])
        np.save(cache, agg)
        print("saved", cache, flush=True)
        return agg

    tr = extract_lik(train_text, OUT / "q4b_lik_train.npz")
    te = extract_lik(test_text, OUT / "q4b_lik_test.npz")
    del model
    torch.cuda.empty_cache()

    import lightgbm as lgb

    oof = np.zeros(len(y), dtype=np.float64)
    tests = []
    for f in range(5):
        tr_i = np.flatnonzero(folds != f)
        va_i = np.flatnonzero(folds == f)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.02,
                               num_leaves=7, max_depth=3, min_child_samples=35,
                               subsample=0.9, colsample_bytree=0.8, reg_alpha=0.8, reg_lambda=3.0,
                               random_state=20260827 + f, n_jobs=8, verbosity=-1)
        m.fit(tr[tr_i], y[tr_i])
        oof[va_i] = m.predict_proba(tr[va_i])[:, 1]
        tests.append(m.predict_proba(te)[:, 1])
    np.save(OUT / "q4b_lik_oof.npy", oof)
    np.save(OUT / "q4b_lik_replay.npy", rank01(np.mean(tests, axis=0)))
    print("q4b_lik OOF AUC", round(float(roc_auc_score(y, oof)), 6), flush=True)
    with open(OUT / "q4b_lik_report.json", "w", encoding="utf-8") as fh:
        json.dump({"oof_auc": float(roc_auc_score(y, oof))}, fh)
    print("P6 DONE", flush=True)


if __name__ == "__main__":
    main()
