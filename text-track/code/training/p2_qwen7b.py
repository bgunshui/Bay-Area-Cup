#!/usr/bin/env python3
"""Phase 2b': Qwen2.5-7B-Instruct (fp16) likelihood + hidden-state features
(14B GPTQ lacks CUDA kernels, too slow; 7B fp16 is the biggest usable model).

Saves: runs/q7b_lik_oof.npy / q7b_lik_replay.npy, q7b_hidden_{mean,first,last}_oof/replay, q7b_all_*
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
MODEL_DIR = BASE / "文本赛道" / "models" / "Qwen2.5-7B-Instruct"
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
    fold_path = OUT / "folds.npy"
    if not fold_path.exists():
        sys.path.insert(0, "/data/湾区杯/文本赛道/repro")
        from wanqu_first_submission.folds import make_folds, text_near_duplicate_groups

        folds = make_folds(y, text_near_duplicate_groups(train_text), 5, 20260823)
        np.save(fold_path, folds)
    else:
        folds = np.load(fold_path)
    print("rows", len(train_text), len(test_text), flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(DEV).eval()
    print("model loaded (AutoModel, headless)", flush=True)
    torch.cuda.empty_cache()

    # ---------- A: hidden embeddings (256) ----------
    def extract_hidden(texts, cache):
        if cache.exists():
            z = np.load(cache)
            if all(z[n].shape[0] == len(texts) for n in ("mean", "first", "last")):
                print("reuse", cache, flush=True)
                return {n: np.asarray(z[n], dtype=np.float32) for n in ("mean", "first", "last")}
        arrays = {"mean": [], "first": [], "last": []}
        t0 = time.time()
        for start in range(0, len(texts), 4):
            chunk = texts[start:start + 4]
            tokenized = tok(chunk, padding=True, truncation=True, max_length=256, return_tensors="pt")
            tokenized = {k: v.to(DEV) for k, v in tokenized.items()}
            with torch.inference_mode():
                out = model(**tokenized, output_hidden_states=True, use_cache=False)
                hidden = out.hidden_states[-1]
                mask = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                first = hidden[:, 0, :]
                last_idx = tokenized["attention_mask"].sum(1).clamp_min(1) - 1
                last = hidden[torch.arange(hidden.shape[0], device=DEV), last_idx, :]
            arrays["mean"].append(torch.nn.functional.normalize(mean.float(), dim=1).cpu().numpy().astype(np.float32))
            arrays["first"].append(torch.nn.functional.normalize(first.float(), dim=1).cpu().numpy().astype(np.float32))
            arrays["last"].append(torch.nn.functional.normalize(last.float(), dim=1).cpu().numpy().astype(np.float32))
            if start % 40 == 0:
                print(f"  hidden {start + len(chunk)}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
        res = {n: np.concatenate(arrays[n], axis=0) for n in arrays}
        np.savez(cache, **res)
        return res

    train_h = extract_hidden(train_text, OUT / "q7b_hidden_train.npz")
    test_h = extract_hidden(test_text, OUT / "q7b_hidden_test.npz")
    print("hidden done", flush=True)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260824)
    for name in ("mean", "first", "last"):
        best = (0.0, -1.0)
        for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
            oof = np.zeros(len(y), dtype=np.float64)
            for tr, va in cv.split(train_h[name], y):
                clf = make_pipeline(StandardScaler(with_mean=False),
                                    LogisticRegression(C=c, max_iter=1200, class_weight="balanced", solver="liblinear"))
                clf.fit(train_h[name][tr], y[tr])
                oof[va] = clf.decision_function(train_h[name][va])
            a = roc_auc_score(y, oof)
            if a > best[1]:
                best = (c, a)
        c, a = best
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, va in cv.split(train_h[name], y):
            clf = make_pipeline(StandardScaler(with_mean=False),
                                LogisticRegression(C=c, max_iter=1600, class_weight="balanced", solver="liblinear"))
            clf.fit(train_h[name][tr], y[tr])
            oof[va] = clf.decision_function(train_h[name][va])
        clf = make_pipeline(StandardScaler(with_mean=False),
                            LogisticRegression(C=c, max_iter=1600, class_weight="balanced", solver="liblinear"))
        clf.fit(train_h[name], y)
        np.save(OUT / f"q7b_hidden_{name}_oof.npy", oof)
        np.save(OUT / f"q7b_hidden_{name}_replay.npy", rank01(clf.decision_function(test_h[name])))
        print("head", name, "C", c, "auc", round(a, 6), flush=True)

    # ---------- B: likelihood (need lm_head back) ----------
    del model
    torch.cuda.empty_cache()
    model2 = AutoModelForCausalLM.from_pretrained(
        str(MODEL_DIR), trust_remote_code=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa").to(DEV).eval()
    print("model2 loaded", flush=True)

    def extract_lik(texts, cache, win=256):
        if cache.exists() and np.load(cache).shape[0] == len(texts):
            print("reuse lik", cache, flush=True)
            return np.load(cache)
        rows = []
        for i, s in enumerate(texts):
            ids = tok.encode(str(s or ""), add_special_tokens=False)[:2048]
            for wd in _windows(ids, win):
                rows.append((i, wd))
        feats, di = [], []
        t0 = time.time()
        for start in range(0, len(rows), 6):
            batch = rows[start:start + 6]
            width = max(len(r[1]) for r in batch)
            pad = tok.pad_token_id
            ids = torch.full((len(batch), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, (_, wd) in enumerate(batch):
                ids[j, :len(wd)] = torch.tensor(wd, dtype=torch.long)
                mask[j, :len(wd)] = 1
            inp = {"input_ids": ids.to(DEV), "attention_mask": mask.to(DEV)}
            with torch.inference_mode():
                logits = model2(**inp).logits
            for row in range(logits.shape[0]):
                feats.append(_row_features(logits[row], inp["input_ids"][row], inp["attention_mask"][row]))
            di.append(np.asarray([r[0] for r in batch]))
            del logits
            if start % 180 == 0:
                print(f"  lik {start + len(batch)}/{len(rows)} {time.time()-t0:.0f}s", flush=True)
        allf = np.stack(feats)
        alld = np.concatenate(di)
        agg = np.zeros((len(texts), allf.shape[1] * 3), dtype=np.float32)
        for doc_idx in range(len(texts)):
            v = allf[alld == doc_idx]
            agg[doc_idx] = np.concatenate([v.mean(0), v.min(0), v.max(0)])
        np.save(cache, agg)
        return agg

    tr_lik = extract_lik(train_text, OUT / "q7b_lik_train.npz")
    te_lik = extract_lik(test_text, OUT / "q7b_lik_test.npz")
    del model2
    torch.cuda.empty_cache()

    sys.path.insert(0, "/data/湾区杯/文本赛道/repro")
    import lightgbm as lgb  # noqa: E402

    def fit_lgb(X, y, Xte):
        oof = np.zeros(len(y), dtype=np.float64)
        tests = []
        for f in range(5):
            tr = np.flatnonzero(folds != f)
            va = np.flatnonzero(folds == f)
            m = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.02,
                                   num_leaves=7, max_depth=3, min_child_samples=35,
                                   subsample=0.9, colsample_bytree=0.8, reg_alpha=0.8, reg_lambda=3.0,
                                   random_state=20260827 + f, n_jobs=8, verbosity=-1)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
            tests.append(m.predict_proba(Xte)[:, 1])
        return oof, np.mean(tests, axis=0)

    oof_lik, te_lik_scr = fit_lgb(tr_lik, y, te_lik)
    np.save(OUT / "q7b_lik_oof.npy", oof_lik)
    np.save(OUT / "q7b_lik_replay.npy", rank01(te_lik_scr))
    print("q7b lik AUC", round(float(roc_auc_score(y, oof_lik)), 6), flush=True)

    hidden_tr = np.concatenate([train_h[n] for n in ("mean", "first", "last")], axis=1)
    hidden_te = np.concatenate([test_h[n] for n in ("mean", "first", "last")], axis=1)
    Xtr = np.concatenate([hidden_tr, tr_lik], axis=1).astype(np.float32)
    Xte = np.concatenate([hidden_te, te_lik], axis=1).astype(np.float32)
    oof_all, te_all = fit_lgb(Xtr, y, Xte)
    np.save(OUT / "q7b_all_oof.npy", oof_all)
    np.save(OUT / "q7b_all_replay.npy", rank01(te_all))
    print("q7b all AUC", round(float(roc_auc_score(y, oof_all)), 6), flush=True)
    with open(OUT / "p2b_report.json", "w", encoding="utf-8") as fh:
        json.dump({"lik": float(roc_auc_score(y, oof_lik)), "all": float(roc_auc_score(y, oof_all))}, fh)
    print("P2b DONE", flush=True)


if __name__ == "__main__":
    main()
