#!/usr/bin/env python3
"""Phase 2b: Qwen2.5-14B-Instruct (GPTQ) features.

A) hidden-state embeddings (mean/first/last, max_len) -> LR heads -> OOF + replay rank
B) windowed token NLL/entropy/rank features -> branch scores (npz for fusion)
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
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BASE / "models" / "Qwen2.5-14B-Instruct"
DEV = torch.device("cuda:0")

from auto_gptq import AutoGPTQForCausalLM  # noqa: E402


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def load_14b(keep_lm_head: bool):
    tok = AutoTokenizer = None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR), trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoGPTQForCausalLM.from_quantized(
        str(MODEL_DIR),
        device="cuda:0",
        use_safetensors=True,
        use_triton=False,
        inject_fused_attention=False,
        inject_fused_mlp=False,
        disable_exllama=True,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.eval()
    if not keep_lm_head:
        wrapper = model.model
        if hasattr(wrapper, "lm_head"):
            wrapper.lm_head = None
    return tokenizer, model


# ---------------- A: hidden embeddings ----------------
def extract_hidden(texts, tokenizer, backbone, batch_size, max_length, cache: Path):
    if cache.exists():
        z = np.load(cache)
        if all(z[n].shape[0] == len(texts) for n in ("mean", "first", "last")):
            print("reuse hidden", cache, flush=True)
            return {n: np.asarray(z[n], dtype=np.float32) for n in ("mean", "first", "last")}
    arrays = {"mean": [], "first": [], "last": []}
    t0 = time.time()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        tokenized = tokenizer(chunk, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        tokenized = {k: v.to(DEV) for k, v in tokenized.items()}
        with torch.inference_mode():
            outputs = backbone(**tokenized, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = outputs.hidden_states[-1]
            mask = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            first = hidden[:, 0, :]
            last_idx = tokenized["attention_mask"].sum(1).clamp_min(1) - 1
            last = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx, :]
        arrays["mean"].append(torch.nn.functional.normalize(mean.float(), dim=1).cpu().numpy().astype(np.float32))
        arrays["first"].append(torch.nn.functional.normalize(first.float(), dim=1).cpu().numpy().astype(np.float32))
        arrays["last"].append(torch.nn.functional.normalize(last.float(), dim=1).cpu().numpy().astype(np.float32))
        if start % (batch_size * 30) == 0:
            print(f"  hidden {start + len(chunk)}/{len(texts)} {time.time() - t0:.0f}s", flush=True)
    result = {n: np.concatenate(arrays[n], axis=0) for n in arrays}
    np.savez(cache, **result)
    print("saved hidden", cache, flush=True)
    return result


# ---------------- B: windowed likelihood ----------------
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
    starts = {0, len(ids) - win}
    mid = (len(ids) - win) // 2
    if mid > 0:
        starts.add(mid)
    return [ids[s:s + win] for s in starts]


def extract_likelihood(texts, tokenizer, model, batch_size=4, win=256, cache: Path = None):
    if cache and cache.exists():
        z = np.load(cache)
        if z.shape[0] == len(texts):
            print("reuse lik", cache, flush=True)
            return z
    rows = []
    t0 = time.time()
    for i, s in enumerate(texts):
        ids = tokenizer.encode(str(s or ""), add_special_tokens=False)[:2048]
        for wd in _windows(ids, win):
            rows.append((i, wd))
    feats = []
    di = []
    for start in range(0, len(rows), batch_size):
        batch = rows[start:start + batch_size]
        doc = [r[0] for r in batch]
        width = max(len(r[1]) for r in batch)
        pad = tokenizer.pad_token_id
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
        di.append(np.asarray(doc))
        del logits
        if start % (batch_size * 40) == 0:
            print(f"  lik {start + len(batch)}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    all_feats = np.stack(feats)
    all_di = np.concatenate(di)
    agg = np.zeros((len(texts), all_feats.shape[1] * 3), dtype=np.float32)
    for doc_idx in range(len(texts)):
        v = all_feats[all_di == doc_idx]
        agg[doc_idx] = np.concatenate([v.mean(0), v.min(0), v.max(0)])
    if cache:
        np.save(cache, agg)
    print("saved lik", cache or "mem", flush=True)
    return agg


def fit_lgb(X, y, Xte, folds, seed=20260827, name=""):
    try:
        import lightgbm as lgb
        oof = np.zeros(len(y), dtype=np.float64)
        tests = []
        for f in range(5):
            tr = np.flatnonzero(folds != f)
            va = np.flatnonzero(folds == f)
            m = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.02,
                                   num_leaves=7, max_depth=3, min_child_samples=35,
                                   subsample=0.9, colsample_bytree=0.8, reg_alpha=0.8, reg_lambda=3.0,
                                   random_state=seed + f, n_jobs=8, verbosity=-1)
            m.fit(X[tr], y[tr])
            oof[va] = m.predict_proba(X[va])[:, 1]
            tests.append(m.predict_proba(Xte)[:, 1])
        return oof, np.mean(tests, axis=0)
    except Exception as exc:
        print("lgb fail", exc, flush=True)
        return None, None


def main():
    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    train_text = train.text.fillna("").astype(str).tolist()
    test_text = test.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    print("rows", len(train_text), len(test_text), flush=True)

    tokenizer, model = load_14b(keep_lm_head=False)
    try:
        backbone = model.model.model
    except AttributeError:
        backbone = model.model
    print("backbone class", type(backbone).__name__, flush=True)
    backbone.eval()
    dev = next(backbone.parameters()).device
    print("device", dev, flush=True)

    # A: hidden embeddings
    train_h = extract_hidden(train_text, tokenizer, backbone, 4, 256, OUT / "q14b_hidden_train.npz")
    test_h = extract_hidden(test_text, tokenizer, backbone, 4, 256, OUT / "q14b_hidden_test.npz")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260824)
    hidden_scores = {}
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
            print("head", name, "C", c, "auc", round(a, 6), flush=True)
            if a > best[1]:
                best = (c, a)
        c, a = best
        clf = make_pipeline(StandardScaler(with_mean=False),
                            LogisticRegression(C=c, max_iter=1600, class_weight="balanced", solver="liblinear"))
        clf.fit(train_h[name], y)
        hidden_scores[name] = {
            "oof": None, "replay": rank01(clf.decision_function(test_h[name])), "C": c, "auc": a}
        oof_vals = np.zeros(len(y), dtype=np.float64)
        for tr, va in cv.split(train_h[name], y):
            clf2 = make_pipeline(StandardScaler(with_mean=False),
                                 LogisticRegression(C=c, max_iter=1600, class_weight="balanced", solver="liblinear"))
            clf2.fit(train_h[name][tr], y[tr])
            oof_vals[va] = clf2.decision_function(train_h[name][va])
        hidden_scores[name]["oof"] = oof_vals
        np.save(OUT / f"q14b_hidden_{name}_oof.npy", oof_vals)
        np.save(OUT / f"q14b_hidden_{name}_replay.npy", hidden_scores[name]["replay"])
    hidden_equ = np.mean([hidden_scores[n]["replay"] for n in hidden_scores], axis=0)
    np.save(OUT / "q14b_hidden_equal_replay.npy", hidden_equ)
    print("hidden done", flush=True)

    # B: likelihood (reload with LM head)
    del model, backbone
    torch.cuda.empty_cache()
    model2 = None
    tokenizer2, model2 = load_14b(keep_lm_head=True)
    model2.eval()
    tr_lik = extract_likelihood(train_text, tokenizer2, model2, 4, 256, OUT / "q14b_lik_train.npz")
    te_lik = extract_likelihood(test_text, tokenizer2, model2, 4, 256, OUT / "q14b_lik_test.npz")
    del model2
    torch.cuda.empty_cache()

    # folds (repro-compatible group folds)
    try:
        sys.path.insert(0, "/data/湾区杯/文本赛道/repro")
        from wanqu_first_submission.folds import make_folds, text_near_duplicate_groups

        folds = make_folds(y, text_near_duplicate_groups(train_text), 5, 20260823)
    except Exception as exc:
        print("folds fallback", repr(exc), flush=True)
        folds = cv.fit_transform(np.zeros(len(y)), y)
    np.save(OUT / "folds.npy", folds)
    oof_lik, te_lik_scr = fit_lgb(tr_lik, y, te_lik, folds, 20260827, "q14b_lik")
    if oof_lik is not None:
        np.save(OUT / "q14b_lik_oof.npy", oof_lik)
        np.save(OUT / "q14b_lik_replay.npy", rank01(te_lik_scr))
        print("q14b lik sample AUC", round(float(roc_auc_score(y, oof_lik)), 6), flush=True)

    # combined hidden+lik features for a stronger single-branch head
    hidden_train = np.concatenate([train_h[n] for n in ("mean", "first", "last")], axis=1)
    hidden_test = np.concatenate([test_h[n] for n in ("mean", "first", "last")], axis=1)
    Xtr = np.concatenate([hidden_train, tr_lik], axis=1).astype(np.float32)
    Xte = np.concatenate([hidden_test, te_lik], axis=1).astype(np.float32)
    oof_all, te_all = fit_lgb(Xtr, y, Xte, folds, 20260828, "q14b_all")
    if oof_all is not None:
        np.save(OUT / "q14b_all_oof.npy", oof_all)
        np.save(OUT / "q14b_all_replay.npy", rank01(te_all))
        print("q14b all sample AUC", round(float(roc_auc_score(y, oof_all)), 6), flush=True)

    # write candidate CSV
    te_id = test.id.astype(str).tolist()
    final = rank01(te_all) if te_all is not None else hidden_equ
    pd.DataFrame({"id": te_id, "score": final}).to_csv(OUT / "q14b_candidate.csv", index=False)
    with open(OUT / "p2_report.json", "w", encoding="utf-8") as f:
        json.dump({"hidden_heads": {k: v["auc"] for k, v in hidden_scores.items()},
                   "lik_oof_auc": float(roc_auc_score(y, oof_lik)) if oof_lik is not None else None,
                   "all_oof_auc": float(roc_auc_score(y, oof_all)) if oof_all is not None else None},
                  f, ensure_ascii=False, indent=2)
    print("P2 DONE", flush=True)


if __name__ == "__main__":
    main()
