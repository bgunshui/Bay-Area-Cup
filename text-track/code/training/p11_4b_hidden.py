#!/usr/bin/env python3
"""Phase 11: Qwen3-4B-Base hidden-state heads (mean/first/last) - llm family upgrade."""
import glob
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
from transformers import AutoModel, AutoTokenizer

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
MODEL_DIR = BASE / "文本赛道" / "models" / "Qwen3-4B-Base"
DEV = torch.device("cuda:0")


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def main():
    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    train_text = train.text.fillna("").astype(str).tolist()
    test_text = test.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    print("rows", len(train_text), len(test_text), flush=True)

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR), local_files_only=True, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    model = AutoModel.from_pretrained(str(MODEL_DIR), local_files_only=True,
                                      trust_remote_code=True, torch_dtype=torch.bfloat16,
                                      attn_implementation="sdpa").to(DEV).eval()
    print("loaded 4B", flush=True)

    def extract(texts, cache):
        if cache.exists():
            z = np.load(cache)
            if all(z[n].shape[0] == len(texts) for n in ("mean", "first", "last")):
                print("reuse", cache, flush=True)
                return {n: np.asarray(z[n], dtype=np.float32) for n in ("mean", "first", "last")}
        arrays = {"mean": [], "first": [], "last": []}
        t0 = time.time()
        for start in range(0, len(texts), 8):
            chunk = texts[start:start + 8]
            tokenized = tok(chunk, padding=True, truncation=True, max_length=384, return_tensors="pt")
            tokenized = {k: v.to(DEV) for k, v in tokenized.items()}
            with torch.inference_mode():
                last_hidden = model(**tokenized).last_hidden_state
            mask = tokenized["attention_mask"].unsqueeze(-1).to(last_hidden.dtype)
            mean = (last_hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            first = last_hidden[:, 0, :]
            last_idx = tokenized["attention_mask"].sum(1).clamp_min(1) - 1
            last = last_hidden[torch.arange(last_hidden.shape[0], device=DEV), last_idx, :]
            arrays["mean"].append(torch.nn.functional.normalize(mean.float(), dim=1).cpu().numpy().astype(np.float32))
            arrays["first"].append(torch.nn.functional.normalize(first.float(), dim=1).cpu().numpy().astype(np.float32))
            arrays["last"].append(torch.nn.functional.normalize(last.float(), dim=1).cpu().numpy().astype(np.float32))
            if start % 160 == 0:
                print(f"  {start + len(chunk)}/{len(texts)} {time.time()-t0:.0f}s", flush=True)
        res = {n: np.concatenate(arrays[n], axis=0) for n in arrays}
        np.savez(cache, **res)
        return res

    train_h = extract(train_text, OUT / "q4b_hidden_train.npz")
    test_h = extract(test_text, OUT / "q4b_hidden_test.npz")
    del model
    torch.cuda.empty_cache()
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
        np.save(OUT / f"q4b_hidden_{name}_oof.npy", oof)
        np.save(OUT / f"q4b_hidden_{name}_replay.npy", rank01(clf.decision_function(test_h[name])))
        print("head", name, "C", c, "auc", round(a, 6), flush=True)
    print("P11 DONE", flush=True)


if __name__ == "__main__":
    main()
