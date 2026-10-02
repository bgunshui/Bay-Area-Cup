#!/usr/bin/env python3
"""Phase 40: structural + public-window LGB fingerprint (new information source).

Features per text (on replay 20000 + official 1000):
 - char/word length, punct ratio, digit ratio, unique char ratio
 - sentence count, mean/std/median sentence length
 - repeated 4-gram ratio, repeated 10-gram ratio (AI repetitiveness)
 - char entropy, word-ish entropy (via jieba-lite tokenize on chars)
 - public detector window features (mean/max/min/q75/std of each public branch window scores)
 - KNN retrieval score (done separately if available; skip if not)
Train 5-fold LGB on OFFICIAL y (as p4 score_public does), predict replay.
Output: runs/fus_lgb_oof.npy + fus_lgb_replay.npy (logit-diff score = AI-high).
"""
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import glob
import numpy as np
import pandas as pd
from collections import Counter

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
INIT = BASE / "initial_text"

try:
    import lightgbm as lgb
except Exception:
    lgb = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import NearestNeighbors

def window_feats(mat):
    f = np.asarray(mat, dtype=np.float64)
    out = []
    for i in range(f.shape[0]):
        v = f[i][~np.isnan(f[i])]
        if v.size == 0:
            v = np.asarray([0.5])
        out.append([v.mean(), v.max(), v.min(), np.quantile(v, 0.75), v.size,
                    v[0], v[-1], v.std() if v.size > 1 else 0.0])
    return np.asarray(out, dtype=np.float64)

def struct_feats(texts):
    """statistical structure features (text-only, no model needed)."""
    rows = []
    for t in texts:
        s = t
        n = max(len(s), 1)
        cnt_sent = s.count("。") + s.count(".") 
        sents = [x for x in s.replace(".", "。").split("。") if x]
        sl = [len(x) for x in sents] if sents else [n]
        # repeated n-gram ratio
        def rep_ratio(k):
            cs = Counter(s[i:i+k] for i in range(max(len(s)-k+1, 0)))
            rep = sum(v for v in cs.values() if v > 1)
            tot = sum(cs.values())
            return rep / tot if tot else 0.0
        # char entropy
        chars = Counter(s)
        ent = -sum((c/n) * np.log2(c/n) for c in chars.values())
        rows.append([
            n, len(set(s)) / n, n / 1000.0,
            sum(c in "。，、；：？！,.?!;:" for c in s) / n,
            sum(c.isdigit() for c in s) / n,
            sum(c.isalpha() and '\u4e00' <= c <= '\u9fff' for c in s) / n,
            cnt_sent, len(sents),
            float(np.mean(sl)) / 100.0, float(np.std(sl)) / 100.0, float(np.median(sl)) / 100.0,
            rep_ratio(4), rep_ratio(8), rep_ratio(16),
            ent,
            sum(1 for c in s if c in "。！？") / n,
        ])
    return np.asarray(rows, dtype=np.float64)

def main():
    # texts
    sample_csv = glob.glob(str(INIT / "**" / "text_sample_data.csv"), recursive=True)
    train = pd.read_csv(sample_csv[0], encoding="utf-8-sig")
    y = train.label.astype(int).to_numpy()
    official_texts = train.text.fillna("").astype(str).tolist()
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(RUNS / "folds.npy")
    n_o, n_r = len(official_texts), len(test_texts)
    print("official", n_o, "replay", n_r, flush=True)

    # (a) structure features
    Fs = struct_feats(official_texts + test_texts)
    print("struct feats", Fs.shape, flush=True)

    # (b) public detector window features (from *_replay.npy / *_sample.npy style files)
    pub_tags = []
    for p in sorted(RUNS.glob("*_sample.npy")):
        tag = p.name[:-len("_sample.npy")]
        if "fold" in tag:
            continue
        rp = RUNS / f"{tag}_replay.npy"
        if rp.exists():
            pub_tags.append(tag)
    print("public tags:", pub_tags, flush=True)
    pub_off = []
    pub_rep = []
    for tag in pub_tags:
        sm = np.load(RUNS / f"{tag}_sample.npy")
        rm = np.load(RUNS / f"{tag}_replay.npy")
        if sm.ndim == 2 and sm.shape[1] == 2:
            sm = sm[:, 1:2]; rm = rm[:, 1:2]
        if sm.ndim == 1:
            sm = sm[:, None]; rm = rm[:, None]
        if sm.shape[0] == n_o and rm.shape[0] == n_r:
            pub_off.append(window_feats(sm))
            pub_rep.append(window_feats(rm))
    if pub_off:
        Foff = np.hstack(pub_off)
        Frep = np.hstack(pub_rep)
    else:
        Foff = np.zeros((n_o, 1)); Frep = np.zeros((n_r, 1))
    print("pub feats", Foff.shape, flush=True)

    # (c) TF-IDF -> SVD-ish reduced semantic distance (train on both)
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), min_df=3, max_features=100000,
                          sublinear_tf=True, dtype=np.float32)
    M = vec.fit_transform(official_texts + test_texts)
    print("tfidf", M.shape, flush=True)

    Xo = np.hstack([Fs[:n_o], Foff])
    Xr = np.hstack([Fs[n_o:], Frep])
    print("Xo", Xo.shape, flush=True)

    # LGB 5-fold on official
    oof = np.zeros(n_o, dtype=np.float64)
    tests = []
    for f in range(5):
        tr = np.flatnonzero(folds != f)
        va = np.flatnonzero(folds == f)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=400, learning_rate=0.03,
                               num_leaves=9, max_depth=4, min_child_samples=30,
                               reg_alpha=0.5, reg_lambda=2.0, random_state=20260827 + f,
                               n_jobs=8, verbosity=-1)
        m.fit(Xo[tr], y[tr])
        oof[va] = m.predict_proba(Xo[va])[:, 1]
        tests.append(m.predict_proba(Xr)[:, 1])
    rep = np.mean(tests, axis=0)
    auc = roc_auc_score(y, oof)
    print("fus_lgb OOF AUC", round(auc, 6), flush=True)
    np.save(RUNS / "fus_lgb_oof.npy", oof)
    np.save(RUNS / "fus_lgb_replay.npy", rep)
    print("FUS_LGB DONE", flush=True)

if __name__ == "__main__":
    main()
