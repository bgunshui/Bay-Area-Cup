#!/usr/bin/env python3
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path('/data/湾区杯')
OUT = BASE / 'active_text'


def rank01(values):
    values = np.asarray(values, dtype=np.float64)
    return rankdata(values, method='average') / (len(values) + 1.0)


def load_rank(path: Path, n: int):
    if not path.exists():
        return None
    values = np.load(path) if path.suffix == '.npy' else pd.read_csv(path).score.to_numpy(float)
    if len(values) != n or not np.isfinite(values).all():
        return None
    return rank01(values)


def main():
    sample = pd.read_csv(glob.glob(str(BASE / 'initial_text' / '**' / 'text_sample_data.csv'), recursive=True)[0], encoding='utf-8-sig')
    test = pd.read_csv(BASE / 'replay_data' / '文本赛道' / 'text_test.csv', encoding='utf-8-sig')
    y = sample.label.astype(int).to_numpy()
    X = np.load(BASE / 'text_qwen7b' / 'train_emb_256.npy').astype('float32')
    Xt = np.load(BASE / 'text_qwen7b' / 'replay_emb_256.npy').astype('float32')
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260825)
    q_oof = {}
    q_test = {}
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        oof = np.zeros(len(y), dtype=np.float64)
        for train_idx, valid_idx in cv.split(X, y):
            clf = make_pipeline(
                StandardScaler(with_mean=False),
                LogisticRegression(C=c, max_iter=1200, class_weight='balanced', solver='liblinear'),
            )
            clf.fit(X[train_idx], y[train_idx])
            oof[valid_idx] = clf.decision_function(X[valid_idx])
        final = make_pipeline(
            StandardScaler(with_mean=False),
            LogisticRegression(C=c, max_iter=1600, class_weight='balanced', solver='liblinear'),
        )
        final.fit(X, y)
        q_oof[c] = rank01(oof)
        q_test[c] = rank01(final.decision_function(Xt))
        print('qwen_oof', c, f'{roc_auc_score(y, oof):.6f}', flush=True)

    bases = {
        'anx': load_rank(OUT / 'anx_sample.npy', len(y)),
        'paper': load_rank(OUT / 'paperpass_sample.npy', len(y)),
        'mokawa': load_rank(OUT / 'mokawa_sample.npy', len(y)),
        'ppl': load_rank(OUT / 'ppl_sample.npy', len(y)),
    }
    tests = {
        'anx': load_rank(OUT / 'anx_replay.npy', len(test)),
        'paper': load_rank(OUT / 'paperpass_replay.npy', len(test)),
        'mokawa': load_rank(OUT / 'mokawa_replay.npy', len(test)),
        'ppl': load_rank(OUT / 'ppl_replay.npy', len(test)),
    }
    print('available', [k for k, v in bases.items() if v is not None], flush=True)
    records = []
    # Keep the grid small enough to avoid selecting a brittle 1,000-row fit.
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0):
        q = q_oof[c]
        qt = q_test[c]
        names = ['anx', 'paper', 'mokawa', 'ppl']
        active = [n for n in names if bases[n] is not None and tests[n] is not None]
        if 'anx' not in active:
            raise RuntimeError('Anx score is required as the stable anchor')
        # Search qwen weight and distribute the remainder among detector anchors.
        for wq in np.arange(0.0, 0.56, 0.05):
            for wp in np.arange(0.0, 0.31, 0.05):
                for wm in np.arange(0.0, 0.31, 0.05):
                    for wl in np.arange(0.0, 0.31, 0.05):
                        if 'paper' not in active and wp > 0: continue
                        if 'mokawa' not in active and wm > 0: continue
                        if 'ppl' not in active and wl > 0: continue
                        wa = 1.0 - wq - wp - wm - wl
                        if wa < 0.44: continue
                        score = wq * q + wa * bases['anx']
                        test_score = wq * qt + wa * tests['anx']
                        if 'paper' in active: score += wp * bases['paper']; test_score += wp * tests['paper']
                        if 'mokawa' in active: score += wm * bases['mokawa']; test_score += wm * tests['mokawa']
                        if 'ppl' in active: score += wl * bases['ppl']; test_score += wl * tests['ppl']
                        records.append((roc_auc_score(y, score), c, wq, wa, wp, wm, wl, test_score))
    records.sort(key=lambda x: x[0], reverse=True)
    for row in records[:20]:
        print('BEST', *(f'{x:.6f}' if isinstance(x, float) else x for x in row[:7]), flush=True)
    ids = test.id.astype(str).tolist()
    for idx, row in enumerate(records[:12]):
        auc, c, wq, wa, wp, wm, wl, test_score = row
        name = f'text_active_{idx:02d}_auc{auc:.6f}_q{wq:.2f}_a{wa:.2f}_p{wp:.2f}_m{wm:.2f}_l{wl:.2f}.csv'
        pd.DataFrame({'id': ids, 'score': test_score}).to_csv(OUT / name, index=False)
    print('DONE', len(records), flush=True)


if __name__ == '__main__':
    main()
