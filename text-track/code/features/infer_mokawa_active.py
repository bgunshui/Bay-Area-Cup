#!/usr/bin/env python3
import glob
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from scipy.special import softmax
from scipy.stats import rankdata
from transformers import AutoTokenizer


BASE = Path('/data/湾区杯')
MODEL = BASE / 'active_models' / 'mokawa'
OUT = BASE / 'active_text'
TOKENIZER = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
SESSION = ort.InferenceSession(str(MODEL / 'onnx' / 'model.onnx'), providers=['CPUExecutionProvider'])


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method='average') / (len(x) + 1.0)


def score(texts, batch=32):
    values = []
    for start in range(0, len(texts), batch):
        z = TOKENIZER(texts[start:start + batch], padding=True, truncation=True, max_length=512, return_tensors='np')
        inputs = {key: value.astype(np.int64) for key, value in z.items()}
        logits = SESSION.run(None, inputs)[0]
        values.extend(softmax(logits, axis=-1)[:, 1].tolist())
        if start == 0 or start + len(z['input_ids']) == len(texts) or start % (batch * 100) == 0:
            print('mokawa', start + len(z['input_ids']), '/', len(texts), flush=True)
    return np.asarray(values, dtype=np.float64)


def main():
    sample = Path(glob.glob(str(BASE / 'initial_text' / '**' / 'text_sample_data.csv'), recursive=True)[0])
    test = BASE / 'replay_data' / '文本赛道' / 'text_test.csv'
    train = pd.read_csv(sample, encoding='utf-8-sig')
    replay = pd.read_csv(test, encoding='utf-8-sig')
    train_score = score(train.text.fillna('').astype(str).tolist())
    replay_score = score(replay.text.fillna('').astype(str).tolist())
    np.save(OUT / 'mokawa_sample.npy', train_score)
    np.save(OUT / 'mokawa_replay.npy', replay_score)
    pd.DataFrame({'id': replay.id.astype(str), 'score': rank01(replay_score)}).to_csv(OUT / 'text_mokawa.csv', index=False)
    print('sample_auc', float(pd.Series(train_score).corr(train.label.astype(float), method='spearman')), flush=True)
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
