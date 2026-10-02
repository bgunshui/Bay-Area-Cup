#!/usr/bin/env python3
from __future__ import annotations

import glob
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from scipy.special import softmax
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from tokenizers import Tokenizer
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE = Path('/data/湾区杯')
OUT = BASE / 'new_text'
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method='average') / (len(x) + 1.0)


def torch_score(model_dir, texts, batch=64):
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, local_files_only=True, torch_dtype=torch.float16)
    model.to(DEVICE).eval()
    values = []
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        z = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors='pt')
        z = {k: v.to(DEVICE) for k, v in z.items()}
        with torch.inference_mode():
            p = torch.softmax(model(**z).logits.float(), dim=1).cpu().numpy()
        values.append(p)
        if start == 0 or start + len(chunk) == len(texts) or start % (batch * 50) == 0:
            print(model_dir.name, start + len(chunk), '/', len(texts), flush=True)
    del model, tok
    torch.cuda.empty_cache() if DEVICE.type == 'cuda' else None
    return np.concatenate(values, axis=0)


def onnx_score(model_dir, texts, batch=64):
    tok = Tokenizer.from_file(str(model_dir / 'tokenizer.json'))
    tok.enable_truncation(max_length=512)
    tok.enable_padding(length=None, pad_id=0, pad_token='[PAD]')
    sess = ort.InferenceSession(str(model_dir / 'onnx' / 'model_quantized.onnx'), providers=['CPUExecutionProvider'])
    values = []
    for start in range(0, len(texts), batch):
        enc = tok.encode_batch(texts[start:start + batch])
        max_len = max(len(e.ids) for e in enc)
        ids = np.asarray([e.ids + [0] * (max_len - len(e.ids)) for e in enc], dtype=np.int64)
        mask = np.asarray([e.attention_mask + [0] * (max_len - len(e.attention_mask)) for e in enc], dtype=np.int64)
        types = np.asarray([e.type_ids + [0] * (max_len - len(e.type_ids)) for e in enc], dtype=np.int64)
        logits = sess.run(None, {'input_ids': ids, 'attention_mask': mask, 'token_type_ids': types})[0]
        values.append(softmax(logits, axis=-1))
        if start == 0 or start + len(enc) == len(texts) or start % (batch * 50) == 0:
            print('aigc_zh_onnx', start + len(enc), '/', len(texts), flush=True)
    return np.concatenate(values, axis=0)


def save_pair(name, sample_probs, replay_probs, train, test, ai_col):
    sample_score = sample_probs[:, ai_col]
    replay_score = replay_probs[:, ai_col]
    print(name, 'sample_auc', f'{roc_auc_score(train.label.astype(int), sample_score):.6f}', flush=True)
    np.save(OUT / f'{name}_sample.npy', sample_score)
    np.save(OUT / f'{name}_replay.npy', replay_score)
    pd.DataFrame({'id': test.id.astype(str), 'score': rank01(replay_score)}).to_csv(OUT / f'{name}.csv', index=False)


def main():
    sample_path = Path(glob.glob(str(BASE / 'initial_text' / '**' / 'text_sample_data.csv'), recursive=True)[0])
    test_path = BASE / 'replay_data' / '文本赛道' / 'text_test.csv'
    train = pd.read_csv(sample_path, encoding='utf-8-sig')
    test = pd.read_csv(test_path, encoding='utf-8-sig')
    train_text = train.text.fillna('').astype(str).tolist()
    test_text = test.text.fillna('').astype(str).tolist()
    print('rows', len(train), len(test), 'device', DEVICE, flush=True)
    yong = torch_score(BASE / 'new_models' / 'yongchao_text', train_text + test_text)
    save_pair('yongchao_ai_class0', yong[:len(train)], yong[len(train):], train, test, 0)
    save_pair('yongchao_class1_probe', yong[:len(train)], yong[len(train):], train, test, 1)
    followsci = torch_score(BASE / 'new_models' / 'followsci_text', train_text + test_text)
    save_pair('followsci_class1', followsci[:len(train)], followsci[len(train):], train, test, 1)
    save_pair('followsci_class0_probe', followsci[:len(train)], followsci[len(train):], train, test, 0)
    zh = onnx_score(BASE / 'new_models' / 'aigc_zh_onnx', train_text + test_text)
    save_pair('aigc_zh_class1', zh[:len(train)], zh[len(train):], train, test, 1)
    save_pair('aigc_zh_class0_probe', zh[:len(train)], zh[len(train):], train, test, 0)
    print('DONE', OUT, flush=True)


if __name__ == '__main__':
    main()
