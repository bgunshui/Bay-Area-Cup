#!/usr/bin/env python3
import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE = Path('/data/湾区杯')
MODEL = Path('/data/tjx_exp/models/Qwen/Qwen2.5-1.5B-Instruct')
OUT = BASE / 'active_text'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def score(model, tokenizer, texts, batch=8):
    values = []
    start_time = time.time()
    for start in range(0, len(texts), batch):
        chunk = texts[start:start + batch]
        encoded = tokenizer(chunk, return_tensors='pt', padding=True, truncation=True, max_length=512)
        ids = encoded.input_ids.to(DEVICE)
        mask = encoded.attention_mask.to(DEVICE)
        with torch.inference_mode():
            logits = model(input_ids=ids, attention_mask=mask).logits
            log_probs = torch.log_softmax(logits[:, :-1], dim=-1)
            targets = ids[:, 1:]
            valid = mask[:, 1:].bool()
            nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            values.extend((nll * valid).sum(1).div(valid.sum(1).clamp_min(1)).float().cpu().tolist())
        if start == 0 or start + len(chunk) == len(texts) or start % (batch * 100) == 0:
            print('ppl', start + len(chunk), '/', len(texts), 'sec', round(time.time() - start_time, 1), flush=True)
    return np.asarray(values, dtype=np.float64)


def main():
    sample = Path(glob.glob(str(BASE / 'initial_text' / '**' / 'text_sample_data.csv'), recursive=True)[0])
    test = BASE / 'replay_data' / '文本赛道' / 'text_test.csv'
    train = pd.read_csv(sample, encoding='utf-8-sig')
    replay = pd.read_csv(test, encoding='utf-8-sig')
    tok = AutoTokenizer.from_pretrained(MODEL, local_files_only=True, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    model.to(DEVICE).eval()
    train_score = score(model, tok, train.text.fillna('').astype(str).tolist())
    test_score = score(model, tok, replay.text.fillna('').astype(str).tolist())
    np.save(OUT / 'ppl_sample.npy', train_score)
    np.save(OUT / 'ppl_replay.npy', test_score)
    print('sample range', float(train_score.min()), float(train_score.max()), 'test range', float(test_score.min()), float(test_score.max()), flush=True)
    del model
    if DEVICE.type == 'cuda':
        torch.cuda.empty_cache()
    print('DONE', flush=True)


if __name__ == '__main__':
    main()
