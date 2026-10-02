#!/usr/bin/env python3
import os, glob, json, time, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

BASE = '/data/湾区杯'
OUT = os.path.join(BASE, 'text_qwen7b')
os.makedirs(OUT, exist_ok=True)
sample_path = glob.glob(BASE + '/initial_text/**/text_sample_data.csv', recursive=True)[0]
test_path = BASE + '/replay_data/文本赛道/text_test.csv'
train = pd.read_csv(sample_path, encoding='utf-8-sig')
test = pd.read_csv(test_path, encoding='utf-8-sig')
train_text = train['text'].fillna('').astype(str).tolist()
test_text = test['text'].fillna('').astype(str).tolist()
y = train['label'].astype(int).to_numpy()
print('train_n', len(train), 'test_n', len(test), 'label_mean', float(y.mean()), flush=True)

model_path = '/data/tjx_exp/models/Qwen/Qwen2___5-7B-Instruct'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu', flush=True)
tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = 'right'
model = AutoModel.from_pretrained(model_path, trust_remote_code=True,
                                  torch_dtype=torch.float16, low_cpu_mem_usage=True)
model.to(device)
model.eval()

def encode(items, out_path, batch_size=8, max_length=256):
    if os.path.exists(out_path):
        arr = np.load(out_path, mmap_mode='r')
        if arr.shape[0] == len(items):
            print('reuse', out_path, arr.shape, flush=True)
            return np.asarray(arr, dtype=np.float32)
    part_dir = out_path + '.parts'
    os.makedirs(part_dir, exist_ok=True)
    parts = []
    t0 = time.time()
    part_size = 1000
    for pstart in range(0, len(items), part_size):
        pend = min(pstart + part_size, len(items))
        part_path = os.path.join(part_dir, f'part_{pstart:07d}_{pend:07d}.npy')
        if os.path.exists(part_path):
            part = np.load(part_path)
            if part.shape[0] == pend - pstart:
                print('reuse part', pstart, pend, flush=True)
                parts.append(part.astype(np.float32))
                continue
        part_chunks = []
        for start in range(pstart, pend, batch_size):
            chunk = items[start:start + batch_size]
            z = tok(chunk, padding=True, truncation=True, max_length=max_length, return_tensors='pt')
            z = {k: v.to(device) for k, v in z.items()}
            with torch.inference_mode():
                with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=(device.type == 'cuda')):
                    h = model(**z).last_hidden_state
                mask = z['attention_mask'].unsqueeze(-1).to(h.dtype)
                mean = (h * mask).sum(1) / mask.sum(1).clamp_min(1)
                first = h[:, 0, :]
                last_idx = z['attention_mask'].sum(1).clamp_min(1) - 1
                last = h[torch.arange(h.shape[0], device=h.device), last_idx, :]
                emb = torch.cat([mean, first, last], dim=1)
                emb = torch.nn.functional.normalize(emb.float(), dim=1).cpu().numpy()
            part_chunks.append(emb.astype(np.float32))
        part = np.concatenate(part_chunks, axis=0)
        np.save(part_path, part)
        print('saved part', pstart, pend, 'sec', round(time.time() - t0, 1), flush=True)
        parts.append(part)
    arr = np.concatenate(parts, axis=0)
    np.save(out_path, arr)
    print('saved', out_path, arr.shape, flush=True)
    return arr

Etr = encode(train_text, os.path.join(OUT, 'train_emb_256.npy'))
Ete = encode(test_text, os.path.join(OUT, 'replay_emb_256.npy'))
del model
if device.type == 'cuda':
    torch.cuda.empty_cache()

def rank_norm(x):
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, method='average') / (len(x) + 1.0)

# Fit several regularized linear heads. The OOF scores are diagnostic only;
# final heads are fitted on all 1,000 official labeled samples.
heads = []
for C in (0.03, 0.1, 0.3, 1.0, 3.0):
    clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=C, max_iter=1000,
                         class_weight='balanced', solver='liblinear'))
    clf.fit(Etr, y)
    raw = clf.decision_function(Ete)
    score = rank_norm(raw)
    heads.append((C, score))
    print('head', C, 'train_auc', roc_auc_score(y, clf.decision_function(Etr)), flush=True)

# Existing public detector candidates are already aligned to replay IDs.
def read_score(path):
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    if len(d) != len(test) or d['id'].astype(str).tolist() != test['id'].astype(str).tolist():
        print('skip misaligned', path, len(d), flush=True)
        return None
    col = 'score' if 'score' in d.columns else d.columns[-1]
    return rank_norm(d[col].to_numpy())

anx = read_score(os.path.join(BASE, 'text_anx.csv'))
paper = read_score(os.path.join(BASE, 'text_paperpass.csv'))
old = read_score(os.path.join(BASE, 'text_replay_submit.csv'))

for C, q in heads:
    stem = 'text_qwen7b_c' + str(C).replace('.', '')
    pd.DataFrame({'id': test['id'], 'score': q}).to_csv(os.path.join(OUT, stem + '.csv'), index=False)
    # Keep a small grid of conservative blends. They are candidates, not claims
    # of official performance; all retain the exact replay test ID order.
    blends = [('qwen_only', q)]
    if anx is not None:
        blends += [('qwen_anx70', 0.30*q + 0.70*anx),
                   ('qwen_anx50', 0.50*q + 0.50*anx),
                   ('qwen_anx30', 0.70*q + 0.30*anx)]
    if anx is not None and paper is not None:
        blends += [('qwen_anx_paper', 0.45*q + 0.40*anx + 0.15*paper),
                   ('qwen_anx_paper2', 0.35*q + 0.50*anx + 0.15*paper)]
    if anx is not None and old is not None:
        blends += [('qwen_anx_old', 0.35*q + 0.55*anx + 0.10*old)]
    for tag, s in blends:
        pd.DataFrame({'id': test['id'], 'score': np.asarray(s, dtype=np.float64)}).to_csv(
            os.path.join(OUT, stem + '_' + tag + '.csv'), index=False)

meta = {'model': model_path, 'max_length': 384, 'train_n': len(train), 'test_n': len(test),
        'device': str(device), 'id_source': test_path,
        'heads': [float(x[0]) for x in heads]}
with open(os.path.join(OUT, 'run_meta.json'), 'w', encoding='utf-8') as f:
    json.dump(meta, f, ensure_ascii=True, indent=2)
print('DONE', OUT, flush=True)
