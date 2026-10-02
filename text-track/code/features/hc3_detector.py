#!/usr/bin/env python3
import glob, json, os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
from sklearn.metrics import roc_auc_score

BASE = '/data/湾区杯'
OUT = os.path.join(BASE, 'text_hc3')
os.makedirs(OUT, exist_ok=True)
paths = glob.glob(BASE + '/external_data/HC3-Chinese/**/all.jsonl', recursive=True)
paths += glob.glob(BASE + '/external_data/HC3-Chinese/._____temp/all.jsonl')
if not paths:
    paths = glob.glob('/data/*/external_data/HC3-Chinese/**/all.jsonl', recursive=True)
if not paths:
    raise FileNotFoundError('HC3-Chinese all.jsonl not found')
rows = []
with open(paths[0], encoding='utf-8') as f:
    for line in f:
        try:
            obj = json.loads(line)
        except Exception:
            continue
        humans = obj.get('human_answers') or []
        bots = obj.get('chatgpt_answers') or []
        for s in humans:
            if str(s).strip(): rows.append((str(s), 0))
        for s in bots:
            if str(s).strip(): rows.append((str(s), 1))
ext = pd.DataFrame(rows, columns=['text', 'label']).drop_duplicates('text')
print('external rows', len(ext), ext['label'].value_counts().to_dict(), flush=True)

sample = glob.glob(BASE + '/initial_text/**/text_sample_data.csv', recursive=True)[0]
replay = BASE + '/replay_data/文本赛道/text_test.csv'
tr = pd.read_csv(sample, encoding='utf-8-sig')
te = pd.read_csv(replay, encoding='utf-8-sig')

def rn(x):
    x = np.asarray(x, dtype=np.float64)
    return rankdata(x, method='average') / (len(x) + 1.)

vec = FeatureUnion([
    ('char', TfidfVectorizer(analyzer='char', ngram_range=(2, 5), min_df=2,
                             max_features=280000, sublinear_tf=True, norm='l2', dtype=np.float32)),
    ('word', TfidfVectorizer(analyzer='word', token_pattern=r'(?u)\b\w+\b', ngram_range=(1, 2),
                             min_df=2, max_features=120000, sublinear_tf=True, norm='l2', dtype=np.float32)),
])
X = vec.fit_transform(ext['text'].tolist())
clf = LogisticRegression(C=2.0, max_iter=700, class_weight='balanced', solver='liblinear')
clf.fit(X, ext['label'].to_numpy())
print('external train auc', roc_auc_score(ext['label'], clf.decision_function(X)), X.shape, flush=True)

def score(items):
    return rn(clf.decision_function(vec.transform(items)))

sample_s = score(tr['text'].fillna('').astype(str).tolist())
replay_s = score(te['text'].fillna('').astype(str).tolist())
pd.DataFrame({'id': te['id'], 'score': replay_s}).to_csv(os.path.join(BASE, 'text_hc3.csv'), index=False)
np.save(os.path.join(OUT, 'initial_scores.npy'), sample_s)
np.save(os.path.join(OUT, 'replay_scores.npy'), replay_s)
anx_path = os.path.join(BASE, 'text_anx.csv')
old_path = os.path.join(BASE, 'text_replay_submit.csv')
anx = pd.read_csv(anx_path)['score'].to_numpy() if os.path.exists(anx_path) else None
old = pd.read_csv(old_path)['score'].to_numpy() if os.path.exists(old_path) else None
for w in (0.05, 0.10, 0.15, 0.20, 0.30):
    s = w * replay_s
    if anx is not None: s = s + (1-w)*rn(anx)
    pd.DataFrame({'id': te['id'], 'score': s}).to_csv(os.path.join(BASE, f'text_hc3_anx_{int(w*100)}.csv'), index=False)
for w in (0.05, 0.10, 0.15):
    if anx is not None and old is not None:
        s = (1-w-0.05)*rn(anx) + w*replay_s + 0.05*rn(old)
        pd.DataFrame({'id': te['id'], 'score': s}).to_csv(os.path.join(BASE, f'text_hc3_anx_old_{int(w*100)}.csv'), index=False)
print('DONE', flush=True)
