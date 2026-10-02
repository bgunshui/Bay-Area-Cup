import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = '/data/湾区杯'
test = pd.read_csv(BASE + '/replay_data/文本赛道/text_test.csv')
names = ['text_new.csv','text_next.csv','text_next_alt.csv','text_anx.csv',
         'text_replay_submit.csv','text_hc3.csv','text_hc3_anx_5.csv',
         'text_hc3_anx_10.csv','text_hc3_anx_15.csv','text_hc3_anx_old_5.csv']
names += sorted([os.path.relpath(p, BASE) for p in glob.glob(BASE + '/text_qwen7b/*_qwen_only.csv')])[:5]
data = {}
for n in names:
    p = os.path.join(BASE, n)
    if os.path.exists(p):
        d = pd.read_csv(p)
        if len(d) == len(test) and d['id'].astype(str).tolist() == test['id'].astype(str).tolist():
            data[n] = d['score'].to_numpy(float)
print('valid', list(data))
for n, a in data.items():
    print(n, 'q', np.quantile(a, [.01,.1,.5,.9,.99]).round(4).tolist(), 'mean', round(float(a.mean()),4))
base = data.get('text_new.csv')
if base is not None:
    for n, a in data.items():
        if n != 'text_new.csv':
            print('corr new', n, round(float(spearmanr(base, a).statistic), 5))
if base is not None:
    for n, a in data.items():
        if n in ('text_new.csv', 'text_next.csv', 'text_next_alt.csv'):
            continue
        for w in (.05, .10, .15, .20):
            s = (1-w) * base + w * a
            out = f'text_blend_{n[:-4]}_{int(w*100)}.csv'.replace('/', '_')
            pd.DataFrame({'id': test['id'], 'score': s}).to_csv(os.path.join(BASE, out), index=False)
            print('wrote', out)
