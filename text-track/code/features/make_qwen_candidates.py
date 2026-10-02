import os
import numpy as np
import pandas as pd
from scipy.stats import rankdata

BASE = '/data/湾区杯'
test = pd.read_csv(BASE + '/replay_data/文本赛道/text_test.csv')
def rn(x):
    x = np.asarray(x, dtype=float)
    return rankdata(x, method='average') / (len(x)+1.)
q = rn(pd.read_csv(BASE + '/text_qwen7b/text_qwen7b_c003_qwen_only.csv')['score'])
anx = rn(pd.read_csv(BASE + '/text_anx.csv')['score'])
paper = rn(pd.read_csv(BASE + '/text_paperpass.csv')['score'])
old = rn(pd.read_csv(BASE + '/text_replay_submit.csv')['score'])
candidates = {
    'text_qwen_anx_paper10.csv': .40*q + .50*anx + .10*paper,
    'text_qwen_anx_paper10_old5.csv': .38*q + .47*anx + .10*paper + .05*old,
    'text_qwen_anx50.csv': .40*q + .60*anx,
    'text_qwen_anx_old5.csv': .38*q + .57*anx + .05*old,
}
for name, score in candidates.items():
    pd.DataFrame({'id': test['id'], 'score': score}).to_csv(os.path.join(BASE, name), index=False)
    print(name, len(score), float(score.min()), float(score.max()), flush=True)
