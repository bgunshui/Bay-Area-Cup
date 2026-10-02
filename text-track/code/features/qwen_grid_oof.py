import numpy as np, pandas as pd, glob, os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

BASE = '/data/湾区杯'
y = pd.read_csv(glob.glob(BASE + '/initial_text/**/text_sample_data.csv', recursive=True)[0])['label'].to_numpy()
X = np.load(BASE + '/text_qwen7b/train_emb_256.npy').astype('float32')
def rn(x):
    return rankdata(np.asarray(x), method='average') / (len(x)+1.)
o = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=20260823).split(X, y):
    c = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000, tol=1e-4,
                      class_weight='balanced', random_state=1)
    c.fit(X[a], y[a]); o[b] = c.decision_function(X[b])
q = rn(o)
anx = rn(np.load(BASE + '/public_models/text_anx_initial.npy'))
paper = rn(np.load(BASE + '/public_models/paperpass_initial.npy'))
best = (0, None)
for wq in np.arange(0, 0.81, .05):
  for wp in np.arange(0, 0.31, .05):
    wa = 1-wq-wp
    if wa < 0: continue
    auc = roc_auc_score(y, wq*q + wa*anx + wp*paper)
    if auc > best[0]: best = (auc, (wq,wa,wp))
    if auc >= .9910: print('candidate',round(float(auc),6),round(float(wq),2),round(float(wa),2),round(float(wp),2))
print('BEST',best)
