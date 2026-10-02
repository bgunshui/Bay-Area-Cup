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
for aidx, bidx in StratifiedKFold(5, shuffle=True, random_state=20260823).split(X, y):
    c = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000, tol=1e-4,
                      class_weight='balanced', random_state=1)
    c.fit(X[aidx], y[aidx]); o[bidx] = c.decision_function(X[bidx])
q = rn(o)
a = rn(np.load(BASE + '/public_models/text_anx_initial.npy'))
p = rn(np.load(BASE + '/public_models/paperpass_initial.npy'))
m = rn(np.load(BASE + '/public_models/mokawa_initial.npy'))
best = (0, None)
for wq in np.arange(0, .61, .05):
    for wm in np.arange(0, .31, .05):
        for wp in np.arange(0, .21, .05):
            wa = 1-wq-wm-wp
            if wa < 0: continue
            z = wq*q + wm*m + wp*p + wa*a
            auc = roc_auc_score(y, z)
            if auc > best[0]: best = (auc, (wq,wa,wp,wm))
print('BEST', best)
for cfg in [(0.4,.5,.1,0),(.3,.5,.1,.1),(.35,.45,.1,.1),(.3,.55,.05,.1),(.25,.55,.05,.15),(.4,.45,.05,.1)]:
    wq,wa,wp,wm = cfg
    print(cfg, roc_auc_score(y, wq*q + wa*a + wp*p + wm*m))
