import numpy as np, pandas as pd, glob, os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata

BASE = '/data/湾区杯'
y = pd.read_csv(glob.glob(BASE + '/initial_text/**/text_sample_data.csv', recursive=True)[0])['label'].to_numpy()
X = np.load(BASE + '/text_qwen7b/train_emb_256.npy').astype('float32')
def rn(x):
    return rankdata(np.asarray(x), method='average') / (len(x) + 1.)
o = np.zeros(len(y))
for a, b in StratifiedKFold(5, shuffle=True, random_state=20260823).split(X, y):
    c = SGDClassifier(loss='log_loss', alpha=1e-5, max_iter=1000, tol=1e-4,
                      class_weight='balanced', random_state=1)
    c.fit(X[a], y[a])
    o[b] = c.decision_function(X[b])
for n in ['text_anx_initial.npy', 'paperpass_initial.npy', 'text_public_zh_initial.npy']:
    p = BASE + '/public_models/' + n
    if os.path.exists(p):
        z = np.load(p)
        print(n, 'q', roc_auc_score(y, z))
        for w in np.arange(0, 1.01, .05):
            print(n, round(float(w),2), roc_auc_score(y, (1-w) * rn(z) + w * rn(o)))
print('qwen', roc_auc_score(y, o))
