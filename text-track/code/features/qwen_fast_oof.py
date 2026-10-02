import numpy as np, pandas as pd, glob, os
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_auc_score

BASE = '/data/湾区杯'
y = pd.read_csv(glob.glob(BASE + '/initial_text/**/text_sample_data.csv', recursive=True)[0])['label'].to_numpy()
X = np.load(BASE + '/text_qwen7b/train_emb_256.npy').astype('float32')
for alpha in (1e-5, 3e-5, 1e-4, 3e-4, 1e-3):
    o = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=20260823).split(X, y):
        clf = SGDClassifier(loss='log_loss', alpha=alpha, max_iter=1000, tol=1e-4,
                            class_weight='balanced', random_state=1)
        clf.fit(X[a], y[a])
        o[b] = clf.decision_function(X[b])
    print('qwen_sgd', alpha, roc_auc_score(y, o), flush=True)
for n in ['text_anx_initial.npy', 'paperpass_initial.npy', 'text_public_zh_initial.npy']:
    p = BASE + '/public_models/' + n
    if os.path.exists(p):
        print(n, roc_auc_score(y, np.load(p)))
p = BASE + '/text_hc3/initial_scores.npy'
if os.path.exists(p):
    print('hc3', roc_auc_score(y, np.load(p)))
