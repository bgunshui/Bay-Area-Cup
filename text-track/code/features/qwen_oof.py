import numpy as np, pandas as pd, os, glob
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
BASE='/data/湾区杯'; y=pd.read_csv(glob.glob(BASE+'/initial_text/**/text_sample_data.csv',recursive=True)[0])['label'].to_numpy()
X=np.load(BASE+'/text_qwen7b/train_emb_256.npy')
print('shape',X.shape)
for C in (.003,.01,.03,.1,.3,1.,3.):
 o=np.zeros(len(y)); sk=StratifiedKFold(5,shuffle=True,random_state=20260823)
 for a,b in sk.split(X,y):
  clf=make_pipeline(StandardScaler(with_mean=False),LogisticRegression(C=C,max_iter=1000,class_weight='balanced',solver='liblinear'))
  clf.fit(X[a],y[a]); o[b]=clf.decision_function(X[b])
 print(C,roc_auc_score(y,o))
for n in ['text_anx_initial.npy','paperpass_initial.npy','text_public_zh_initial.npy']:
 p=BASE+'/public_models/'+n
 if os.path.exists(p): print(n,roc_auc_score(y,np.load(p)))
