#!/usr/bin/env python3
import os, glob, json, math, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion
import joblib

BASE="/data/湾区杯"
OUT=os.path.join(BASE,"text_retrain")
os.makedirs(OUT,exist_ok=True)
sample=glob.glob(BASE+"/initial_text/**/text_sample_data.csv",recursive=True)[0]
testp=glob.glob(BASE+"/initial_text/**/text_test.csv",recursive=True)[0]
train=pd.read_csv(sample,encoding="utf-8-sig")
test=pd.read_csv(testp,encoding="utf-8-sig")
train_text=train["text"].fillna("").astype(str).tolist()
test_text=test["text"].fillna("").astype(str).tolist()
y=train["label"].astype(int).to_numpy()
print("train",len(train_text),"test",len(test_text),"label_mean",float(y.mean()),flush=True)

def rank_norm(x):
    x=np.asarray(x,dtype=np.float64)
    return rankdata(x,method="average")/(len(x)+1.0)

print("tfidf char",flush=True)
char=TfidfVectorizer(analyzer="char",ngram_range=(2,5),min_df=2,max_features=220000,
                     sublinear_tf=True,norm="l2",dtype=np.float32)
Xc=char.fit_transform(train_text)
Tc=char.transform(test_text)
cc=LogisticRegression(C=3.0,max_iter=700,class_weight="balanced",solver="liblinear")
cc.fit(Xc,y)
sc=cc.decision_function(Tc)
print("char done",Xc.shape,Tc.shape,flush=True)

print("tfidf word",flush=True)
word=TfidfVectorizer(analyzer="word",token_pattern=r"(?u)\b\w+\b",ngram_range=(1,2),
                     min_df=1,max_features=100000,sublinear_tf=True,norm="l2",dtype=np.float32)
Xw=word.fit_transform(train_text)
Tw=word.transform(test_text)
cw=LogisticRegression(C=2.0,max_iter=700,class_weight="balanced",solver="liblinear")
cw.fit(Xw,y)
sw=cw.decision_function(Tw)
print("word done",Xw.shape,Tw.shape,flush=True)

def style_row(s):
    n=max(len(s),1)
    chars=np.fromiter((ord(c) for c in s),dtype=np.int32)
    if chars.size:
        ascii_ratio=float((chars<128).mean())
        digit_ratio=float(np.isin(chars,np.arange(48,58)).mean())
        alpha_ratio=float(((chars>=65)&(chars<=122)).mean())
    else: ascii_ratio=digit_ratio=alpha_ratio=0.0
    lines=s.splitlines() or [s]
    lens=np.asarray([len(z) for z in lines],dtype=np.float32)
    punct=sum(1 for c in s if c in "，。！？；：、“”‘’（）《》,.!?;:()[]{}\"'")/n
    spaces=sum(1 for c in s if c.isspace())/n
    repeats=1.0-len(set(s))/n
    return [math.log1p(n),math.log1p(len(lines)),float(lens.mean()),float(lens.std()),
            ascii_ratio,digit_ratio,alpha_ratio,punct,spaces,repeats,
            s.count("\n")/n,s.count("。")/n,s.count("，")/n,s.count("!")/n,
            s.count("?")/n]
Xs=np.asarray([style_row(s) for s in train_text],dtype=np.float32)
Ts=np.asarray([style_row(s) for s in test_text],dtype=np.float32)
mu=Xs.mean(0); sd=Xs.std(0)+1e-5
Xs=(Xs-mu)/sd; Ts=(Ts-mu)/sd
cs=LogisticRegression(C=1.0,max_iter=500,class_weight="balanced")
cs.fit(Xs,y); ss=cs.decision_function(Ts)

cls_score=0.55*rank_norm(sc)+0.30*rank_norm(sw)+0.15*rank_norm(ss)
pd.DataFrame({"id":test["id"],"score":cls_score.astype(np.float64)}).to_csv(os.path.join(OUT,"text_cls.csv"),index=False)
joblib.dump({"char":char,"char_clf":cc,"word":word,"word_clf":cw,"style_clf":cs,"style_mu":mu,"style_sd":sd},os.path.join(OUT,"text_cls.joblib"),compress=3)
print("classical candidate written",flush=True)

# Qwen 3B frozen representation. If unavailable, keep classical candidate.
qscore=None
try:
    import torch
    from transformers import AutoTokenizer, AutoModel
    qpath="/data/tjx_exp/models/Qwen/Qwen2.5-3B-Instruct"
    print("loading qwen",qpath,flush=True)
    tok=AutoTokenizer.from_pretrained(qpath,trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    model=AutoModel.from_pretrained(qpath,trust_remote_code=True,torch_dtype=torch.float16)
    device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device); model.eval()
    def enc(items,batch=8):
        out=[]
        t0=time.time()
        for st in range(0,len(items),batch):
            chunk=items[st:st+batch]
            z=tok(chunk,padding=True,truncation=True,max_length=256,return_tensors="pt")
            z={k:v.to(device) for k,v in z.items()}
            with torch.inference_mode():
                h=model(**z).last_hidden_state
                m=z["attention_mask"].unsqueeze(-1).to(h.dtype)
                e=(h*m).sum(1)/m.sum(1).clamp_min(1)
                e=torch.nn.functional.normalize(e.float(),dim=1).cpu()
            out.append(e)
            if (st//batch)%50==0: print("qwen",st,"/",len(items),"elapsed",round(time.time()-t0,1),flush=True)
        return torch.cat(out,0).numpy()
    Etr=enc(train_text)
    Ete=enc(test_text)
    qclf=LogisticRegression(C=1.0,max_iter=500,class_weight="balanced")
    qclf.fit(Etr,y)
    qraw=qclf.decision_function(Ete)
    qscore=rank_norm(qraw)
    pd.DataFrame({"id":test["id"],"score":qscore.astype(np.float64)}).to_csv(os.path.join(OUT,"text_qwen3b.csv"),index=False)
    torch.save({"weight":torch.from_numpy(qclf.coef_.astype(np.float32)),"bias":torch.from_numpy(qclf.intercept_.astype(np.float32))},os.path.join(OUT,"text_qwen3b_head.pt"))
    with open(os.path.join(OUT,"text_qwen3b_meta.json"),"w",encoding="utf-8") as f:
        json.dump({"model":qpath,"max_length":256,"train_n":len(train_text)},f,ensure_ascii=True,indent=2)
    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    print("qwen candidate written",flush=True)
except Exception as e:
    print("QWEN_FAILED",repr(e),flush=True)

if qscore is None:
    final=cls_score
else:
    final=0.55*qscore+0.45*cls_score
pd.DataFrame({"id":test["id"],"score":final.astype(np.float64)}).to_csv(os.path.join(BASE,"text_retrain_submit.csv"),index=False)
pd.DataFrame({"id":test["id"],"score":final.astype(np.float64)}).to_csv(os.path.join(OUT,"text_blend.csv"),index=False)
with open(os.path.join(OUT,"run_meta.json"),"w",encoding="utf-8") as f:
    json.dump({"sample":sample,"test":testp,"train_n":len(train),"test_n":len(test),"qwen":qscore is not None},f,ensure_ascii=True,indent=2)
print("FINAL",os.path.join(BASE,"text_retrain_submit.csv"),"rows",len(test),flush=True)
