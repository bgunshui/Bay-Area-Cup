#!/usr/bin/env python3
import os,glob,time,json,warnings
warnings.filterwarnings("ignore")
import numpy as np,pandas as pd,joblib
from scipy.stats import rankdata
BASE='/data/湾区杯'
out=BASE+'/text_replay'
os.makedirs(out,exist_ok=True)
testp=glob.glob(BASE+'/replay_data/**/text_test.csv',recursive=True)[0]
te=pd.read_csv(testp,encoding='utf-8-sig')
texts=te.text.fillna('').astype(str).tolist()
print('replay text',len(te),flush=True)
bundle=joblib.load(BASE+'/text_retrain/text_cls.joblib')
print('char transform',flush=True)
Xc=bundle['char'].transform(texts)
rawc=bundle['char_clf'].decision_function(Xc)
print('word transform',flush=True)
Xw=bundle['word'].transform(texts)
raww=bundle['word_clf'].decision_function(Xw)
Xs=np.asarray([__import__('math').log1p(max(len(s),1)) for s in texts],dtype=np.float32)
# classical decision uses the saved style head; reconstruct exact style vector.
def style_row(s):
 n=max(len(s),1); chars=np.fromiter((ord(c) for c in s),dtype=np.int32)
 if chars.size:
  ar=float((chars<128).mean()); dr=float(np.isin(chars,np.arange(48,58)).mean()); al=float(((chars>=65)&(chars<=122)).mean())
 else: ar=dr=al=0.0
 lines=s.splitlines() or [s]; lens=np.asarray([len(z) for z in lines],dtype=np.float32)
 punct=sum(1 for c in s if c in "，。！？；：、“”‘’（）《》,.!?;:()[]{}\"'")/n
 spaces=sum(1 for c in s if c.isspace())/n; repeats=1.0-len(set(s))/n
 return [np.log1p(n),np.log1p(len(lines)),float(lens.mean()),float(lens.std()),ar,dr,al,punct,spaces,repeats,s.count("\\n")/n,s.count("。")/n,s.count("，")/n,s.count("!")/n,s.count("?")/n]
raws=bundle['style_clf'].decision_function((np.asarray([style_row(s) for s in texts],dtype=np.float32)-bundle['style_mu'])/bundle['style_sd'])
def rn(x): return rankdata(x,method='average')/(len(x)+1.0)
cls=.55*rn(rawc)+.30*rn(raww)+.15*rn(raws)
pd.DataFrame({'id':te.id.astype(str),'score':cls}).to_csv(out+'/text_replay_cls.csv',index=False)
# Qwen 3B head
import torch
from transformers import AutoTokenizer,AutoModel
qpath='/data/tjx_exp/models/Qwen/Qwen2.5-3B-Instruct'
tok=AutoTokenizer.from_pretrained(qpath,trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModel.from_pretrained(qpath,trust_remote_code=True,torch_dtype=torch.float16)
device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device).eval()
head=torch.load(BASE+'/text_retrain/text_qwen3b_head.pt',map_location='cpu')
w=head['weight'].float(); b=head['bias'].float()
qraw=[]; t0=time.time()
for st in range(0,len(texts),8):
 z=tok(texts[st:st+8],padding=True,truncation=True,max_length=256,return_tensors='pt')
 z={k:v.to(device) for k,v in z.items()}
 with torch.inference_mode():
  h=model(**z).last_hidden_state; m=z['attention_mask'].unsqueeze(-1).to(h.dtype)
  e=torch.nn.functional.normalize((h*m).sum(1)/m.sum(1).clamp_min(1),dim=1).float().cpu()
 qraw.extend((e@w.T+b).numpy().ravel().tolist())
 if st%400==0: print('qwen',st,'/',len(texts),'sec',round(time.time()-t0,1),flush=True)
q=rn(np.asarray(qraw))
pd.DataFrame({'id':te.id.astype(str),'score':q}).to_csv(out+'/text_replay_qwen3b.csv',index=False)
final=.45*cls+.55*q
pd.DataFrame({'id':te.id.astype(str),'score':final}).to_csv(BASE+'/text_replay_submit.csv',index=False)
pd.DataFrame({'id':te.id.astype(str),'score':final}).to_csv(out+'/text_replay_blend.csv',index=False)
print('FINAL',BASE+'/text_replay_submit.csv',len(te),flush=True)
