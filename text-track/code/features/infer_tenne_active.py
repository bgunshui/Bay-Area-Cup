#!/usr/bin/env python3
import glob
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

BASE=Path('/data/湾区杯'); ROOT=BASE/'new_models'/'tenne'; OUT=BASE/'new_text'; DEV=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def rn(x): return rankdata(np.asarray(x,dtype=float),method='average')/(len(x)+1.)
def run(model_dir,texts):
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float16); model.to(DEV).eval(); out=[]
 for st in range(0,len(texts),64):
  z=tok(texts[st:st+64],padding=True,truncation=True,max_length=512,return_tensors='pt'); z={k:v.to(DEV) for k,v in z.items()}
  with torch.inference_mode(): out.append(torch.softmax(model(**z).logits.float(),-1).cpu().numpy())
 del model,tok; torch.cuda.empty_cache() if DEV.type=='cuda' else None
 return np.concatenate(out)
def main():
 sp=Path(glob.glob(str(BASE/'initial_text'/'**'/'text_sample_data.csv'),recursive=True)[0]); tp=BASE/'replay_data'/'文本赛道'/'text_test.csv'; tr=pd.read_csv(sp,encoding='utf-8-sig'); te=pd.read_csv(tp,encoding='utf-8-sig'); texts=tr.text.fillna('').astype(str).tolist()+te.text.fillna('').astype(str).tolist()
 for ver in ('Tenne-CH3-V1','Tenne-CH3-V2'):
  p=run(ROOT/ver,texts); s,r=p[:len(tr)],p[len(tr):]
  for col in (0,1):
   auc=roc_auc_score(tr.label.astype(int),s[:,col]); print(ver,'class',col,'sample_auc',f'{auc:.6f}',flush=True); np.save(OUT/f'tenne_{ver[-2:].lower()}_c{col}_sample.npy',s[:,col]); np.save(OUT/f'tenne_{ver[-2:].lower()}_c{col}_replay.npy',r[:,col]); pd.DataFrame({'id':te.id.astype(str),'score':rn(r[:,col])}).to_csv(OUT/f'tenne_{ver[-2:].lower()}_c{col}.csv',index=False)
 print('DONE',flush=True)
if __name__=='__main__': main()
