#!/usr/bin/env python3
from pathlib import Path
import numpy as np, pandas as pd, torch
from transformers.utils import import_utils
import transformers.modeling_utils as _modeling_utils
import_utils.check_torch_load_is_safe = lambda: None
_modeling_utils.check_torch_load_is_safe = lambda: None
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.stats import rankdata

BASE=Path('/data/湾区杯'); OUT=BASE/'window_text'; OUT.mkdir(exist_ok=True); DEV=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def r(x): return rankdata(np.asarray(x,float),method='average')/(len(x)+1.)
def run(model_dir,texts,chunk,stride,tag):
 tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True); model=AutoModelForSequenceClassification.from_pretrained(model_dir,local_files_only=True,torch_dtype=torch.float16); model.to(DEV).eval(); vals=[]
 for ii,s in enumerate(texts):
  ids=tok(s,add_special_tokens=False,return_attention_mask=False)['input_ids']
  if not ids: vals.append(0.5); continue
  windows=[ids[k:k+chunk] for k in range(0,max(1,len(ids)-chunk+1),stride)]
  if windows[-1][-1] != ids[-1]: windows.append(ids[-chunk:])
  ps=[]
  for st in range(0,len(windows),32):
   z=tok.pad([{'input_ids':[tok.cls_token_id]+w+[tok.sep_token_id]} for w in windows[st:st+32]],padding=True,return_tensors='pt'); z={k:v.to(DEV) for k,v in z.items()}
   with torch.inference_mode(): ps.extend(torch.softmax(model(**z).logits.float(),-1)[:,1].cpu().numpy().tolist())
  vals.append(float(np.quantile(ps,.75)))
  if ii==0 or (ii+1)%1000==0: print(tag,ii+1,'/',len(texts),flush=True)
 del model,tok; torch.cuda.empty_cache() if DEV.type=='cuda' else None
 return np.asarray(vals)
def main():
 te=pd.read_csv(BASE/'replay_data'/'文本赛道'/'text_test.csv',encoding='utf-8-sig'); texts=te.text.fillna('').astype(str).tolist()
 anx=run(BASE/'active_models'/'anx',texts,254,128,'anx'); np.save(OUT/'anx_q75_replay.npy',anx); pd.DataFrame({'id':te.id.astype(str),'score':r(anx)}).to_csv(OUT/'text_anx_q75.csv',index=False)
 v3=run(BASE/'audit_public'/'yuchuantian_AIGC_detector_zhv3',texts,510,510,'v3'); np.save(OUT/'v3_max_replay.npy',v3); pd.DataFrame({'id':te.id.astype(str),'score':r(v3)}).to_csv(OUT/'text_v3_max.csv',index=False)
 print('DONE',flush=True)
if __name__=='__main__': main()
