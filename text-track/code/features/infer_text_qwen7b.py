import os,glob,time,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from scipy.stats import rankdata
from transformers import AutoTokenizer,AutoModel
BASE='/data/湾区杯'; MODEL='/data/tjx_exp/models/Qwen/Qwen2___5-7B-Instruct'
dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True)
sp=glob.glob(BASE+'/initial_text/**/text_sample_data.csv',recursive=True)[0]
tp=glob.glob(BASE+'/replay_data/**/text_test.csv',recursive=True)[0]
tr=pd.read_csv(sp,encoding='utf-8-sig'); te=pd.read_csv(tp,encoding='utf-8-sig')
a=tr.text.fillna('').astype(str).tolist(); b=te.text.fillna('').astype(str).tolist(); y=tr.label.astype(int).to_numpy()
tok=AutoTokenizer.from_pretrained(MODEL,trust_remote_code=True)
if tok.pad_token is None: tok.pad_token=tok.eos_token
model=AutoModel.from_pretrained(MODEL,trust_remote_code=True,torch_dtype=torch.float16,device_map='auto',low_cpu_mem_usage=True)
dev=next(model.parameters()).device
model.eval()
def enc(items,bs=2):
 out=[]; t=time.time()
 for st in range(0,len(items),bs):
  z=tok(items[st:st+bs],padding=True,truncation=True,max_length=256,return_tensors='pt')
  z={k:v.to(dev) for k,v in z.items()}
  with torch.inference_mode():
   h=model(**z).last_hidden_state; m=z['attention_mask'].unsqueeze(-1).to(h.dtype)
   e=(h*m).sum(1)/m.sum(1).clamp_min(1); e=torch.nn.functional.normalize(e.float(),dim=1).cpu()
  out.append(e)
  if st==0 or st%200==0: print('qwen7b',st,'/',len(items),'sec',round(time.time()-t,1),flush=True)
 return torch.cat(out).numpy()
Etr=enc(a); Ete=enc(b)
clf=LogisticRegression(C=1.0,max_iter=500,class_weight='balanced',solver='liblinear')
clf.fit(Etr,y); s=clf.decision_function(Ete)
def rn(x): return rankdata(np.asarray(x),method='average')/(len(x)+1.)
q=rn(s); ids=te.id.astype(str).tolist()
pd.DataFrame({'id':ids,'score':q}).to_csv(BASE+'/text_qwen7b.csv',index=False)
oldp=BASE+'/text_replay_submit.csv'
if os.path.exists(oldp):
 old=pd.read_csv(oldp)
 if old.id.astype(str).tolist()==ids:
  oldr=rn(old.score.to_numpy(float))
  for w in [.2,.35,.5,.65,.8]:
   pd.DataFrame({'id':ids,'score':w*q+(1-w)*oldr}).to_csv(BASE+f'/text_qwen7b_blend_{int(w*100):02d}.csv',index=False)
np.save(BASE+'/text_qwen7b_train.npy',Etr); torch.save({'coef':clf.coef_,'intercept':clf.intercept_},BASE+'/text_qwen7b_head.pt')
print('done qwen7b',len(ids),flush=True)
