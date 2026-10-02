import os,glob,time,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from transformers import AutoTokenizer,AutoModelForSequenceClassification
BASE='/data/湾区杯'; MODEL=BASE+'/public_models/chinese_ai_bert'
dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True)
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,torch_dtype=torch.float16)
model.to(dev).eval()
def run(items,bs=32):
 out=[]; t=time.time()
 for st in range(0,len(items),bs):
  z=tok(items[st:st+bs],padding=True,truncation=True,max_length=512,return_tensors='pt')
  z={k:v.to(dev) for k,v in z.items()}
  with torch.inference_mode(): p=torch.softmax(model(**z).logits.float(),1)[:,1]
  out.extend(p.cpu().numpy().tolist())
  if st==0 or st%640==0 or st+bs>=len(items): print('infer',st+len(z['input_ids']),'/',len(items),'sec',round(time.time()-t,1),flush=True)
 return np.asarray(out,dtype=float)
sp=glob.glob(BASE+'/initial_text/**/text_sample_data.csv',recursive=True)[0]; tr=pd.read_csv(sp,encoding='utf-8-sig')
sv=run(tr.text.fillna('').astype(str).tolist()); print('initial_auc',roc_auc_score(tr.label.astype(int),sv),flush=True); np.save(BASE+'/public_models/text_anx_initial.npy',sv)
tp=glob.glob(BASE+'/replay_data/**/text_test.csv',recursive=True)[0]; te=pd.read_csv(tp,encoding='utf-8-sig'); ids=te.id.astype(str).tolist(); pv=run(te.text.fillna('').astype(str).tolist())
pd.DataFrame({'id':ids,'score':pv}).to_csv(BASE+'/text_anx.csv',index=False)
def rn(x): return rankdata(np.asarray(x),method='average')/(len(x)+1.)
oldp=BASE+'/text_replay_submit.csv'
if os.path.exists(oldp):
 old=pd.read_csv(oldp)
 if old.id.astype(str).tolist()==ids:
  for w in [.1,.2,.3,.4,.5,.65,.8,1.0]:
   pd.DataFrame({'id':ids,'score':(1-w)*rn(old.score.to_numpy(float))+w*rn(pv)}).to_csv(BASE+f'/text_anx_blend_{int(w*100):02d}.csv',index=False)
print('done',len(ids),flush=True)
