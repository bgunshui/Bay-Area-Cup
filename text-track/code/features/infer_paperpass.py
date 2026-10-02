import os,glob,time,warnings
warnings.filterwarnings('ignore')
import numpy as np,pandas as pd,torch
from sklearn.metrics import roc_auc_score
from scipy.stats import rankdata
from transformers import AutoTokenizer,AutoModelForSequenceClassification
BASE='/data/湾区杯'; MODEL=BASE+'/public_models/paperpass_v3'
dev=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'); print('device',dev,flush=True)
tok=AutoTokenizer.from_pretrained(MODEL)
model=AutoModelForSequenceClassification.from_pretrained(MODEL,torch_dtype=torch.float16)
model.to(dev).eval()
def run(items,bs=16):
 out=[]; t=time.time()
 for st in range(0,len(items),bs):
  z=tok(items[st:st+bs],padding=True,truncation=True,max_length=512,return_tensors='pt')
  z={k:v.to(dev) for k,v in z.items()}
  with torch.inference_mode(): p=torch.softmax(model(**z).logits.float(),1)[:,1]
  out.extend(p.cpu().numpy().tolist())
  if st==0 or st%320==0 or st+bs>=len(items): print('infer',st+len(z['input_ids']),'/',len(items),'sec',round(time.time()-t,1),flush=True)
 return np.asarray(out,dtype=float)
sp=glob.glob(BASE+'/initial_text/**/text_sample_data.csv',recursive=True)[0]; tr=pd.read_csv(sp,encoding='utf-8-sig')
sv=run(tr.text.fillna('').astype(str).tolist()); print('initial_auc',roc_auc_score(tr.label.astype(int),sv),flush=True); np.save(BASE+'/public_models/paperpass_initial.npy',sv)
tp=glob.glob(BASE+'/replay_data/**/text_test.csv',recursive=True)[0]; te=pd.read_csv(tp,encoding='utf-8-sig'); ids=te.id.astype(str).tolist(); pv=run(te.text.fillna('').astype(str).tolist())
pd.DataFrame({'id':ids,'score':pv}).to_csv(BASE+'/text_paperpass.csv',index=False)
def rn(x): return rankdata(np.asarray(x),method='average')/(len(x)+1.)
pp=rn(pv)
for n,w in [('anx',.5),('anx75',.75),('base',.25)]:
 pass
an=pd.read_csv(BASE+'/text_anx.csv'); old=pd.read_csv(BASE+'/text_replay_submit.csv')
assert an.id.astype(str).tolist()==ids==old.id.astype(str).tolist()
ar=rn(an.score.to_numpy(float)); br=rn(old.score.to_numpy(float))
os.makedirs('/data/wq_candidates',exist_ok=True)
for wa,wp,wb,name in [
 (.0,1.0,.0,'text_paperpass_100'),
 (.25,.75,.0,'text_pp_anx_25'),
 (.5,.5,.0,'text_pp_anx_50'),
 (.75,.25,.0,'text_pp_anx_75'),
 (.2,.6,.2,'text_pp_anx_base'),
 (.1,.8,.1,'text_pp_anx_base80')]:
 d=pd.DataFrame({'id':ids,'score':wp*pp+wa*ar+wb*br}); d.to_csv(BASE+'/'+name+'.csv',index=False); d.to_csv('/data/wq_candidates/'+name+'.csv',index=False)
print('done',len(ids),flush=True)
