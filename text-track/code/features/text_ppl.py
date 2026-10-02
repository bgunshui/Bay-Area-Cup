import csv,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
src='/data/湾区杯/replay_data/文本赛道/text_test.csv'; out='/data/湾区杯/text_submit.csv'; md='/data/tjx_exp/models/Qwen/Qwen2.5-1.5B-Instruct'
tok=AutoTokenizer.from_pretrained(md,local_files_only=True,use_fast=True)
m=AutoModelForCausalLM.from_pretrained(md,local_files_only=True,torch_dtype=torch.float16,device_map={'':'cuda:0'}).eval()
rows=[]
with open(src,encoding='utf-8-sig',newline='') as f:
 for x in csv.DictReader(f): rows.append((x['id'],x['text']))
ss=[]
with torch.inference_mode():
 for st in range(0,len(rows),8):
  b=rows[st:st+8]; e=tok([x[1] for x in b],return_tensors='pt',padding=True,truncation=True,max_length=512)
  ids=e.input_ids.to('cuda:0'); mask=e.attention_mask.to('cuda:0'); z=m(input_ids=ids,attention_mask=mask).logits
  lp=torch.log_softmax(z[:,:-1],-1); tg=ids[:,1:]; mm=mask[:,1:].bool(); n=-lp.gather(-1,tg.unsqueeze(-1)).squeeze(-1)
  ss += (-(n*mm).sum(1)/mm.sum(1).clamp_min(1)).float().cpu().tolist()
  if st%1600==0: print(st,flush=True)
lo,hi=min(ss),max(ss)
with open(out,'w',encoding='utf-8',newline='') as f:
 w=csv.writer(f);w.writerow(['id','score'])
 for (id,_),s in zip(rows,ss):w.writerow([id,f'{(s-lo)/(hi-lo+1e-9):.10f}'])
print('DONE',len(rows),lo,hi)
