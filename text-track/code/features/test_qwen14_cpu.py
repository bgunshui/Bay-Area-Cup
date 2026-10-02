import time, torch
from auto_gptq import AutoGPTQForCausalLM
from transformers import AutoTokenizer
path="/data/湾区杯/models/Qwen2.5-14B-Instruct"
t=time.time(); tok=AutoTokenizer.from_pretrained(path,trust_remote_code=True); model=AutoGPTQForCausalLM.from_quantized(path,device="cpu",use_safetensors=True,use_triton=False,inject_fused_attention=False,inject_fused_mlp=False,disable_exllama=True,trust_remote_code=True); print("load",time.time()-t,flush=True)
z=tok(["这是一段用于测试文本生成概率的句子。"],return_tensors="pt"); t=time.time()
with torch.inference_mode(): out=model.model(**z,output_hidden_states=True,use_cache=False)
print("forward",time.time()-t,"hidden",out.last_hidden_state.shape,flush=True)
