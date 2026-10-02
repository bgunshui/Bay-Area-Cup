#!/usr/bin/env python3
"""在 chroot 内运行:验证修复后的依赖闭包能否真正跑通推理代码路径。"""
import importlib.metadata as md
import os
import sys
import traceback

print("=" * 78)
print("python:", sys.version.replace("\n", " "))
print("=" * 78)

PKGS = [
    "torch", "transformers", "peft", "tokenizers", "accelerate",
    "fastapi", "uvicorn", "pydantic", "pydantic_core", "starlette",
    "sentencepiece", "scipy", "numpy", "huggingface_hub", "safetensors",
    "filelock", "fsspec", "jinja2", "MarkupSafe", "networkx", "sympy",
    "mpmath", "typing_extensions", "psutil", "PyYAML", "regex", "requests",
    "urllib3", "certifi", "charset_normalizer", "idna", "h11", "click",
    "colorama", "exceptiongroup", "anyio", "sniffio", "tqdm", "triton",
    "nvidia-cublas-cu12", "nvidia-cudnn-cu12", "nvidia-cuda-runtime-cu12",
]
print("--- 发行版版本 ---")
bad = []
for p in PKGS:
    try:
        print(f"  {p:26s} {md.version(p)}")
    except Exception as e:
        print(f"  {p:26s} !! {type(e).__name__}: {e}")
        bad.append(p)
print(f"  无法解析的发行版: {bad if bad else '无'}")

print("\n--- 关键 import(与 server.py 完全一致的前置顺序)---")
import torch
print("  torch", torch.__version__, "cuda_capable", torch.version.cuda)
try:
    import torch.distributed.tensor  # noqa
    print("  torch.distributed.tensor  预加载 OK")
except Exception as e:
    print("  torch.distributed.tensor  预加载失败:", type(e).__name__, e)

fail = []
for mod in ["numpy", "scipy", "filelock", "fsspec", "jinja2", "networkx", "sympy",
            "yaml", "regex", "requests", "psutil", "tqdm", "safetensors",
            "huggingface_hub", "tokenizers", "transformers", "peft", "triton",
            "fastapi", "starlette", "uvicorn", "pydantic"]:
    try:
        __import__(mod)
        print(f"  import {mod:20s} OK")
    except Exception as e:
        print(f"  import {mod:20s} FAIL {type(e).__name__}: {e}")
        fail.append(mod)

import transformers
import peft
print(f"\n  transformers {transformers.__version__}   peft {peft.__version__}")

print("\n--- 真实 tokenizer 加载 ---")
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
tok = AutoTokenizer.from_pretrained("/app/model/Qwen2.5-7B-Instruct", trust_remote_code=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"
enc = tok("这是一段用于验证的中文文本,检查分词器是否正常工作。", return_tensors="pt",
          truncation=True, max_length=512)
print("  input_ids shape:", tuple(enc["input_ids"].shape), "pad_token_id:", tok.pad_token_id)

print("\n--- 用真实 adapter 权重做 PeftModel 加载(缩小层数的同构模型)---")
cfg = AutoConfig.from_pretrained("/app/model/Qwen2.5-7B-Instruct", trust_remote_code=True)
print("  hidden_size:", cfg.hidden_size, "intermediate_size:", cfg.intermediate_size,
      "num_layers(真实):", cfg.num_hidden_layers)
cfg.num_hidden_layers = 1
cfg.num_labels = 2
cfg.vocab_size = 1000
cfg.torch_dtype = torch.bfloat16
torch.manual_seed(0)
base = AutoModelForSequenceClassification.from_config(cfg)
base.config.pad_token_id = tok.pad_token_id
n_par = sum(p.numel() for p in base.parameters())
print(f"  缩小模型参数量 {n_par/1e6:.1f}M")

from peft import PeftModel
adir = "/app/adapters/lora7b_distill_r3a_fold0_adapter"
model = PeftModel.from_pretrained(base, adir, adapter_name="f0", ignore_mismatched_sizes=True)
model.eval()

print("\n--- 前向计算 ---")
ids = torch.randint(0, 1000, (1, 32))
with torch.inference_mode():
    out = model(input_ids=ids)
    print("  logits:", out.logits.float().tolist())
    s = float(out.logits.float()[0, 0])
    p = 1.0 / (1.0 + pow(2.718281828459045, -max(-30.0, min(30.0, s))))
    print(f"  logit[0]={s:.6f}  sigmoid={p:.6f}")

print("\n--- 打印 adapter_config.json 关键字段 ---")
import json
ac = json.load(open(os.path.join(adir, "adapter_config.json")))
for k in ("peft_type", "r", "lora_alpha", "lora_dropout", "target_modules",
          "task_type", "peft_version", "base_model_name_or_path", "bias"):
    print(f"  {k} = {ac.get(k)}")

print("\n" + "=" * 78)
print("失败模块:", fail if fail else "无")
print("结果:", "ALL PASS" if (not fail and not bad) else "HAS FAILURES")
print("=" * 78)
