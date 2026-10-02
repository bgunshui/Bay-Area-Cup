#!/usr/bin/env python3
"""端到端集成验证:导入 /app/server.py,用真实 adapter 权重 + 同构缩小模型,
真正走一遍 0.90×14B + 0.10×7B 的融合推理路径与 FastAPI 响应构造。"""
import json
import os
import sys
import traceback

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, "/app")

import torch
import torch.distributed as _td
try:
    import torch.distributed.tensor  # noqa: F401   ← peft 0.20.0 必需的前置预加载
except Exception as e:
    print("预加载 torch.distributed.tensor 失败:", e)

from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel

FOLDS = int(os.environ.get("E2E_FOLDS", "2"))


class TokShim:
    """真实 tokenizer,但把 token id 取模映射进缩小模型的词表范围,
    从而在不加载 41GB 真权重的前提下走通 server.py 的完整推理路径。"""

    def __init__(self, tok, vocab):
        object.__setattr__(self, "_tok", tok)
        object.__setattr__(self, "_vocab", vocab)

    def __call__(self, text, **kw):
        enc = self._tok(text, **kw)
        enc["input_ids"] = enc["input_ids"] % self._vocab
        return enc

    @property
    def eos_token_id(self):
        e = object.__getattribute__(self, "_tok").eos_token_id
        v = object.__getattribute__(self, "_vocab")
        return e if (e is not None and e < v) else 0

    def __getattr__(self, k):
        return getattr(object.__getattribute__(self, "_tok"), k)


ROOT = "/app/adapters"
AD14 = "lora14b_p4"
AD7A = "lora7b_distill_r3a"
AD7B = "lora7b_distill_r3b"

print("--- 14B adapter_config ---")
ac = json.load(open(f"{ROOT}/{AD14}_fold0_adapter/adapter_config.json"))
print({k: ac.get(k) for k in ("peft_type", "r", "lora_alpha", "target_modules", "task_type",
                              "peft_version", "inference_mode", "corda_config")})
print("--- 7B adapter_config ---")
ac7 = json.load(open(f"{ROOT}/{AD7A}_fold0_adapter/adapter_config.json"))
print({k: ac7.get(k) for k in ("peft_type", "r", "lora_alpha", "target_modules", "task_type",
                               "peft_version", "inference_mode", "corda_config")})


def build(model_dir, prefix, folds, tag, extra_prefix=None, extra_folds=0):
    """真实 tokenizer + 真实 adapter 权重,但模型只保留 1 层(形状一致),CPU bf16。"""
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"  {prefix}: hidden={cfg.hidden_size} inter={cfg.intermediate_size} "
          f"真实层数={cfg.num_hidden_layers}")
    cfg.num_hidden_layers = 1
    cfg.num_labels = 2
    cfg.vocab_size = 2000
    cfg.torch_dtype = torch.bfloat16
    torch.manual_seed(0)
    model = AutoModelForSequenceClassification.from_config(cfg)
    model.config.pad_token_id = tok.pad_token_id

    dirs = [(f, f"{ROOT}/{prefix}_fold{f}_adapter") for f in range(folds)
            if os.path.isdir(f"{ROOT}/{prefix}_fold{f}_adapter")]
    names = [f"{tag}0"]
    model = PeftModel.from_pretrained(model, dirs[0][1], adapter_name=names[0],
                                      ignore_mismatched_sizes=True)
    for f, d in dirs[1:]:
        nm = f"{tag}{f}"
        model.load_adapter(d, adapter_name=nm)
        names.append(nm)
    if extra_prefix:
        for f in range(extra_folds):
            d = f"{ROOT}/{extra_prefix}_fold{f}_adapter"
            if os.path.isdir(d):
                nm = f"b_{tag}{f}"
                model.load_adapter(d, adapter_name=nm)
                names.append(nm)
    model.eval()
    print(f"  {prefix}: 载入 {len(names)} 个 adapter -> {names}")
    return model, tok, names


print("\n--- 构建 14B 侧 ---")
m14, t14, a14 = build("/app/model/Qwen2.5-14B-Instruct", AD14, FOLDS, "f14_")
print("--- 构建 7B 侧 ---")
m7, t7, a7 = build("/app/model/Qwen2.5-7B-Instruct", AD7A, FOLDS, "a7_", extra_prefix=AD7B, extra_folds=FOLDS)

print("\n--- 导入 server.py 并注入 STATE ---")
import server
server.STATE.update(m14=m14, t14=TokShim(t14, 2000), a14=a14,
                    m7=m7, t7=TokShim(t7, 2000), a7=a7, ready=True)
print("  server.MAX_LEN =", server.MAX_LEN, " 权重:", server.W_14B, server.W_7B)

print("\n--- 探针 ---")
print("  /health/liveness  ->", server.health_liveness())
print("  /health/startup   ->", server.health_startup())
print("  /health/readiness ->", server.health_readiness())

print("\n--- POST /predict (真实融合路径,CPU/bf16/缩小模型)---")
TEXTS = [
    "这是一段由人类撰写的评论,表达了对产品的真实使用感受,语气自然。",
    "本研究基于多模态特征融合方法,提出了一个新的检测框架,实验表明该方法具有显著优势。",
    "The quick brown fox jumps over the lazy dog. " * 8,
]
for i, t in enumerate(TEXTS):
    r = server.predict(server.PredictReq(req_id=f"r{i}", data=t))
    if hasattr(r, "body"):
        print(f"  req{i}: HTTP {r.status_code} {json.loads(r.body)}")
    else:
        print(f"  req{i}: {r}")

print("\n--- 边界情况 ---")
r = server.predict(server.PredictReq(req_id="empty", data=""))
print("  空串:", r if not hasattr(r, "body") else json.loads(r.body))
server.STATE["ready"] = False
r = server.predict(server.PredictReq(req_id="x", data="abc"))
print("  未就绪:", r.status_code, json.loads(r.body))
print("\nE2E 完成")
