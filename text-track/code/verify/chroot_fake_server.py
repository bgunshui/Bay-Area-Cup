#!/usr/bin/env python3
"""在 chroot 内:用同构缩小模型 + 真实 adapter 权重替换 server.py 的加载函数,
然后启动镜像里真实的 FastAPI 应用(server.app),以便通过真实 HTTP 验证。"""
import os
import sys

sys.path.insert(0, "/app")

import torch  # noqa: E402
import torch.distributed as _td  # noqa: E402

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
import torch.distributed.tensor  # noqa: F401,E402

from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer  # noqa: E402
from peft import PeftModel  # noqa: E402

import server  # noqa: E402

ROOT = "/app/adapters"
FOLDS = int(os.environ.get("E2E_FOLDS", "2"))


class TokShim:
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


def build(model_dir, prefix, folds, tag, vocab=1500):
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    cfg = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    cfg.num_hidden_layers = 1
    cfg.num_labels = 2
    cfg.vocab_size = vocab
    cfg.torch_dtype = torch.bfloat16
    torch.manual_seed(0)
    m = AutoModelForSequenceClassification.from_config(cfg)
    m.config.pad_token_id = tok.pad_token_id
    dirs = [(f, "%s/%s_fold%d_adapter" % (ROOT, prefix, f)) for f in range(folds)
            if os.path.isdir("%s/%s_fold%d_adapter" % (ROOT, prefix, f))]
    names = ["%s0" % tag]
    m = PeftModel.from_pretrained(m, dirs[0][1], adapter_name=names[0],
                                  ignore_mismatched_sizes=True)
    for f, d in dirs[1:]:
        nm = "%s%d" % (tag, f)
        m.load_adapter(d, adapter_name=nm)
        names.append(nm)
    m.eval()
    print("[fake] %s: %d adapters" % (prefix, len(names)), flush=True)
    return m, TokShim(tok, vocab), names


def fake_load():
    try:
        m14, t14, a14 = build("/app/model/Qwen2.5-14B-Instruct", "lora14b_p4", FOLDS, "f14_")
        m7, t7, a7 = build("/app/model/Qwen2.5-7B-Instruct", "lora7b_distill_r3a", FOLDS, "a7_")
        for f in range(FOLDS):
            d = "%s/lora7b_distill_r3b_fold%d_adapter" % (ROOT, f)
            if os.path.isdir(d):
                nm = "b7_%d" % f
                m7.load_adapter(d, adapter_name=nm)
                a7.append(nm)
        server.STATE.update(m14=m14, t14=t14, a14=a14, m7=m7, t7=t7, a7=a7, ready=True)
        print("[fake] 注入完成 14B=%d 7B=%d" % (len(a14), len(a7)), flush=True)
    except Exception:
        import traceback
        server.STATE["error"] = traceback.format_exc()[-800:]
        print("[fake] 失败:", traceback.format_exc(), flush=True)


server._load_worker = fake_load

import uvicorn  # noqa: E402

uvicorn.run(server.app, host="0.0.0.0", port=8080, log_level="warning")
