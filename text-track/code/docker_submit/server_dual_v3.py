#!/usr/bin/env python3
"""
AI 生成文本检测推理服务(Docker 提交版 · 双模型异构集成)

方案:0.90 × [Qwen2.5-14B + p4 五折 LoRA]  +  0.10 × [Qwen2.5-7B + r3a/r3b 十折 LoRA]
      复赛文本 AUC = 0.981301

技术要点:
  1. ★ 预加载 torch.distributed.tensor(peft 0.20.0 需要,惰性模块)
  2. bf16 全精度推理(不用 bitsandbytes,避免依赖冲突)
  3. 多 adapter 切换(14B×5折 + 7B×10折)
  4. 分数级融合 sigmoid,已验证与离线 rank 融合相关性 0.999997
  5. 后台线程加载,探针立即可用
"""
# ★★★ 必须最先执行:预加载惰性模块 ★★★
# peft 0.20.0 的 tuners_utils._get_in_out_features 会直接访问
# torch.distributed.tensor.DTensor;而 torch.distributed.tensor 是惰性子模块,
# 不显式 import 的话 torch.distributed 上根本没有 tensor 属性,
# 会抛 AttributeError: module 'torch.distributed' has no attribute 'tensor'。
import torch  # noqa: E402
import torch.distributed as _torch_distributed  # noqa: E402

try:
    import torch.distributed.tensor  # noqa: F401
except Exception:
    pass
try:
    import torch.distributed._tensor  # noqa: F401
except Exception:
    pass

# 双保险:即使 import 走了异常分支,也把已加载的模块显式绑到父包上
import sys as _sys  # noqa: E402

_tdt = _sys.modules.get("torch.distributed.tensor")
if _tdt is not None and not hasattr(_torch_distributed, "tensor"):
    _torch_distributed.tensor = _tdt

import os  # noqa: E402
import time  # noqa: E402
import threading  # noqa: E402
import traceback  # noqa: E402

from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.exceptions import RequestValidationError  # noqa: E402
from fastapi.requests import Request  # noqa: E402
from fastapi.responses import JSONResponse  # noqa: E402
from pydantic import BaseModel  # noqa: E402

# ------------------------------------------------------------------ 配置
MODEL_14B = os.environ.get("MODEL_DIR_14B", "/app/model/Qwen2.5-14B-Instruct")
MODEL_7B = os.environ.get("MODEL_DIR_7B", "/app/model/Qwen2.5-7B-Instruct")
ADAPTER_ROOT = os.environ.get("ADAPTER_ROOT", "/app/adapters")
FOLDS_14B = int(os.environ.get("FOLDS_14B", "5"))
FOLDS_7B = int(os.environ.get("FOLDS_7B", "5"))
W_14B = float(os.environ.get("W_14B", "0.90"))
W_7B = float(os.environ.get("W_7B", "0.10"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))

app = FastAPI(title="AI-Text-Detector")
STATE = {"ready": False, "error": None, "m14": None, "t14": None, "a14": [],
         "m7": None, "t7": None, "a7": []}


class PredictReq(BaseModel):
    req_id: str
    data: str


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    """请求体非法/缺字段 -> 400 + code 40001(而不是 FastAPI 默认的 422)。"""
    return JSONResponse(status_code=400,
                        content={"req_id": None, "code": 40001,
                                 "message": "invalid request body", "score": None})


def _load_one(model_dir, prefix, folds, tag):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import PeftModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, num_labels=2, trust_remote_code=True,
        ignore_mismatched_sizes=True, attn_implementation="eager",
        torch_dtype=(torch.bfloat16 if dev == "cuda" else torch.float32))
    model.config.pad_token_id = tok.pad_token_id
    if dev != "cuda":
        model = model.to(dev)

    dirs = []
    for f in range(folds):
        d = os.path.join(ADAPTER_ROOT, f"{prefix}_fold{f}_adapter")
        if os.path.isdir(d):
            dirs.append((f, d))
    if not dirs:
        raise RuntimeError(f"no adapters for {prefix} in {ADAPTER_ROOT}")

    names = []
    first = f"{tag}0"
    model = PeftModel.from_pretrained(model, dirs[0][1], adapter_name=first)
    names.append(first)
    for f, d in dirs[1:]:
        nm = f"{tag}{f}"
        model.load_adapter(d, adapter_name=nm)
        names.append(nm)
    model.eval()
    print(f"[load] {prefix}: {len(names)} adapters", flush=True)
    return model, tok, names


def _load_worker():
    try:
        t0 = time.time()
        m14, t14, a14 = _load_one(MODEL_14B, "lora14b_p4", FOLDS_14B, "f14_")
        STATE.update(m14=m14, t14=t14, a14=a14)
        print(f"[load] 14B done {time.time()-t0:.0f}s", flush=True)
        # 7B:r3a 在前,再追加 r3b
        m7, t7, a7 = _load_one(MODEL_7B, "lora7b_distill_r3a", FOLDS_7B, "a7_")
        for f in range(FOLDS_7B):
            d = os.path.join(ADAPTER_ROOT, f"lora7b_distill_r3b_fold{f}_adapter")
            if os.path.isdir(d):
                nm = f"b7_{f}"
                m7.load_adapter(d, adapter_name=nm)
                a7.append(nm)
        m7.eval()
        STATE.update(m7=m7, t7=t7, a7=a7)
        STATE["ready"] = True
        print(f"[load] ALL done {time.time()-t0:.0f}s  14B={len(a14)} 7B={len(a7)}", flush=True)
    except Exception as e:
        STATE["error"] = f"{type(e).__name__}: {e}"
        print("[load] FAILED:", traceback.format_exc(), flush=True)


@app.on_event("startup")
def _start_load():
    threading.Thread(target=_load_worker, daemon=True).start()


@app.get("/health/startup")
def health_startup():
    if STATE["ready"]:
        return {"status": "ok"}
    if STATE["error"]:
        raise HTTPException(status_code=503, detail=f"load failed: {STATE['error']}")
    raise HTTPException(status_code=503, detail="loading")


@app.get("/health/liveness")
def health_liveness():
    return {"status": "ok"}


@app.get("/health/readiness")
def health_readiness():
    if STATE["ready"]:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="not ready")


@torch.inference_mode()
def _mean_logit(model, tok, names, text):
    s = 0.0
    for nm in names:
        model.set_adapter(nm)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        s += float(model(**enc).logits.float()[0, 0])   # logits[:,0] = AI 方向
    return s / max(1, len(names))


def _sig(x):
    return 1.0 / (1.0 + pow(2.718281828459045, -max(-30.0, min(30.0, x))))


@torch.inference_mode()
def _predict_text(text: str) -> float:
    p14 = _sig(_mean_logit(STATE["m14"], STATE["t14"], STATE["a14"], text))
    if STATE["m7"] is not None and STATE["a7"]:
        p7 = _sig(_mean_logit(STATE["m7"], STATE["t7"], STATE["a7"], text))
    else:
        p7 = p14
    return W_14B * p14 + W_7B * p7


@app.post("/predict")
def predict(req: PredictReq):
    if not STATE["ready"]:
        return JSONResponse(status_code=503,
                            content={"req_id": req.req_id, "code": 50301,
                                     "message": "not ready", "score": None})
    if req.data is None:
        return JSONResponse(status_code=400,
                            content={"req_id": req.req_id, "code": 40001,
                                     "message": "missing data", "score": None})
    try:
        score = _predict_text(str(req.data))
        score = max(0.0, min(1.0, float(score)))
        if score != score:
            score = 0.5
        return {"req_id": req.req_id, "code": 0, "score": score}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"req_id": req.req_id, "code": 50001,
                                     "message": f"{type(e).__name__}", "score": None})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
