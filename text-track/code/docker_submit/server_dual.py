#!/usr/bin/env python3
"""
AI 生成文本检测推理服务(Docker 提交版 · 双模型异构集成)

方案:0.90 × [Qwen2.5-14B + p4 五折 LoRA]  +  0.10 × [Qwen2.5-7B + r3a/r3b 十折 LoRA]
      复赛文本 AUC = 0.981301(文本榜第 4 名,与提交文件一致)

设计要点:
  1. 两个模型均 8-bit 量化加载(14B≈14GB + 7B≈7.5GB ≈ 22GB 显存)
  2. 多 adapter 切换推理(14B×5折 + 7B×10折)
  3. 分数级融合(sigmoid),已验证与离线 rank 融合相关性 0.999997
  4. 后台线程加载,探针立即可用
"""
import os
import time
import threading
import traceback

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ------------------------------------------------------------------ 配置
MODEL_14B = os.environ.get("MODEL_DIR_14B", "/app/model/Qwen2.5-14B-Instruct")
MODEL_7B = os.environ.get("MODEL_DIR_7B", "/app/model/Qwen2.5-7B-Instruct")
ADAPTER_ROOT = os.environ.get("ADAPTER_ROOT", "/app/adapters")
FOLDS_14B = int(os.environ.get("FOLDS_14B", "5"))
FOLDS_7B = int(os.environ.get("FOLDS_7B", "5"))
W_14B = float(os.environ.get("W_14B", "0.90"))
W_7B = float(os.environ.get("W_7B", "0.10"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
LOAD_8BIT = os.environ.get("LOAD_8BIT", "1") == "1"

app = FastAPI(title="AI-Text-Detector")
STATE = {"ready": False, "error": None, "m14": None, "t14": None, "a14": [],
         "m7": None, "t7": None, "a7": [], "log": []}


class PredictReq(BaseModel):
    req_id: str
    data: str


def _load_one(model_dir, adapter_prefix, folds, tag):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              BitsAndBytesConfig)
    from peft import PeftModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    kwargs = dict(num_labels=2, trust_remote_code=True,
                  ignore_mismatched_errors=True if False else None,
                  ignore_mismatched_sizes=True, attn_implementation="eager")
    if LOAD_8BIT and dev == "cuda":
        kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        kwargs["device_map"] = {"": 0}
        kwargs["torch_dtype"] = torch.bfloat16
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForSequenceClassification.from_pretrained(model_dir, **kwargs)
    model.config.pad_token_id = tok.pad_token_id
    if not (LOAD_8BIT and dev == "cuda"):
        model = model.to(dev)

    dirs = []
    for f in range(folds):
        d = os.path.join(ADAPTER_ROOT, f"{adapter_prefix}_fold{f}_adapter")
        if os.path.isdir(d):
            dirs.append((f, d))
    if not dirs:
        raise RuntimeError(f"no adapters for {adapter_prefix} under {ADAPTER_ROOT}")

    names = []
    first = f"{tag}_fold{dirs[0][0]}"
    model = PeftModel.from_pretrained(model, dirs[0][1], adapter_name=first)
    names.append(first)
    for f, d in dirs[1:]:
        nm = f"{tag}_fold{f}"
        model.load_adapter(d, adapter_name=nm)
        names.append(nm)
    model.eval()
    print(f"[load] {adapter_prefix}: {len(names)} adapters", flush=True)
    return model, tok, names


def _load_worker():
    try:
        t0 = time.time()
        m14, t14, a14 = _load_one(MODEL_14B, "lora14b_p4", FOLDS_14B, "f14")
        STATE.update(m14=m14, t14=t14, a14=a14)
        print(f"[load] 14B done {time.time()-t0:.0f}s", flush=True)
        m7, t7, a7 = _load_one(MODEL_7B, "lora7b_distill_r3a", FOLDS_7B, "a7")
        # 追加 r3b 的 adapter
        extra = []
        for f in range(FOLDS_7B):
            d = os.path.join(ADAPTER_ROOT, f"lora7b_distill_r3b_fold{f}_adapter")
            if os.path.isdir(d):
                nm = f"b7_fold{f}"
                m7.load_adapter(d, adapter_name=nm)
                extra.append(nm)
        STATE.update(m7=m7, t7=t7, a7=a7 + extra)
        STATE["ready"] = True
        print(f"[load] ALL done in {time.time()-t0:.0f}s  "
              f"14B={len(a14)} 7B={len(a7)+len(extra)}", flush=True)
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
        try:
            enc = {k: v.to(model.device) for k, v in enc.items()}
        except Exception:
            enc = {k: v.cuda() for k, v in enc.items()}
        out = model(**enc)
        s += float(out.logits.float()[0, 0])   # logits[:,0] = AI 方向
    return s / max(1, len(names))


def _sig(x):
    return 1.0 / (1.0 + float(torch.exp(torch.tensor(-max(-30.0, min(30.0, x))))))


@torch.inference_mode()
def _predict_text(text: str) -> float:
    l14 = _mean_logit(STATE["m14"], STATE["t14"], STATE["a14"], text)
    p14 = _sig(l14)
    if STATE["m7"] is not None and STATE["a7"]:
        l7 = _mean_logit(STATE["m7"], STATE["t7"], STATE["a7"], text)
        p7 = _sig(l7)
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
