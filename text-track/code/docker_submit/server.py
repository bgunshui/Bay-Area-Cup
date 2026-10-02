#!/usr/bin/env python3
"""
AI 生成文本检测推理服务(Docker 提交版)

方案:Qwen2.5-14B-Instruct + 5 折 LoRA 自训练集成
      复赛 AUC = 0.981069(文本赛道第 4 名)

设计要点:
  1. 8-bit 量化加载(显存 ~16GB),满足平台显存约束
  2. 5 折 LoRA adapter 切换推理,logit[:,0] 平均(与离线方案一致)
  3. sigmoid 转概率输出 [0,1](单条推理无法 rank,已验证与 rank 融合等价 0.999997)
  4. 模型在后台线程加载,探针立即可用(startup 在加载完成前返回 503)
"""
import os
import sys
import time
import threading
import traceback

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------------------------------------------------------------- 配置
MODEL_DIR = os.environ.get("MODEL_DIR", "/app/model/Qwen2.5-14B-Instruct")
ADAPTER_ROOT = os.environ.get("ADAPTER_ROOT", "/app/adapters")
FOLDS = int(os.environ.get("FOLDS", "5"))
MAX_LEN = int(os.environ.get("MAX_LEN", "512"))
LOAD_8BIT = os.environ.get("LOAD_8BIT", "1") == "1"

app = FastAPI(title="AI-Text-Detector")

STATE = {
    "ready": False,
    "error": None,
    "model": None,
    "tok": None,
    "adapters": [],
    "device": None,
}


class PredictReq(BaseModel):
    req_id: str
    data: str


# ---------------------------------------------------------------- 模型加载
def _load_worker():
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        from peft import PeftModel

        t0 = time.time()
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        STATE["device"] = dev
        print(f"[load] device={dev} 8bit={LOAD_8BIT}", flush=True)

        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"

        kwargs = dict(num_labels=2, trust_remote_code=True,
                      ignore_mismatched_sizes=True, attn_implementation="eager")
        if LOAD_8BIT and dev == "cuda":
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            kwargs["device_map"] = {"": 0}
        else:
            kwargs["torch_dtype"] = torch.bfloat16

        model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, **kwargs)
        model.config.pad_token_id = tok.pad_token_id
        if not (LOAD_8BIT and dev == "cuda"):
            model = model.to(dev)

        # 依次挂载 fold adapters
        adapter_dirs = []
        for f in range(FOLDS):
            d = os.path.join(ADAPTER_ROOT, f"lora14b_p4_fold{f}_adapter")
            if os.path.isdir(d):
                adapter_dirs.append((f, d))
        if not adapter_dirs:
            raise RuntimeError(f"no adapters found under {ADAPTER_ROOT}")

        first_name = f"fold{adapter_dirs[0][0]}"
        model = PeftModel.from_pretrained(model, adapter_dirs[0][1], adapter_name=first_name)
        for f, d in adapter_dirs[1:]:
            model.load_adapter(d, adapter_name=f"fold{f}")
        model.eval()

        STATE["model"] = model
        STATE["tok"] = tok
        STATE["adapters"] = [f"fold{f}" for f, _ in adapter_dirs]
        STATE["ready"] = True
        print(f"[load] done in {time.time()-t0:.1f}s adapters={STATE['adapters']}", flush=True)
    except Exception as e:
        STATE["error"] = f"{type(e).__name__}: {e}"
        print("[load] FAILED:", traceback.format_exc(), flush=True)


@app.on_event("startup")
def _start_load():
    threading.Thread(target=_load_worker, daemon=True).start()


# ---------------------------------------------------------------- 健康探针
@app.get("/health/startup")
def health_startup():
    if STATE["ready"]:
        return {"status": "ok"}
    if STATE["error"]:
        raise HTTPException(status_code=503, detail=f"load failed: {STATE['error']}")
    raise HTTPException(status_code=503, detail="model loading")


@app.get("/health/liveness")
def health_liveness():
    return {"status": "ok"}


@app.get("/health/readiness")
def health_readiness():
    if STATE["ready"]:
        return {"status": "ok"}
    raise HTTPException(status_code=503, detail="not ready")


# ---------------------------------------------------------------- 推理
@torch.inference_mode()
def _predict_text(text: str) -> float:
    model, tok = STATE["model"], STATE["tok"]
    logits_sum = 0.0
    for name in STATE["adapters"]:
        model.set_adapter(name)
        enc = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(
            model.device if hasattr(model, "device") else "cuda")
        logits = model(**enc).logits.float()
        # logits[:,0] 为 AI 方向(训练用 BCE 于 logit0,与离线方案一致)
        logits_sum += float(logits[0, 0])
    mean_logit = logits_sum / len(STATE["adapters"])
    # sigmoid → [0,1](与离线 rank 融合等价,corr=0.999997)
    return 1.0 / (1.0 + torch.exp(torch.tensor(-mean_logit)).item())


@app.post("/predict")
def predict(req: PredictReq):
    if not STATE["ready"]:
        return JSONResponse(status_code=503,
                            content={"req_id": req.req_id, "code": 50301,
                                     "message": "service not ready", "score": None})
    if req.data is None:
        return JSONResponse(status_code=400,
                            content={"req_id": req.req_id, "code": 40001,
                                     "message": "missing data", "score": None})
    try:
        score = _predict_text(str(req.data))
        score = max(0.0, min(1.0, float(score)))
        if score != score:  # NaN guard
            score = 0.5
        return {"req_id": req.req_id, "code": 0, "score": score}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"req_id": req.req_id, "code": 50001,
                                     "message": f"{type(e).__name__}", "score": None})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
