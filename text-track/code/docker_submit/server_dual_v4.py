#!/usr/bin/env python3
"""
AI 生成文本检测推理服务(Docker 提交版 · 双模型异构集成)

方案:0.90 × [Qwen2.5-14B + p4 五折 LoRA]  +  0.10 × [Qwen2.5-7B + r3a/r3b 十折 LoRA]
      复赛文本 AUC = 0.981301

相对早期版本修掉的三处会导致交付失败的问题:
  1. ★ 设备放置:transformers 在不传 device_map 时会把模型留在 CPU
     (modeling_utils.py: `if device_map is None: param_device = "cpu"`)。
     早期写法 `if dev != "cuda": model = model.to(dev)` 在有 GPU 时反而不搬运,
     结果是 14B 在 CPU 上跑 15 次前向,单请求远超 90 秒上限。
     现在显式 device_map={"":0} 流式加载到显存,并保留 CPU->GPU 搬运作为兜底。
  2. ★ peft 0.20.0 需要 torch.distributed.tensor 先被导入(惰性模块),
     否则 load_adapter 抛 AttributeError: module 'torch.distributed' has no attribute 'tensor'。
  3. 空串/纯空白文本会产生空张量导致前向 RuntimeError,现用 eos 兜底。

其它要点:
  * bf16 全精度推理(不用 bitsandbytes,避免其 triton.ops 依赖冲突)
  * 多 adapter 切换(14B×5折 + 7B×10折),串行化以免 set_adapter 竞态
  * 分数级融合 sigmoid,已验证与离线 rank 融合相关性 0.999997
  * 后台线程加载模型,三个探针立即可用
"""
# ★★★ 必须最先执行:预加载 peft 需要的惰性模块 ★★★
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

import sys as _sys  # noqa: E402

_tdt = _sys.modules.get("torch.distributed.tensor")
if _tdt is not None and not hasattr(_torch_distributed, "tensor"):
    _torch_distributed.tensor = _tdt

import os  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402
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
# set_adapter 会改动共享模型状态,并发请求下必须串行化
_INFER_LOCK = threading.Lock()


class PredictReq(BaseModel):
    req_id: str
    data: str


@app.exception_handler(RequestValidationError)
async def _validation_handler(request: Request, exc: RequestValidationError):
    """请求体非法/缺字段 -> 400 + code 40001(而不是 FastAPI 默认的 422)。"""
    return JSONResponse(status_code=400,
                        content={"req_id": None, "code": 40001,
                                 "message": "invalid request body", "score": None})


def _from_pretrained(model_dir, dtype, dev):
    """加载基座模型,并确保它真的落在 dev 上。"""
    from transformers import AutoModelForSequenceClassification

    base_kw = dict(num_labels=2, trust_remote_code=True,
                   ignore_mismatched_sizes=True, attn_implementation="eager",
                   torch_dtype=dtype)
    if dev == "cuda":
        # 直接流式加载进显存:主机内存峰值低,启动也更快
        try:
            m = AutoModelForSequenceClassification.from_pretrained(
                model_dir, device_map={"": 0}, low_cpu_mem_usage=True, **base_kw)
            print(f"[load] {model_dir} 经 device_map 加载到 cuda:0", flush=True)
        except Exception as e:
            print(f"[load] device_map 加载失败({type(e).__name__}: {e}),"
                  f"回退为先 CPU 再搬运", flush=True)
            m = AutoModelForSequenceClassification.from_pretrained(model_dir, **base_kw)
    else:
        m = AutoModelForSequenceClassification.from_pretrained(model_dir, **base_kw)

    # 兜底:无论走哪条分支,都确认一遍设备。已经在目标设备上时几乎零开销。
    try:
        m = m.to(dev)
    except Exception as e:
        print(f"[load] model.to({dev}) 失败:{type(e).__name__}: {e}", flush=True)
    return m


def _load_one(model_dir, prefix, folds, tag):
    from transformers import AutoTokenizer
    from peft import PeftModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32

    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"

    model = _from_pretrained(model_dir, dtype, dev)
    model.config.pad_token_id = tok.pad_token_id
    print(f"[load] {prefix} 基座就位 device={model.device} dtype={dtype}", flush=True)

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
    print(f"[load] {prefix}: {len(names)} adapters device={model.device}", flush=True)
    return model, tok, names


def _load_worker():
    try:
        t0 = time.time()
        print(f"[load] torch={torch.__version__} cuda_available={torch.cuda.is_available()}", flush=True)
        if torch.cuda.is_available():
            print(f"[load] GPU={torch.cuda.get_device_name(0)} "
                  f"mem={torch.cuda.get_device_properties(0).total_memory/2**30:.1f}GiB", flush=True)

        m14, t14, a14 = _load_one(MODEL_14B, "lora14b_p4", FOLDS_14B, "f14_")
        STATE.update(m14=m14, t14=t14, a14=a14)
        print(f"[load] 14B done {time.time()-t0:.0f}s", flush=True)

        m7, t7, a7 = _load_one(MODEL_7B, "lora7b_distill_r3a", FOLDS_7B, "a7_")
        for f in range(FOLDS_7B):
            d = os.path.join(ADAPTER_ROOT, f"lora7b_distill_r3b_fold{f}_adapter")
            if os.path.isdir(d):
                nm = f"b7_{f}"
                m7.load_adapter(d, adapter_name=nm)
                a7.append(nm)
        m7.eval()
        STATE.update(m7=m7, t7=t7, a7=a7)

        # 用一条真实样本预热,确保 readiness 返回 200 后第一个请求就能正常处理
        _ = _predict_text("预热样本")
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
        if enc["input_ids"].numel() == 0:
            # 空串/纯空白:分词结果为空,空张量前向会 RuntimeError,用 eos 兜底
            eos = tok.eos_token_id if tok.eos_token_id is not None else 0
            enc["input_ids"] = torch.tensor([[eos]], dtype=torch.long)
            enc["attention_mask"] = torch.ones_like(enc["input_ids"])
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
        with _INFER_LOCK:                     # set_adapter 非线程安全,串行化
            score = _predict_text(str(req.data))
        score = max(0.0, min(1.0, float(score)))
        if score != score:                    # NaN 保护
            score = 0.5
        return {"req_id": req.req_id, "code": 0, "score": score}
    except Exception as e:
        return JSONResponse(status_code=500,
                            content={"req_id": req.req_id, "code": 50001,
                                     "message": f"{type(e).__name__}", "score": None})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
