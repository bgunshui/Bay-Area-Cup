#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""server.py —— 图像 AI 生成置信度推理服务（湾区杯复赛 · 图像赛道 · Docker 交付）。

严格遵循《分类模型 Docker 交付标准 v1.2》：
  · 监听 0.0.0.0:8080
  · GET  /health/startup    模型加载完成后 200（加载期非 200，平台会重试）
  · GET  /health/liveness   仅反映进程存活，**极轻量**（不碰模型）
  · GET  /health/readiness  200 即保证紧接着的 /predict 能处理
  · POST /predict           {"req_id","data"(base64 图像)} → {"req_id","code":0,"score"}
  · 全程**不联网**；权重与查表全部在镜像内
  · 单张推理实测 < 1 秒（远低于 90 秒上限）

score 语义：**AI 生成的置信度**，越大越倾向 AI（已 clip 到 [0,1]）。
"""
import base64
import io
import json
import os
import sys
import threading
import time
import traceback

import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

APP_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS = os.environ.get("WEIGHTS_DIR", os.path.join(APP_DIR, "weights"))
TABLES = os.environ.get("TABLES_DIR", os.path.join(APP_DIR, "tables"))
BACKBONES = os.environ.get("BACKBONES_DIR", os.path.join(APP_DIR, "backbones"))
DEVICE = os.environ.get("DEVICE", "cuda")
DTYPE = os.environ.get("DTYPE", "float16")

STATE = {"ready": False, "err": None, "t_load": None, "n": 0}
_LOCK = threading.Lock()
_M = None
_F = None

app = FastAPI(title="AI-image-detector", version="1.0")


def _load():
    global _M, _F
    t0 = time.time()
    try:
        import torch
        from models import ComponentEnsemble
        from fusion import Fusion

        print(f"[startup] torch {torch.__version__} cuda={torch.cuda.is_available()}", flush=True)
        if DEVICE.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("要求 GPU 但 torch.cuda.is_available()=False")
        if torch.cuda.is_available():
            print(f"[startup] GPU = {torch.cuda.get_device_name(0)}", flush=True)

        _F = Fusion(TABLES)
        print(f"[startup] 融合查表就绪：{len(_F.terms)} 个分量项，prior_w={_F.prior_w}", flush=True)
        _M = ComponentEnsemble(WEIGHTS, BACKBONES,
                               device=DEVICE, dtype=getattr(torch, DTYPE))
        # ⚠️ 预热：**必须用 8 个分量自己的键**，且**逐个都跑一遍**。
        #   2026-09-16 真跑 HTTP 测试抓到的致命 bug：这里原来写的是
        #       _M.score("cliph_rerun", Image.new(...))
        #   —— 短键名。而 `models.py` 的键在修 bug #3 时已统一成 **CSV basename**
        #   （`img_cliph_rerun`）⇒ 预热直接 `KeyError: 'cliph_rerun'` ⇒ `_load()` 抛异常
        #   ⇒ `STATE["ready"]` 永远是 False ⇒ **启动/就绪探针永远 503，服务完全不可用**。
        #   （教训同 §0.9：同一件事散落在多处时，改一处必须全改。）
        #   顺带好处：8 个都过一遍会走通 `_swap_lora` 的换入分支（见 models.py 的说明）。
        for _k in _M.heads:
            _M.score(_k, Image.new("RGB", (256, 256), (128, 128, 128)))
        STATE["ready"] = True
        STATE["t_load"] = time.time() - t0
        print(f"[startup] 就绪，用时 {STATE['t_load']:.1f}s", flush=True)
    except Exception as e:
        STATE["err"] = f"{type(e).__name__}: {e}"
        traceback.print_exc()


# ───────────────────────── 探针 ─────────────────────────
@app.get("/health/startup")
def h_startup():
    if STATE["ready"]:
        return {"status": "ok"}
    return JSONResponse({"status": "loading", "err": STATE["err"]}, status_code=503)


@app.get("/health/liveness")
def h_liveness():
    # ⚠️ 规范要求：只反映进程存活、必须极轻量、禁止在这里做推理
    return {"status": "ok"}


@app.get("/health/readiness")
def h_readiness():
    if STATE["ready"]:
        return {"status": "ok"}
    return JSONResponse({"status": "not_ready", "err": STATE["err"]}, status_code=503)


# ───────────────────────── 推理 ─────────────────────────
def _err(req_id, http, code, msg):
    return JSONResponse({"req_id": req_id, "code": code, "message": msg, "score": None},
                        status_code=http)


@app.post("/predict")
async def predict(request: Request):
    if not STATE["ready"]:
        return _err(None, 503, 50301, "service not ready")
    try:
        body = await request.json()
    except Exception:
        return _err(None, 400, 40001, "request body is not valid JSON")
    if not isinstance(body, dict):
        return _err(None, 400, 40001, "request body must be a JSON object")
    req_id = body.get("req_id")
    data = body.get("data")
    if not isinstance(req_id, str) or not req_id:
        return _err(req_id if isinstance(req_id, str) else None, 400, 40001, "missing/invalid req_id")
    if not isinstance(data, str) or not data:
        return _err(req_id, 400, 40001, "missing/invalid data")

    try:
        raw = base64.b64decode(data, validate=False)
        if not raw:
            raise ValueError("empty after base64 decode")
        nbytes = len(raw)
        im = Image.open(io.BytesIO(raw))
        im.load()
        im = im.convert("RGB")
        w, h = im.size
    except Exception as e:
        return _err(req_id, 400, 40002, f"invalid base64 image data: {type(e).__name__}")

    try:
        with _LOCK:                      # 单模型实例串行，避免并发下显存抖动
            comp = {k: _M.score(k, im) for k in _M.heads}
            score = _F.fuse(comp, w, h, nbytes)
        score = float(min(1.0, max(0.0, score)))
        if not np.isfinite(score):
            raise ValueError("non-finite score")
        STATE["n"] += 1
        return {"req_id": req_id, "code": 0, "score": score}
    except Exception as e:
        traceback.print_exc()
        return _err(req_id, 500, 50001, f"inference error: {type(e).__name__}: {e}")


@app.get("/debug/info")
def debug_info():
    """自查用（规范未要求，不影响评测）。"""
    return {"ready": STATE["ready"], "err": STATE["err"], "load_seconds": STATE["t_load"],
            "n_predict": STATE["n"], "components": list(_M.heads) if _M else [],
            "prior_weight": _F.prior_w if _F else None,
            "lb_score_of_reference": (_F.meta.get("lb_score") if _F else None)}


if __name__ == "__main__":
    import uvicorn
    threading.Thread(target=_load, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")),
                log_level="info", access_log=False)
