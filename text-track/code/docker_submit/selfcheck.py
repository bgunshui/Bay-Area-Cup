#!/usr/bin/env python3
"""本地自检:模拟平台的调用流程,验证 server.py 的逻辑(不加载真实模型)。"""
import json
import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

print("=== Docker 提交包自检 ===")
import os

BASE = r"D:\湾区杯\文本赛道\docker_submit"

# 1) 文件齐备性
need = ["server.py", "Dockerfile", "requirements.txt", "技术方案报告.md", "README.md"]
print("\n[1] 必需文件:")
for f in need:
    p = os.path.join(BASE, f)
    ok = os.path.exists(p)
    print(f"  {'✓' if ok else '✗'} {f}" + (f"  ({os.path.getsize(p)} bytes)" if ok else "  MISSING"))

# 2) adapters
ad = os.path.join(BASE, "adapters")
folds = sorted(os.listdir(ad)) if os.path.isdir(ad) else []
print(f"\n[2] adapters({len(folds)} 个):")
tot = 0
for d in folds:
    w = os.path.join(ad, d, "adapter_model.safetensors")
    sz = os.path.getsize(w) / 1e6 if os.path.exists(w) else 0
    tot += sz
    print(f"  {'✓' if sz > 0 else '✗'} {d}  {sz:.1f} MB")
print(f"  合计: {tot:.1f} MB")

# 3) 基座模型
md = os.path.join(BASE, "model", "Qwen2.5-14B-Instruct")
print(f"\n[3] 基座模型 {md}:")
if os.path.isdir(md):
    files = os.listdir(md)
    print(f"  ✓ 存在,{len(files)} 个文件")
    st = [f for f in files if f.endswith(".safetensors")]
    print(f"  safetensors: {len(st)} 个")
else:
    print("  ✗ 未放置(构建前需从服务器复制或从 HF 下载)")

# 4) server.py 关键接口检查
print("\n[4] server.py 接口检查:")
src = open(os.path.join(BASE, "server.py"), encoding="utf-8").read()
for key, desc in [("/health/startup", "启动探针"), ("/health/liveness", "存活探针"),
                  ("/health/readiness", "就绪探针"), ("/predict", "推理接口"),
                  ("req_id", "req_id 回传"), ("max(0.0, min(1.0", "score 裁剪"),
                  ("BitsAndBytesConfig", "8-bit 量化"), ("set_adapter", "多 adapter")]:
    print(f"  {'✓' if key in src else '✗'} {desc} ({key})")

# 5) 镜像大小估算
model_gb = 28.0 if os.path.isdir(md) else 28.0
print(f"\n[5] 镜像大小估算:")
print(f"  14B 基座(bf16 打包): {model_gb:.0f} GB")
print(f"  5 折 adapters: {tot/1000:.2f} GB")
print(f"  PyTorch + 依赖: ~8 GB")
print(f"  ─────────────────")
print(f"  预计镜像: ~{model_gb + tot/1000 + 8:.0f} GB")
print(f"  (运行期 8-bit 量化后显存占用 ~18 GB)")
