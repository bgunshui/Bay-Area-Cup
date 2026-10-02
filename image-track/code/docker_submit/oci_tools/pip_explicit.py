#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pip_explicit.py —— 用 `--no-deps` + **显式包列表**安装，绕开 pip 的依赖解析。

为什么放弃"正常装 + dry-run 迭代"（2026-09-16 实测）：
    带 `--platform manylinux2014_x86_64 --only-binary=:all:` 时，pip 必须把**所有候选 wheel 的元数据**
    都拉下来做回溯，实测 **21 分钟还停在第一轮**，而且 `open_clip_torch` 依赖 `torch`
    ⇒ 它还会把 PyPI 的 torch（~2.5 GB + nvidia 依赖）一起拖下来，**覆盖官方基座自带的那份**。

做法：
    1. **显式列出全部需要的包**（顶层 + 传递依赖），`--no-deps` 逐个装 —— 不解析、不回溯，秒级完成。
    2. **基座已有的（torch/numpy/requests/tqdm/...）一律不列**，由官方镜像提供。
    3. 装完由 `oci_build3.py` 的"剪除与基座重名项"再兜一道底。

⚠️ `--no-deps` 的代价：如果漏了某个传递依赖，镜像里会 `ImportError`。
   所以下面这份列表是**按各包声明的依赖手工展开**的，并且刻意多列了几个（多装无害，重名的会被剪掉）。
"""
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/build/work/piproot"
PYVER = sys.argv[2] if len(sys.argv) > 2 else "3.11"
ABI = "cp" + PYVER.replace(".", "")
INDEX = "https://mirrors.aliyun.com/pypi/simple/"

# 冻结版本（能钉就钉；镜像上没有的只写包名让 pip 自选）
PKGS = [
    # ── 核心（影响权重加载/推理语义）──
    "transformers==5.17.0",
    "tokenizers==0.23.2",
    "peft==0.20.0",
    "accelerate==1.15.0",
    "huggingface_hub==1.31.0",
    "safetensors==0.8.0",
    "open_clip_torch==3.3.0",
    "regex==2026.9.10",
    # ── transformers/open_clip 的传递依赖（基座里**没有**的）──
    "ftfy",
    "protobuf",
    "pyyaml",
    "chardet",              # ftfy 依赖
    "wcwidth",              # ftfy 依赖
    # ── 数值（基座有 numpy，其余要补）──
    "scipy",
    "pandas",
    "scikit-learn",
    "joblib",
    "threadpoolctl",
    "python-dateutil",
    "pytz",
    "tzdata",
    "six",
    # ── 图像 ──
    "pillow==11.0.0",
    # ── 服务框架 ──
    "fastapi==0.115.6",
    "starlette",
    "pydantic",
    "pydantic-core",
    "annotated-types",
    "anyio",
    "sniffio",
    "idna",
    "exceptiongroup",
    "uvicorn==0.34.0",
    "click",
    "h11",
    "httptools",
    "python-dotenv",
    "uvloop",
    "watchfiles",
    "websockets",
    # ── 杂项（accelerate/peft 会用到）──
    "psutil",
    "typing-extensions",
]

cmd = ([sys.executable, "-m", "pip", "install", "-q", "--no-deps",
        "--target", TARGET, "--upgrade",
        "--python-version", PYVER, "--implementation", "cp", "--abi", ABI,
        "--platform", "manylinux2014_x86_64", "--only-binary=:all:", "-i", INDEX] + PKGS)
print(f"安装 {len(PKGS)} 个包（--no-deps，不解析依赖图）…", flush=True)
r = subprocess.run(cmd)
if r.returncode != 0:
    # 逐个重试，精确报出哪个包装不上
    print("\n整批失败 → 逐个安装以定位问题包", flush=True)
    ok, fail = [], []
    for p in PKGS:
        rr = subprocess.run(cmd[:-len(PKGS)] + [p], capture_output=True, text=True)
        (ok if rr.returncode == 0 else fail).append(p)
        if rr.returncode != 0:
            print(f"  ❌ {p}: {(rr.stdout + rr.stderr).strip().splitlines()[-1][:140]}", flush=True)
    print(f"\n成功 {len(ok)}，失败 {len(fail)}：{fail}")
    if fail:
        sys.exit(f"❌ 有 {len(fail)} 个包装不上")

with open("/hy-tmp/build/final_specs.txt", "w") as f:
    f.write("\n".join(PKGS) + "\n")
print("\n✅ 安装完成。实际装入的版本：")
subprocess.run([sys.executable, "-m", "pip", "list", "--path", TARGET, "--format", "columns"])
