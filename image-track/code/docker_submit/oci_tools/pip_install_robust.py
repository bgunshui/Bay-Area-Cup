#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pip_install_robust.py —— 用冻结版本装依赖，**装不上的包自动解除版本锁定**。

背景（2026-09-16 实测）：
    我从训练实例的 `pip list` 冻结了版本，但**镜像上不一定有**：
        `scipy==1.17.1` → 阿里云 pypi 最高只到 1.16.3 ⇒ pip 直接失败
    ⚠️ 我第一次想"自己解析索引 HTML 来查版本"，**结果解析器本身不可靠**
       （把 scipy 判成"可用"，还把 scikit-learn 解析出个 `1.4.0rc1` 的 RC 版）。
    ⇒ **不要自己猜版本，让 pip 自己说话**：跑 `--dry-run`，从它的报错里精确取出
      "Could not find a version" 的包名，**只给这些包解锁**，其余保持冻结，迭代重试。

为什么可以解锁：版本敏感性只对 `peft` / `transformers` 有一点（LoRA 键名层级），
    而我们的 `load_lora_into()` 已经对 `base_model.model.` / `vision_model.` / `.default`
    做归一化，**对 peft 版本差异是健壮的**；其余包（scipy/sklearn/pandas/pillow/fastapi/uvicorn）
    对我们的代码完全版本无关。
"""
import re
import subprocess
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
TARGET = sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/build/work/piproot"
PYVER = sys.argv[2] if len(sys.argv) > 2 else "3.11"
ABI = "cp" + PYVER.replace(".", "")
INDEX = "https://mirrors.aliyun.com/pypi/simple/"

FROZEN = [
    # ── 关键（影响权重加载/推理语义）—— **必须钉住** ──
    "transformers==5.17.0", "tokenizers==0.23.2", "peft==0.20.0",
    "accelerate==1.15.0", "huggingface_hub==1.31.0", "safetensors==0.8.0",
    "open_clip_torch==3.3.0", "regex==2026.9.10", "numpy==2.2.4",
    # ── 非关键（对我们的代码完全版本无关，且实测镜像上没有冻结版本） ──
    # ⚠️ 2026-09-16：`scipy==1.17.1` / `pandas==3.0.5` / `scikit-learn==1.9.1`
    #    在阿里云 pypi 上**都不存在**（那台训练机的这几个包是从别的源装的）。
    #    迭代解锁能把它们找出来，但每轮 dry-run 都要重新解析整张依赖图（实测第 4 轮卡了 7 分钟），
    #    ⇒ 干脆**从一开始就不锁定**，让 pip 一次选好。
    "scipy", "pandas", "scikit-learn", "pillow==11.0.0",
    "fastapi==0.115.6", "uvicorn[standard]==0.34.0",
]


def base_specs():
    spec = [sys.executable, "-m", "pip", "install", "-q",
            "--python-version", PYVER, "--implementation", "cp", "--abi", ABI,
            "--platform", "manylinux2014_x86_64", "--only-binary=:all:", "-i", INDEX]
    return spec


specs = list(FROZEN)

for it in range(1, 9):
    print(f"\n===== 第 {it} 轮解析（dry-run） =====", flush=True)
    cmd = base_specs() + ["--dry-run", "--target", TARGET, "--upgrade"] + specs
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode == 0:
        print("  ✅ 依赖图可解，进入正式安装")
        break
    # 精确取出 pip 报"找不到版本"的包名
    missing = set(re.findall(r"Could not find a version that satisfies the requirement "
                             r"([A-Za-z0-9_.\-]+)==", r.stdout + r.stderr))
    missing |= set(re.findall(r"No matching distribution found for ([A-Za-z0-9_.\-]+)==",
                              r.stdout + r.stderr))
    print(f"  镜像上没有的包：{sorted(missing) if missing else '（未能从报错里解析出来）'}")
    if not missing:
        print("  ❌ 无法自动定位，打印 pip 输出尾部：")
        print("\n".join((r.stdout + r.stderr).splitlines()[-25:]))
        sys.exit(2)
    changed = False
    for pkg in missing:
        for i, s in enumerate(specs):
            name = re.split(r"[=<>!\[]", s)[0].replace("_", "-").lower()
            if name == pkg.replace("_", "-").lower():
                specs[i] = re.split(r"==", s)[0]      # 只解锁这一个
                print(f"  → 解除锁定：{s}  ⇒  {specs[i]}（让 pip 自选可用版本）")
                changed = True
    if not changed:
        sys.exit("  ❌ 解锁后规格没变，放弃")

print("\n===== 正式安装 =====")
print("  最终规格：")
for s in specs:
    print(f"    {s}")
cmd = base_specs() + ["--target", TARGET, "--upgrade"] + specs
r = subprocess.run(cmd)
if r.returncode != 0:
    sys.exit(f"❌ 安装失败 rc={r.returncode}")

with open("/hy-tmp/build/final_specs.txt", "w") as f:
    f.write("\n".join(specs) + "\n")
print(f"\n✅ 安装完成；实际规格写入 /hy-tmp/build/final_specs.txt")
print("  （若某包被解锁，实际装入的版本见下面的 pip freeze）")
subprocess.run([sys.executable, "-m", "pip", "list", "--path", TARGET])
