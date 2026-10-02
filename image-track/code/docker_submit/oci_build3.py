#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oci_build3.py —— OCI 手法组装镜像（pytorch 官方基座 + **零解包** + 硬链接）。

相对 v2 的三处磁盘优化（gpu4090b 的 /hy-tmp 只有 43 GB 可用，必须省）：
    1. **不解包 base 的 rootfs**（省 20 GB）：
       只需要 (a) `etc/passwd` / `etc/group` 用来加 appuser —— 用 `tar -xOf` 单文件抽取；
              (b) 探测 site-packages 路径 —— 用 `tar -tf` 列目录名。
    2. app → layer2 用 **硬链接**（`cp -al`）而非拷贝（省 10 GB）。
    3. 做完 layer2.tar 后立刻删掉 app 源与 buildkit.tar.gz，再导出（再省 11 GB）。

峰值占用估算：base 压缩层 7 + app 10.3 + layer2.tar 10 = 27 GB；导出 tar 约 30 GB。
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path("/hy-tmp/build")
BASE_OCI = R / "base_oci_pt"
WORK = R / "work"
LAYER2 = WORK / "layer2"
FINAL = R / "final_oci"
ALIYUN = "https://mirrors.aliyun.com/pypi/simple/"

PKGS = [
    "transformers==5.17.0", "tokenizers==0.23.2", "peft==0.20.0",
    "accelerate==1.15.0", "huggingface_hub==1.31.0", "safetensors==0.8.0",
    "open_clip_torch==3.3.0", "regex==2026.9.10",
    "numpy==2.2.4", "scipy==1.17.1", "pandas==3.0.5",
    "scikit-learn==1.9.1", "pillow==11.0.0",
    "fastapi==0.115.6", "uvicorn[standard]==0.34.0",
]


def step(n, t):
    print(f"\n{'='*72}\n【{n}】{t}\n{'='*72}", flush=True)


def run(cmd, **kw):
    print("  $ " + (" ".join(cmd[:10]) + " …" if len(cmd) > 10 else " ".join(cmd)), flush=True)
    return subprocess.run(cmd, **kw)


def df():
    o = subprocess.run(["df", "-h", "/hy-tmp"], capture_output=True, text=True).stdout
    print("  [磁盘] " + o.strip().splitlines()[-1], flush=True)


def load_oci(d):
    idx = json.loads((d / "index.json").read_text())
    mf = json.loads((d / "blobs/sha256" / idx["manifests"][0]["digest"].split(":")[1]).read_text())
    cfg = json.loads((d / "blobs/sha256" / mf["config"]["digest"].split(":")[1]).read_text())
    return mf, cfg


def tar_of(blob, mt):
    return ["tar", ("-tzf" if "gzip" in mt else "-tf"), str(blob)]


def main():
    t0 = time.time()
    if not (BASE_OCI / "index.json").exists():
        sys.exit(f"❌ 缺 {BASE_OCI}")
    src = R / "buildkit/docker_submit"
    if not (src / "app").exists():
        sys.exit(f"❌ 缺 {src}/app —— 先解包 buildkit")
    mf, cfg = load_oci(BASE_OCI)
    print(f"基座 {len(mf['layers'])} 层；app 源 {src/'app'}")

    # ---------- 1 ----------
    step(1, "从 base 层里**只抽两个文件**（不解包 rootfs）")
    pw = gr = None
    for L in mf["layers"]:
        blob = BASE_OCI / "blobs/sha256" / L["digest"].split(":")[1]
        for name in ("etc/passwd", "etc/group"):
            if (name == "etc/passwd" and pw is not None) or (name == "etc/group" and gr is not None):
                continue
            r = subprocess.run(["tar", "-xzOf", str(blob), name], capture_output=True)
            if r.returncode == 0 and r.stdout:
                if name == "etc/passwd":
                    pw = r.stdout.decode("utf-8", "replace")
                else:
                    gr = r.stdout.decode("utf-8", "replace")
    if pw is None:
        pw = "root:x:0:0:root:/root:/bin/bash\n"
    if gr is None:
        gr = "root:x:0:\n"
    print(f"  etc/passwd {len(pw)} B；etc/group {len(gr)} B")

    # ---------- 2 ----------
    step(2, "用 `tar -tf` 探测 site-packages（官方镜像在 conda 里）")
    cands = set()
    for L in mf["layers"]:
        blob = BASE_OCI / "blobs/sha256" / L["digest"].split(":")[1]
        r = subprocess.run(tar_of(blob, L["mediaType"]), capture_output=True, text=True)
        for line in r.stdout.splitlines():
            m = re.search(r"((?:opt/conda|usr/local|usr)/lib/python3\.\d+/site-packages)/", line)
            if m:
                cands.add(m.group(1))
    if not cands:
        sys.exit("❌ 探测不到 site-packages")
    # 挑含 torch 的那个
    pick = None
    for c in sorted(cands, key=len):
        for L in mf["layers"]:
            blob = BASE_OCI / "blobs/sha256" / L["digest"].split(":")[1]
            r = subprocess.run(tar_of(blob, L["mediaType"]), capture_output=True, text=True)
            if f"{c}/torch/version.py" in r.stdout:
                pick = c
                break
        if pick:
            break
    pick = pick or sorted(cands, key=len)[0]
    pyver = re.search(r"python(3\.\d+)", pick).group(1)
    print(f"  ✅ site-packages = {pick}   python {pyver}")
    print(f"  候选：{sorted(cands)}")

    # ---------- 3 ----------
    step(3, f"pip 补装依赖（{pick}）")
    piproot = WORK / "piproot"
    # ⚠️ 2026-09-16 加的：**piproot 已经装好就直接复用**。
    #    起因：第一次跑到这一步时 pip 因 `scipy==1.17.1` 在镜像上不存在而失败，
    #    修好后如果整条流水线重跑，会把这十几分钟的安装白做一遍。
    #    判据用"dist-info 数量 ≥ 10"，避免半装状态被误当成完成。
    n_exist = sum(1 for _ in piproot.rglob("*.dist-info")) if piproot.exists() else 0
    if n_exist >= 10:
        print(f"  ⏩ piproot 已有 {n_exist} 个包，跳过 pip 安装", flush=True)
    else:
        shutil.rmtree(piproot, ignore_errors=True)
        piproot.mkdir(parents=True, exist_ok=True)
        # ⚠️ 2026-09-16 改：**优先用 `pip_explicit.py`**（`--no-deps` + 42 个显式包，秒级）。
        #   原先优先的 `pip_install_robust.py`（dry-run 迭代）实测**21 分钟还停在第一轮**：
        #   带 `--platform manylinux2014_x86_64 --only-binary=:all:` 时 pip 要为回溯拉全部候选元数据，
        #   而且 `open_clip_torch` 依赖 `torch` ⇒ 还会把 PyPI 的 torch（~2.5 GB + nvidia 依赖）
        #   一起拖下来，**覆盖官方基座自带的那份**。
        expl = R / "pip_explicit.py"
        if expl.exists():
            r = run([sys.executable, str(expl), str(piproot), pyver])
        elif (R / "pip_install_robust.py").exists():
            r = run([sys.executable, str(R / "pip_install_robust.py"), str(piproot), pyver])
        else:
            abi = "cp" + pyver.replace(".", "")
            r = run([sys.executable, "-m", "pip", "install", "-q", "--target", str(piproot),
                     "--upgrade", "--python-version", pyver, "--implementation", "cp", "--abi", abi,
                     "--platform", "manylinux2014_x86_64", "--only-binary=:all:", "-i", ALIYUN] + PKGS)
        if r.returncode != 0:
            sys.exit("❌ pip 失败")

        # ⚠️ **把 scikit-learn 钉到 1.6.1**：`app/tables/gbm.pkl` 是 sklearn **1.6.1** 存的，
        #   装更新的版本会在 `Fusion.__init__` 打 `InconsistentVersionWarning`。
        #   实测 1.6.1 与 1.7.2 的 `predict_proba` 在**同一 numpy 下逐位相同**（见技术报告 §9），
        #   所以钉版本只为"消除跨版本告警 + 与冻结产物完全同源"，不改变任何数值。
        r = run([sys.executable, "-m", "pip", "install", "-q", "--no-deps", "--upgrade",
                 "--target", str(piproot), "-i", ALIYUN,
                 "scikit-learn==1.6.1", "joblib==1.4.2"])
        if r.returncode != 0:
            sys.exit("❌ scikit-learn 钉版本失败")
    n = sum(1 for _ in piproot.rglob("*.dist-info"))
    sz = sum(f.stat().st_size for f in piproot.rglob("*") if f.is_file())
    print(f"  ✅ {n} 个包  {sz/1e9:.2f} GB")
    df()

    # ---------- 3.5 融合表自检 ----------
    step(3.5, "融合表自检（用**容器那套包**跑，能解开 gbm.pkl 并算出一个合法分数）")
    # ⚠️ 2026-09-16 修正：**必须用 piproot 那套包来跑自检，不能用宿主 python**。
    #   起因：宿主 python3 装的是 scikit-learn 1.9.1，它反序列化 1.6.1 存的 `gbm.pkl` 时
    #   抛 `ModuleNotFoundError: No module named '_loss'`（1.9 重排了 sklearn._loss 的内部模块名），
    #   于是自检**报了一个容器里根本不会发生的假失败**。
    #   ——教训：自检的判据必须复刻运行环境，否则它测的是"宿主机"，不是"镜像"。
    #   容器里 site-packages = 基座自带的 + 我们这一层（piproot 合并进去），
    #   所以这里用 `PYTHONPATH=piproot` 近似替身；numpy 这类基座包用宿主的顶上。
    CHK = r'''
import sys, warnings
warnings.simplefilter("ignore")
sys.path.insert(0, SRC_APP)
import numpy as np
from fusion import Fusion
F = Fusion(TABLES)
keys = [k for k, _ in F.terms]
assert len(keys) >= 8, "分量项太少：%d" % len(keys)
dummy = {k: 0.5 for k in keys}
sc = F.fuse(dummy, 1024, 768, 200000)
assert 0.0 <= sc <= 1.0, "分数越界 %r" % sc
# 再验两个极端尺寸/字节数，确认先验那条路（gbm + bucket + 三个 ECDF）能走通
for (w, h, nb) in ((1024, 1024, 300000), (256, 256, 9000), (1920, 1080, 900000)):
    v = F.fuse(dummy, w, h, nb)
    assert 0.0 <= v <= 1.0, "尺寸 %dx%d 分数越界 %r" % (w, h, v)
import sklearn
print("  sklearn %s | numpy %s | python %s" % (
    sklearn.__version__, np.__version__, ".".join(map(str, sys.version_info[:3]))))
print("  ✅ 融合表可用：%d 个分量项，prior_w=%s，全 0.5 输入的分数 = %.6f"
      % (len(keys), F.prior_w, sc))
'''
    chk = CHK.replace("SRC_APP", repr(str(src / "app"))).replace(
        "TABLES", repr(str(src / "app" / "tables")))
    env2 = dict(os.environ, PYTHONPATH=str(piproot))
    r = subprocess.run([sys.executable, "-c", chk], capture_output=True, text=True, env=env2)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        sys.stderr.write(r.stderr[-3000:])
        sys.exit("❌ 融合表自检失败（容器环境替身）")

    # ---------- 3.7 ----------
    # ⚠️⚠️ 2026-09-16 **必须做的一步**：剪掉与基座 site-packages 重名的顶层项。
    #   原因：我们的新层会被合并到**同一个** `opt/conda/lib/python3.11/site-packages`。
    #   而 `open_clip_torch` 依赖 `torch`、`transformers` 依赖 `numpy` 等 —— pip 会把
    #   **PyPI 版的 torch/torchvision/nvidia** 一起装进 piproot，合并后**覆盖官方基座自带的那份**
    #   ⇒ 违背"用 pytorch 官方镜像"的初衷，且可能引入不匹配的 CUDA 运行时。
    #   规则：**凡基座已有的顶层名，一律从我们的层里删掉**（基座永远优先）。
    step(3.7, "剪掉与基座重名的顶层项（绝不覆盖官方镜像的 torch）")
    base_names = set()
    for L in mf["layers"]:
        blob = BASE_OCI / "blobs/sha256" / L["digest"].split(":")[1]
        flag = "-tzf" if "gzip" in L["mediaType"] else "-tf"
        r = subprocess.run(["tar", flag, str(blob)], capture_output=True, text=True)
        pre = pick + "/"
        for line in r.stdout.splitlines():
            if line.startswith(pre):
                rest = line[len(pre):]
                if rest:
                    base_names.add(rest.split("/")[0])
    print(f"  基座 site-packages 顶层名 {len(base_names)} 个")

    # ⚠️⚠️ 2026-09-16 修复的**交付级 bug**（第 2 个"不报错的失败"）：
    #   "凡基座已有的顶层名一律删掉"这条规则**对纯 Python 工具包是错的** ——
    #   基座是 2024-12 构建的，我们装的是 2026 年的版本；有些包**新版本是硬需求**。
    #   实证：基座的 `typing_extensions.py` 里只有 `class _Sentinel`（**没有**公开的 `Sentinel`，
    #   那是 typing_extensions 4.12 才加的），而 piproot 里 `pydantic_core` 的
    #   `__init__.py` 第一行就是 `from typing_extensions import Sentinel` ⇒
    #   **容器里 `import fastapi` 直接 ImportError，服务根本起不来**。
    #   （真跑才会暴露：宿主机上 import 成功是因为宿主机另有一份旧 typing_extensions，
    #     直到把 piproot 提前到 PYTHONPATH 才炸出来。）
    #   ⇒ 解法：给"必须让我们这层赢"的包开白名单，**永不剪除**。
    PRUNE_KEEP = {
        "typing_extensions.py", "typing_extensions",
        "typing_extensions-*.dist-info",
    }
    removed = []
    for entry in list(piproot.iterdir()):
        nm = entry.name
        if nm in PRUNE_KEEP:
            print(f"  ⏭ 保留（白名单）{nm}")
            continue
        # dist-info 是 pip 元数据，重名也删（否则 pip 以为装了却没文件）
        if nm in base_names or nm.split("-")[0].lower().replace("_", "-") in \
                {b.lower().replace("_", "-") for b in base_names}:
            removed.append((nm, sum(f.stat().st_size for f in entry.rglob("*")
                                    if f.is_file()) if entry.is_dir() else entry.stat().st_size))
            if entry.is_dir():
                shutil.rmtree(entry, ignore_errors=True)
            else:
                entry.unlink(missing_ok=True)
    removed.sort(key=lambda x: -x[1])
    if removed:
        print(f"  剪掉 {len(removed)} 项（合计 {sum(s for _, s in removed)/1e9:.2f} GB）：")
        for nm, s in removed[:15]:
            print(f"    - {nm:40s} {s/1e6:9.1f} MB")
    else:
        print("  无需剪除")

    # ---- 硬断言：白名单里的包必须真的还在，且**确实能用**（专门防上面这个 bug 复活）----
    #   ⚠️ 判据用**真的 import 一次**，不要用正则去匹配源码形态 ——
    #   第一版写成 `re.search(r"^\s*class\s+Sentinel\b")`，结果在 typing_extensions **4.16.0**
    #   上误报：新版是 `Sentinel = sentinel`（赋值），不是 `class Sentinel`。
    #   而 `pydantic_core` 真正做的是 `from typing_extensions import Sentinel` ⇒ 那就测这个。
    te = piproot / "typing_extensions.py"
    if not te.exists():
        sys.exit("❌ typing_extensions 不在我们层里 —— 容器里 pydantic_core 会 ImportError")
    chk = (f"import sys; sys.path.insert(0, {str(piproot)!r}); "
           "from typing_extensions import Sentinel as S; print('Sentinel OK ->', S)")
    r = run([sys.executable, "-c", chk])
    if r.returncode != 0:
        sys.exit("❌ 我们层里的 typing_extensions 无法 `from typing_extensions import Sentinel`"
                 " —— pydantic_core 会 ImportError（基座那份太旧，必须用我们的）")
    print(f"  ✅ 断言通过：typing_extensions {te.stat().st_size} B，"
          f"`from typing_extensions import Sentinel` 成功")

    # ---- 硬断言：**依赖必须闭合**（缺任何一个硬依赖 = 容器里 ImportError）----
    #   ⚠️ 这是 2026-09-16 第二次踩同类坑后加的。`pip_explicit.py` 用 `--no-deps` 装包，
    #   依赖列表是人肉展开的；漏掉传递依赖时，**在宿主机上跑验证根本发现不了**
    #   （宿主机恰好也有那份包，import 会静默落到宿主机上）。
    #   实证漏过：timm(open_clip) / httpx+hf-xet(huggingface_hub) / typing-inspection(pydantic)
    #             / typer(transformers) / cloudpickle(joblib)。
    dep = R / "depcheck.py"
    if dep.exists():
        r = run([sys.executable, str(dep), str(piproot)])
        if r.returncode != 0:
            sys.exit("❌ 依赖不闭合 —— 容器里必定 ImportError（清单见 /hy-tmp/build/missing.txt）")
    else:
        print("  ⚠️ 没有 depcheck.py，**跳过依赖闭合检查**"
              "（不建议：--no-deps 装包很容易漏传递依赖）")

    # ---------- 4 ----------
    step(4, "组装 layer2/（app 用**硬链接**，不占额外空间）")
    shutil.rmtree(LAYER2, ignore_errors=True)
    LAYER2.mkdir(parents=True, exist_ok=True)
    r = run(["cp", "-al", str(src / "app"), str(LAYER2 / "app")])
    if r.returncode != 0:
        r = run(["cp", "-a", str(src / "app"), str(LAYER2 / "app")])
    if r.returncode != 0 or not (LAYER2 / "app").is_dir():
        sys.exit("❌ app 源拷贝失败")

    # ⚠️⚠️ 2026-09-16 修复的**致命 bug**（第一次组装时真实发生、且静默通过）：
    #   这里原来直接 `cp -al <骨干> <LAYER2>/app/backbones/<名字>`，**忘了先 mkdir backbones**，
    #   于是三个 `cp` 全部报
    #       cp: cannot create directory '…/layer2/app/backbones/clip_h14_cache': No such file or directory
    #   而**返回值没有被检查** ⇒ 9.3 GB 骨干一个都没进层，镜像照常"成功"产出（层只有 1.47 GB）。
    #   这样的镜像跑起来必然找不到骨干权重。
    #   ——教训同 AGENTS.md §0.9-A：**不报错的失败最阴；每个 cp/mv 都必须看 returncode。**
    bb = R / "backbones"
    if not bb.is_dir():
        sys.exit(f"❌ 找不到骨干目录 {bb}")
    bb_dst = LAYER2 / "app" / "backbones"
    bb_dst.mkdir(parents=True, exist_ok=True)
    for sub in sorted(bb.iterdir()):
        r = run(["cp", "-al", str(sub), str(bb_dst / sub.name)])
        if r.returncode != 0:
            r = run(["cp", "-a", str(sub), str(bb_dst / sub.name)])
        if r.returncode != 0:
            sys.exit(f"❌ 骨干 {sub.name} 拷贝失败")

    # ---- 硬断言：骨干必须**真的**进层（专门防上面那个 bug 复活）----
    src_files = [f for f in bb.rglob("*") if f.is_file()]
    dst_files = [f for f in bb_dst.rglob("*") if f.is_file()]
    sb = sum(f.stat().st_size for f in src_files)
    db = sum(f.stat().st_size for f in dst_files)
    print(f"  骨干校验：{len(dst_files)}/{len(src_files)} 个文件，{db/1e9:.2f}/{sb/1e9:.2f} GB")
    if len(dst_files) != len(src_files) or db != sb:
        sys.exit("❌ 骨干拷贝不全 —— 拒绝继续（否则会产出不可用的镜像）")
    for sub in sorted(bb.iterdir()):
        if not (bb_dst / sub.name).is_dir():
            sys.exit(f"❌ 骨干子目录缺失：{sub.name}")

    dst_site = LAYER2 / pick
    dst_site.parent.mkdir(parents=True, exist_ok=True)
    # ⚠️ 2026-09-16 改：原来这里是 `shutil.move(piproot, dst_site)` —— piproot 被**搬**进层里，
    #   层一删（step 5.5）piproot 就跟着没了，于是**每次重跑都要重装一遍依赖**（上一次就踩了）。
    #   改成硬链接拷贝：几乎不占额外空间，且 piproot 原样保留，重跑直接命中 step 3 的"跳过"分支。
    r = run(["cp", "-al", str(piproot), str(dst_site)])
    if r.returncode != 0:
        print("  ⚠️ 硬链接失败，退回整体搬移（piproot 将不再保留）")
        shutil.move(str(piproot), str(dst_site))
    etc = LAYER2 / "etc"; etc.mkdir(parents=True, exist_ok=True)
    (etc / "passwd").write_text(pw.rstrip("\n") + "\nappuser:x:1000:1000::/home/appuser:/bin/sh\n")
    (etc / "group").write_text(gr.rstrip("\n") + "\nappuser:x:1000:\n")

    # ---- 硬断言：整层不能太小（骨干 9.3 GB + app 0.95 GB + site-packages 0.49 GB）----
    tot = sum(f.stat().st_size for f in LAYER2.rglob("*") if f.is_file())
    print(f"  layer2/ 合计 {tot/1e9:.2f} GB"
          f"（骨干 {db/1e9:.2f} + site-packages + app）")
    if tot < 5e9:
        sys.exit(f"❌ layer2 只有 {tot/1e9:.2f} GB，骨干几乎肯定没进去 —— 拒绝继续")
    df()

    # ---------- 4.5 【2026-09-18 新增闸门】以 **uid 1000（容器运行用户）** 真加载一次模型 ----------
    # ⚠️⚠️ 这道闸门是**一次真实交付失败**换来的：
    #   第一版镜像交付后被反馈「镜像权重崩溃」。根因不在算法、也不在权重内容 ——
    #   `bb_fp16.py` / `fix_bigg_st.py` 生成 `.safetensors` 时的 **umask 是 077**，
    #   于是三个骨干被写成 `-rw------- root:root`；而镜像 config 里 `User=appuser`（uid 1000），
    #   ⇒ 容器里 **PermissionError**，直接崩在"加载骨干"这一步。
    #   （`.bin` 是复制来的、保留了 0644，所以**只有 safetensors 中招**，极其隐蔽。）
    #
    #   而我此前**所有**验证都是在 **root 身份**下做的 ⇒ **永远看不到这个错**。
    #   ⇒ 硬规则：**权限/可读性问题必须在"最终运行身份"下验证，不能以 root 验。**
    step(4.5, "【闸门】以 uid 1000（= 容器里的 appuser）逐个试开层内每个文件")
    # ⚠️⚠️ 这道闸门是**一次真实交付失败**换来的（2026-09-18）：
    #   首版镜像交付后被反馈「**镜像权重崩溃**」。根因不在算法、也不在权重内容 ——
    #   `bb_fp16.py` / `fix_bigg_st.py` 生成 `.safetensors` 时的 **umask=077**，
    #   于是三个骨干被写成 `-rw------- root:root`；而镜像 config 里 `User=appuser`（uid 1000），
    #   ⇒ 容器里 **PermissionError**，直接崩在"加载骨干"那一步。
    #   （`.bin` 是复制来的、保留了 0644，所以**只有 safetensors 中招**，极其隐蔽。）
    #   而我此前**所有**验证都是在 **root 身份**下做的 ⇒ **永远看不到这个错**。
    #   ⇒ 硬规则：**权限/可读性问题必须在"最终运行身份"下验证，不能以 root 验。**
    #
    # 判据为什么是"逐个 open()"而不是"跑一遍模型加载"：
    #   构建 shell 被 cgroup 限死在 **2 GiB**（`/sys/fs/cgroup/memory.max`），
    #   在 CPU 上加载 8 个模型必被 OOM 杀（实测日志只有一个 `Killed`）；
    #   而 appuser 也拿不到 GPU。⇒ 用"逐个试开"来验：**恰好覆盖本次的失败模式，
    #   且比加载模型覆盖得更全**（backbones / weights / tables / site-packages 全扫）。
    if subprocess.run(["id", "1000"], capture_output=True).returncode != 0:
        subprocess.run(["useradd", "-u", "1000", "-M", "-s", "/bin/sh", "appuser"],
                       capture_output=True)
    run(["chmod", "-R", "a+rX", str(LAYER2)])          # 修复动作本身
    probe = WORK / "permcheck.py"
    probe.write_text(
        "import os, sys\n"
        "root = sys.argv[1]\n"
        "bad = []\n"
        "n = 0\n"
        "for dp, dn, fn in os.walk(root):\n"
        "    for f in fn:\n"
        "        p = os.path.join(dp, f)\n"
        "        n += 1\n"
        "        if os.path.islink(p):\n"
        "            continue\n"
        "        try:\n"
        "            with open(p, 'rb') as fh:\n"
        "                fh.read(1)\n"
        "        except Exception as e:\n"
        "            bad.append(f'{type(e).__name__}: {p}')\n"
        "print('SCANNED', n, 'UNREADABLE', len(bad))\n"
        "for b in bad[:20]:\n"
        "    print('  !!', b)\n"
        "sys.exit(1 if bad else 0)\n",
        encoding="utf-8")
    r = run(["su", "-s", "/bin/bash", "appuser", "-c",
             f"PYTHONDONTWRITEBYTECODE=1 python3 {probe} {LAYER2}"])
    if r.returncode != 0:
        sys.exit("❌ 以 uid 1000 有文件读不到 —— 镜像起来就会「权重崩溃」（文件权限）")
    print("  ✅ 非 root 身份下**层内每个文件都可读**")
    df()

    # ---------- 5 ----------
    step(5, "打 layer2.tar 并 gzip")
    lay = WORK / "layer2.tar"
    r = run(["tar", "-C", str(LAYER2), "--numeric-owner", "--owner=0", "--group=0",
             "-cf", str(lay), "."])
    if r.returncode != 0:
        sys.exit("❌ 打层失败")
    # diff_id = **未压缩** tar 的摘要（写进 config.rootfs.diff_ids，不能是压缩后的）
    h = hashlib.sha256()
    with lay.open("rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            h.update(b)
    diff_id = "sha256:" + h.hexdigest()
    print(f"  未压缩 {lay.stat().st_size/1e9:.2f} GB  diff_id={diff_id[:23]}…")

    # ⚠️⚠️ 2026-09-16 修复：**层必须 gzip，且必须标成 OCI 的 gzip 类型**。
    #   第一次导出死在 skopeo：
    #       fatal: creating an updated image manifest: Unknown media type during
    #              manifest conversion: "application/vnd.docker.image.rootfs.diff.tar"
    #   原因：我们把层标成了 **docker** 的"未压缩"类型（`...rootfs.diff.tar`），
    #   却放在 **OCI** 布局里让 skopeo 转成 docker-archive —— skopeo 的转换器不认这个组合。
    #   改法：gzip 后标 `application/vnd.oci.image.layer.v1.tar+gzip`，与基座那 5 层**同类**。
    #   副产品：12.75 GB → ~11 GB，下载也更快。
    gz = WORK / "layer2.tar.gz"
    gz.unlink(missing_ok=True)
    prog = "pigz" if shutil.which("pigz") else "gzip"
    print(f"  用 {prog} 压缩…")
    r = run([prog, "-1", "-f", str(lay)])
    if not gz.exists():
        cand = sorted(WORK.glob("layer2.tar.*"))
        if r.returncode != 0 or not cand:
            sys.exit("❌ 压缩失败")
        gz = cand[0]
    hb = hashlib.sha256()
    with gz.open("rb") as f:
        while True:
            b = f.read(1 << 22)
            if not b:
                break
            hb.update(b)
    blob = hb.hexdigest()
    print(f"  压缩后 {gz.stat().st_size/1e9:.2f} GB  blob={blob[:23]}…")

    step(5.5, "清掉不再需要的中间物（**保留 app 源**，只删 layer2 目录与 buildkit.tar.gz）")
    shutil.rmtree(LAYER2, ignore_errors=True)
    # ⚠️ 2026-09-16：**不再删 `src/app`**。
    #   上一次就是这一行把 app 源（含 8 个权重共 950 MB）删掉的，
    #   结果是"想重跑就得先把 900 MB 重传一遍"（实测上传只有 0.4 MB/s ≈ 37 分钟）。
    #   磁盘够（导出前 76 GB 可用），留着当保险；真要清理也是最后一步的事。
    for f in R.glob("buildkit.tar.gz"):
        f.unlink()
        print(f"  已删 {f}")
    df()

    # ---------- 6 ----------
    step(6, "写 OCI 布局")
    shutil.rmtree(FINAL, ignore_errors=True)
    (FINAL / "blobs/sha256").mkdir(parents=True)
    (FINAL / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    layers = []
    for L in mf["layers"]:
        d = L["digest"].split(":")[1]
        s = BASE_OCI / "blobs/sha256" / d
        t = FINAL / "blobs/sha256" / d
        try:
            os.link(s, t)
        except OSError:
            shutil.copy2(s, t)
        layers.append(L)
    shutil.move(str(gz), str(FINAL / "blobs/sha256" / blob))
    layers.append({"mediaType": "application/vnd.oci.image.layer.v1.tar+gzip",
                   "size": (FINAL / "blobs/sha256" / blob).stat().st_size,
                   "digest": "sha256:" + blob})
    nc = json.loads(json.dumps(cfg))
    c = nc.setdefault("config", {})
    keep = [e for e in c.get("Env", []) if not e.startswith(
        ("PYTHONUNBUFFERED", "HF_", "TRANSFORMERS_", "TOKENIZERS_", "OMP_", "PYTHONDONTWRITE"))]
    c["Env"] = keep + ["PYTHONUNBUFFERED=1", "PYTHONDONTWRITEBYTECODE=1", "HF_HUB_OFFLINE=1",
                       "TRANSFORMERS_OFFLINE=1", "HF_HOME=/app/.hf",
                       "TOKENIZERS_PARALLELISM=false", "OMP_NUM_THREADS=8"]
    c["Cmd"] = ["python", "server.py"]
    c["WorkingDir"] = "/app"
    c["ExposedPorts"] = {"8080/tcp": {}}
    c["User"] = "appuser"
    c["Healthcheck"] = {"Test": ["CMD-SHELL",
                                 "curl -f http://localhost:8080/health/liveness || exit 1"],
                        "Interval": 10_000_000_000, "Timeout": 3_000_000_000,
                        "StartPeriod": 300_000_000_000, "Retries": 3}
    nc["rootfs"] = {"type": "layers", "diff_ids": list(cfg["rootfs"]["diff_ids"]) + [diff_id]}
    nc["history"] = list(cfg.get("history", [])) + [{
        "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "created_by": "oci_build3.py (no docker daemon)",
        "comment": "baycup image track on pytorch official base"}]
    nc.pop("container", None)
    cb = json.dumps(nc, separators=(",", ":")).encode()
    cd = hashlib.sha256(cb).hexdigest()
    (FINAL / "blobs/sha256" / cd).write_bytes(cb)
    nmf = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
           "config": {"mediaType": "application/vnd.oci.image.config.v1+json",
                      "size": len(cb), "digest": "sha256:" + cd}, "layers": layers}
    nb = json.dumps(nmf, separators=(",", ":")).encode()
    nd = hashlib.sha256(nb).hexdigest()
    (FINAL / "blobs/sha256" / nd).write_bytes(nb)
    (FINAL / "index.json").write_text(json.dumps({
        "schemaVersion": 2,
        "manifests": [{"mediaType": "application/vnd.oci.image.manifest.v1+json",
                       "size": len(nb), "digest": "sha256:" + nd,
                       "annotations": {"org.opencontainers.image.ref.name": "0.918887"}}]}, indent=2))
    print(f"  config {len(cb)} B  manifest {len(nb)} B  层数 {len(layers)}")
    df()

    # ---------- 7 ----------
    step(7, "OCI → docker-archive（最后一步，产出 ~30 GB tar）")
    out = R / "baycup_image_aidet_0.918887.tar"
    if out.exists():
        out.unlink()
    r = run(["skopeo", "copy", f"oci:{FINAL}:0.918887",
             f"docker-archive:{out}:baycup-image-aidet:0.918887"])
    if r.returncode != 0:
        sys.exit("❌ 导出失败")
    print(f"\n✅ {out}  {out.stat().st_size/1e9:.2f} GB")
    print("   sha256:", subprocess.run(["sha256sum", str(out)], capture_output=True,
                                       text=True).stdout.split()[0])
    df()
    print(f"\n总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
