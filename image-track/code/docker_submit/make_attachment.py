#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""make_attachment.py —— 把全部交付材料打成一个压缩包：**图像附件资料.tar.gz**

用户要求（2026-09-16）：「一整份资料要的是压缩包，命名图像附件资料」。

结构（解压后）：
    图像附件资料/
    ├── 00_交付说明.md                    ← 一页纸：对应哪两项要求、怎么用
    ├── MANIFEST.md                       ← 全部文件字节数 + sha256 台账
    ├── Docker镜像/
    │   └── baycup_image_aidet_0.918887.tar
    ├── 技术方案报告/
    │   ├── AS2_技术方案_TECHNICAL_REPORT.md
    │   └── AS3_提交说明_README.md
    └── 构建与复现套件/
        ├── AS1_构建套件_buildkit.tar.gz
        ├── docker_submit/…              ← 代码 + 8 权重 + 查表 + OCI 工具
        └── <10 个冻结 CSV>              ← 放这里是为了 `verify_reproduce.py`
                                            （它取 `WS = docker_submit/ 的上一级`）

⚠️ 校验纪律：先压到临时文件 → `tar -tzf` 能列出内容 → 再改名到位。
   绝不直接覆盖成品（避免"压到一半失败却留下一个看起来正常的包"）。
"""
import hashlib
import os
import shutil
import subprocess
import sys
import tarfile
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

DESK = Path(r"C:\Users\38232\Desktop\湾区杯_提交资料")
DIST = Path(r"D:\湾区杯\图像赛道\docker_submit\dist")
SRC = Path(r"D:\湾区杯\图像赛道\docker_submit")
LB = "0.918887"
TOP = "图像附件资料"
OUT = DESK / f"{TOP}.tar.gz"
IMG = DESK / "baycup_image_aidet_0.918887.tar"

REPRO_CSVS = [
    "v100_gpp20_seed4_c75_prior200.csv",
    "img_cliph_rerun.csv", "img_cliph_s20261101.csv",
    "img_cliph_s20261102.csv", "img_cliph_s20261103.csv",
    "img_gpp_siglip2.csv", "img_bigg_rerun2.csv", "img_bigg_rerun3.csv",
    "img_bigg_s384_retrain.csv",
    "test_specs.csv",
]


def sha256(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(blk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def add_tree(tf, root, arc, exclude=()):
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(x in rel.parts for x in exclude):
            continue
        if "__pycache__" in rel.parts or p.suffix in (".pyc",):
            continue
        # ⚠️ 运行期残留：`verify_reproduce.py` 跑完会 `np.save(HERE/"_verify_from_csv.npy")`，
        #    `--http` 模式还会写 `_verify_http.npy`。它们**只写不读**（脚本里只有 np.save，没有 np.load），
        #    不影响复核正确性，但**不该进交付包**（2026-09-19 审计时才发现包里混进了一个 156 KB 的）。
        if p.suffix == ".npy" and p.name.startswith("_verify"):
            print(f"  ⏭ 跳过运行期残留 {rel.as_posix()}")
            continue
        tf.add(p, arcname=f"{arc}/{rel.as_posix()}")
        n += 1
    return n


# ⚠️ 必须排除的目录（2026-09-19 查出的**重复交付**）：
#   `dist/` 里放着 AS1/AS2/AS3 三个套件包与旧总包 —— 而它们在本包里
#   **已经按 `技术方案报告/`、`构建与复现套件/` 各放了一份**。
#   不排除的话 `add_tree` 会把 901 MB 的 AS1 再塞一遍、把 898 MB 的旧总包整个塞进来
#   （旧总包自己还含一份 docker_submit）⇒ 包内凭空多出约 2.7 GB 的纯重复。
EXCLUDE_DIRS = ("dist",)


def main():
    for p in (IMG,):
        if not p.exists():
            raise SystemExit(f"❌ 缺少 {p}（镜像是否还没下载完？）")
    print(f"镜像 {IMG.name}  {IMG.stat().st_size/1e9:.2f} GB")
    print(f"  先核对 sha256（约 1 分钟）…", flush=True)
    h = sha256(IMG)
    # 期望值从 `docker_submit/dist/IMAGE_SHA256.txt` 读（由 _patch_image.py 在打完补丁层后写出），
    # 读不到才用下面的常量。这样"换过镜像"时不会忘改常量而误判。
    expfile = DIST / "IMAGE_SHA256.txt"
    exp = "47fc14e74f273e9091eb54a71913168f521958595dd1a6cc62f4b3c6f68a0871"
    if expfile.exists():
        exp = expfile.read_text(encoding="utf-8").split()[0].strip()
        print(f"  （期望值取自 {expfile.name}）")
    print(f"  sha256 = {h}")
    if h != exp:
        raise SystemExit(f"❌ 镜像 sha256 与远端不一致！期望 {exp}")
    print("  ✅ 与远端一致")

    tmp = DESK / f"{TOP}.tar.gz.part"
    tmp.unlink(missing_ok=True)
    t0 = time.time()
    with tarfile.open(tmp, "w:gz", compresslevel=1) as tf:
        pre = f"{TOP}"
        for f in ("00_交付说明.md", "MANIFEST.md"):
            src = SRC / f if (SRC / f).exists() else DIST / f
            if src.exists():
                tf.add(src, arcname=f"{pre}/{f}")
        tf.add(IMG, arcname=f"{pre}/Docker镜像/{IMG.name}")
        for f in ("AS2_技术方案_TECHNICAL_REPORT.md", "AS3_提交说明_README.md"):
            tf.add(DIST / f, arcname=f"{pre}/技术方案报告/{f}")
        # ⚠️ 不再单独放 `AS1_构建套件_buildkit.tar.gz`（901 MB）。
        #   它就是"docker_submit/ + 10 个 CSV"的压缩版，而这两样在本包里
        #   **已经原样解开着**（下面 add_tree 与 REPRO_CSVS）⇒ 放进来是纯重复，
        #   而且它还是在文档修订**之前**打的，内容已经过期。
        #   需要那个 tar.gz 随时可以重新生成：
        #     tar czf AS1_构建套件_buildkit.tar.gz docker_submit <10 个 CSV>
        n = add_tree(tf, SRC, f"{pre}/构建与复现套件/docker_submit", EXCLUDE_DIRS)
        m = 0
        for c in REPRO_CSVS:
            p = SRC.parent / c
            if p.exists():
                tf.add(p, arcname=f"{pre}/构建与复现套件/{c}")
                m += 1
        print(f"  已写入：docker_submit {n} 个文件，复现 CSV {m}/{len(REPRO_CSVS)} 个")
    el = time.time() - t0
    print(f"  压缩完成 {tmp.stat().st_size/1e9:.2f} GB  用时 {el/60:.1f} 分钟", flush=True)

    # 校验：能列出内容才算成功
    r = subprocess.run(["tar", "-tzf", str(tmp)], capture_output=True, text=True)
    if r.returncode != 0:
        raise SystemExit(f"❌ 压缩包不可读：{r.stderr[:300]}")
    names = r.stdout.splitlines()
    need = [f"{TOP}/Docker镜像/baycup_image_aidet_{LB}.tar",
            f"{TOP}/MANIFEST.md", f"{TOP}/00_交付说明.md"]
    miss = [x for x in need if x not in names]
    if miss:
        raise SystemExit(f"❌ 包内缺：{miss}")
    print(f"  ✅ 可读，共 {len(names)} 个条目，关键项齐全")

    OUT.unlink(missing_ok=True)
    os.replace(tmp, OUT)
    print(f"\n✅ {OUT}  {OUT.stat().st_size/1e9:.2f} GB")
    hs = sha256(OUT)
    print(f"   sha256 = {hs}")
    # 校验文件放在**包外面**（包自己的摘要不能写进包里 —— 那会自指，且改一个字节就全变）
    side = DESK / f"{TOP}.tar.gz.sha256.txt"
    side.write_text(
        f"{hs}  {OUT.name}\n"
        f"bytes: {OUT.stat().st_size}\n"
        f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"校验: certutil -hashfile {OUT.name} SHA256\n",
        encoding="utf-8")
    print(f"   校验文件 {side.name}")
    print(f"   解压：tar -xzf {TOP}.tar.gz")


if __name__ == "__main__":
    main()
