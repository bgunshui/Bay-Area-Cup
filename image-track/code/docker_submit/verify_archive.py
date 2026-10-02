#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_archive.py —— 从**成品压缩包本身**验证它是否自足。

判据（全过才算通过）：
  1. 包内必需条目齐全（镜像 / 报告 / 说明 / MANIFEST / 构建套件 / docker_submit / 10 个 CSV）
  2. 能把非镜像部分解压出来（跳过 18.98 GB 的镜像，省时间省磁盘）
  3. 在解压出来的树里跑 `verify_reproduce.py --from-csv`，**必须输出 spearman = 1.0000000000**

为什么要做第 3 步：前两次交付都栽在"包看起来没问题、其实解压后跑不起来"
（漏了 10 个复现 CSV / 总包是"包中包"）。**必须用成品本身验，不能用源目录验。**
"""
import os
import re
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
ARCH = Path(r"C:\Users\38232\Desktop\湾区杯_提交资料\图像附件资料.tar.gz")
DEST = Path(r"D:\湾区杯\图像赛道\_verify_extract")
TOP = "图像附件资料"

REQUIRED = [
    f"{TOP}/00_交付说明.md",
    f"{TOP}/MANIFEST.md",
    f"{TOP}/Docker镜像/baycup_image_aidet_0.918887.tar",
    f"{TOP}/技术方案报告/AS2_技术方案_TECHNICAL_REPORT.md",
    f"{TOP}/技术方案报告/AS3_提交说明_README.md",
    # ⚠️ 2026-09-19：`AS1_构建套件_buildkit.tar.gz` **已不再收录**。
    #    它就是"docker_submit/ + 10 个 CSV"的压缩版，而这两样本包里已经原样解开着
    #    ⇒ 放进来是 901 MB 的纯重复（用户要求"有重复就删掉"）。
    #    这条检查清单**必须跟着包一起改** —— 否则清单会变成一个"永远失败"的假告警，
    #    这正是 AGENTS.md §0.4 记的那类"监控自己坏了、还安静地说错话"。
    f"{TOP}/构建与复现套件/docker_submit/app/models.py",
    f"{TOP}/构建与复现套件/docker_submit/app/fusion.py",
    f"{TOP}/构建与复现套件/docker_submit/app/server.py",
    f"{TOP}/构建与复现套件/docker_submit/verify_reproduce.py",
    f"{TOP}/构建与复现套件/docker_submit/verify_container.sh",
    f"{TOP}/构建与复现套件/docker_submit/pip_freeze_image.txt",
    f"{TOP}/构建与复现套件/docker_submit/app/tables/fusion_meta.json",
    f"{TOP}/构建与复现套件/v100_gpp20_seed4_c75_prior200.csv",
    f"{TOP}/构建与复现套件/img_cliph_rerun.csv",
    f"{TOP}/构建与复现套件/img_cliph_s20261101.csv",
    f"{TOP}/构建与复现套件/img_cliph_s20261102.csv",
    f"{TOP}/构建与复现套件/img_cliph_s20261103.csv",
    f"{TOP}/构建与复现套件/img_gpp_siglip2.csv",
    f"{TOP}/构建与复现套件/img_bigg_rerun2.csv",
    f"{TOP}/构建与复现套件/img_bigg_rerun3.csv",
    f"{TOP}/构建与复现套件/img_bigg_s384_retrain.csv",
    f"{TOP}/构建与复现套件/test_specs.csv",
]
# 明确**不该**出现的条目（重复交付）
FORBIDDEN = [
    f"{TOP}/构建与复现套件/AS1_构建套件_buildkit.tar.gz",
    f"{TOP}/构建与复现套件/docker_submit/dist",
]
WEIGHTS = [f"{TOP}/构建与复现套件/docker_submit/app/weights/{n}" for n in (
    "cliph_rerun.pt", "cliph_s20261101.pt", "cliph_s20261102.pt", "cliph_s20261103.pt",
    "gpp.pt", "bigg_rerun2.pt", "bigg_rerun3.pt", "bigg_s384_retrain.pt")]


def main():
    print(f"压缩包 {ARCH.name}  {ARCH.stat().st_size/1e9:.2f} GB")
    t = tarfile.open(ARCH)
    names = set(t.getnames())
    print(f"  条目 {len(names)} 个")

    miss = [x for x in REQUIRED + WEIGHTS if x not in names]
    if miss:
        print("  ❌ 包内缺以下条目：")
        for m in miss:
            print("     ", m)
        return 1
    print(f"  ✅ 必需条目齐全（{len(REQUIRED)} 项 + {len(WEIGHTS)} 个权重）")

    bad = [x for x in FORBIDDEN if any(n == x or n.startswith(x + "/") for n in names)]
    if bad:
        print("  ❌ 包里出现了应当被排除的重复条目：")
        for b in bad:
            print("     ", b)
        return 1
    print("  ✅ 无重复条目（AS1 / dist 均已排除）")

    if DEST.exists():
        shutil.rmtree(DEST)
    DEST.mkdir(parents=True)
    print(f"  解压非镜像部分 → {DEST}（跳过 18.98 GB 的镜像）…", flush=True)
    n = 0
    for m in t:
        if not m.isfile():
            continue
        if "/Docker镜像/" in m.name:
            continue
        t.extract(m, DEST)
        n += 1
    t.close()
    print(f"  解压 {n} 个文件")

    work = DEST / TOP / "构建与复现套件"
    cmd = [sys.executable, "docker_submit/verify_reproduce.py", "--from-csv"]
    print(f"  在 {work} 里跑：{' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=str(work), capture_output=True, text=True, encoding="utf-8",
                       errors="replace")
    tail = (r.stdout or "").strip().splitlines()[-6:]
    for line in tail:
        print("     " + line)
    if r.returncode != 0:
        print("  ❌ verify_reproduce 退出码非 0")
        print((r.stderr or "")[-800:])
        return 1
    m = re.search(r"spearman\s*=\s*([0-9.]+)", r.stdout or "")
    if not m:
        print("  ❌ 输出里找不到 spearman")
        return 1
    val = float(m.group(1))
    ok = abs(val - 1.0) < 1e-9
    print(f"  spearman = {val:.10f}  →  {'✅ 通过（包自足）' if ok else '❌ 不达标'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
