#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""package_submit.py —— 生成最终提交压缩包。

附件提交要求（用户转述）：
    ① Docker 镜像（按《分类模型 Docker 交付标准 v1.2》）
    ② 技术方案（模型架构 / 训练策略 / 关键实验结果）
    统一打包提交。

产出（`dist/`）：
    AS1_构建套件_buildkit.tar.gz      ← **核心交付物**：Dockerfile + 代码 + 8 个权重 + 查表
                                         （不含 3 个骨干，约 1 GB；骨干由 build.sh 下载或从实例盘拷贝）
    AS2_技术方案_TECHNICAL_REPORT.md
    AS3_提交说明_README.md
    AS4_镜像sha256.txt                 ← 若 dist/ 下已放好镜像 tar，则一并记录
    MANIFEST.md                        ← 全部文件的 sha256 / 字节数
    baycup_attachment_0.918887.tar.gz  ← 以上全部打成一个包

用法：
    python package_submit.py
    python package_submit.py --with-image dist/baycup_image_aidet_0.918887.tar
"""
import argparse
import hashlib
import shutil
import sys
import tarfile
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
DIST = HERE / "dist"
LB = "0.918887"

# ⚠️ 2026-09-16 补：这些 CSV 必须放在 **tar 的根目录**（与 `docker_submit/` 平级）。
#   原因：`verify_reproduce.py` 里 `WS = HERE.parent`（HERE = docker_submit/），
#   它从**上一级**读 `v100_gpp20_seed4_c75_prior200.csv` 与 8 个分量 CSV、`test_specs.csv`。
#   之前打包漏了它们 ⇒ 解压后那条"**不需要 GPU/权重就能复核成绩**"的命令会直接报缺文件，
#   而那恰恰是评委最可能先跑的一条。教训同 §0.9-A：不报错的失败最阴。
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


def add_tree(tf, root, arc_prefix, exclude=()):
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if any(part in exclude for part in rel.parts):
            continue
        if "__pycache__" in rel.parts or p.suffix in (".pyc", ".npy"):
            continue
        tf.add(p, arcname=f"{arc_prefix}/{rel}")
        n += 1
    return n


def add_kit(tf, verbose=False):
    """把构建套件写进 tar：`docker_submit/**` + **tar 根目录**的 10 个复现 CSV。

    ⚠️ 顶层清单是**动态收集**的（凡 .py/.sh/.md + Dockerfile + requirements.txt）。
    原来是一个硬编码的 12 个文件名元组 —— 结果 `oci_build3.py`、`bb_fp16.py`、
    `fix_openclip.py`、`fix_bigg_st.py`、`make_prior.py` 这些后加的脚本**静默漏掉**，
    而打包照样"成功"。（又一次"不报错的失败"。）
    """
    kitfiles = sorted(
        (p for p in HERE.iterdir()
         if p.is_file() and (p.suffix in (".py", ".sh", ".md")
                             or p.name in ("Dockerfile", "requirements.txt"))),
        key=lambda q: q.name)
    for p in kitfiles:
        tf.add(p, arcname=f"docker_submit/{p.name}")
    n = add_tree(tf, HERE / "app", "docker_submit/app", exclude=("backbones",))
    m = add_tree(tf, HERE / "oci_tools", "docker_submit/oci_tools")
    nr = 0
    for nm in REPRO_CSVS:
        p = HERE.parent / nm
        if p.exists():
            tf.add(p, arcname=nm)
            nr += 1
        elif verbose:
            print(f"  ⚠️ 缺复现 CSV：{p}")
    if verbose:
        print(f"  顶层 {len(kitfiles)} 个文件")
        print(f"  app/ 下 {n} 个文件（**排除 backbones**；骨干由 build.sh 下载，"
              f"或 `--skip-download` 时从本地拷贝）")
        if m:
            print(f"  oci_tools/ 下 {m} 个文件"
                  f"（**无 Docker 权限时**的 OCI 组装手法，附证据）")
        print(f"  根目录 {nr}/{len(REPRO_CSVS)} 个复现用 CSV"
              f"（供 `python docker_submit/verify_reproduce.py --from-csv`）")
    return nr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-image", default=None, help="已构建好的镜像 tar 路径")
    ap.add_argument("--image-record", default=None,
                    help="只**登记**镜像的 sha256/字节数（不拷贝、不打进总包）")
    ap.add_argument("--image-sha256", default=None,
                    help="直接给出远端算好的镜像 sha256（连本地文件都不需要）")
    ap.add_argument("--image-size", type=int, default=None, help="镜像字节数")
    ap.add_argument("--image-name", default=f"baycup_image_aidet_{LB}.tar")
    a = ap.parse_args()
    DIST.mkdir(exist_ok=True)

    print("=" * 70)
    print("① 构建套件（buildkit）—— Dockerfile + 代码 + 权重 + 查表")
    print("=" * 70)
    kit = DIST / "AS1_构建套件_buildkit.tar.gz"
    with tarfile.open(kit, "w:gz", compresslevel=3) as tf:
        add_kit(tf, verbose=True)
    print(f"  ✅ {kit.name}  {kit.stat().st_size/1e6:.1f} MB")
    print(f"     sha256 = {sha256(kit)}")

    print()
    print("=" * 70)
    print("② 技术方案与说明")
    print("=" * 70)
    outs = [kit]
    for src, dst in (("TECHNICAL_REPORT.md", f"AS2_技术方案_TECHNICAL_REPORT.md"),
                     ("README.md", "AS3_提交说明_README.md")):
        d = DIST / dst
        shutil.copy2(HERE / src, d)
        print(f"  ✅ {dst}  {d.stat().st_size/1024:.1f} KB")
        outs.append(d)

    img = None
    rec = None
    if a.with_image:
        img = Path(a.with_image)
        if not img.exists():
            raise SystemExit(f"❌ 找不到镜像 {img}")
        d = DIST / img.name
        if img.resolve() != d.resolve():
            print(f"  拷贝镜像（{img.stat().st_size/1e9:.1f} GB，可能要几分钟）…")
            shutil.copy2(img, d)
        print(f"  ✅ {d.name}  {d.stat().st_size/1e9:.2f} GB  sha256={sha256(d)}")
        outs.append(d)
    elif a.image_sha256:
        if not a.image_size:
            raise SystemExit("❌ 给了 --image-sha256 就必须同时给 --image-size")
        print(f"  镜像**单独交付**（登记远端算好的摘要，不重算、不压缩）："
              f"{a.image_name}  {a.image_size/1e9:.2f} GB")
        print(f"     sha256 = {a.image_sha256}")
        rec = (a.image_name, a.image_size, a.image_sha256)
    elif a.image_record:
        # ⚠️ 2026-09-16 加：**只登记、不打包**。
        #   镜像 tar 已经 19 GB，而它的每一层在 OCI 里本来就是 gzip 过的
        #   （只有基座那 5 层被 skopeo 按 docker-archive 规范解压了）。
        #   再把它塞进总包 gzip 一遍，要花几十分钟且几乎压不动 —— 得不偿失。
        #   所以镜像单独交付，这里只把文件名/字节数/sha256 记进 MANIFEST。
        rec = Path(a.image_record)
        if not rec.exists():
            raise SystemExit(f"❌ 找不到镜像 {rec}")
        print(f"  镜像**单独交付**（不重复压缩）：{rec.name}  {rec.stat().st_size/1e9:.2f} GB")
        print(f"     sha256 = {sha256(rec)}")
        rec = (rec.name, rec.stat().st_size, sha256(rec))
    else:
        print("  （未提供 --with-image / --image-record，跳过镜像）")

    # ---- MANIFEST ----
    print()
    print("=" * 70)
    print("③ MANIFEST")
    print("=" * 70)
    lines = ["# 提交材料清单 —— 湾区杯复赛 · 图像赛道", "",
             f"- 对应 LB 成绩：**AUC = {LB}**",
             f"- 提交文件：`v100_gpp20_seed4_c75_prior200.csv`",
             "- 交付标准：《分类模型 Docker 交付标准 v1.2》", "",
             "| 文件 | 字节 | sha256 |", "|---|---|---|"]
    for p in outs:
        lines.append(f"| `{p.name}` | {p.stat().st_size} | `{sha256(p)}` |")
    if rec is not None:
        rn, rs, rh = rec
        lines.append(f"| `{rn}`（**单独交付**，见下） | {rs} | `{rh}` |")
        lines += ["", "## 关于 Docker 镜像为什么单独一个文件", "",
                  f"`{rn}`（{rs/1e9:.2f} GB）是**已构建好的完整镜像**，",
                  "可直接 `docker load -i` 使用。它不塞进总包的原因：",
                  "镜像里的每一层**本来就是 gzip 过的**（基座 5 层 + 我们那层），",
                  "再整体 gzip 一遍要花几十分钟且几乎压不动 —— 只是白费时间。",
                  "", "```bash",
                  f"docker load -i {rn}",
                  "docker run --gpus all -p 8080:8080 baycup-image-aidet:0.918887",
                  "```", ""]
    lines += ["", "## 怎么用（解压 `baycup_attachment_0.918887.tar.gz` 之后）", "",
              "总包是**扁平结构**：`docker_submit/` 与 10 个复现 CSV 都在解压后的**根目录**，",
              "所以下面两条命令**解压一次就能直接跑**。", "",
              "### 不需要 GPU / 不需要权重，复核成绩（**建议先跑这条**）", "",
              "```bash",
              "python docker_submit/verify_reproduce.py --from-csv",
              "# 必须输出 spearman = 1.0000000000（逐图重建 LB 0.918887 的提交文件）",
              "```", "",
              "### 自己构建镜像（linux/amd64 + Docker + 联网）", "",
              "```bash",
              "cd docker_submit",
              "bash build.sh          # 下载 3 个骨干(约12GB) → build → 冒烟 → docker save",
              "```", "",
              "**若 hf-mirror 不通**：先把骨干拷到 `app/backbones/` 再 `bash build.sh --skip-download`：", "",
              "```bash",
              "scp -P <port> -r root@<host>:/hy-tmp/sjmodels/siglip2_384    app/backbones/siglip2_384",
              "scp -P <port> -r root@<host>:/hy-tmp/sjmodels/clip_h14_cache app/backbones/clip_h14_cache",
              "# clip_bigg 需另行取得（HF: laion/CLIP-ViT-bigG-14-laion2B-39B-b160k）",
              "```", "",
              "> `AS1_构建套件_buildkit.tar.gz` 也在 `dist/` 里留了一份**独立的**套件包",
              "> （内容与总包里的 `docker_submit/` + CSV 相同），只是为了单独分发方便；",
              "> 总包**没有**再套一层，避免「包中包」导致解压一次跑不了命令。", ""]
    (DIST / "MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print("  ✅ MANIFEST.md")
    outs.append(DIST / "MANIFEST.md")

    # ---- 总包 ----
    print()
    print("=" * 70)
    print("④ 打成总包")
    print("=" * 70)
    total = DIST / f"baycup_attachment_{LB}.tar.gz"
    with tarfile.open(total, "w:gz") as tf:
        # ⚠️ 2026-09-16 改：总包改成**扁平结构**。
        #   原来总包里装的是 `AS1_构建套件_buildkit.tar.gz`（**包里套包**）——
        #   解压一次只拿到 AS1 自己，想跑 `verify_reproduce.py` 还得再解一层，
        #   而"不需要 GPU 就能复核成绩"恰恰是评委最可能先跑的一条 ⇒ 会直接报找不到文件。
        #   （这个坑是**实测**出来的：解压后跑那条命令，报 `can't open file ...verify_reproduce.py`。）
        #   现在总包**直接**含 `docker_submit/**` + 根目录 10 个 CSV + 报告 + 交付说明 + MANIFEST。
        add_kit(tf, verbose=False)
        for p in outs:
            if p.name in (kit.name, "MANIFEST.md"):
                continue                      # AS1 的内容已平铺在上面；MANIFEST 下面单独加
            tf.add(p, arcname=p.name)
        # 交付说明在**根目录**再放一份，让解压后一眼就能看到
        d0 = HERE / "00_交付说明.md"
        if d0.exists():
            tf.add(d0, arcname=d0.name)
        tf.add(DIST / "MANIFEST.md", arcname="MANIFEST.md")
    print(f"  ✅ {total.name}  {total.stat().st_size/1e6:.1f} MB")
    print(f"     sha256 = {sha256(total)}")
    print()
    print("dist/ 内容：")
    for p in sorted(DIST.iterdir()):
        print(f"  {p.name:44s} {p.stat().st_size/1e6:10.1f} MB")


if __name__ == "__main__":
    main()
