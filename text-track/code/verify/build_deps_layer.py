#!/usr/bin/env python3
"""把 D:\\deps_wheels2 里的 wheel 解包成 docker 镜像的一个新层。

背景:原镜像主层里 torch/scipy/tokenizers/sentencepiece/functorch 全部是
cpython-312 的编译产物,而镜像是 Python 3.10 —— ABI 不匹配,import torch 直接失败。
因此本层必须同时提供这些包的 **cp310** 版本以覆盖下层,并补齐全部传递依赖。

处理要点:
  * .data/{purelib,platlib,scripts,data,headers} 重定位
  * 保留可执行位与符号链接
  * 跳过镜像已有且无需替换的发行版(packaging/setuptools/wheel/pip),
    避免出现两个不同版本的 dist-info
  * 对会被替换但名字不同的 dist-info 打 whiteout(torch-2.5.1+cu124.dist-info),
    保证只剩一份元数据

用法(WSL, root):
    python3 build_deps_layer.py
"""
import os
import shutil
import stat
import sys
import tarfile
import time
import zipfile

WHEELS = "/mnt/d/deps_wheels2"
STAGE = "/root/deplayer"
OUT_TAR = "/mnt/d/deps_layer.tar"
LIST_FILE = "/mnt/d/deps_layer_list.txt"

SITE = "usr/local/lib/python3.10/site-packages"
BINP = "usr/local/bin"
HEAD = "usr/local/include"

# 镜像主层已有、且版本可接受 —— 不再放一份(否则两个不同版本的 dist-info 并存)
ALREADY = {"packaging", "setuptools", "wheel", "pip"}

# 会被本层替换、但 dist-info 目录名不同的旧元数据 —— 用 whiteout 清掉
WHITEOUTS = ["torch-2.5.1+cu124.dist-info"]

# 额外覆盖进镜像的文件树(相对镜像根 /),例如加固版 app/server.py
EXTRA = "/mnt/d/applayer"


def norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


def main():
    if os.path.exists(STAGE):
        shutil.rmtree(STAGE)
    for d in (SITE, BINP, HEAD):
        os.makedirs(os.path.join(STAGE, d), exist_ok=True)

    wheels = sorted(f for f in os.listdir(WHEELS) if f.endswith(".whl"))
    print(f"wheel 总数 {len(wheels)}")

    placed, skipped = [], []
    for w in wheels:
        dn = w.split("-")[0]
        if norm(dn) in ALREADY:
            skipped.append(w)
            continue
        n_files = 0
        with zipfile.ZipFile(os.path.join(WHEELS, w)) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                name = info.filename
                mode = (info.external_attr >> 16) & 0xFFFF
                is_link = stat.S_ISLNK(mode)

                if ".data/" in name:
                    _, tail = name.split(".data/", 1)
                    kind, _, rest = tail.partition("/")
                    if kind in ("purelib", "platlib"):
                        target = os.path.join(STAGE, SITE, rest)
                    elif kind == "scripts":
                        target = os.path.join(STAGE, BINP, rest)
                    elif kind == "headers":
                        target = os.path.join(STAGE, HEAD, rest)
                    elif kind == "data":
                        # .data/data 相对 sys.prefix,即 /usr/local
                        target = os.path.join(STAGE, "usr/local", rest)
                    else:
                        print(f"  !! 未知 .data 类型 {kind}: {name}")
                        continue
                else:
                    target = os.path.join(STAGE, SITE, name)

                os.makedirs(os.path.dirname(target), exist_ok=True)
                if is_link:
                    if os.path.lexists(target):
                        os.remove(target)
                    os.symlink(z.read(info).decode("utf-8"), target)
                else:
                    with z.open(info) as src, open(target, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    if mode:
                        os.chmod(target, mode & 0o7777)
                n_files += 1
        placed.append((w, n_files))
        print(f"  + {w:78s} {n_files:5d} files", flush=True)

    print(f"\n放入 {len(placed)} 个 wheel;跳过 {len(skipped)} 个: {skipped}")

    # whiteout:空文件 .wh.<name>
    for name in WHITEOUTS:
        p = os.path.join(STAGE, SITE, ".wh." + name)
        with open(p, "wb"):
            pass
        print(f"  whiteout: {os.path.relpath(p, STAGE)}")

    # 附加覆盖文件(如加固版 app/server.py),目录结构相对镜像根 /
    if os.path.isdir(EXTRA):
        n_extra = 0
        for root, _dirs, files in os.walk(EXTRA):
            for fn in files:
                src = os.path.join(root, fn)
                rel = os.path.relpath(src, EXTRA)
                dst = os.path.join(STAGE, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)
                n_extra += 1
                print(f"  extra: {rel}")
        print(f"  附加覆盖文件 {n_extra} 个")

    print("\n打包 deps_layer.tar ...", flush=True)
    t0 = time.time()
    if os.path.exists(OUT_TAR):
        os.remove(OUT_TAR)
    rc = os.system(f"tar --numeric-owner --owner=0 --group=0 -cf {OUT_TAR} -C {STAGE} .")
    if rc != 0:
        print(f"!! tar 失败 rc={rc}")
        return 1
    size = os.path.getsize(OUT_TAR)
    print(f"deps_layer.tar = {size:,} bytes ({size/2**30:.2f} GiB), 耗时 {time.time()-t0:.0f}s")

    with open(LIST_FILE, "w") as f:
        f.write(f"# deps layer wheels={len(placed)} size={size}\n")
        for b, n in placed:
            f.write(f"{b}\t{n}\n")
    print(f"清单写入 {LIST_FILE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
