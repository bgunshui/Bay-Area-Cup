#!/usr/bin/env python3
"""扫描镜像主层:找出所有 cp312/ABI 不匹配的编译产物,并检查系统库缺失。"""
import tarfile
import collections

IMG = "/mnt/d/oci_build/ai-text-detector-0.981301.tar"
outer = tarfile.open(IMG, "r:")
inner = tarfile.open(fileobj=outer.extractfile("./newlayer.tar"), mode="r:")

SP = "./usr/local/lib/python3.10/site-packages/"
EXT = (".so", ".pyd")
hits = collections.defaultdict(lambda: collections.defaultdict(list))
allso = collections.Counter()
for m in inner:
    n = m.name
    if not n.startswith(SP):
        continue
    if n.endswith(EXT) or ".cpython-" in n:
        rel = n[len(SP):]
        top = rel.split("/")[0]
        if "cpython-312" in n:
            hits["cp312"][top].append(rel)
        if "cpython-310" in n:
            hits["cp310"][top].append(rel)
        if "abi3" in n:
            hits["abi3"][top].append(rel)
        if n.endswith(".so") and "cpython-" not in n:
            hits["other-so"][top].append(rel)

for k in ("cp312", "cp310", "abi3", "other-so"):
    d = hits[k]
    print(f"\n=== {k}: {len(d)} 个顶层包, {sum(len(v) for v in d.values())} 个文件 ===")
    for top in sorted(d):
        print(f"  {top:24s} {len(d[top]):4d}  {d[top][:2]}")
