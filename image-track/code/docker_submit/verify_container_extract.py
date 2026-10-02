#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""verify_container_extract.py —— 把 docker-archive 镜像摊成一个可 chroot 的 rootfs。

    python3 verify_container_extract.py <镜像 tar> <目标目录>

docker-archive 的形态（skopeo / docker save 都是这个）：
    manifest.json   {"Config":"<cfg>.json","RepoTags":[…],"Layers":["<diff_id>.tar", …]}
    <diff_id>.tar   每个成员的内容是**一层**（未压缩 tar，或 gzip 过的 tar）
后写的层覆盖先写的 ⇒ 与 overlay 语义一致。
"""
import json
import os
import subprocess
import sys
import tarfile
import time

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
IMG, ROOT = sys.argv[1], sys.argv[2]
os.makedirs(ROOT, exist_ok=True)
t0 = time.time()
print(f"  镜像 {IMG}  {os.path.getsize(IMG)/1e9:.2f} GB → {ROOT}", flush=True)

t = tarfile.open(IMG, "r:")
man = t.getmember("manifest.json")
spec = json.loads(t.extractfile(man).read())
o = spec[0] if isinstance(spec, list) else spec
layers = o["Layers"]
print(f"  {len(layers)} 层", flush=True)

for i, name in enumerate(layers, 1):
    m = t.getmember(name)
    src = t.extractfile(m)
    src.seek(0)
    head = src.read(2)
    gz = head == b"\x1f\x8b"
    p = subprocess.Popen(["tar", "-x", "-C", ROOT] + (["-z"] if gz else []),
                         stdin=subprocess.PIPE)
    buf = head + src.read(1 << 22)
    try:
        while buf:
            p.stdin.write(buf)
            buf = src.read(1 << 22)
    except (BrokenPipeError, OSError):
        pass
    p.stdin.close()
    rc = p.wait()
    print(f"    [{i}/{len(layers)}] {name[:20]}… {m.size/1e9:.2f} GB rc={rc}  "
          f"{time.time()-t0:.0f}s", flush=True)
    if rc != 0:
        sys.exit(f"❌ 解 {name} 失败")
print(f"  ✅ 摊开完成 {time.time()-t0:.0f}s，顶层：{sorted(os.listdir(ROOT))}", flush=True)
