"""检查 docker-archive 镜像：外层 tar + 内层 newlayer.tar 的随机访问能力。

关键点：外层用 r: 随机访问，内层用 outer.extractfile() 拿到可 seek 的 fileobj，
再以 r: 打开，这样 tarfile 会用 seek 跳过数据而不是读 44GB。
"""
import tarfile
import sys
import time

IMG = r"D:\oci_build\ai-text-detector-0.981301.tar"

t0 = time.time()
outer = tarfile.open(IMG, "r:")
print("外层成员：")
for m in outer.getmembers():
    print(f"  {m.name:24s} type={m.type} size={m.size:,} mtime={m.mtime}")

print("\n--- config.json ---")
cfg = outer.extractfile("./config.json").read().decode("utf-8")
print(cfg[:4000])

print("\n--- manifest.json ---")
man = outer.extractfile("./manifest.json").read().decode("utf-8")
print(man)
print(f"\n外层读取耗时 {time.time()-t0:.1f}s")

inner_f = outer.extractfile("./newlayer.tar")
print(f"\n内层 fileobj: {type(inner_f).__name__} seekable={inner_f.seekable()}")
t1 = time.time()
inner = tarfile.open(fileobj=inner_f, mode="r:")
n = 0
top = {}
for m in inner:
    n += 1
    parts = m.name.split("/")
    key = "/".join(parts[:2]) if len(parts) > 1 else parts[0]
    e = top.setdefault(key, [0, 0])
    e[0] += 1
    e[1] += m.size
print(f"内层成员总数 {n}，索引耗时 {time.time()-t1:.1f}s（若为秒级说明 seek 生效）")
print("\n前 30 个顶层路径：")
for i, (k, v) in enumerate(sorted(top.items())):
    if i < 30:
        print(f"  {k:40s} files={v[0]:6d} bytes={v[1]:,}")
print(f"\n顶层路径总数 {len(top)}")
