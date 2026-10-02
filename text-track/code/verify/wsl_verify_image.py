#!/usr/bin/env python3
"""验证最终镜像:结构一致性 + 两层 sha256 + 重新抽取环境 + 应用 whiteout。

成员名前缀兼容 "name" 与 "./name" 两种写法。

用法(WSL, root):
    python3 wsl_verify_image.py
"""
import hashlib
import json
import os
import shutil
import sys
import tarfile
import time

IMG = "/mnt/d/oci_build_fixed/ai-text-detector-0.981301.tar"
DEST = sys.argv[1] if len(sys.argv) > 1 else "/root/imgtest2/rootfs"
DEPS_FROM = "deps_layer.tar"
SMALL = 25 * 1024 * 1024
KEEP = ("bin", "etc", "lib", "lib64", "sbin", "usr", "var", "root", "opt")
BUF = 8 << 20

problems = []


def norm(name):
    return name[2:] if name.startswith("./") else name


def sha256_stream(f, size, label):
    h = hashlib.sha256()
    left = size
    while left > 0:
        b = f.read(min(BUF, left))
        if not b:
            raise EOFError(f"{label} 提前 EOF")
        h.update(b)
        left -= len(b)
    return h.hexdigest()


def main():
    total = os.path.getsize(IMG)
    print(f"镜像: {IMG}  {total:,} bytes ({total/2**30:.2f} GiB)")
    outer = tarfile.open(IMG, "r:")
    members = {norm(m.name): m for m in outer.getmembers()}
    print("外层成员:", sorted(members))

    man = json.loads(outer.extractfile(members["manifest.json"]).read())
    cfg_name = man[0]["Config"]
    layers = man[0]["Layers"]
    print(f"manifest: Config={cfg_name} RepoTags={man[0]['RepoTags']} Layers={layers}")

    if cfg_name not in members:
        problems.append(f"manifest 指向的 config 文件 {cfg_name} 不在包里")
    cfg = json.loads(outer.extractfile(members[cfg_name]).read())
    diff_ids = cfg["rootfs"]["diff_ids"]
    print(f"config: diff_ids={diff_ids}")
    print(f"config: Cmd={cfg['config'].get('Cmd')} WorkingDir={cfg['config'].get('WorkingDir')}")
    print(f"config: Healthcheck={cfg['config'].get('Healthcheck', {}).get('Test')}")
    print(f"config: history 条数={len(cfg.get('history', []))}")

    if len(diff_ids) != len(layers):
        problems.append(f"diff_ids({len(diff_ids)}) 与 Layers({len(layers)}) 数量不一致")
    if len(cfg.get("history", [])) != len(diff_ids):
        problems.append("history 条数与 diff_ids 不一致")
    if "config.json" not in members:
        problems.append("缺少兼容用的 config.json")

    for i, (lyr, did) in enumerate(zip(layers, diff_ids)):
        m = members[lyr]
        f = outer.extractfile(m)
        t0 = time.time()
        hexd = sha256_stream(f, m.size, lyr)
        ok = ("sha256:" + hexd) == did
        print(f"[层{i}] {lyr:18s} {m.size:>14,} bytes  sha256={hexd[:16]}…  "
              f"{'匹配 diff_id ✓' if ok else '与 diff_id 不符 ✗'}  ({time.time()-t0:.0f}s)")
        if not ok:
            problems.append(f"{lyr} 的 sha256 与 config.diff_ids[{i}] 不一致")

    if os.path.exists(DEST):
        shutil.rmtree(DEST)
    os.makedirs(DEST, exist_ok=True)
    t0 = time.time()

    def apply_layer(layer_name, pick):
        tf = tarfile.open(fileobj=outer.extractfile(members[layer_name]), mode="r:")
        n = 0
        for m in tf:
            rel = norm(m.name)
            base = os.path.basename(rel)
            if base.startswith(".wh."):
                target = os.path.join(DEST, os.path.dirname(rel), base[4:])
                if os.path.exists(target):
                    shutil.rmtree(target, ignore_errors=True)
                    print(f"    whiteout 生效:已删除 {base[4:]}")
                continue
            if not pick(rel, m):
                continue
            tf.extract(m, DEST, set_attrs=True, numeric_owner=True, filter="fully_trusted")
            n += 1
        return n

    n1 = apply_layer("newlayer.tar", lambda rel, m: (
        rel == "." or rel.split("/")[0] in KEEP or rel == "app/server.py"
        or rel.startswith("app/adapters/") or (rel.startswith("app/model/") and m.size < SMALL)))
    print(f"[抽取] 主层 {n1} 个成员, {time.time()-t0:.0f}s")
    n2 = apply_layer(DEPS_FROM, lambda rel, m: rel.startswith("usr/local")
                     or rel.startswith("app/"))
    print(f"[抽取] 依赖层 {n2} 个成员, 共 {time.time()-t0:.0f}s")

    for d in ("dev", "proc", "sys", "tmp"):
        os.makedirs(os.path.join(DEST, d), exist_ok=True)

    sp = os.path.join(DEST, "usr/local/lib/python3.10/site-packages")
    entries = sorted(os.listdir(sp))
    torch_di = [e for e in entries if e.startswith("torch-") and e.endswith(".dist-info")]
    print(f"\n[检查] site-packages 顶层 {len(entries)} 项")
    print(f"[检查] torch dist-info: {torch_di}")
    if len(torch_di) != 1:
        problems.append(f"torch dist-info 不是唯一一份: {torch_di}")
    if os.path.exists(os.path.join(sp, ".wh.torch-2.5.1+cu124.dist-info")):
        problems.append("whiteout 文件残留(.wh. 未被消费)")
    for need in ("torch", "transformers", "peft", "fastapi", "numpy", "nvidia", "triton",
                 "tokenizers", "sentencepiece", "scipy", "uvicorn", "pydantic", "starlette"):
        if not os.path.exists(os.path.join(sp, need)):
            problems.append(f"site-packages 缺少 {need}")
    for need in ("usr/local/bin/python3.10", "usr/local/bin/healthcheck.py", "app/server.py"):
        if not os.path.exists(os.path.join(DEST, need)):
            problems.append(f"缺少 {need}")

    with open(os.path.join(DEST, "app/server.py")) as f:
        srv = f.read()
    if "device_map" not in srv:
        problems.append("app/server.py 不是加固版(缺 device_map 设备放置修复)")

    print(f"\n[结果] 问题: {problems if problems else '无'}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
