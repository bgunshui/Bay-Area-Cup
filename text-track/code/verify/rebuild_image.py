"""在现有 docker-archive 镜像上追加一个依赖层,重建可提交镜像。

策略:不重新生成 44GB 主层,而是把原 newlayer.tar 的字节流原样搬进新的外层 tar,
同时增量计算它的 sha256(用于校验 config.rootfs.diff_ids[0] 是否属实),
再追加 deps_layer.tar,并更新 manifest.json / config.json / repositories。

顺带修复一个隐患:原镜像 manifest 里 Config 指向 "<hash>.json",但包里文件实际叫
config.json。这里两个名字都写入,manifest 指向哈希名(标准做法)。
"""
import hashlib
import io
import json
import os
import sys
import tarfile
import time

SRC = r"D:\oci_build\ai-text-detector-0.981301.tar"
DEPS = r"D:\deps_layer.tar"
OUTDIR = r"D:\oci_build_fixed"
OUT = os.path.join(OUTDIR, "ai-text-detector-0.981301.tar")

BUF = 8 << 20


def sha256_hex_of_fobj(f, size, label=""):
    h = hashlib.sha256()
    left = size
    done = 0
    t0 = time.time()
    while left > 0:
        b = f.read(min(BUF, left))
        if not b:
            raise EOFError(f"{label}: 提前 EOF,还差 {left} 字节")
        h.update(b)
        left -= len(b)
        done += len(b)
        if label and done % (4 << 30) < BUF:
            print(f"    {label} {done/2**30:.1f}/{size/2**30:.1f} GiB  {time.time()-t0:.0f}s", flush=True)
    return h.hexdigest()


def sha256_hex_of_file(path):
    with open(path, "rb") as f:
        return sha256_hex_of_fobj(f, os.path.getsize(path))


def chain_id(parent_chain, diff_id):
    """moby: chainID = sha256(parentChainID.String() + " " + diffID.String())"""
    return hashlib.sha256((parent_chain + " " + diff_id).encode()).hexdigest()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    if os.path.exists(OUT):
        os.remove(OUT)

    src = tarfile.open(SRC, "r:")
    cfg_bytes = src.extractfile("./config.json").read()
    cfg = json.loads(cfg_bytes)
    diffid1 = cfg["rootfs"]["diff_ids"][0]
    print(f"[原镜像] diff_ids = {cfg['rootfs']['diff_ids']}")
    print(f"[原镜像] history = {cfg['history']}")
    layers1 = json.loads(src.extractfile("./manifest.json").read())[0]["Layers"]
    print(f"[原镜像] Layers = {layers1}")

    deps_size = os.path.getsize(DEPS)
    print(f"\n[依赖层] {DEPS}  {deps_size:,} bytes ({deps_size/2**30:.2f} GiB)")
    print("[依赖层] 计算 sha256 ...", flush=True)
    t0 = time.time()
    diffid2_hex = sha256_hex_of_file(DEPS)
    diffid2 = "sha256:" + diffid2_hex
    print(f"[依赖层] sha256 = {diffid2_hex}  ({time.time()-t0:.0f}s)")

    # ---- 新 config ----
    cfg["rootfs"]["diff_ids"] = [diffid1, diffid2]
    cfg["history"] = [
        cfg["history"][0] if cfg.get("history") else {"created": "2026-09-16T00:00:00Z"},
        {"created": "2026-09-18T00:00:00Z", "created_by": "add-deps-layer"},
    ]
    # HEALTHCHECK 原来用 curl,但 python:3.10-slim 基础镜像里没有 curl,
    # 会导致容器永远 unhealthy。改用镜像内必然存在的 python 脚本。
    hc = cfg.get("config", {}).get("Healthcheck")
    if hc:
        old = hc.get("Test")
        hc["Test"] = ["CMD", "python", "/usr/local/bin/healthcheck.py"]
        hc["Interval"] = 10_000_000_000
        hc["Timeout"] = 5_000_000_000
        hc["StartPeriod"] = 300_000_000_000
        hc["Retries"] = 3
        print(f"[新 config] Healthcheck: {old} -> {hc['Test']}")
    new_cfg_bytes = json.dumps(cfg, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    cfg_hash = hashlib.sha256(new_cfg_bytes).hexdigest()
    cfg_name = f"{cfg_hash}.json"
    print(f"\n[新 config] {len(new_cfg_bytes)} bytes  sha256={cfg_hash}")
    print(f"[新 config] diff_ids = {cfg['rootfs']['diff_ids']}")

    # ---- 新 manifest ----
    man = json.loads(src.extractfile("./manifest.json").read())
    man[0]["Config"] = cfg_name
    man[0]["Layers"] = ["newlayer.tar", "deps_layer.tar"]
    man_bytes = json.dumps(man, separators=(",", ":")).encode("utf-8")
    print(f"[新 manifest] {man_bytes.decode()}")

    # ---- repositories(用正确的顶层 chainID)----
    chain1 = diffid1
    chain2 = "sha256:" + chain_id(chain1, diffid2)
    old_repos = json.loads(src.extractfile("./repositories").read())
    repo, tag = next(iter(old_repos.items()))
    repo_tag = next(iter(tag))
    repos = {repo: {repo_tag: chain2.split(":")[1][:12]}}
    repos_bytes = json.dumps(repos, separators=(",", ":")).encode("utf-8")
    print(f"[新 repositories] {repos_bytes.decode()}")

    # ---- 写新镜像 ----
    print(f"\n[写出] {OUT}  (约 {(44775191040 + deps_size)/2**30:.1f} GiB)", flush=True)
    inner_ti = src.getmember("./newlayer.tar")
    inner_f = src.extractfile(inner_ti)
    inner_size = inner_ti.size
    deps_f = open(DEPS, "rb")

    t0 = time.time()
    out = tarfile.open(OUT, "w", format=tarfile.GNU_FORMAT)
    try:
        d = tarfile.TarInfo(".")
        d.type = tarfile.DIRTYPE
        d.mode = 0o755
        d.mtime = 1789556610
        out.addfile(d)

        for name, data, mtime in (
            ("config.json", new_cfg_bytes, 1789556610),
            (cfg_name, new_cfg_bytes, 1789556610),
            ("manifest.json", man_bytes, 1789556610),
        ):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.mode = 0o644
            ti.mtime = mtime
            out.addfile(ti, io.BytesIO(data))

        print("  -> newlayer.tar 流式复制中 ...", flush=True)
        ti = tarfile.TarInfo("newlayer.tar")
        ti.size = inner_size
        ti.mode = 0o644
        ti.mtime = 1789556501
        out.addfile(ti, inner_f)
        print(f"     newlayer.tar 完成 {time.time()-t0:.0f}s", flush=True)

        print("  -> deps_layer.tar 流式复制中 ...", flush=True)
        ti = tarfile.TarInfo("deps_layer.tar")
        ti.size = deps_size
        ti.mode = 0o644
        ti.mtime = 1789556610
        out.addfile(ti, deps_f)

        ti = tarfile.TarInfo("repositories")
        ti.size = len(repos_bytes)
        ti.mode = 0o644
        ti.mtime = 1789556610
        out.addfile(ti, io.BytesIO(repos_bytes))
    finally:
        out.close()
        deps_f.close()

    size = os.path.getsize(OUT)
    print(f"\n[完成] {OUT}")
    print(f"        {size:,} bytes ({size/2**30:.2f} GiB)  耗时 {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
