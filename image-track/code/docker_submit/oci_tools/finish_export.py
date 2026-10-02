#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""finish_export.py —— 复用已打好的 `layer2.tar`，只重做 **step 5b(gzip) / 6(OCI) / 7(导出)**。

为什么要单独一个脚本：`oci_build3.py` 的 step 1–4（探测 site-packages → 剪重名项 →
组装 layer2（含 11.30 GB 骨干拷贝与硬断言）→ 打 12.77 GB tar）实测要 ~15 分钟，
而这次失败的**只是最后一步**（层的 mediaType 写错成 docker 未压缩类型）。
已经打好的 `layer2.tar` 完全可以复用，没必要重跑前面。

修正内容与 `oci_build3.py` 一致：
    · 层 **gzip** 压缩；
    · mediaType 改成 `application/vnd.oci.image.layer.v1.tar+gzip`（与基座 5 层同类）；
    · `config.rootfs.diff_ids` 仍用**未压缩** tar 的摘要（这是 OCI 规范要求的）。
"""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
R = Path("/hy-tmp/build")
BASE_OCI = R / "base_oci_pt"
WORK = R / "work"
FINAL = R / "final_oci"


def step(n, t):
    print(f"\n{'='*72}\n【{n}】{t}\n{'='*72}", flush=True)


def df():
    r = subprocess.run(["df", "-h", "/hy-tmp"], capture_output=True, text=True)
    print("  [磁盘]", r.stdout.strip().splitlines()[-1], flush=True)


def sha(p, blk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            b = f.read(blk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def main():
    t0 = time.time()
    lay = WORK / "layer2.tar"
    if not lay.exists():
        sys.exit(f"❌ 找不到 {lay}")
    print(f"复用已打好的层：{lay}  {lay.stat().st_size/1e9:.2f} GB")

    idx = json.loads((BASE_OCI / "index.json").read_text())
    md = idx["manifests"][0]["digest"].split(":")[1]
    mf = json.loads((BASE_OCI / "blobs/sha256" / md).read_text())
    if "manifests" in mf:                      # 套了一层 index
        md = mf["manifests"][0]["digest"].split(":")[1]
        mf = json.loads((BASE_OCI / "blobs/sha256" / md).read_text())
    cfg = json.loads((BASE_OCI / "blobs/sha256" /
                      mf["config"]["digest"].split(":")[1]).read_text())
    print(f"  基座 {len(mf['layers'])} 层，config {mf['config']['size']} B")

    step("5b", "gzip layer2.tar（修 skopeo 不认 docker 未压缩类型的问题）")
    diff_id = "sha256:" + sha(lay)
    print(f"  diff_id(未压缩) = {diff_id[:23]}…")
    gz = WORK / "layer2.tar.gz"
    gz.unlink(missing_ok=True)
    prog = "pigz" if shutil.which("pigz") else "gzip"
    print(f"  {prog} -1 -f …（12.77 GB，要几分钟）", flush=True)
    r = subprocess.run([prog, "-1", "-f", str(lay)])
    if not gz.exists():
        cand = sorted(WORK.glob("layer2.tar.*"))
        if r.returncode != 0 or not cand:
            sys.exit("❌ 压缩失败")
        gz = cand[0]
    if gz.stat().st_size < 5e9:
        sys.exit(f"❌ 压缩产物只有 {gz.stat().st_size/1e9:.2f} GB，层不完整")
    blob = sha(gz)
    print(f"  压缩后 {gz.stat().st_size/1e9:.2f} GB  blob={blob[:23]}…")
    df()

    step(6, "写 OCI 布局")
    shutil.rmtree(FINAL, ignore_errors=True)
    (FINAL / "blobs/sha256").mkdir(parents=True)
    (FINAL / "oci-layout").write_text('{"imageLayoutVersion":"1.0.0"}')
    layers = []
    for L in mf["layers"]:
        d = L["digest"].split(":")[1]
        s, t = BASE_OCI / "blobs/sha256" / d, FINAL / "blobs/sha256" / d
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

    step(7, "OCI → docker-archive")
    out = R / "baycup_image_aidet_0.918887.tar"
    out.unlink(missing_ok=True)
    r = subprocess.run(["skopeo", "copy", f"oci:{FINAL}:0.918887",
                        f"docker-archive:{out}:baycup-image-aidet:0.918887"])
    if r.returncode != 0 or not out.exists():
        sys.exit("❌ 导出失败")
    print(f"\n✅ {out}  {out.stat().st_size/1e9:.2f} GB")
    print("   sha256:", sha(out))
    df()
    print(f"\n总用时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
