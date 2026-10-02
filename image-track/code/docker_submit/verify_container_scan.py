# -*- coding: utf-8 -*-
"""verify_container_scan.py —— 在容器内以**最终运行身份 appuser** 审计 /app。

由 `verify_container.sh` 拷进 rootfs 的 /tmp 后以 uid 1000 运行。
判据是"真的打开每个文件并读首字节 + 末字节（顺手抓截断）"，
因为 2026-09-18 那次「镜像权重崩溃」的失败点就是**读不到文件**（PermissionError）。
"""
import glob
import json
import os
import struct
import time
import zipfile

print("运行身份：uid=%d gid=%d" % (os.getuid(), os.getgid()), flush=True)
t0 = time.time()
APP = "/app"
bad, n, tot = [], 0, 0
for r, d, f in os.walk(APP):
    for name in f:
        p = os.path.join(r, name)
        n += 1
        try:
            st = os.stat(p)
            with open(p, "rb") as fh:
                fh.read(1)
                if st.st_size > 1:
                    fh.seek(st.st_size - 1)
                    fh.read(1)
            tot += st.st_size
        except OSError as e:
            bad.append("%s | %s: %s" % (p, type(e).__name__, e.strerror))
print("SCANNED %d  BYTES %.2f GB  UNREADABLE %d  用时 %.1fs"
      % (n, tot / 1e9, len(bad), time.time() - t0), flush=True)
for b in bad[:60]:
    print("   BAD", b)
print("❌ 有不可读文件 —— 同 2026-09-18 事故，绝不能交付" if bad
      else "✅ appuser 可读 /app 下全部文件")

print("\n---- 骨干 safetensors：头解析 + 长度一致性 + 真读首/末张量 ----")
try:
    from safetensors import safe_open
except Exception as e:
    safe_open = None
    print("  （safetensors 不可用：%s）" % e)
for p in sorted(glob.glob("/app/backbones/*/*.safetensors")):
    size = os.path.getsize(p)
    with open(p, "rb") as fh:
        hlen = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(hlen))
    md = hdr.pop("__metadata__", {})
    data = sum(v["data_offsets"][1] - v["data_offsets"][0] for v in hdr.values())
    tag = "LEN-OK" if 8 + hlen + data == size else "LEN-MISMATCH"
    line = ("ST %-42s tensors=%-5d data=%7.3fGB file=%7.3fGB %s"
            % (p.replace("/app/backbones/", ""), len(hdr), data / 1e9, size / 1e9, tag))
    if safe_open:
        with safe_open(p, framework="pt") as f:
            ks = list(f.keys())
            a, b = f.get_tensor(ks[0]), f.get_tensor(ks[-1])
        line += "  首/末: %s%s / %s%s" % (ks[0][:24], tuple(a.shape), ks[-1][:24], tuple(b.shape))
    print(line)

print("\n---- app/weights/*.pt：zip 索引（不整载，避开小内存机器）----")
for p in sorted(glob.glob("/app/weights/*")):
    try:
        z = zipfile.ZipFile(p)
        print("PT %-34s %8.1f MB 条目=%d" % (os.path.basename(p), os.path.getsize(p) / 1e6,
                                             len(z.namelist())))
    except Exception as e:
        print("PT %-34s ✗ %s: %s" % (os.path.basename(p), type(e).__name__, e))

print("\n---- app/tables ----")
for p in sorted(glob.glob("/app/tables/*")):
    print("TB %-38s %8.1f KB" % (os.path.basename(p), os.path.getsize(p) / 1e3))

print("\n---- 真读一个最小权重（走 torch.load 真实路径）----")
try:
    import torch
    p = sorted(glob.glob("/app/weights/*.pt"), key=os.path.getsize)[0]
    sd = torch.load(p, map_location="cpu", weights_only=False)
    ks = list(sd.keys()) if isinstance(sd, dict) else []
    print("TORCHLOAD %s  %.1f MB  顶层键=%d %s"
          % (os.path.basename(p), os.path.getsize(p) / 1e6, len(ks), ks[:3]))
except Exception as e:
    import traceback
    traceback.print_exc()
    print("❌ torch.load 失败：", type(e).__name__, e)

print("\n---- 运行期依赖逐个 import ----")
import importlib
MODS = ["numpy", "PIL", "torch", "torchvision", "transformers", "peft", "open_clip",
        "safetensors", "sklearn", "scipy", "pandas", "fastapi", "uvicorn", "pydantic",
        "starlette", "timm", "httpx", "huggingface_hub"]
nok = 0
for m in MODS:
    try:
        mod = importlib.import_module(m)
        nok += 1
        print("OK   %-18s %s" % (m, getattr(mod, "__version__", "-")), flush=True)
    except Exception as e:
        print("FAIL %-18s %s: %s" % (m, type(e).__name__, e), flush=True)
print("import %d/%d 通过" % (nok, len(MODS)))
