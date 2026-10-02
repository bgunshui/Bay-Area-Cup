#!/usr/bin/env python3
"""B/C/D) 镜像内部审计:app 树、脚本内容比对、safetensors 权重完整性、index 对照。"""
import hashlib
import json
import os
import sys
import tarfile

IMG = r"D:\oci_build_fixed\ai-text-detector-0.981301.tar"
LOCAL_SERVER = r"D:\湾区杯\文本赛道\docker_submit\server_dual_v4.py"
LOCAL_HEALTH = r"D:\applayer\usr\local\bin\healthcheck.py"
problems = []


def norm(n):
    return n[2:] if n.startswith("./") else n


outer = tarfile.open(IMG, "r:")
out_members = {norm(m.name): m for m in outer.getmembers()}

inner = tarfile.open(fileobj=outer.extractfile(out_members["newlayer.tar"]), mode="r:")
main = {norm(m.name): m for m in inner}
inner2 = tarfile.open(fileobj=outer.extractfile(out_members["deps_layer.tar"]), mode="r:")
deps = {norm(m.name): m for m in inner2}

print("#" * 78)
print("B) server.py / healthcheck.py / requirements.txt 生效内容")
print("#" * 78)
with open(LOCAL_SERVER, "rb") as f:
    want_srv = f.read()
with open(LOCAL_HEALTH, "rb") as f:
    want_hc = f.read()

srv_main = inner.extractfile(main["app/server.py"]).read() if "app/server.py" in main else None
srv_deps = inner2.extractfile(deps["app/server.py"]).read() if "app/server.py" in deps else None
eff = srv_deps if srv_deps else srv_main
print("  主层 app/server.py     : %s bytes" % (len(srv_main) if srv_main else "缺失"))
print("  依赖层 app/server.py   : %s bytes" % (len(srv_deps) if srv_deps else "缺失"))
print("  本地 server_dual_v4.py : %d bytes" % len(want_srv))
print("  生效版本 == 本地 v4    :", eff == want_srv)
print("  生效版本 sha256        :", hashlib.sha256(eff).hexdigest())
print("  本地     sha256        :", hashlib.sha256(want_srv).hexdigest())
for probe, desc in ((b"device_map", "device_map 设备放置修复"),
                    (b'import torch.distributed.tensor', "distributed.tensor 预加载"),
                    (b'_tdt = _sys.modules.get', "预加载后显式绑定"),
                    (b"RequestValidationError", "400/40001 校验处理"),
                    (b"_INFER_LOCK", "推理串行化锁"),
                    (b"numel() == 0", "空串 eos 兜底"),
                    (b'eos_token_id', "eos_token_id 兜底"),
                    (b'W_14B', "0.90 权重"), (b'W_7B', "0.10 权重")):
    ok = probe in eff
    print("    [%s] %s" % ("OK" if ok else "X", desc))
    if not ok:
        problems.append("server.py 缺少 " + desc)
if eff != want_srv:
    problems.append("/app/server.py 生效版本与本地 v4 不一致")

hc = inner2.extractfile(deps["usr/local/bin/healthcheck.py"]).read() if "usr/local/bin/healthcheck.py" in deps else None
print("\n  healthcheck.py 存在:", hc is not None, " 与本地一致:", hc == want_hc if hc else False)
if hc != want_hc:
    problems.append("healthcheck.py 与本地不一致或缺缺失")

req = inner2.extractfile(deps["app/requirements.txt"]).read().decode() if "app/requirements.txt" in deps else None
print("  requirements.txt 行数:", len(req.splitlines()) if req else "缺失")
for pin in ("torch==2.5.1", "transformers==4.46.0", "peft==0.20.0", "tokenizers==0.20.3"):
    if req and pin not in req:
        problems.append("requirements.txt 缺 " + pin)

print("\n" + "#" * 78)
print("C) safetensors 权重完整性(头解析 + 字节区间连续性 + 大小自洽)")
print("#" * 78)
weights = [(k, v) for k, v in main.items() if k.endswith(".safetensors")]
weights.sort()
print("文件数:", len(weights))
bad = 0
total_t = 0
for name, m in weights:
    f = inner.extractfile(m)
    head = f.read(8)
    n = int.from_bytes(head, "little")
    hdr_raw = f.read(n)
    try:
        hdr = json.loads(hdr_raw.decode("utf-8"))
    except Exception as e:
        print("  [X] %s 头解析失败 %s" % (name, e))
        bad += 1
        continue
    tensors = {k: v for k, v in hdr.items() if k != "__metadata__"}
    spans = []
    ok_fields = True
    for k, v in tensors.items():
        if not all(x in v for x in ("dtype", "shape", "data_offsets")):
            ok_fields = False
            break
        spans.append(tuple(v["data_offsets"]))
    spans.sort()
    contiguous = all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))
    end = spans[-1][1] if spans else 0
    expect = 8 + n + end
    ok = ok_fields and contiguous and expect == m.size
    if not ok:
        bad += 1
        problems.append("%s 权重完整性异常 (size=%d expect=%d contiguous=%s)" % (name, m.size, expect, contiguous))
    total_t += len(tensors)
    meta = hdr.get("__metadata__", {})
    print("  [%s] %-62s %13d B  tensors=%4d  fmt=%s" % (
        "OK" if ok else "X", name.replace("app/", ""), m.size, len(tensors), meta.get("format")))
print("合计 tensor:", total_t, " 异常:", bad)

print("\n" + "#" * 78)
print("D) index.json 与实际分片对照")
print("#" * 78)
present = set(k for k in main if k.endswith(".safetensors"))
for k in sorted(main):
    if k.endswith("model.safetensors.index.json"):
        data = json.loads(inner.extractfile(main[k]).read())
        wm = data.get("weight_map", {})
        base = k.rsplit("/", 1)[0]
        shards = sorted(set(wm.values()))
        real = sorted(x.rsplit("/", 1)[1] for x in present if x.startswith(base + "/"))
        missing = [s for s in shards if base + "/" + s not in present]
        print("  %s" % k)
        print("    index 分片 %d: %s" % (len(shards), shards))
        print("    实际 分片 %d: %s" % (len(real), real))
        print("    权重条目 %d  缺失: %s" % (len(wm), missing if missing else "无"))
        if missing:
            problems.append("%s 缺分片 %s" % (k, missing))
        if shards != real:
            problems.append("%s 分片列表不一致" % k)

print("\n" + "#" * 78)
print("B/C/D 阶段问题:", problems if problems else "无")
print("#" * 78)
with open(r"D:\湾区杯\文本赛道\_audit_problems2.json", "w", encoding="utf-8") as f:
    json.dump(problems, f, ensure_ascii=False, indent=1)
sys.exit(0 if not problems else 1)
