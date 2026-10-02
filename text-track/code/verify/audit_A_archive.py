#!/usr/bin/env python3
"""提交包终审:
A) 包层:成员/大小/文件名 UTF-8 字节/pax 头/从包内取出的报告与信息文件是否与本地一致
B) 镜像层:config 全字段、app 树、server.py 与 healthcheck.py 内容比对
C) 权重完整性:14 个基座 shard + 15 个 adapter 的 safetensors 头解析 + 字节区间和 == 文件大小
D) index.json weight_map 引用的 shard 是否都存在
"""
import hashlib
import io
import json
import os
import sys
import tarfile

ARCHIVE = r"D:\文本附件资料.tar"
IMG_IN_ARCHIVE = "ai-text-detector-0.981301.tar"
LOCAL_REPORT = r"D:\湾区杯\文本赛道\docker_submit\技术方案报告.md"
LOCAL_INFO = r"D:\湾区杯\文本赛道\docker_submit\镜像信息.txt"
LOCAL_SERVER = r"D:\湾区杯\文本赛道\docker_submit\server_dual_v4.py"

problems = []


def sha256_of(fobj, size):
    h = hashlib.sha256()
    left = size
    while left > 0:
        b = fobj.read(min(8 << 20, left))
        if not b:
            break
        h.update(b)
        left -= len(b)
    return h.hexdigest()


print("#" * 78)
print("A) 提交包层")
print("#" * 78)
arc = tarfile.open(ARCHIVE, "r:")
members = arc.getmembers()
print("包大小: {:,} bytes ({:.2f} GiB)".format(os.path.getsize(ARCHIVE), os.path.getsize(ARCHIVE) / 2**30))
print("成员数:", len(members))
for m in members:
    raw = m.name.encode("utf-8", "surrogateescape")
    try:
        dec = raw.decode("utf-8")
        enc_ok = (dec == m.name or True)
        utf8 = "UTF-8 合法"
    except Exception:
        utf8 = "[!] 非 UTF-8"
        problems.append("成员名非 UTF-8: " + repr(raw))
    print("  {:<42s} {:>14,} bytes  type={}  {}".format(
        raw.decode("utf-8", "replace"), m.size, m.type.decode(), utf8))
    print("      pax_headers keys =", sorted((m.pax_headers or {}).keys()))

names = {m.name: m for m in members}
for need in (IMG_IN_ARCHIVE, "技术方案报告.md", "镜像信息.txt"):
    if need not in names:
        problems.append("包内缺少 " + need)

# 从包里取出小文件,与本地比对
for arc_name, local in (("技术方案报告.md", LOCAL_REPORT), ("镜像信息.txt", LOCAL_INFO)):
    if arc_name not in names:
        continue
    got = arc.extractfile(names[arc_name]).read()
    with open(local, "rb") as f:
        want = f.read()
    same = got == want
    print("  [比对] {:<20s} 包内 {:>7,} 本地 {:>7,}  {} ".format(
        arc_name, len(got), len(want), "一致" if same else "[!] 不一致"))
    if not same:
        problems.append(arc_name + " 包内与本地不一致(可能打进了旧版)")
    if arc_name == "技术方案报告.md":
        txt = got.decode("utf-8")
        for probe in ("0.981301", "★ 最终提交方案", "最终提交成绩:0.981301",
                      "70.1 秒", "0.68 秒", "41,868 MiB", "中间结果"):
            if probe not in txt:
                problems.append("报告缺少关键内容: " + probe)
            else:
                print("      含关键内容 [OK] " + probe)
        if txt.count("0.981069") > 2:
            print("      [!] 0.981069 出现", txt.count("0.981069"), "次,偏多")

print("\n内存中取出镜像并校验(流式)...", flush=True)
img_in = arc.extractfile(names[IMG_IN_ARCHIVE])
img_bytes = names[IMG_IN_ARCHIVE].size
h = hashlib.sha256()
left = img_bytes
while left > 0:
    b = img_in.read(min(8 << 20, left))
    if not b:
        break
    h.update(b)
    left -= len(b)
got = h.hexdigest()
want = "a163a653d2aa05b7bd51ec6016fbc5aed51379f7802fbfbb9a39c231dc691f68"
print("  包内镜像 sha256 =", got)
print("  期望           =", want)
print("  ", "一致" if got == want else "[!] 不一致")
if got != want:
    problems.append("包内镜像 sha256 不匹配")
if left != 0:
    problems.append("镜像成员读取不完整,还差 %d 字节" % left)
arc.close()
print("\n阶段 A 问题:", problems if problems else "无")
json.dump(problems, open(r"D:\湾区杯\文本赛道\_audit_problems.json", "w"))
sys.exit(0 if not problems else 1)
