#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""probe_ms.py —— 探测 ModelScope 上是否有我们要的三个骨干（本地 hf-mirror 不通，MS 通）。"""
import json
import sys
import urllib.request

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
UA = {"User-Agent": "Mozilla/5.0"}
CAND = [
    ("siglip2", ["AI-ModelScope/siglip2-so400m-patch14-384",
                 "google/siglip2-so400m-patch14-384",
                 "AI-ModelScope/siglip2-so400m-patch14-384-ms"]),
    ("cliph14", ["AI-ModelScope/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
                 "AI-ModelScope/clip-vit-h-14"]),
    ("bigg14", ["AI-ModelScope/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k",
                "AI-ModelScope/clip-vit-bigg-14"]),
]


def ls(repo):
    url = (f"https://modelscope.cn/api/v1/models/{repo}/repo/tree"
           f"?Revision=master&Recursive=true")
    try:
        r = urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=25)
        d = json.loads(r.read())
        files = d.get("Data", {}).get("Files", []) or []
        return [f.get("Path", "") for f in files]
    except Exception as e:
        return f"ERR {type(e).__name__}: {e}"


for name, repos in CAND:
    print(f"=== {name} ===")
    for repo in repos:
        f = ls(repo)
        if isinstance(f, str):
            print(f"  {repo:52s} {f[:70]}")
            continue
        big = [p for p in f if p.endswith((".bin", ".safetensors", ".json"))]
        print(f"  {repo:52s} ✅ {len(f)} 个文件；关键文件：")
        for p in big[:8]:
            print(f"        {p}")
        break
