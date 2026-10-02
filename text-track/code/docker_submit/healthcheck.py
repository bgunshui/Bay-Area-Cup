#!/usr/bin/env python3
"""容器 HEALTHCHECK:探测 /health/liveness。

镜像内没有 curl(基础镜像 python:3.10-slim 不含),因此用 Python 标准库实现,
不引入任何新依赖。存活探针失败 -> 退出码 1。
"""
import sys
import urllib.request

URL = "http://127.0.0.1:8080/health/liveness"

try:
    with urllib.request.urlopen(URL, timeout=3) as r:
        sys.exit(0 if r.status == 200 else 1)
except Exception:
    sys.exit(1)
