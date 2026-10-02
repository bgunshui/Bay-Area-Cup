#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""depcheck.py —— 检查 piproot 里所有包的**硬依赖**能否由 (piproot ∪ 基座镜像) 满足。

为什么必须有这个检查（2026-09-16 的教训）：
    `pip_explicit.py` 用 `--no-deps` 装包，依赖列表是**人肉展开**的。
    一旦漏掉某个传递依赖，镜像里就会 `ImportError` —— 而**在宿主机上跑验证根本发现不了**，
    因为宿主机另有那份包（`import` 静默地落到宿主机的 dist-packages 上）。
    这是最阴的一类失败："验证通过、交付必崩"。

    实证：2026-09-16 漏了
        · `timm`（open_clip_torch 硬依赖）
        · `httpx` / `hf-xet`（huggingface_hub 硬依赖）
        · `typing-inspection`（pydantic 硬依赖）
        · `typer`（transformers 硬依赖）

用法：python3 depcheck.py [piproot 路径]   → 打印缺失清单，缺则为非 0 退出
"""
import json
import re
import subprocess
import sys
from pathlib import Path

PIPROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "/hy-tmp/build/work/piproot")
BASE_OCI = Path("/hy-tmp/build/base_oci_pt")

# 发行版名 → 顶层模块名 的别名（大多数情况两者只是 -/_ 的区别）
ALIAS = {
    "pyyaml": "yaml", "pillow": "PIL", "scikit-learn": "sklearn",
    "open-clip-torch": "open_clip", "python-dateutil": "dateutil",
    "charset-normalizer": "charset_normalizer", "typing-extensions": "typing_extensions",
    "pydantic-core": "pydantic_core", "hf-xet": "hf_xet", "attrs": "attr",
    "protobuf": "google", "opencv-python": "cv2", "markdown-it-py": "markdown_it",
    "msgpack": "msgpack", "python-multipart": "multipart", "uvicorn": "uvicorn",
    "sentencepiece": "sentencepiece", "einops": "einops", "pytz": "pytz",
}


def norm(s):
    return s.lower().replace("-", "_").replace(".", "_")


def parse_req(spec):
    """`foo>=1.2,<2 ; extra == "x"` → ("foo", "<2,>=1.2")；可选依赖返回 None。"""
    if "extra ==" in spec:
        return None
    body = spec.split(";")[0].strip()
    m = re.match(r"^\s*([A-Za-z0-9_.\-]+)\s*(\(.*\))?\s*(.*)$", body)
    if not m:
        return None
    name, _, vers = m.group(1), m.group(2), m.group(3)
    vers = vers.strip()
    if vers.startswith("(") and vers.endswith(")"):
        vers = vers[1:-1]
    return name, vers


def installed_versions():
    """piproot 里每个发行版的版本（从 dist-info 目录名解析）。"""
    out = {}
    for d in PIPROOT.glob("*.dist-info"):
        parts = d.name[:-len(".dist-info")].rsplit("-", 1)
        if len(parts) == 2:
            out[norm(parts[0])] = parts[1]
    return out


def base_toplevel():
    """基座 site-packages 的顶层名（与 oci_build3.py step 3.7 同算法）。"""
    idx = json.loads((BASE_OCI / "index.json").read_text())
    md = idx["manifests"][0]["digest"].split(":")[1]
    mf = json.loads((BASE_OCI / "blobs/sha256" / md).read_text())
    if "manifests" in mf:
        md = mf["manifests"][0]["digest"].split(":")[1]
        mf = json.loads((BASE_OCI / "blobs/sha256" / md).read_text())
    names = set()
    for L in mf["layers"]:
        # 只需要"含 site-packages 的那几层"，但 tar -t 本身不贵
        blob = BASE_OCI / "blobs/sha256" / L["digest"].split(":")[1]
        flag = "-tzf" if "gzip" in L["mediaType"] else "-tf"
        r = subprocess.run(["tar", flag, str(blob)], capture_output=True, text=True)
        for line in r.stdout.splitlines():
            if "site-packages/" in line:
                rest = line.split("site-packages/", 1)[1]
                if rest:
                    names.add(rest.split("/")[0].split(".")[0])
    return names


def piproot_toplevel():
    n = set()
    for e in PIPROOT.iterdir():
        if e.is_dir():
            n.add(e.name)
        elif e.suffix == ".py":
            n.add(e.stem)
    return n


def requires_of(meta_text):
    out = []
    for line in meta_text.splitlines():
        if not line.startswith("Requires-Dist:"):
            continue
        spec = line[len("Requires-Dist:"):].strip()
        if "extra ==" in spec:            # 可选依赖，跳过
            continue
        m = re.match(r"^\s*([A-Za-z0-9_.\-]+)", spec)
        if m:
            out.append(m.group(1))
    return out


def main():
    base = base_toplevel()
    pip = piproot_toplevel()
    have = {norm(x) for x in (base | pip)}
    have |= {norm(v) for v in ALIAS.values()}
    have |= {norm(k) for k in ALIAS}
    vers = installed_versions()

    missing = {}
    badver = {}
    for d in sorted(PIPROOT.glob("*.dist-info")):
        mt = d / "METADATA"
        if not mt.exists():
            continue
        dist = d.name.rsplit("-", 2)[0]
        for line in mt.read_text(encoding="utf-8", errors="replace").splitlines():
            if not line.startswith("Requires-Dist:"):
                continue
            pr = parse_req(line[len("Requires-Dist:"):].strip())
            if pr is None:
                continue
            req, spec = pr
            key = norm(req)
            cand = {key, norm(ALIAS.get(req.lower(), req))}
            if not (cand & have):
                missing.setdefault(req, set()).add(dist)
                continue
            # ⚠️ **版本**也要对：只看"在不在"会漏掉下面这种会直接崩的错配 ——
            #   `pydantic 2.13.5` 要求 `pydantic-core==2.46.5`，而 piproot 里装的是 2.49.0
            #   ⇒ `import pydantic` 抛 SystemError ⇒ `import fastapi` 失败 ⇒ 服务起不来。
            if not spec:
                continue
            cur = vers.get(key)
            if cur is None:
                continue                       # 由基座提供，版本无法在这里核对
            try:
                from packaging.requirements import Requirement
                if not Requirement(f"{req}{spec}").specifier.contains(cur, prereleases=True):
                    badver.setdefault(f"{req}{spec}", []).append(f"{dist}: 现装 {cur}")
            except Exception:
                pass

    print(f"基座顶层名 {len(base)} 个；piproot 顶层名 {len(pip)} 个，发行版 {len(vers)} 个")
    rc = 0
    if missing:
        print(f"\n❌ 缺 {len(missing)} 个硬依赖（← 谁需要它）：")
        for k, v in sorted(missing.items()):
            print(f"   - {k:28s} ← {', '.join(sorted(v))}")
        Path("/hy-tmp/build/missing.txt").write_text(
            "\n".join(sorted(missing)) + "\n", encoding="utf-8")
        print("   已写 /hy-tmp/build/missing.txt")
        rc = 1
    if badver:
        print(f"\n❌ {len(badver)} 处**版本不满足**（← 谁要求的 / 现装版本）：")
        for k, v in sorted(badver.items()):
            print(f"   - 要求 {k:28s} ← {', '.join(sorted(v))}")
        rc = 1
    if rc == 0:
        print("✅ 所有硬依赖都能由 (piproot ∪ 基座) 提供，且 piproot 内的版本满足约束")
    return rc


if __name__ == "__main__":
    sys.exit(main())
