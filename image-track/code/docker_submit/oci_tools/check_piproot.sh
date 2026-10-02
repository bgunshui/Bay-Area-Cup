echo "=== pip 装的是什么（进程命令行）==="
ps -eo pid,etimes,cmd | grep -E "pip instal[l]" | grep -v grep | head -2 | cut -c1-400
echo
echo "=== piproot 现状 ==="
du -sh /hy-tmp/build/work/piproot 2>/dev/null
find /hy-tmp/build/work/piproot -maxdepth 1 -name "*.dist-info" 2>/dev/null | sed 's#.*/##' | sort | head -40
echo "  包数: $(find /hy-tmp/build/work/piproot -maxdepth 1 -name '*.dist-info' 2>/dev/null | wc -l)"
echo
echo "=== piproot 里有没有 torch/nvidia（会覆盖基座）==="
ls -d /hy-tmp/build/work/piproot/torch /hy-tmp/build/work/piproot/torchvision \
       /hy-tmp/build/work/piproot/nvidia /hy-tmp/build/work/piproot/triton 2>/dev/null || echo "  还没有"
du -sh /hy-tmp/build/work/piproot/* 2>/dev/null | sort -h | tail -12
echo
echo "=== 基座自带了哪些顶层包（torch 等）==="
for L in $(ls /hy-tmp/build/base_oci_pt/blobs/sha256/); do
  :
done
python3 - <<'PY'
import json, pathlib, subprocess, re
p = pathlib.Path("/hy-tmp/build/base_oci_pt")
idx = json.loads((p/"index.json").read_text()); m = idx["manifests"][0]
mf = json.loads((p/"blobs"/"sha256"/m["digest"].split(":")[1]).read_text())
names = set()
for L in mf["layers"]:
    blob = p/"blobs"/"sha256"/L["digest"].split(":")[1]
    flag = "-tzf" if "gzip" in L["mediaType"] else "-tf"
    r = subprocess.run(["tar", flag, str(blob)], capture_output=True, text=True)
    for line in r.stdout.splitlines():
        mm = re.match(r"opt/conda/lib/python3\.11/site-packages/([^/]+)/", line)
        if mm:
            names.add(mm.group(1))
        mm2 = re.match(r"opt/conda/lib/python3\.11/site-packages/([^/]+\.(?:py|pth|so))$", line)
        if mm2:
            names.add(mm2.group(1))
keep = sorted(n for n in names if not n.endswith(".dist-info") and not n.startswith("_"))
print(f"  基座 site-packages 顶层名 {len(keep)} 个")
key = [n for n in keep if n.lower().startswith(("torch","nvidia","triton","sympy","networkx",
       "jinja2","filelock","fsspec","typing","packaging","pyyaml","regex","requests","tqdm",
       "huggingface","safetensors","numpy","pillow","scipy","sklearn","pandas"))]
print("  与我们的依赖有交集的：")
for n in key:
    print(f"    {n}")
PY
echo
df -h /hy-tmp | tail -1
