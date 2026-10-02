#!/usr/bin/env python3
"""在 chroot 内运行:用镜像里真实的 /app/server.py 启 uvicorn,走真实 HTTP 打接口。

阶段 1:严格按镜像 CMD(cwd=/app, `python server.py`)启动,但把模型目录指向不存在的路径,
        验证进程能起、能绑 8080、探针语义正确(liveness 200 / startup 503 且带错误详情)。
阶段 2:用同构缩小模型 + 真实 adapter 权重注入 STATE,启动真实 FastAPI 应用,
        验证 readiness 200、/predict 返回合法 score、healthcheck.py 退出码 0。
"""
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

APP = "/app"
ENV_BASE = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "PYTHONUNBUFFERED": "1",
    "PATH": "/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
    "FOLDS_14B": "5", "FOLDS_7B": "5", "W_14B": "0.90", "W_7B": "0.10", "MAX_LEN": "512",
}


def http(path, payload=None, timeout=10):
    url = "http://127.0.0.1:8080" + path
    try:
        if payload is None:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.status, r.read().decode()
        rq = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                    headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(rq, timeout=timeout) as r:
            return r.status, r.read().decode()
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()


def wait_free_port(seconds=20):
    t0 = time.time()
    while time.time() - t0 < seconds:
        try:
            http("/health/liveness", None, 2)
            return False
        except Exception:
            return True
        time.sleep(1)
    return True


def kill(p):
    p.terminate()
    try:
        p.wait(timeout=20)
    except Exception:
        p.kill()


print("=== 内存 ===")
os.system("head -2 /proc/meminfo")

# ---------------------------------------------------------------- 阶段 1
print("\n=== 阶段 1:镜像 CMD 原样启动(cwd=/app, python server.py),模型目录指向不存在路径 ===")
env = dict(ENV_BASE)
env["MODEL_DIR_14B"] = "/nonexistent14b"
env["MODEL_DIR_7B"] = "/nonexistent7b"
env["ADAPTER_ROOT"] = "/app/adapters"
log1 = open("/tmp/phase1.log", "w")
t0 = time.time()
p1 = subprocess.Popen([sys.executable, "server.py"], cwd=APP, env=env,
                      stdout=log1, stderr=subprocess.STDOUT)
bound = False
while time.time() - t0 < 60:
    try:
        s, b = http("/health/liveness", None, 3)
        if s == 200:
            bound = True
            print("  uvicorn 已监听 8080, /health/liveness -> 200 ✓  (%.1fs)" % (time.time() - t0))
            break
    except Exception:
        pass
    time.sleep(2)
if not bound:
    print("  !! 60 秒内未监听 8080")
for path in ("/health/liveness", "/health/readiness", "/health/startup"):
    try:
        s, b = http(path, None, 10)
        print("  %-20s -> %s %s" % (path, s, b[:110]))
    except Exception as e:
        print("  %-20s -> 异常 %s" % (path, type(e).__name__))
kill(p1)
log1.close()
print("  阶段 1 日志尾部:")
with open("/tmp/phase1.log") as f:
    for line in f.readlines()[-6:]:
        print("     ", line.rstrip()[:150])

wait_free_port()

# ---------------------------------------------------------------- 阶段 2
print("\n=== 阶段 2:真实 FastAPI 应用 + 同构缩小模型 + 真实 adapter,真实 HTTP ===")
env2 = dict(ENV_BASE)
env2["E2E_FOLDS"] = os.environ.get("E2E_FOLDS", "2")
env2["MODEL_DIR_14B"] = "/app/model/Qwen2.5-14B-Instruct"
env2["MODEL_DIR_7B"] = "/app/model/Qwen2.5-7B-Instruct"
env2["ADAPTER_ROOT"] = "/app/adapters"
log2 = open("/tmp/phase2.log", "w")
t0 = time.time()
p2 = subprocess.Popen([sys.executable, "/tmp/fake_server.py"], cwd=APP, env=env2,
                      stdout=log2, stderr=subprocess.STDOUT)
ready = False
while time.time() - t0 < 600:
    if p2.poll() is not None:
        print("  !! 进程退出 returncode =", p2.returncode)
        break
    try:
        s, b = http("/health/startup", None, 5)
        if s == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(4)
print("  就绪: %s  耗时 %.1fs" % (ready, time.time() - t0))

if ready:
    for path in ("/health/liveness", "/health/readiness", "/health/startup"):
        s, b = http(path, None, 10)
        print("  %-20s -> %s %s" % (path, s, b[:80]))

    TEXTS = [("人类", "昨天和朋友去了那家新开的川菜馆,排队等了一个多小时才坐下,菜的分量挺足但太辣了。"),
             ("AI", "本研究提出了一种基于多模态特征融合的生成文本检测框架,实验表明该方法具有显著优势。"),
             ("短", "今天天气不错。"),
             ("空", "")]
    for name, t in TEXTS:
        t1 = time.time()
        s, b = http("/predict", {"req_id": name, "data": t}, 300)
        print("  predict[%s] %.2fs -> %s %s" % (name, time.time() - t1, s, b[:110]))

    # 边界:非法请求体(缺 req_id)
    s, b = http("/predict", {"data": "abc"}, 30)
    print("  缺 req_id -> %s %s" % (s, b[:110]))

    # healthcheck.py 在服务在线时应退出 0
    r = subprocess.run([sys.executable, "/usr/local/bin/healthcheck.py"], timeout=30)
    print("  服务在线时 healthcheck.py 退出码 = %d (期望 0)" % r.returncode)

kill(p2)
print("  阶段 2 日志尾部:")
log2.close()
with open("/tmp/phase2.log") as f:
    for line in f.readlines()[-8:]:
        print("     ", line.rstrip()[:150])

print("\n=== 服务已停,检查 healthcheck.py 离线退出码 ===")
r = subprocess.run([sys.executable, "/usr/local/bin/healthcheck.py"], timeout=30)
print("  无服务时 healthcheck.py 退出码 = %d (期望非 0)" % r.returncode)
