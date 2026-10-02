"""在真机 4090 上验证加固版 server.py:设备放置、启动耗时、单条推理耗时、融合分数。"""
import json
import os
import subprocess
import sys
import time
import urllib.request

SP = "/root/sj-tmp/rootfs/usr/local/lib/python3.10/site-packages"
ROOT = "/root/sj-tmp/rootfs/app"
LOG = "/root/sj-tmp/gpu_server.log"

env = dict(os.environ)
env.update({
    "PYTHONPATH": SP,
    "MODEL_DIR_14B": ROOT + "/model/Qwen2.5-14B-Instruct",
    "MODEL_DIR_7B": ROOT + "/model/Qwen2.5-7B-Instruct",
    "ADAPTER_ROOT": ROOT + "/adapters",
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "TOKENIZERS_PARALLELISM": "false",
    "FOLDS_14B": "5", "FOLDS_7B": "5",
    "W_14B": "0.90", "W_7B": "0.10", "MAX_LEN": "512",
})


def gpu_used():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        return o.stdout.strip()
    except Exception as e:
        return "n/a " + str(e)


print("GPU 载入前:", gpu_used(), flush=True)
log = open(LOG, "w")
p = subprocess.Popen([sys.executable, "-S", "/root/sj-tmp/server_v4.py"],
                     env=env, stdout=log, stderr=subprocess.STDOUT, cwd="/root/sj-tmp")
print("server pid", p.pid, flush=True)


def req(path, data=None, timeout=10):
    url = "http://127.0.0.1:8080" + path
    if data is None:
        r = urllib.request.urlopen(url, timeout=timeout)
    else:
        rq = urllib.request.Request(url, data=json.dumps(data).encode(),
                                    headers={"Content-Type": "application/json"})
        r = urllib.request.urlopen(rq, timeout=timeout)
    with r:
        return r.status, r.read().decode()


t0 = time.time()
ready = False
while time.time() - t0 < 900:
    if p.poll() is not None:
        print("!! 服务进程已退出, returncode =", p.returncode, flush=True)
        break
    try:
        s, b = req("/health/startup", None, 5)
        if s == 200:
            ready = True
            break
    except Exception:
        pass
    time.sleep(5)

startup = time.time() - t0
print("STARTUP_SECONDS", round(startup, 1), "ready", ready, flush=True)
print("GPU 载入后:", gpu_used(), flush=True)

if ready:
    for probe in ("/health/liveness", "/health/readiness", "/health/startup"):
        try:
            s, b = req(probe, None, 10)
            print(f"  {probe} -> {s} {b[:60]}", flush=True)
        except Exception as e:
            print(f"  {probe} -> 异常 {type(e).__name__}: {e}", flush=True)

    TEXTS = [
        ("人类-评论", "昨天下午和朋友去了那家新开的川菜馆,排队等了一个多小时才坐下。菜的分量挺足,但是真的太辣了,我喝了两瓶冰水才缓过来。服务员态度还不错,就是上菜有点慢。"),
        ("AI-论文", "本研究提出了一种基于多模态特征融合的生成文本检测框架。通过引入跨模态注意力机制与对抗训练策略,该方法在多个公开数据集上取得了显著优于基线的性能,证明了所提方法的有效性与鲁棒性。"),
        ("短文本", "今天天气不错。"),
        ("英文", "The quick brown fox jumps over the lazy dog. " * 6),
        ("空串", ""),
    ]
    times = []
    for name, t in TEXTS:
        t1 = time.time()
        try:
            s, b = req("/predict", {"req_id": name, "data": t}, 300)
            dt = time.time() - t1
            times.append(dt)
            print(f"  predict[{name}] {dt:.2f}s -> {b[:120]}", flush=True)
        except Exception as e:
            print(f"  predict[{name}] 异常 {type(e).__name__}: {e}", flush=True)

    if times:
        print("PREDICT_SECONDS min=%.2f max=%.2f avg=%.2f" % (min(times), max(times), sum(times) / len(times)), flush=True)

log.flush()
p.terminate()
try:
    p.wait(timeout=30)
except Exception:
    p.kill()
print("已停止服务", flush=True)
print("===== server 日志(前 40 行)=====")
with open(LOG) as f:
    for i, line in enumerate(f):
        if i >= 40:
            break
        print("   ", line.rstrip())
print("===== server 日志(尾 15 行)=====")
with open(LOG) as f:
    lines = f.readlines()
for line in lines[-15:]:
    print("   ", line.rstrip())
