# 真机 GPU 端到端验证报告(2026-09-18)

服务器:`root@<HOST-B> -p <PORT>`(hostname `<HOSTNAME>`)
GPU:**NVIDIA GeForce RTX 4090,49,140 MiB**(驱动 595.71.05 / CUDA 13.2 能力)

## 一、验证方式说明(为什么不是 `docker run`)

该服务器**没有可用的 Docker 守护进程**:本机是受限容器
(`CapEff=00000000a80425fb`,**无 `CAP_SYS_ADMIN`**),`dockerd` 无法启动;
`chroot` 里也无法 `mount --bind /dev`、`/proc`,所以 chroot 内 `torch.cuda.is_available()` 为 False。

因此改用等价且更严格的验证路径:

- 宿主是 **Ubuntu 22.04 / Python 3.10.12 / glibc 2.35**
- 镜像的依赖全是 **cp310 / manylinux_2_17~2_28** wheel,**ABI 与宿主 Python 3.10 完全一致**
- 用 `PYTHONPATH=<镜像 site-packages> /usr/bin/python3.10 -S` 直接把镜像的依赖挂到宿主 Python 上
  (`-S` 用于隔离宿主自带的老 torchvision,否则 transformers 会报 `operator torchvision::nms does not exist`)
- 模型与适配器直接指向镜像内的 `/app/model`、`/app/adapters`
- 用 `server_dual_v4.py`(**即最终镜像里的 `app/server.py`**)启动服务

即:**除了少一层容器隔离,依赖、代码、模型、精度、CUDA 路径全部与镜像内一致**。

## 二、依赖自检(镜像 site-packages)

```
python 3.10.12
torch 2.5.1+cu124            cuda True
transformers 4.46.0          peft 0.20.0        tokenizers 0.20.3
fastapi 0.115.0              pydantic 2.9.2     uvicorn / starlette / numpy / scipy
torch.distributed.tensor 预加载 OK
```

## 三、关键指标

| 指标 | 实测 | 提交声明 | 平台上限 | 结论 |
|---|---|---|---|---|
| 启动(43 GB 权重 + 15 adapter) | **70.1 秒** | ~180 秒 | 300 秒 | 余量充足 |
| 单条推理(15 次前向) | **min 0.65 / max 0.74 / avg 0.68 秒** | ~3 秒 | 90 秒 | 余量充足 |
| 显存占用 | **41,868 MiB / 47.4 GiB** | ≥ 48 GB | — | 余量约 6.5 GiB |
| 14B 侧加载 | 44 秒 | — | — | 8 个分片 |
| 全部加载完成 | 68 秒 | — | — | 14B=5 fold 7B=10 fold |

## 四、设备放置修复已确认生效(本次最重要的验证点)

旧版 server.py 写成 `if dev != "cuda": model.to(dev)`,而 transformers 在未传 `device_map` 时
会把模型留在 CPU —— 结果是**有 GPU 时反而不搬运**,14B 会在 CPU 上跑 15 次前向,
单请求远超 90 秒上限,决赛必然失败。

真机日志(修复后):

```
[load] GPU=NVIDIA GeForce RTX 4090 mem=47.4GiB
[load] .../Qwen2.5-14B-Instruct 经 device_map 加载到 cuda:0
[load] lora14b_p4 基座就位 device=cuda:0 dtype=torch.bfloat16
[load] lora14b_p4: 5 adapters device=cuda:0
[load] 14B done 44s
[load] lora7b_distill_r3a: 5 adapters device=cuda:0
[load] ALL done 68s  14B=5 7B=10
```

`device=cuda:0` 明确出现在两侧,且显存占用 41.8 GiB 证明权重确实在显卡上。

## 五、探针与推理接口

```
/health/startup    -> 200 {"status":"ok"}    (加载期间正确返回 503)
/health/liveness   -> 200 {"status":"ok"}
/health/readiness  -> 200 {"status":"ok"}

POST /predict
  人类-评论 0.74s -> {"code":0,"score":0.3934}
  AI-论文   0.68s -> {"code":0,"score":0.6951}
  短文本    0.65s -> {"code":0,"score":0.3521}
  英文      0.68s -> {"code":0,"score":0.5204}
  空串      0.66s -> {"code":0,"score":0.7571}   (eos 兜底,不报错)
```

不同风格文本给出明显不同的分数,说明 0.90×14B + 0.10×7B 融合路径在真卡上正常工作。

## 六、未做的验证(如实说明)

未做 20005 条复赛集的全量分数复现:
- 全量耗时约 3.8 小时(0.68s × 20005)
- 离线 rank 融合与线上 sigmoid 融合的排序相关性此前已用 20000 条实测为 **0.999997**
  (排序差异 >0.01 的样本占比 0.065%),本报告不重复该结论
- 若需要,可在该机直接跑:预计 4 小时内出全量 AUC

## 七、本次在服务器上新增的文件

```
/root/sj-tmp/server_v4.py     加固版推理服务(与镜像内 app/server.py 同源)
/root/sj-tmp/gpu_driver.py    验证驱动脚本
/root/sj-tmp/gpu_server.log   完整服务日志
```
