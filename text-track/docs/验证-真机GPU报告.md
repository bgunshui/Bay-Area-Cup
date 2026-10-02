# 真机 GPU 小样本验证记录（历史测试 2026-09-18）

> 2026-10-02 交接核对：保留原报告的指标与日志摘录，并按入库驱动限定其覆盖。完整服务日志、完整驱动输出和原始设备记录未入库；本次没有重新运行 GPU 测试。

历史服务器为 `root@<HOST-B> -p <PORT>`，hostname `<HOSTNAME>`。原报告写 GPU 为 NVIDIA GeForce RTX 4090、49,140 MiB，驱动 595.71.05；这些是原记录，待原始 `nvidia-smi` 核对设备型号和实际容量。

## 一、实际验证路径与版本

历史报告称服务器没有可用 Docker 守护进程，受限容器无法正常启动 dockerd 或为 chroot 绑定 GPU 设备，因此改用：

- Ubuntu 22.04 / Python 3.10.12 宿主。
- `PYTHONPATH` 指向镜像抽取的 Python 3.10 site-packages，`-S` 隔离宿主依赖。
- 模型和 adapter 路径指向抽取的镜像资产。
- 启动加固版 v4。仓库驱动实际启动 `/root/sj-tmp/server_v4.py`，未在脚本中检查该运行文件哈希。

当前 `server_dual_v4.py` 源码 SHA256 为 `7d868df187907a411ac5cf1f0631ed0470befb732cebe9ad14a440e785b636ed`，与历史终审报告所写的最终镜像服务哈希一致。当前工作区旧单层镜像的服务是 v2；新最终双层镜像（53,317,416,960 B，a163…91f68）未交接本地。

这是宿主环境的真模型烟测记录，不是 `docker run`。宿主 Python 补丁版本、动态库搜索与容器隔离条件存在差别，不能称比真实容器验证“等价且更严格”。

## 二、原报告的依赖摘录

```text
python 3.10.12
torch 2.5.1+cu124            cuda True
transformers 4.46.0          peft 0.20.0        tokenizers 0.20.3
fastapi 0.115.0              pydantic 2.9.2
torch.distributed.tensor 预加载 OK
```

上述为报告摘录，不是本次重新采集输出。镜像自身 Python 的历史记录是 3.10.21。

## 三、原报告记录的指标及适用范围

| 指标 | 历史记录 | 可解释的范围 |
|---|---|---|
| 启动等待 | 70.1 秒 | 驱动每 5 秒轮询 startup 的一次观测，包含加载/预热，非精确分阶段性能基准 |
| 单条请求 | min 0.65 / max 0.74 / avg 0.68 秒 | 五条短样本逐条各请求一次，含 HTTP 往返；不是任意长度或并发性能 |
| 显存使用 | 41,868 MiB | 约 40.89 GiB；驱动读取整卡 `memory.used`，不是独立测量服务进程峰值 |
| 14B 加载 / 全部就位 | 44 秒 / 68 秒 | 来自服务摘录；完整配置为 14B 5 adapter、7B 10 adapter |

原报告同时写“总量 49,140 MiB”和日志“47.4 GiB”：前者换算约 47.99 GiB，二者不一致，暂不据此计算精确余量。两个基座文件合计约 41.70 GiB，15 个 adapter 文件合计 907,824,320 B（约 865.77 MiB）；磁盘体积与运行显存不是同一指标。

## 四、设备放置的源码与历史日志

v2 的 `if dev != "cuda": model.to(dev)` 没有在 CUDA 分支明确迁移；v4 增加 `device_map={"":0}` 与 `.to(dev)` 兜底。不要把未经测量的旧版 CPU 延迟写成“必然超过 90 秒”；完整推理是 14B 五次加 7B 十次前向。

原报告保留的修复后日志摘录：

```text
[load] GPU=NVIDIA GeForce RTX 4090 mem=47.4GiB
[load] .../Qwen2.5-14B-Instruct 经 device_map 加载到 cuda:0
[load] lora14b_p4 基座就位 device=cuda:0 dtype=torch.bfloat16
[load] lora14b_p4: 5 adapters device=cuda:0
[load] 14B done 44s
[load] lora7b_distill_r3a: 5 adapters device=cuda:0
[load] ALL done 68s  14B=5 7B=10
```

这些摘录支持历史测试使用 GPU 路径的记录，但不是独立完整日志。v4 仍捕获 `.to(dev)` 失败后返回模型，没有强制断言最终设备。

## 五、五条样本的历史接口记录

入库 `code/verify/gpu_driver.py` 的 `TEXTS` 有且只有以下五类样本，顺序运行；“人类”“AI”是手写样本名称，不是经标注评测集：

| 样本名称 | 时间 | 记录的成功分数 |
|---|---|---|
| 中文评论短文 | 0.74 s | 0.3934 |
| 论文风格短文 | 0.68 s | 0.6951 |
| 短句 | 0.65 s | 0.3521 |
| 英文短文 | 0.68 s | 0.5204 |
| 空串 | 0.66 s | 0.7571 |

原报告称三个探针均返回 200、加载期间 startup 返回 503。不同分数不能单独证明融合权重正确、检测质量良好或校准有效；对应源码可核查融合式，完整输出尚待取得。

## 六、未覆盖的项目与残留问题

- 没有真实 `docker load` / `docker run`；没有容器 GPU 注入与完整健康检查证据。
- 没有 512 tokens 长输入、纯空白专项、并发吞吐、排队延迟或重复试验。
- 没有 20000 条复赛集 HTTP 全量复现，更没有复赛公开标签可供独立 AUC 计算。不能用五短样本 0.68 秒线性保证全量耗时。
- 2026-10-02独立核对冻结CSV与线上数组：Spearman 0.9998697005807853；严格反序218,292对、单侧并列变化279对，合计约0.10929%。这不是GPU HTTP全量测试；旧0.999997与旧204,591对不作为当前复现结果。
- v4 已有锁与 EOS 兜底，但 NaN 防护仍先夹取后检查，adapter 缺失仍跳过，没有方案完整性的强制校验。

## 七、可交接与待交接文件

已入库：`code/docker_submit/server_dual_v4.py`、`code/verify/gpu_driver.py` 与本报告摘录。

仅在历史报告记录了服务器路径，未入库：`/root/sj-tmp/gpu_server.log`、完整驱动 stdout、原始设备信息与实际运行脚本哈希。需要交接这些原始材料，才能将历史烟测记录提升为可独立核验的运行证据。
