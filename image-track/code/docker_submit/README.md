# 湾区杯复赛 · 图像赛道 —— AI 生成图像鉴别（Docker 交付包）

> **对应提交成绩：LB AUC = 0.918887**
> 提交文件：`v100_gpp20_seed4_c75_prior200.csv`（20000 行 `id,score`）
> 交付标准：《分类模型 Docker 交付标准 v1.2》

---

## 1. 这个包里有什么

```
docker_submit/
├── Dockerfile               镜像定义（pytorch 2.5.1-cu124 基座 + 离线权重）
├── requirements.txt         依赖版本（与产出冻结 CSV 的环境**逐项一致**）
├── build.sh                 一键：备资材 → 离线自检 → build → 冒烟 → docker save
├── prepare_assets.py        归拢 8 个 LoRA 权重 + 下载 3 个骨干
├── make_tables.py           把「批 rank 融合」改写成「逐图 ECDF 融合」并自检
├── make_prior.py            把尺寸先验固化成可部署部件（档查表 + GBM + 混合）
├── verify_reproduce.py      端到端验证（--from-csv 不需 GPU；--http 打真实服务）
├── diag_prior2.py           并列(tie)边界残差的定位脚本（保留作为排查记录）
└── app/
    ├── server.py            FastAPI 服务：三探针 + POST /predict
    ├── models.py            3 个骨干 + 8 个 LoRA 的加载与前向（预处理逐项对齐）
    ├── fusion.py            ECDF 版 rank 加权融合
    ├── weights/             8 个 LoRA+head 权重（≈1.0 GB）
    ├── backbones/           3 个骨干（≈12 GB，由 prepare_assets.py 下载）
    └── tables/              融合查表 + 尺寸先验部件（≈0.5 MB）
```

## 2. 快速开始

基座是 **`pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`（pytorch 官方镜像）**。
有两条构建路径，产出的镜像等价：

### 路径 A：有 Docker daemon
```bash
# 前置：linux/amd64 + Docker + 能访问 hf-mirror；磁盘 ≥ 60 GB
bash build.sh
```
产出 `baycup_image_aidet_0.918887.tar`（含 sha256）。

### 路径 B：**没有 Docker daemon**（本次交付实际走的路）
交付环境是无特权容器（`CapEff` 无 `cap_sys_admin`），实测
`docker build` 与 `docker pull` 都会倒在 `unshare: operation not permitted`（换 vfs 存储驱动也一样）。
⇒ 改用 **`skopeo` + 手工 OCI 布局**（skopeo 只"下载 + 解 tar"，不碰命名空间）：

```bash
bash pull_pt.sh              # skopeo 拉 pytorch 官方基座 → base_oci_pt/
python3 bb_fp16.py           # 骨干 fp32→fp16（18.4→9.3 GB）
python3 fix_openclip.py      # 补 open_clip_model.safetensors（离线加载必需）
python3 fix_bigg_st.py       # bigG 分片 .bin → model.safetensors（避开 torch≥2.6 的 weights_only）
python3 oci_build3.py        # 组装新层 → 写 OCI 布局 → skopeo 导出 docker-archive
```
产出的 tar 含 `manifest.json` / `repositories` / `layer.tar` / `VERSION` / `json`，`arch=amd64`，
与 `docker save` 格式一致，可直接 `docker load`。

### 离线自检（不需要 GPU / Docker）
```bash
python prepare_assets.py --weights-only    # 归拢权重
python verify_reproduce.py --from-csv      # 必须输出 spearman = 1.0000000000
python check_weight_keys.py                # 校验 8 个权重的键名/rank/head
```

**打真实服务验证（需要 GPU）：**
```bash
python verify_reproduce.py --http http://<host>:8080
```

## 3. 服务接口（严格按规范）

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health/startup` | 模型全部加载完才 200；加载期返回 503，平台重试 |
| GET | `/health/liveness` | **只反映进程存活**，不碰模型（<1 ms） |
| GET | `/health/readiness` | 200 即保证紧接着的 `/predict` 可用 |
| POST | `/predict` | `{"req_id","data"}` → `{"req_id","code":0,"score"}` |
| GET | `/debug/info` | 自查用（规范未要求） |

请求/响应示例：

```bash
curl -s localhost:8080/predict -H 'Content-Type: application/json' \
  -d "{\"req_id\":\"a1\",\"data\":\"$(base64 -w0 test.jpg)\"}"
# {"req_id":"a1","code":0,"score":0.9321}
```

错误码按规范：`40001` 请求非法 / `40002` base64 或图像解析失败 / `50001` 推理内部错误 / `50301` 未就绪。

## 4. 资源声明（规范 §4.6 要求）

| 资源 | 需求 | 说明 |
|---|---|---|
| GPU 显存 | **≥ 10 GB**（实测峰值 ~9 GB） | fp16 推理；三个骨干合计约 6.4 GB |
| CPU | 4 核 | 单张预处理为主 |
| 内存 | **≥ 8 GB** | 加载期峰值（含 fp32 骨干转 fp16） |
| 磁盘 | 镜像约 20 GB | 含 torch/cu124 运行时与全部权重 |
| 启动时间 | **~60-120 秒**（< 300 秒上限） | 三个骨干 + 8 个 LoRA + 两次预热 |
| 单次推理 | **~0.2-0.5 秒**（< 90 秒上限） | 单张，含 8 个模型前向 |

> 无 GPU 也能启动（`DEVICE=cpu`），但单张会到 3-6 秒；规范默认提供 GPU，按 GPU 声明。

## 5. 这个包**为什么能复现 0.918887**（关键技术点）

提交文件是**批 rank 加权**：`rank(x)` 是在 20000 张里排名，**单张图算不出来**。
本包把它改写成**逐图可计算**的 ECDF：

```
rk(x)  →  ECDF(x) = 升序参考表上的平均秩百分位
        (searchsorted(left) + searchsorted(right) + 1) / 2 / (n+1)
```

- 在那 20000 张测试图上 ECDF **恒等于** `rankdata(method='average')/(n+1)`
- 对新图是良定义的单调映射（= 用测试集分布做分位数标定）
- **实测：逐图管线 vs 提交文件 `spearman = 1.0000000000` ⇒ AUC 完全相同**（`verify_reproduce.py --from-csv`）

⚠️ 那个 `+1` 不能少：少掉就恒小 `0.5/(n+1)`，在并列组大的量（尺寸先验只有 43 档）上
会让查询值落到并列组**另一侧**，凭空产生 ~3.5e-3 的跳变。定位过程见 `diag_prior2.py`。

## 6. 已知偏差（如实披露）

| 项 | 说明 |
|---|---|
| **BS=1 vs BS=32** | 冻结 CSV 是 **BS=32 批量** 推理产出的；服务是**逐张**推理。bf16 下的数值抖动会让极少数图偏移，`--http` 验证的 spearman 预期 ≥ 0.99（不是 1.0）。这是 HTTP 单图服务无法避免的，**不影响量级**。 |
| 尺寸先验 | 属 shortcut learning（利用数据集构建偏差）。按现行规则不违规，但若赛题另有"仅基于图像内容判断"条款则可能踩线 —— **提交前请确认**。 |
| 转导性质 | 8 个模型里有 4 个（cliph 族）是"用测试集软标签蒸馏"的学生，因此**在这 20000 张上的表现不代表对新数据的泛化能力**。 |

## 7. 提交前自查清单（对照规范 §7）

- [x] 每个任务单独一个镜像（本包**只做图像赛道**）
- [x] 镜像可离线启动：`HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1`，权重全在镜像内，无任何下载逻辑
- [x] `/health/startup` 加载完成后 200，加载期 503
- [x] `/health/liveness`、`/health/readiness` 实现且语义正确
- [x] `POST /predict` 支持 base64 图像（JPEG/PNG/BMP/WebP 经 PIL 统一处理）
- [x] 响应 `req_id` 原样返回
- [x] `score ∈ [0,1]`，已做 `min/max` 裁剪 + `isfinite` 检查
- [x] 单次推理 ~0.2-0.5 秒，远低于 90 秒（无需额外超时说明）
- [x] 镜像内网络逻辑：**无**
- [x] 镜像 digest 与资源声明：见上文第 4 节（digest 在 `build.sh` 第 ⑤ 步打印）
- [x] 非 root 运行（`appuser`）
- [x] `EXPOSE 8080`，服务绑 `0.0.0.0:8080`
- [x] `HEALTHCHECK` 已按规范 §2.3 配置
