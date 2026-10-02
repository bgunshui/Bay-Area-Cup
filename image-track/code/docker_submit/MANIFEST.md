# 提交材料清单（MANIFEST）—— 湾区杯复赛 · 图像赛道

- 对应 LB 成绩：**AUC = 0.918887**
- 提交文件：`v100_gpp20_seed4_c75_prior200.csv`（20000 行 `id,score`）
- 交付标准：《分类模型 Docker 交付标准 v1.2》
- 统一压缩包：**`图像附件资料.tar.gz`**（用户要求：一整份资料一个包，命名"图像附件资料"）
- 打包时间：**2026-09-18 21:28:57**

---

## 一、包内结构（解压后）

```
图像附件资料/
├── 00_交付说明.md                      一页纸：对应哪两项要求、怎么用
├── MANIFEST.md                         本文件：全部文件的字节数 + sha256 台账
├── Docker镜像/
│   └── baycup_image_aidet_0.918887.tar  ① Docker 镜像（docker load 可直接用，7 层）
├── 技术方案报告/
│   ├── AS2_技术方案_TECHNICAL_REPORT.md ② 技术方案报告（模型/数据/融合/工程/复现）
│   └── AS3_提交说明_README.md           交付包使用说明（接口、资源声明、自查清单）
└── 构建与复现套件/
    ├── docker_submit/                   完整源码 + 8 个权重 + 查表 + 构建/复检脚本
    │   ├── Dockerfile                   ★ 与交付镜像逐项对应（官方 pytorch 基座）
    │   ├── requirements.txt             依赖版本（**以镜像内实测为准**）
    │   ├── pip_freeze_image.txt         镜像内 `pip freeze` 原始输出（151 行，权威凭据）
    │   ├── build.sh / prepare_assets.py / make_tables.py / make_prior.py
    │   ├── oci_build3.py                无 Docker 特权时的 OCI 组装路径（本次实际用的）
    │   ├── verify_reproduce.py          端到端复核（--from-csv 不需 GPU/权重）
    │   ├── verify_container.sh          ★ 不需 GPU 的镜像复检（chroot 代替 docker run）
    │   ├── app/                         server.py / models.py / fusion.py
    │   │   ├── weights/                 8 个 LoRA+head 权重
    │   │   └── tables/                  融合查表 + 尺寸先验部件
    │   └── oci_tools/                   构建闸门与导出工具
    └── <10 个冻结 CSV>                  逐图重建 0.918887 的输入（见下）
```

> **关于 `AS1_构建套件_buildkit.tar.gz`**：它只是"`docker_submit/` + 10 个 CSV"的压缩版，
> 而这两样本包里**已经原样解开着**，放进来是纯重复（约 900 MB），故**不重复收录**。
> 如需该文件，一条命令即可再生：
> `tar czf AS1_构建套件_buildkit.tar.gz docker_submit v100_*.csv img_*.csv test_specs.csv`

---

## 二、Docker 镜像（①）

| 项 | 值 |
|---|---|
| 文件 | `Docker镜像/baycup_image_aidet_0.918887.tar` |
| 字节数 | **18,984,816,640** |
| sha256 | **`fe2fc57e5805f059ec62992ae838a90745142f908b778a9fbcc10b528f562fc1`** |
| 格式 | `docker save` 兼容的 docker-archive（`manifest.json` + 7 层 + config） |
| 架构 | `linux/amd64` |
| 基座 | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`（**pytorch 官方镜像**） |
| 镜像 tag | `baycup-image-aidet:0.918887` |
| 层数 | **7**（基座 3 层 + pip 依赖层 + 应用层 + 2 个元数据层 + **curl 补丁层**） |
| 运行身份 | `appuser`（uid 1000，非 root） |

```bash
docker load -i baycup_image_aidet_0.918887.tar
docker run --gpus all -p 8080:8080 baycup-image-aidet:0.918887
# 就绪后：GET /health/{startup,liveness,readiness} + POST /predict
```

### 这一版镜像修掉了什么（两次真实交付事故）

| # | 现象 | 根因 | 修复 |
|---|---|---|---|
| 1 | **「镜像权重崩溃」**（上一版被驳回的原因） | 3 个骨干 `.safetensors` 源文件是 `-rw------- root:root`（fp16 转换脚本 umask=077；`.bin` 是 `cp` 来的所以是 0644，**只有 safetensors 中招**），而容器以 `User=appuser` 运行 ⇒ 加载骨干时 `PermissionError` | 构建时 `chmod -R a+rX` + **构建闸门：以 appuser 逐个 `open()` 层内每个文件，`UNREADABLE != 0` 直接失败**；Dockerfile 里也补了 `chmod` |
| 2 | `HEALTHCHECK` 永远 `unhealthy` | pytorch 官方基座**不含 `curl` 可执行文件**（只有 `curl-config`），而 HEALTHCHECK 就是 `curl -f …` | 新增一层 2.5 MB 的 curl 补丁层（curl 7.81.0 + 10 个缺失动态库，与基座同为 Ubuntu 22.04 jammy，ABI 一致） |

两次修复都做了**独立复检**，方法与证据见 `docker_submit/verify_container.sh` 与技术方案报告 §9。

---

## 三、技术方案报告（②）

| 文件 | 字节数 | sha256 |
|---|---|---|
| `技术方案报告/AS2_技术方案_TECHNICAL_REPORT.md` | `45124` | `6df8810c65ee16e7819ff2d7c45554f8aeeb6705cb25e6bf1b60ab8b0d5967b2` |
| `技术方案报告/AS3_提交说明_README.md` | `9627` | `a53011ffe4ec14de493845bf52cf584507f8e7d822d826697db84ac3fa92e6ea` |

---

## 四、复现 0.918887 只需 6 个 CSV（不需要 GPU、不需要权重）

解压后：

```bash
python 构建与复现套件/docker_submit/verify_reproduce.py --from-csv
# 必须输出 spearman = 1.0000000000
```

包内 `构建与复现套件/` 根目录下的 10 个 CSV：

| 文件 | 作用 |
|---|---|
| `v100_gpp20_seed4_c75_prior200.csv` | **提交文件本身**（0.918887） |
| `img_cliph_rerun.csv`、`img_cliph_s20261101/02/03.csv` | cliph 槽的 4 个分量 |
| `img_gpp_siglip2.csv` | gpp 分量 |
| `img_bigg_rerun2.csv`、`img_bigg_rerun3.csv` | bigg 槽的 2 个分量 |
| `img_bigg_s384_retrain.csv` | s384 槽的分量 |
| `test_specs.csv` | 测试集 20000 张的宽高/字节数（尺寸先验要用） |

---

## 五、镜像复检（不需要 GPU）

```bash
bash 构建与复现套件/docker_submit/verify_container.sh <镜像 tar>
```

它用 `chroot` 代替 `docker run`（因为构建环境拿不到 Docker 特权），逐项检查：

1. 把镜像的每一层摊开成一个 rootfs（层 tar 直接解，后层覆盖前层）
2. **以最终运行身份 `appuser`(uid 1000) 打开 `/app` 下每一个文件**（这正是事故 1 的失败点）
3. 以 appuser 逐个 `import` 运行期依赖
4. 解析 3 个骨干的 `safetensors` 头并用 `safe_open` 真读首/末张量
5. 启动 `server.py`，检查三探针语义
6. 以 appuser 实跑 HEALTHCHECK 那条 `curl -f` 命令

---

## 六、校验

```powershell
certutil -hashfile 图像附件资料.tar.gz SHA256
```

包自身的 sha256 写在**包外面**的 `图像附件资料.tar.gz.sha256.txt` 里
（不能写进包内 —— 那会自指，改一个字节就全变）。
