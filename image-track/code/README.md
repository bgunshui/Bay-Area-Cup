# 图像赛道 · 代码导览

> 三个目录分工明确：**服务化**（交付镜像里跑的那套）、**训练**（8 个分量怎么来的）、
> **融合**（8 个分量怎么合成提交文件）＋**探针**（怎么验证每个分量是对的）。

---

## 一、`docker_submit/` —— 交付镜像里的那一套（最重要）

**这就是打进 Docker 镜像的代码本体**，与交付 tar 内的文件**逐字节一致**。

| 文件 | 作用 | 备注 |
|---|---|---|
| `app/server.py` | FastAPI 服务：三探针 + `POST /predict` | 启动时预热 **8 个分量全部**（曾因预热用短键名导致探针永远 503） |
| `app/models.py` | 3 个骨干 + 8 个 LoRA 的加载与前向 | 含**同家族 LoRA 换入守卫**（`_lora_snapshot`/`_swap_lora`）；按 **(家族, rank)** 缓存（r=16 与 r=32 必须分开） |
| `app/fusion.py` | 逐图 ECDF + 三槽加权 + 尺寸先验 | 交付的核心：把批 rank 改写成单张可算 |
| `app/tables/` | ECDF 查表 + 融合权重 + 尺寸先验部件（476 KB） | `fusion_meta.json` 是**权威配方来源** |
| `Dockerfile` | 镜像定义（**pytorch 官方基座** + 依赖 + 应用层 + curl） | 与交付镜像逐项对应 |
| `requirements.txt` / `pip_freeze_image.txt` | 依赖版本（**以镜像内实测为准**） | 前者是可重建子集，后者是镜像内 `pip freeze` 原始输出 |
| `verify_reproduce.py` | **端到端复现**：`--from-csv` 不需 GPU/权重 | 输出 `spearman = 1.0000000000` 即通过 |
| `verify_container.sh` + `verify_container_extract.py` + `verify_container_scan.py` | **不需要 Docker、不需要 GPU 的成品镜像复检** | 用 chroot 代替 `docker run`，以 `appuser` 逐个 `open()` 镜像内文件 |
| `build.sh` / `prepare_assets.py` / `oci_build3.py` / `oci_tools/` | 构建链路（含无 Docker 特权的 OCI 手工组装路径） | `oci_build3.py` 内有两道硬闸门（依赖闭合、权限可读） |
| `check_weight_keys.py` / `check_submission.py` | 权重键名自检 / 提交文件格式自检 | — |

> ⚠️ 本目录**不含权重与骨干**（体积原因，且 4 个权重超过 GitHub 单文件 100 MiB 限制），见 [`../WEIGHTS.md`](../WEIGHTS.md)。

---

## 二、`training/` —— 8 个分量的训练脚本

| 脚本 | 产出哪个分量 / 作用 |
|---|---|
| `cliph_rerun.py` | **主槽核心** `img_cliph_rerun`：CLIP-H + **测试集软标签转导蒸馏**（教师=早期 anchor），占基底权重 42.25% |
| `bigg_rerun.py` | `img_bigg_rerun2` / `img_bigg_rerun3`：bigG@224 纯监督 LoRA（Tiny 28.8k） |
| `train_s384_sj.py` | `img_bigg_s384_retrain`：bigG@**384** 拉伸、LoRA **r=32**、Tiny 28.8k 纯监督 |
| `train_gpp_siglip2.py` + `infer_gpp.py` | `img_gpp_siglip2`：**SigLIP2-384 + GenImage++**（换骨干+换数据，是主槽里最独立的强分量） |
| `train_clean_s384.py` | 历史 `s384`（0.908114 配方里的那一格），作对照 |
| `train_cliph300k.py` | **纯监督对照**：同骨干换数据到 302k（证明"独立性来自监督方式而非数据量"） |
| `train_distill2.py` | 转导蒸馏的多组对照（d1/d2/d3/e1/e2），用于**证否**"换教师能造独立分量" |
| `extract_s384r_lora.py` | 从 7.5 GB 的 `best.pt` 里抽出 LoRA-only（~100 MB）便于备份 |

**共同约定**（跨脚本必须一致，否则会静默出错）：

- `lora_alpha = 2 × r`（⇒ scaling 恒为 2.0）
- **crop 系必须 `interpolate_pos_encoding=True`**（bigG 原生 224，喂 384 要插值）
- val 划分种子统一 `random_state = 20260925`（否则**跨模型比 val 无意义**）
- 存盘用 `head.state_dict()`（`head` 是 `nn.Linear` 模块，不是张量）

---

## 三、`probe/` —— 怎么验证每个分量是对的

| 脚本 | 作用 |
|---|---|
| `probe_ckpt.py` | 给 HF 系（bigG / SigLIP2）的 LoRA checkpoint 出 20000 行分量 CSV；**断言 LoRA 映射数 == 张量数**，否则抛错 |
| `probe_cliph.py` | 给 open_clip 系（CLIP ViT-H-14）出分量 CSV（键结构与 `probe_ckpt.py` 不同） |
| `probe_matrix.py` | 出**分量间相关矩阵**，用来判断"往哪里注入新分量" |

---

## 四、`fusion/` —— 从 8 个分量到提交文件

| 脚本 | 作用 |
|---|---|
| `build_final.py` | **最终配方（权威）**：`SLOTS`/`RECIPES` 写在文件顶部；产出 `v100_gpp20_seed4_c75_prior200.csv` 及其变体；**自检**：先用旧基底重建已知文件，corr 必须 >0.999999，否则不出文件 |
| `build_slot_multi.py` / `build_slot_blend.py` | 槽内多源/部分混入（往主槽注入独立分量） |
| `size_prior2.py` | 尺寸先验（A+D 混合），OOF AUC 0.8247 |
| `check_submission.py` | 提交文件合规自检（列名/行数/id 集合与顺序/范围/唯一值） |

> `make_tables.py` 与 `make_prior.py` 在 `docker_submit/` 里（它们同时是交付包的一部分）：
> 前者把"批 rank 融合"固化成可部署的 ECDF 查表，后者把尺寸先验固化成查表 + GBM 部件。

---

## 五、最小复现路径

> 每条命令都标注了**执行目录**（都以本仓库的 `image-track/code/` 为基准写）。

```bash
# ① 提交文件逐图可重建（不需要 GPU / 权重 / Docker）—— 在 code/docker_submit/ 下执行
cd docker_submit
python -m venv .venv && .venv/bin/pip install -r requirements-verify.txt   # 约 1 分钟
.venv/bin/python verify_reproduce.py --from-csv
#    → 必须输出 spearman = 1.0000000000
#    ⚠️ 需要 numpy ≥ 2（app/tables/gbm.pkl 是 numpy 2.x 序列化的）

# ② 有 GPU 时：出 8 个分量的 CSV，再重跑融合 —— 在 code/ 下执行
cd ..
python probe/probe_ckpt.py        # HF 系分量（SigLIP2 / bigG），需 TFM/SHORT/CROP 环境变量
python probe/probe_cliph.py       # open_clip 系分量（CLIP ViT-H-14）
python fusion/build_final.py      # 按文件顶部 SLOTS 配方合成提交文件（自带 corr 自检）

# ③ 有 GPU 时：起服务打三探针 + /predict —— 在 code/ 下执行（脚本是 bash，**需 Linux**）
bash docker_submit/verify_container.sh <镜像 tar>     # 用 chroot 摊开镜像，无需 Docker 守护进程
```
