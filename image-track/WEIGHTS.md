# 权重清单与获取方式

> ⚠️ **权重不在本仓库里** —— 两个原因：
> ① 4 个 cliph 权重每个 **160.2 MiB**，**超过 GitHub 单文件 100 MiB 硬限**，普通 git 推不上去；
> ② 即便用 LFS 也会让仓库膨胀到 ~1 GB，而队友"了解项目、准备答辩"并不需要它们。
>
> 需要跑完整推理时，按下表核对 md5 后从**交付镜像**或**本地备份**取。

---

## 一、8 个 LoRA + 分类头（`app/weights/`，合计 948.5 MB）

| 文件名 | 字节 | **md5** | 对应分量（融合权重） |
|---|---|---|---|
| `cliph_rerun.pt` | 167,932,286 | `b25444a889b9cc934422eb8752e422b7` | `img_cliph_rerun`（**0.150000**，占基底 42.25%） |
| `cliph_s20261101.pt` | 167,932,286 | `2e6ba4cd59d48f6789f4b0198ded77ec` | `img_cliph_s20261101`（0.150000） |
| `cliph_s20261102.pt` | 167,932,286 | `8203939b99e01d2ac3c03029e686cde2` | `img_cliph_s20261102`（0.150000） |
| `cliph_s20261103.pt` | 167,932,286 | `bc4aae614d06340f75f05b75601691be` | `img_cliph_s20261103`（0.150000） |
| `gpp.pt` | 17,940,837 | `719b4e4490342493812276b8d4c10a67` | `img_gpp_siglip2`（0.150000） |
| `bigg_rerun2.pt` | 101,619,683 | `4e7af069177009ff4f1421a031a96d04` | `img_bigg_rerun2`（0.071429） |
| `bigg_rerun3.pt` | 101,619,683 | `b6a0281fb5c19b670123fc43b52f9158` | `img_bigg_rerun3`（0.071429） |
| `bigg_s384_retrain.pt` | 101,662,168 | `c229ae5bd9df2035907e9f6a05ce88de` | `img_bigg_s384_retrain`（0.107143） |

> 每个 `.pt` 的结构：`{"lora": {…}, "head": {…}}`（HF 系）或 `{"vision": {…}, "head": {…}}`（cliph 系）。
> **cliph 权重是 fp32，bigG/SigLIP2 的 LoRA 是 fp16**（历史原因，见下"注意"）。

---

## 二、3 个骨干（`app/backbones/`，fp16，合计 9.3 GB）

| 目录 | 文件 | 字节 | **md5** |
|---|---|---|---|
| `clip_h14_cache/` | `open_clip_model.safetensors` | 1,972,298,570 | `2985b68a5c235b80f9f4e9dd2bff0086` |
| `clip_h14_cache/` | `open_clip_pytorch_model.bin`（fallback，运行期不读） | 1,972,452,464 | `2e525f2d18ae90d7f7f1e50cf6add7e3` |
| `clip_bigg/` | `model.safetensors` | 5,079,296,074 | `62ce47c9e2ee8dc28005b3caed673c85` |
| `siglip2_384/` | `model.safetensors` | 2,272,125,644 | `7c94d5dac450ae4c7c486826518ae739` |

**上游来源**（都能从公开源重新取得，且已验证**逐字节一致**）：

| 骨干 | 上游仓库 | 转换脚本 |
|---|---|---|
| `clip_h14_cache` | HF `laion/CLIP-ViT-H-14-laion2B-s32B-b79K`（取 `open_clip_pytorch_model.bin` + `config.json`） | `bb_fp16.py` → `fix_openclip.py` |
| `clip_bigg` | HF `laion/CLIP-ViT-bigG-14-laion2B-39B-b160k`（取两个 `pytorch_model-0000X-of-00002.bin` 分片） | `bb_fp16.py` → `fix_bigg_st.py` |
| `siglip2_384` | HF/ModelScope `google/siglip2-so400m-patch14-384`（`model.safetensors`，源 md5 `df94f80642aae7977664188a9e986ed7`，4,544,143,072 字节） | `bb_fp16.py` |

> 转换脚本在 `code/docker_submit/`（`bb_fp16.py`、`fix_openclip.py`、`fix_bigg_st.py`）。
> **实测**：按上表重建出的三个文件与镜像内**逐字节一致**（md5 全部命中）。

---

## 三、查表与先验部件（`app/tables/`，476 KB，**已在仓库内** ✅）

| 文件 | 字节 | md5 |
|---|---|---|
| `ecdf_final.npz` | 131,485 | `5d1a4427b727de36ba1d24afcfb65aad` |
| `ecdf_models.npz` | 111,392 | `10a6d95b272a5878777d909a337f671e` |
| `ecdf_prior.npz` | 55,329 | `b2da423974be541a98cd10e5a13b8409` |
| `fusion_meta.json`（**权威配方**） | 1,826 | `36e9a83fa56fa1919499dd11c81a2cae` |
| `gbm.pkl` | 163,870 | `c88043901a86b7fb864cef55880a410b` |
| `bucket.json` | 1,197 | `14a9f08756648c30f332117021705c33` |
| `prior_meta.json` | 331 | `221680ed478b91bd3860dc87f907dd18` |

---

## 四、怎么拿到权重（三选一）

| 途径 | 说明 |
|---|---|
| **① 交付镜像（推荐）** | `图像附件资料.tar.gz` 内的 `baycup_image_aidet_0.918887.tar`（18,984,816,640 字节，sha256 `fe2fc57e5805f059ec62992ae838a90745142f908b778a9fbcc10b528f562fc1`）。`docker load` 后从容器里 `docker cp` 出 `/app/weights` 与 `/app/backbones` |
| **② 本地备份** | 队长本机 `D:\湾区杯\图像赛道\docker_submit\app\weights\`（8 个 .pt）与 `_pkg_src\baycup_image_aidet_0.918887.tar`（完整镜像） |
| **③ 重新生成骨干** | 按上表从 HF/ModelScope 取上游文件，依次跑 `bb_fp16.py` / `fix_openclip.py` / `fix_bigg_st.py`，再用 md5 核对 |

---

## 五、注意（踩过的坑，别重犯）

1. **cliph 权重是 `{"vision":…}`，不是 `{"lora":…}`** —— 用错加载器会**静默命中 0 个张量**（只打一行警告）。
   `code/probe/probe_cliph.py` 与 `app/models.py` 都做了键名归一化，并**断言映射数 == 张量数**。
2. **LoRA 的 rank 必须从 checkpoint 推断**（cliph/bigg_rerun 是 r=16，`bigg_s384_retrain` 是 r=32），
   写死会在 `load_state_dict` 时形状不匹配或静默错配。
3. **同家族同 rank 的分量会共享同一份 peft 包装器** ⇒ 打分前**必须换入自己的 LoRA**
   （`models.py` 的 `_swap_lora`），否则 8 个里会错 4 个（实测主分量会从 0.999 掉到 0.335）。
4. **`open_clip_pytorch_model.bin` 是 fallback**：`open_clip_model.safetensors` 存在时运行期不读它，
   所以它与镜像内那份有 439 字节差异（torch 版本序列化差异）**不影响结果**。
