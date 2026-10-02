# 提交文件与分量 CSV（`submission/`）

> 本目录是**整个复现链的输入**：10 个冻结 CSV，不含任何需要 GPU 的东西。
> 只要它们 + `code/docker_submit/app/tables/` 里的查表，就能**逐图重建 LB 0.918887 的提交文件**。

---

## 1. 文件清单

| # | 文件 | 角色 | 最终权重 |
|---|---|---|---|
| 0 | `v100_gpp20_seed4_c75_prior200.csv` | ★ **实际提交给组委会的文件**，`check_submission` 的 id 顺序基准 | — |
| 1 | `img_cliph_rerun.csv` | cliph 槽 · CLIP ViT-H-14 + LoRA，**转导蒸馏**（教师 = 旧 anchor） | 0.150000 |
| 2 | `img_cliph_s20261101.csv` | cliph 槽 · **同配方只换随机种子** `SEED=20261101` | 0.150000 |
| 3 | `img_cliph_s20261102.csv` | cliph 槽 · 同配方 `SEED=20261102` | 0.150000 |
| 4 | `img_cliph_s20261103.csv` | cliph 槽 · 同配方 `SEED=20261103` | 0.150000 |
| 5 | `img_gpp_siglip2.csv` | cliph 槽 · SigLIP2-so400m-384 + GenImage++ | 0.150000 |
| 6 | `img_bigg_rerun2.csv` | bigg 槽 · CLIP ViT-bigG-14 @224（Tiny 28.8k） | 0.071429 |
| 7 | `img_bigg_rerun3.csv` | bigg 槽 · 同上，另一次训练 | 0.071429 |
| 8 | `img_bigg_s384_retrain.csv` | s384 槽 · bigG @384，LoRA `r=32` | 0.107143 |
| — | `test_specs.csv` | **不是提交文件**：(id, w, h, fmt, bytes)，尺寸先验的输入 | — |

> 2/3/4 号是**多 seed 集成**：同配方只换种子训 3 个，两两 rank 相关平均仅 **0.9601**，
> 4 路平均把噪声标准差压缩 1.53%，主分量 AUC 从 0.9125 提到 **0.9158（+0.0033）**。

---

## 2. 最终配方（与 `code/docker_submit/app/tables/fusion_meta.json` 完全一致）

```
槽内 = Σ 分量内权重 × ECDF( 该分量分数 )        # ECDF 就是"在这 20000 张里的百分位"
基底 = Σ 槽权重 × 槽内
最终 = (1 − 0.20) × ECDF(基底)  +  0.20 × ECDF(尺寸先验)
```

| 槽 | 槽权重 | 槽内分量（各占槽内多少） |
|---|---|---|
| `cliph` | **0.750000** | `cliph_rerun` 0.20 / `s20261101` 0.20 / `s20261102` 0.20 / `s20261103` 0.20 / `gpp_siglip2` 0.20 |
| `bigg` | **0.142857** | `bigg_rerun2` 0.50 / `bigg_rerun3` 0.50 |
| `s384` | **0.107143** | `bigg_s384_retrain` 1.00 |
| 尺寸先验 | **0.20**（在槽之外叠加） | 查表 + GBM 混合，OOF AUC 0.8247 |

---

## 3. 哈希台账（**逐字节权威**）

GitHub 上原始下载的文件与下表**逐字节相同**：本仓库根有 `.gitattributes` 声明 `* -text`，
禁止 Git 做任何 CRLF/LF 转换（Windows 默认的 `core.autocrlf=true` 会静默改写字节，
使仓库里的文件与实际提交件不一致）。

| 文件 | 字节 | md5 | sha256 |
|---|---|---|---|
| `v100_gpp20_seed4_c75_prior200.csv` | 480,010 | `9cade3174f33da089883927113d6ce8f` | `067f641a50090e01fb5674a92b45caffffa0a85ca7b54b94aa11d537541cca84` |
| `img_cliph_rerun.csv` | 411,040 | `a587d6c09a400dfcc2faf027d363fcab` | `fb19951a351baa8ab0b27be31a2338c1ec7736a586ca51016106228dbe993690` |
| `img_cliph_s20261101.csv` | 411,095 | `c8a55f6e670f79b12b53c88f6a6e0f90` | `6286b91841e80fdf02d011fb562e3e8a7ed7293bd869830d9ef29fefd9383d82` |
| `img_cliph_s20261102.csv` | 411,028 | `40d44f0cf8b570061f8ea1d809125ed5` | `5970a623c15d8f491ac2649f19c89f8166953a30391090dab812352473d0a49a` |
| `img_cliph_s20261103.csv` | 411,119 | `7b04d843a9a660b1a84f3d86a75b0f2a` | `5d9431c2c06fe067343fa286777cce8913c4fba4b31e0c33cd0f66012f1a308f` |
| `img_gpp_siglip2.csv` | 422,029 | `26e42ac41d73b05f3b0066a6acabc205` | `b64a0b344ec602d84202b5977d213b9d92ce187d553a330f9dcf17a440224894` |
| `img_bigg_rerun2.csv` | 380,587 | `761d9525ca77671723a2f6f26b7d6d62` | `600cd91971a329878b61527571235f0cd680d7d338fc62dd17ef17fcc00eb81f` |
| `img_bigg_rerun3.csv` | 421,999 | `965f029014d49e1a058e01d5fc847fb5` | `ecf2b57048b312b412ac87125d278ae8bb4aa7cfabf0163f2cc438fcf0000920` |
| `img_bigg_s384_retrain.csv` | 435,419 | `476a82aaa73600bbf58a2ee732f80f89` | `1f4ea9d5afe893892ef5bd8d95a35b88137da002e59d0497e6df100e25a2d14a` |
| `test_specs.csv` | 606,789 | `9c44545025f9c5d02e1c8d445d5caf5c` | `9b22cd018fa6e3c23ab5686ea8f65d2be581482216eae43c73e8827bef4daa6f` |

**自己核一遍**：

```bash
# Linux / macOS
md5sum    submission/*.csv
sha256sum submission/*.csv
```
```powershell
# Windows PowerShell
Get-ChildItem submission\*.csv | ForEach-Object {
  "{0}  {1}" -f (Get-FileHash $_.FullName -Algorithm MD5).Hash.ToLower(), $_.Name }
```

---

## 4. 怎么用这些文件（不需要 GPU / 权重 / Docker）

```bash
cd code/docker_submit
python -m venv .venv && .venv/bin/pip install -r requirements-verify.txt
.venv/bin/python verify_reproduce.py --from-csv
#   → 实测输出 spearman = 1.0000000000，max|Δ| = 7.500e-05
```

```bash
# 格式合规自检（列名 / 20000 行 / id 集合与顺序 / 范围 / 唯一值 / md5）
cd ..                                  # 回到 image-track/
python code/fusion/check_submission.py submission/v100_gpp20_seed4_c75_prior200.csv
#   → 通过  行=20000  范围=[0.000050, 0.999950]  唯一值=19981
#     md5 = 9cade3174f33da089883927113d6ce8f      ← 与上表一致
```

---

## 5. 三个必须知道的细节

1. **`score` 的精度与并列情况（实测数据，别凭印象）**
   - 本提交件 **20000 行全部是 10 位小数**（例：`00000.jpg,0.7758112094`）。
   - 唯一值 **19981** 个；并列共 **18 组**（17 组各 2 张 + 1 组 3 张），
     所以「唯一值 = 20000 − 19 = 19981」。⚠️ 这里 **19 是"重复多出来的行数"，不是"并列处数"** ——
     两者极易混淆（本仓库 `docs/05` 早期版本就写成了"19 个并列"，已一并更正）。
   - ⚠️ 比赛规范里那句「保留六位小数」指**指标 AUC 的报送精度**，**不是 score 列**：
     官方 `image_submission_example.csv` 的 score 就是 `0.5`，score 列只要求 ∈[0,1]。
     所以「10 位小数」既不违规也不是缺陷。

2. ⚠️ **`img_bigg_s384_retrain` 的"冻结 CSV"与"仓库里那份权重"是同一配方下的两次不同训练。**
   原因：产出该 CSV 的训练脚本与抽取镜像权重的脚本指向**同一个 `runs/clean_s384/best.pt`**，
   而实测两者输出相差**远超数值抖动** ⇒ 唯一自洽的解释是该 `best.pt` 在两步之间被重复执行覆盖，
   **产出 CSV 的那一份权重已被覆盖、不可取回**。
   它是 s384 槽的独占分量（最终权重 **10.7143%**），镜像输出与冻结 CSV 的一致度为 **0.9785**
   （其余 7 个分量 0.9961~0.9999），对最终排序的影响已精确量化为 **+0.000382**。
   详见 [`../docs/04_镜像复现偏差说明-回组委会.md`](../docs/04_镜像复现偏差说明-回组委会.md) §3.4。

   > 文件名里**不放括号**是两条赛道统一的约定：含 `(` `)` 的文件名会被「截到第一个 `)`」的朴素链接
   > 检查器误报断链（`<>` 写法也挡不住，因为它只包住括号、括号仍在文件名里）。去掉括号后，
   > 任何检查器（含朴素正则）都不会再误报。详见 `README.md` 的提交说明。

3. **本目录里的 CSV 一律是冻结产物，不会被任何重跑替换。**
   理由是"冻结版才有 LB 证据"（例如 `img_gpp_siglip2.csv` 参与过 LB 0.905416 的实测），
   而重跑版没有；且每次重跑都会引入新的 ~0.03% 数值漂移，
   会让"提交文件 = 冻结 CSV 的确定性函数"这个性质失效。
   → **复现能力的价值是"原版丢了能重建"，不是"主动把原版换掉"。**
