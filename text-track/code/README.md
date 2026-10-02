# 代码索引

本目录按**功能**而非时间组织。所有脚本均为项目实际使用过的原始脚本(未做美化改写),
命名沿用了当时的实验编号(`pXX_`)与服务器辅助脚本(`srv_`)前缀。

```
代码/
├── docker_submit/       Docker 镜像内实际运行的服务(server v4 / Dockerfile / 依赖 / healthcheck)
├── fusion/              ★ 最高分 0.981301 的融合链路、提交生成与偏差复算
├── training/            LoRA 训练、伪标签自训练、蒸馏
├── features/            似然/隐藏态特征、公共 AIGC 检测器推理
├── verify/              镜像 / 依赖 / 权重完整性验证工具
└── (实验存档未收录,见文末说明)
```

---

## docker_submit —— 推理服务(交付镜像内运行)

| 文件 | 说明 |
|---|---|
| **`server_dual_v4.py`** | ★ **最终交付版**推理服务。双模型(14B p4 五折 + 7B r3a/r3b 十折)、0.90/0.10 sigmoid 融合、三探针 + `/predict`。相对早期版本修了 3 处会导致交付失败的问题(见下) |
| `server_dual_v3.py` | v3:加入 `torch.distributed.tensor` 显式绑定 + 400/40001 校验处理 |
| `server_dual_v2.py` | v2:加入 `torch.distributed.tensor` 预加载(peft 0.20.0 必需) |
| `server_dual.py` | 早期双模型版本 |
| `server.py` | 最初的单模型版本 |
| `selfcheck.py` | 服务自检脚本 |
| `Dockerfile` | 原始 Dockerfile(构建时用) |
| `requirements.txt` | ★ 精确依赖清单(适配器加载的强制版本) |
| `build_on_server.sh` | 服务器侧构建脚本 |

### v4 相比早期版本修掉的三处问题(每一处都会让交付失败)

| # | 问题 | 后果 | 修法 |
|---|---|---|---|
| 1 | **设备放置**:`if dev != "cuda": model.to(dev)` | transformers 在未传 `device_map` 时把模型留在 CPU,**有 GPU 时反而不搬运** → 14B 在 CPU 上跑 15 次前向,单请求远超 90 秒 | 显式 `device_map={"":0}` 流式加载 + `model.to(dev)` 兜底 |
| 2 | **`torch.distributed.tensor` 未预加载** | peft 0.20.0 抛 `AttributeError: module 'torch.distributed' has no attribute 'tensor'` | 在 import peft 前预加载,并在预加载后显式绑定到父包 |
| 3 | **空串/纯空白文本** | 分词结果为空张量,前向 `RuntimeError` | 用 `eos_token_id` 兜底 |

另加:推理互斥锁(避免 `set_adapter` 并发竞态)、启动预热(保证 readiness 返回 200 后首个请求即可正常处理)。

---

## fusion —— 最高分链路 ★

### 生成提交(按实验顺序)

| 文件 | 说明 |
|---|---|
| **`p97_p4_7b_sweep.py`** | ★ **生成 `submit_p4_90_7b2_10.csv` 的脚本**(即 0.981301 那次提交)。做 p4 与 7B 集成权重的精细扫描(0.025→0.30) |
| **`p101_verify_7b2.py`** | ★ 核验 `submit_p4_90_7b2_10` 的组成 |
| `p96_local_p4_7b.py` / `p95_p4_7b.py` | p4 + 7B 融合的本地预览 |
| `p87_p4_only.py` | 纯 p4(14B p4 五折)提交 |
| `p88_p4_heavy.py` / `p90_p4_mid.py` / `p89_p4_complement.py` | 14B 侧重配比扫描 |
| `p92_p4p5_local.py` | p4 + p5 融合本地预览 |
| `p99_full_rank.py` | 全量 rank 融合 |
| `p100_fusion_modes.py` | 多种融合模式对比 |

### 教师融合与权重搜索(48 分支体系)

| 文件 | 说明 |
|---|---|
| `p4_fusion.py` | ★ 融合主框架(多分支、rank01 归一化、权重搜索) |
| `p40_fus_lgb.py` / `p40_test_fus.py` | LightGBM 融合与测试 |
| `p56_ultimate.py` | 终极融合候选 |
| `p60_allrerun.py` / `p61_w14b_sweep.py` | 全量重跑 / 14B 权重扫描 |
| `p66_v6_nollm.py` ~ `p72_v9_rerun.py` | v6–v9 系列融合变体 |
| `p86_seed_weights.py` | 种子权重搜索 |
| `p64_verify_consistency.py` | 一致性核验 |
| `p70/p82/p91/p94_scoreboard*.py` | 提交成绩记分板(记录每次提交与反馈) |
| `p98_our_score.py` | 本方成绩统计 |

### 伪标签构造

| 文件 | 说明 |
|---|---|
| `mk_pseudo_best.py` / `mk_pseudo_v2.py` / `p73_mk_pseudo_v3.py` / `mk_pseudo_v4.py` | 四轮伪标签构造 |
| `calc_rank.py` / `calc_rank2.py` | rank01 归一化与融合计算 |

### 偏差复算(镜像 vs 离线)

| 文件 | 说明 |
|---|---|
| **`find_offline_formula.py`** | ★ 反推离线的精确聚合方式(确认 `replay.npy = mean_fold(rank01(fold logits))`) |
| **`analyze_final.py`** | ★ 定量两条链路的排序翻转与 AUC 上界(204,591 对 / 0.1023% / 上界 0.0020) |
| `final_check.py` | ★ 逐位重建提交文件(排序 100% 一致、数值最大差 1.1e-16) |
| `analyze_deviation2.py` | 早期偏差分析 |

---

## training —— 训练与自训练

| 文件 | 说明 |
|---|---|
| **`p43_14b_p2.py`** | ★ 14B 第二轮(域自适应)训练 |
| **`p42_self_train.py`** | ★ 伪标签自训练主脚本 |
| `p32_distill.py` / `p32_old_retrain.py` | 蒸馏与重训 |
| `p7_pseudo.py` | 早期伪标签流程 |
| `p75_strict_official.py` | 严格只用官方数据的对照训练 |
| `p51_7b_official_retrain.py` | 7B 官方数据重训 |
| `p30_v2_qwen14b_lora.py` / `p30_qwen14b_lora.py` | 14B LoRA 训练 |
| `p2_qwen14.py` / `p2_qwen7b.py` / `p10_qwen7b_lora.py` | 早期 14B/7B LoRA |
| `p3a_qwen_lora.py` / `p3c_macbert.py` | 1.5B LoRA / MacBERT 对照 |
| `p20_qwen7b_adv2.py` / `p21_qwen7b_pseudo.py` | 对抗增强 / 伪标签 7B |
| `p25_seedens.py` / `p26_seedens2.py` / `p27_seed2_070.py` | ★ 多随机种子集成 |
| `p44_genfp.py` / `p44b_genfp.py` | 生成指纹分支 |
| `p47_p3_preview.py` / `p48_p3_candidates.py` | p3 谱系预览与候选 |
| `p54_final_candidates.py` / `p55_rt_candidates.py` / `p57_rt_variants.py` | 最终候选生成 |
| `p77_reproduce_p2ad.py` | p2ad 复现 |
| `p6_4b_lik.py` / `p11_4b_hidden.py` / `p17_ft7b2.py` | 4B 似然 / 隐藏态 / 7B 特征 |

---

## features —— 特征与外部检测器

| 文件 | 说明 |
|---|---|
| `blend_text_oof.py` | ★ 文本 OOF 融合 |
| `qwen_blend_oof.py` / `qwen_oof.py` / `qwen_grid_oof.py` / `qwen_fast_oof.py` | Qwen 系列 OOF 生成与网格搜索 |
| `qwen7b_replay.py` | 7B 复赛集推理 |
| `qwen_mokawa_grid.py` | 与 mokawa 检测器的网格融合 |
| `infer_text_replay.py` | 复赛集通用推理入口 |
| `infer_text_qwen7b.py` / `infer_text_qwen14b_gptq.py` | 7B / 14B(GPTQ)推理 |
| `infer_text_public_zh.py` / `infer_text_anx.py` / `infer_text_windows.py` | 公共中文 AIGC 检测器 / anx / 窗口化推理 |
| `infer_paperpass*.py` / `infer_mokawa_active.py` / `infer_tenne_active.py` / `infer_active_text.py` | 各外部检测器推理 |
| `infer_ppl_active.py` / `text_ppl.py` | 困惑度特征 |
| `hc3_detector.py` | HC3 数据集检测器 |
| `infer_new_text_models.py` / `make_qwen_candidates.py` / `analyze_text_candidates.py` | 新模型推理与候选生成 |
| `train_initial_text.py` | 初始文本模型训练 |
| `parallel_hf_download.py` | 并行下载 |
| `test_qwen14_cpu.py` | CPU 冒烟 |

---

## verify —— 验证与审计

| 文件 | 说明 |
|---|---|
| **`audit_BCD_image.py`** | ★ 镜像内容审计:**27 个 safetensors / 5,093 个 tensor** 头部解析 + 字节区间连续性 + 大小自洽;`index.json` 与分片对照;脚本内容比对 |
| **`audit_A_archive.py`** | ★ 提交包审计:成员/编码/PAX 头/包内文件与本地比对/镜像 sha256 |
| **`wsl_verify_image.py`** | ★ 镜像结构验证:manifest/config 一致性、分层 sha256 复算、抽取、whiteout 生效 |
| `check_archive.py` | 提交包内镜像 sha256 复算 |
| `scan_abi.py` | 扫描镜像内 **cp312/cp310 编译产物**(ABI 错配检测) |
| `peek_image.py` | 嵌套 tar 随机访问工具(不必解 44 GB 主层) |
| `build_deps_layer.py` | ★ 依赖层构建:59 个 wheel 解包 + `.data` 重定位 + whiteout |
| `rebuild_image.py` | ★ 镜像重建:流式搬运主层 + 追加依赖层 + 重写 manifest/config + 修正 HEALTHCHECK |
| `chroot_test.py` | chroot 内依赖与真实适配器冒烟 |
| `chroot_e2e.py` | chroot 内端到端(注入同构缩小模型) |
| `chroot_http_driver.py` | ★ chroot 内以镜像原样 `CMD` 起服务,走真实 HTTP 打三探针与 `/predict` |
| `chroot_fake_server.py` | 上面用的假加载器(缩小模型 + 真实 adapter) |
| `gpu_driver.py` | ★ 真机 GPU 端到端(启动耗时 / 单条延迟 / 显存 / 设备放置确认) |

---

## 附:实验存档(未收录,约 1100 个脚本)

按时间保存的**全部**实验脚本,包括:

- `p*.py` —— 按实验编号的候选生成、融合、评估脚本
- `patch_*.py` —— 训练脚本的一次性补丁(修 batch/seed/path/调度等)
- `srv_*.py` —— 服务器侧运维辅助脚本(进度查询、备份、重启、显存检查等)
- `*.sh` / `sched*.sh` / `watchdog*.sh` —— 调度与守护脚本

这些脚本**未经整理**,是完整的工作痕迹,可用于追溯任意一个数字是怎么得来的。
如需定位某个结论的出处,建议从 `evidence/关键实验结果.md` 里的脚本链接入手。

---

## 附:命名约定

| 前缀 | 含义 |
|---|---|
| `pNN_` | 第 NN 号实验(Pipeline step) |
| `p4_` / `p5_` | 14B 的第 4 / 5 代伪标签训练谱系 |
| `r3a` / `r3b` | 7B 的第 3 代伪标签谱系(a/b 两个种子) |
| `srv_` | 服务器运维辅助脚本 |
| `patch_` | 对某训练脚本的一次性修补 |
| `mk_pseudo` | 伪标签构造 |
| `rank01` | `rankdata(x)/(N+1)`,全局秩归一化 |

---

## 关于"实验全量存档"

本项目在开发过程中产生了约 **1100 个一次性脚本**(实验候选、训练补丁、服务器运维脚本)。
由于其中**大量嵌有训练服务器的账号口令**,且对理解方案没有增量价值,
**未纳入本仓库**。如需追溯某个数字的完整过程,请找队长索取本地存档。
