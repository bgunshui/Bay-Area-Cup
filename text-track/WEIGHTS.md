# 大文件与权重清单(未入库内容)

> 本仓库只收录**代码、文档、少量结果数组**(合计约 2.7 MB)。
> 下列大文件因体积原因未入库,本文说明**它们的标识、体积与获取方式**,以便队友按需索取。

---

## 一、Docker 交付镜像(最关键,但有备份)

| 项 | 值 |
|---|---|
| 文件 | `ai-text-detector-0.981301.tar` |
| 大小 | **53,317,416,960 字节(49.66 GiB)** |
| sha256 | `a163a653d2aa05b7bd51ec6016fbc5aed51379f7802fbfbb9a39c231dc691f68` |
| 标签 | `ai-text-detector:0.981301` |
| 层次 | `newlayer.tar`(`sha256:42a2dfcc09ec947fb4f54d9a207b75f70c079308e162007e7228f421c413c343`,47,775,191,040 B)<br>`deps_layer.tar`(`sha256:c7a28569236d983d388d52b766a7ee8b2da83f74083d35d601210c6a4b555758`,5,542,215,680 B) |
| 配置对象 | `sha256:e3d6048659f57dad95fcdd4f3295c877459262a797662f9e1a60b736cab5e0eb` |
| 提交包 | 曾打包为 `文本附件资料.tar`(53,317,447,680 字节),内含镜像 + 技术方案报告 + 镜像信息 |
| 获取 | 找队长取(本地保留有一份镜像 tar) |

**镜像内已内置**:两个基座模型(41.7 GiB)+ 15 个 LoRA 适配器(866 MB)+ 完整 Python/CUDA 依赖。

---

## 二、基座模型(公开,可自行下载)

| 模型 | 体积(bf16) | 来源 |
|---|---|---|
| `Qwen2.5-14B-Instruct` | 28 GiB(8 分片) | ModelScope / HuggingFace:`Qwen/Qwen2.5-14B-Instruct` |
| `Qwen2.5-7B-Instruct` | 15 GiB(4 分片) | ModelScope / HuggingFace:`Qwen/Qwen2.5-7B-Instruct` |

> 服务器上已清理(公开可重下)。镜像内仍保留,因此**只要拿到镜像就不需要再下载**。

---

## 三、LoRA 适配器(不可再生,必须有备份)

最终方案使用的 **15 个适配器**(约 866 MB):

| 组 | 数量 | 单个大小 | 目录命名 |
|---|---|---|---|
| 14B p4 五折 | 5 | 100,735,600 B(96.1 MiB) | `lora14b_p4_fold{0..4}_adapter` |
| 7B r3a 五折 | 5 | 40,414,632 B(38.5 MiB) | `lora7b_distill_r3a_fold{0..4}_adapter` |
| 7B r3b 五折 | 5 | 40,414,632 B(38.5 MiB) | `lora7b_distill_r3b_fold{0..4}_adapter` |

**多重备份位置**:

| 备份 | 内容 |
|---|---|
| 交付镜像内 `/app/adapters/` | 全部 15 个(已逐字节审计,5,093 个 tensor 完整性校验通过) |
| `adapters_p4.tar.gz`(445 MB) | 14B p4 五折(每个 `adapter_model.safetensors` 均为 100,735,600 B) |
| `7b_r3_adapters.tar.gz`(357 MB) | 7B r3a/r3b 十折 |
| `adapters_all.tar.gz`(890 MB) | 15 个全含 |
| 附加(消融用) | `lora14b_p3_adapters.tar.gz`(p3 五折)、`v2_adapters.tar.gz`(14B official_v2 五折)、`p2_adapters.tar.gz`(7B/1.5B p2 各五折) |

> **已知丢失**(不影响 0.981301 复现):`lora14b_p5` 五折适配器、`pseudo_best_v4.npy`。
> 均为训练中间产物;最终方案不使用 p5。

---

## 四、竞赛数据边界(哪些进了仓库、哪些没有)

**⚠️ 本仓库现为私有仓库,但仍然不要把下面的"未入库"内容外传。**

### 已入库 ✅

| 内容 | 位置 | 大小 | 说明 |
|---|---|---|---|
| **冻结提交文件** | `submission/submit_p4_90_7b2_10.csv` | 1,065,385 B | 即 0.981301 那次提交,**含测试集 id 与我们的分数**;仅因仓库私有而收录 |
| 我们的模型分数数组 | `evidence/离线预测/*.npy` | 各 8~160 KB | 14B p4 / 7B r3a / 7B r3b 在复赛集上的分数(**无 id、无原文**) |
| 官方样例标签与折划分 | `evidence/离线预测/官方样例_labels.npy` / `_folds.npy` | 8,128 B ×2 | 竞赛提供的 1000 条样例标签,供 OOF 核验 |
| 融合中间结果 | `evidence/离线预测/fusion_offline_submitted.npy` / `fusion_online_image.npy` | 各 160,128 B | 离线链路与线上链路的最终分数(用于偏差复算) |

### 未入库 ❌

| 内容 | 大小 | 原因 |
|---|---|---|
| 复赛测试集原文(20000 条) | 69,100,580 B | **竞赛数据,不得外传** |
| 官方样例原文(1000 条) | 3,746,063 B | **竞赛数据,不得外传** |
| 其余历史提交 CSV(`submissions/` 下约 200 个) | 各 ≈1 MB | 含测试集 id,无增量价值 |

> **如果本仓库以后要改为公开**,请先删除 `submission/` 与 `evidence/离线预测/官方样例_labels.npy`
> —— 前者含测试集 id,后者是竞赛提供的标签。


---

## 五、训练中间产物(部分已备份,部分随服务器清理丢失)

| 内容 | 状态 |
|---|---|
| p3 / p4 / p5 各五折的 `oof/raw/replay` npy 与 `replay_raw_mean` | ✅ 本地有备份 |
| `pseudo_best.npy` / `v2` / `v3` | ✅ 本地有备份 |
| `pseudo_best_v4.npy` | ❌ 丢失(训练中间产物) |
| 训练日志(`oldserver_logs/`,131 个文件) | ✅ 本地有备份 |
| 早期 23 分支融合系统(MARS)代码与结果 | 见 `文档/历史-早期23分支系统结果报告.md` |

---

## 六、如何确认你拿到的是正确版本

```bash
# 镜像
sha256sum ai-text-detector-0.981301.tar
# a163a653d2aa05b7bd51ec6016fbc5aed51379f7802fbfbb9a39c231dc691f68

# 适配器
ls -l lora14b_p4_fold0_adapter/adapter_model.safetensors   # 100735600
ls -l lora7b_distill_r3a_fold0_adapter/adapter_model.safetensors  # 40414632

# 归档格式(第三方解析器)
skopeo inspect docker-archive:ai-text-detector-0.981301.tar
```

---

> 相关:[复现指南](docs/07_复现指南.md) · [镜像信息](docs/镜像信息.txt) ·
> [提交包终审报告](docs/验证-提交包终审报告.md)
