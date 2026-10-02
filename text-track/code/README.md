# 文本赛道代码索引

本目录收录研发阶段的训练、特征、融合和交付脚本，保留了实验编号和旧服务器路径。**目录名或文件名不能代替实际脚本角色，也不能证明某个脚本曾按当前默认参数产生最终模型。** 本索引按现有源码说明职责，区分历史模板、最终服务源码与缺少运行输入的核验工具。

```text
code/
├── docker_submit/   服务源码及历史构建模板
├── fusion/          教师、标签构造、候选融合和分数比较
├── training/        神经模型训练，同时含特征与候选生成脚本
├── features/        特征、公开模型推理、轻量分类头和早期候选
└── verify/          原构建/审计/冒烟工具
```

当前 `training/` 共30个、`features/` 共27个 Python 文件。静态核对覆盖各文件的输入、训练/推理逻辑、默认参数与输出；没有执行训练、模型推理、下载或镜像测试。复现边界详见 [复现指南](../docs/07_复现指南.md)。

## 1. docker_submit：服务与历史构建材料

| 文件 | 源码角色与边界 |
|---|---|
| `server_dual_v4.py` | 当前标注为最终服务的源码。14B p4 五折 + 7B r3a/r3b 各五折，共15个适配器；先按基座求logit均值，再做sigmoid及0.90/0.10融合 |
| `server_dual_v3.py`、`server_dual_v2.py`、`server_dual.py` | 历史双模型版本；预加载/绑定、校验、设备放置等处理需按具体文件比较 |
| `server.py` | 早期单模型服务；不能用它代表最终双模型实现 |
| `healthcheck.py`、`selfcheck.py` | 健康检查/自检工具；脚本存在不等于本次已运行通过 |
| `Dockerfile` | 历史单模型构建模板，基于torch 2.4 CUDA 12.1，复制model目录与server.py；不是最终依赖补全镜像的完整构建说明 |
| `requirements.txt` | 历史依赖清单：transformers 4.44.2、peft 0.12.0、torch>=2.4等；不是最终镜像的精确环境锁 |
| `build_on_server.sh` | 原服务器构建脚本，依赖原路径与模型资产 |

v4源码包含设备放置、`torch.distributed.tensor` 预加载/绑定、空文本兜底、推理锁与预热。它们是源码中的处理机制；实际效果、资源与时延须引用对应运行记录，不能由静态检查直接得出“全部验证通过”。

历史最终环境记录与上述 requirements 不同，详见 [交付说明](../docs/03_交付说明.md) 和 [复现指南B节](../docs/07_复现指南.md)。本轮没有重建或部署镜像。

## 2. fusion：已有分数的融合与标签构造

### 2.1 最终提交相关脚本

| 文件 | 实际职责 |
|---|---|
| `p97_p4_7b_sweep.py` | 读取p4/r3a/r3b的seed replay，再排名并扫描7B权重0.025–0.30；包含输出 `submit_p4_90_7b2_10.csv` 的分支 |
| `p101_verify_7b2.py` | 读取相同谱系预测，核对组成与冻结CSV；不能独立计算复赛AUC |
| `p95_p4_7b.py`、`p96_local_p4_7b.py` | p4与7B已有分数的候选比较 |
| `p87_p4_only.py` | 纯p4候选生成 |
| `p88_p4_heavy.py`、`p90_p4_mid.py`、`p89_p4_complement.py` | 14B候选配比扫描 |
| `p92_p4p5_local.py`、`p99_full_rank.py`、`p100_fusion_modes.py` | 其他候选与融合方式比较 |

原提交源码对各seed保存的replay做 `rank01` 后，融合为：

```text
rank01(0.90 * rank01(p4_replay)
       + 0.10 * (rank01(r3a_replay) + rank01(r3b_replay)) / 2)
```

`p43` 保存的replay又包含 `rank01(mean_fold(rank01(raw_fold)))`。**逐折排名后再聚合，与先求raw均值再排名不是同一算子。** 当前缺7B r3a/r3b replay及逐折raw，不能将入库raw_mean替换后声称逐位重建最终CSV。

`p97/p101` 硬编码 `D:\湾区杯\文本赛道\transfer` 和 `submissions`；`p97` 还从未入库的 `submit_14b_p4_only.csv` 读取id。当前冻结CSV可用于结果检查，源码原样运行还需原始输入及路径布局。

### 2.2 教师、分支与搜索

| 文件 | 实际职责 |
|---|---|
| `p4_fusion.py` | 动态读取可用分支；在官方OOF AUC上搜索家族内/家族间权重并生成候选，不是固定48个非零分量 |
| `p40_fus_lgb.py`、`p40_test_fus.py` | LightGBM融合及候选比较 |
| `p56_ultimate.py` | 汇集已有自训练与特征分支，生成多种融合候选 |
| `p60_allrerun.py`、`p61_w14b_sweep.py` | 融合候选重算/14B权重扫描，不应按文件名理解为完整模型重训 |
| `p66_v6_nollm.py` 至 `p72_v9_rerun.py`、`p81_v9_p3.py`、`p83_tradeoff.py`、`p86_seed_weights.py` | 不同家族、模型谱系与seed权重的候选变体 |
| `p64_verify_consistency.py` | 已有结果一致性比较 |
| `p70_scoreboard.py`、`p82_scoreboard2.py`、`p91_scoreboard3.py`、`p94_scoreboard4.py`、`p98_our_score.py` | 成绩/候选整理；其记录须与原始平台反馈对应，不能因脚本名就视为官方原始证据 |

入库 `p4_report.json` 保存43个分支统计；`p77` 描述的p2AD由4个ft、3个llm及2个伪标签学生输入构成。“48分支候选体系”是团队说明，不能等同于每个教师都同时融合48个模型。

### 2.3 伪标签谱系

| 文件 | 输入 → 产物 |
|---|---|
| `mk_pseudo_best.py` | `submit_v2_ftintF.csv` → `pseudo_best.npy`；注释关联反馈0.973656 |
| `mk_pseudo_v2.py` | `submit_v2_p2AD.csv` → `pseudo_best_v2.npy`；注释关联反馈0.978284 |
| `p73_mk_pseudo_v3.py` | `submit_v8_dual_only.csv` 或p3/p4 replay → `pseudo_best_v3.npy` |
| `mk_pseudo_v4.py` | p3/p4/p5 replay → `pseudo_best_v4.npy`；属于另一个实验变体 |
| `calc_rank.py`、`calc_rank2.py` | 排名与分数比较 |

前三组源码说明存在多代伪标签构造流程，但实际命令、标签文件及运行日志未完整入库。四个构造脚本存在，不代表最终模型使用四轮；p4/p5编号也不能当作第四/第五轮的证据。

公式 `sigmoid(4*(rank-0.5))` 的范围约为 **0.1192–0.8808**，不是0.08–0.92。`conf=clip(4*abs(soft-0.5),0,1)` 是启发式训练权重，不是校准置信概率。

### 2.4 偏差与提交比较

| 文件 | 可以检查什么 / 不能直接声称什么 |
|---|---|
| `find_offline_formula.py` | 比较每折raw、raw_mean与保存replay的候选关系；依赖未入库逐折输入，不应写成已确认唯一公式 |
| `analyze_final.py` | 对两条算法计算秩相关/翻转等；它直接融合逐折rank均值，与p97的seed再排名链不同 |
| `final_check.py` | 原提交输入和原文对齐检查；旧绝对路径、缺失输入，后半段还写对照CSV，不能原样充当克隆目录的只读快速验证命令 |
| `analyze_deviation2.py` | 历史偏差分析 |

本轮复算：旧README近似输入链、`fusion_offline_submitted.npy` 与最终CSV均存在排序及数值差异，不能写“100%排序一致、最大差1.1e-16”。指标及比较代码见 [复现指南A节](../docs/07_复现指南.md)。新增标准库入口为 [verify_frozen.py](../evidence/核对工具/verify_frozen.py)，它只读取原始 CSV/npy，并写入核对报告。

排序翻转对数D给出的AUC差值条件界为 `D/(n_positive*n_negative)`；`analyze_final` 的 `D/1e8` 明确假设正负各10000条。未知复赛标签比例时，不能将0.0020作为无条件严格上界；也不能把随机翻转假设下的量级当作实际AUC变化。

## 3. training：30个文件的实际角色

### 3.1 神经模型训练与对照

| 文件 | 角色 / 关键区别 |
|---|---|
| `p10_qwen7b_lora.py` | 7B监督LoRA，official/advmix可选；默认epochs2、batch2、max-len768、seed20260827，CE分类 |
| `p20_qwen7b_adv2.py` | 7B对抗增强变体，加入改写数据；不能代表纯官方样例训练 |
| `p21_qwen7b_pseudo.py` | 7B早期硬伪标签，选目标域排名两端；不是全量软标签自训练 |
| `p30_qwen14b_lora.py` | 14B早期监督LoRA，softmax输出 |
| `p30_v2_qwen14b_lora.py` | 14B监督变体，预测改为logit差并保存adapter及raw；与BCE学生的logit0约定不同 |
| `p3a_qwen_lora.py` | 参数化Qwen LoRA，可选official/augmix与对抗数据；依赖未入库的 `wanqu_first_submission` 模块 |
| `p3c_macbert.py` | MacBERT对照训练，含外部语料预训练/窗口聚合；不是Qwen LoRA，亦有外部模块依赖 |
| `p7_pseudo.py` | 1.5B早期硬伪标签实验，选排名两端 |
| `p32_distill.py` | 早期软蒸馏，真标签权重4.0、软标签1.0；模型目录缺失时存在回退至14B的逻辑，不能仅凭 `--size` 或tag判定实际基座 |
| `p32_old_retrain.py` | 旧服务器蒸馏重训，标签/seed/suffix/device参数化；与p42/p43权重设计不同 |
| `p42_self_train.py` | 7B/1.5B置信加权软标签自训练；默认pseudo_best、epochs2、seed20260906、suffix_p2 |
| `p43_14b_p2.py` | 14B置信加权自训练；默认pseudo_best_v2、epochs2、seed20260909、suffix_p3、batch4，默认开启梯度检查点 |
| `p51_7b_official_retrain.py` | 仅官方数据的7B重训对照；默认batch4、max-len768、lr1e-4、开启梯度检查点、用logit差输出 |
| `p75_strict_official.py` | 意图对照p10的监督重训；实际max-len默认512而p10默认768，预测又未传该参数而用768；“strict”注释不证明严格同配置 |

**真实选择机制：** p42/p43每epoch计算官方验证折AUC，恢复最高AUC的状态；导出预测时用验证折决定是否翻转方向。p4融合搜索也使用官方OOF。应说样例对复赛微小差异的判断有限，不能说“样例从不用于选择”或“样例选型必错”。

p42/p43对每折全部目标域文本使用软伪标签，真标签权重2.0，BCE训练 `logits[:,0]`。它们是自训练脚本，不能在流程表中列成初始纯监督训练。

默认batch是p42的7B=4/1.5B=8、p43=4，二者默认开启梯度检查点；这些是源码默认值，最终p4/r3a/r3b实际命令未齐，不能统一宣称batch2/关闭检查点。训练encode未显式指定truncation，预测编码则启用截断；配置长度不是已检查的实际训练长度上限。

已有同名折replay会触发跳过训练，且未验证seed/标签是否与当前参数一致。新seed不能只改参数后继续复用旧suffix；同一候选恢复过程也需结合缓存和日志解释。

### 3.2 特征与轻量分类头，不是基座LoRA

| 文件 | 角色 |
|---|---|
| `p2_qwen14.py`、`p2_qwen7b.py` | 14B/7B隐藏态与似然特征；在特征上训练LR/LGB等头，并保存OOF/replay |
| `p6_4b_lik.py`、`p11_4b_hidden.py` | Qwen3-4B似然/隐藏态分支与轻量分类头 |
| `p44_genfp.py`、`p44b_genfp.py` | 1.5B/7B NLL指纹特征及LGB头；p44b拆分特征阶段 |

### 3.3 读取已有预测的集成、分析与候选生成

| 文件 | 角色 |
|---|---|
| `p25_seedens.py`、`p26_seedens2.py`、`p27_seed2_070.py` | 读取已有7B seed预测做集成/权重变体；不训练新seed |
| `p17_ft7b2.py` | 7B重权候选与KNN平滑 |
| `p47_p3_preview.py`、`p48_p3_candidates.py` | p3谱系分析与融合候选 |
| `p54_final_candidates.py`、`p55_rt_candidates.py`、`p57_rt_variants.py` | 已有预测、重训分支替换和候选对照 |
| `p77_reproduce_p2ad.py` | 按p2AD结构组合已有分支，并与原伪标签/提交比较；不是重新训练教师 |

## 4. features：27个文件的角色

| 文件 | 角色 |
|---|---|
| `train_initial_text.py` | 早期字符/词TF-IDF、风格特征和Qwen3B嵌入轻量头；使用初始阶段数据路径，不能代表当前最终训练链 |
| `infer_text_replay.py` | 使用早期分类器/头资产为复赛文本出分数，不是最终14B/7B适配器服务 |
| `infer_text_qwen7b.py`、`qwen7b_replay.py` | 7B嵌入提取、LR等轻量头与候选融合；不微调7B基座 |
| `infer_text_qwen14b_gptq.py` | GPTQ基座特征提取、LR头与OOF选择/融合 |
| `qwen_oof.py`、`qwen_fast_oof.py`、`qwen_blend_oof.py`、`qwen_grid_oof.py`、`qwen_mokawa_grid.py` | 已有嵌入的LR/SGD头、OOF和网格融合；不是Qwen基座训练 |
| `blend_text_oof.py` | 嵌入头与公共分数的OOF融合搜索 |
| `hc3_detector.py` | 外部HC3语料的TF-IDF/LR检测器训练与候选融合 |
| `infer_active_text.py`、`infer_new_text_models.py` | 本地公开分类器推理与候选输出；AI方向按各分支配置处理 |
| `infer_text_anx.py`、`infer_text_public_zh.py`、`infer_paperpass.py`、`infer_paperpass_active.py`、`infer_mokawa_active.py`、`infer_tenne_active.py` | 对本地公开模型/ONNX权重推理及融合；不能据此证明所有历史流程从未使用外部API |
| `infer_text_windows.py` | 窗口化公开检测器推理和聚合 |
| `infer_ppl_active.py`、`text_ppl.py` | 语言模型NLL/困惑度类特征 |
| `make_qwen_candidates.py`、`analyze_text_candidates.py` | 已有分数的候选组合与分析 |
| `parallel_hf_download.py` | 历史模型下载工具；不属于检测推理或训练 |
| `test_qwen14_cpu.py` | GPTQ模型CPU加载/前向探针；存在不等于当前已冒烟通过 |

注意指标名字：`infer_active_text.py` 输出的 `sample_auc_raw` 实际是 `np.corrcoef`；`infer_mokawa_active.py` 的 `sample_auc` 实际是Spearman，均不能引用为ROC AUC。不同脚本的数据划分/随机种子也不同，不应合并成一份共同的“五折验证成绩”。

## 5. verify：原始审计与构建工具

| 文件组 | 实际职责 / 必要边界 |
|---|---|
| `audit_BCD_image.py`、`audit_A_archive.py`、`wsl_verify_image.py`、`check_archive.py`、`scan_abi.py`、`peek_image.py` | 归档、依赖、权重结构与脚本比对；依赖指定镜像/旧路径，不由源码存在直接得出当前资产完整 |
| `build_deps_layer.py`、`rebuild_image.py` | 原依赖层与镜像重建；需原镜像及wheel输入，不是当前requirements的一步构建 |
| `chroot_test.py`、`chroot_e2e.py`、`chroot_http_driver.py`、`chroot_fake_server.py` | 依赖/HTTP冒烟，包括缩小模型或假加载场景；不能替代完整大模型GPU推理验证 |
| `gpu_driver.py` | GPU服务端请求与性能采集；性能数字应对应真实运行日志 |

## 6. 命名、存档与缺口

`pNN` 是实验编号，suffix/tag区分候选与缓存。五折是数据划分，seed是训练随机性，p4/p5编号不能单独证明标签更新轮次；r3a/r3b被团队标注为7B第三代的两个seed，但具体seed数字和最终运行命令仍需日志映射。

团队说明约1100个原始补丁、调度与实验脚本未入库，并提到部分存档含凭据。需要追溯时应索取经过脱敏的运行配置和记录，不能声称当前仓库已支持任意数字的完整追溯。

当前源码、冻结结果、官方成绩与团队历史说明是不同证据层次。最终官方AUC0.981301按交接确认；精确提交分数重建缺7B replay/逐折raw；重训另缺原文、教师完整输入、标签和命令。参见 [复现指南](../docs/07_复现指南.md)、[WEIGHTS](../WEIGHTS.md) 与 [实验结果](../evidence/关键实验结果.md)。
