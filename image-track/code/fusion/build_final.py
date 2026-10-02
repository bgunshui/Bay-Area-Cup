#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_final.py —— 用**新的重跑基底**替换 `v49_rt_prior20.csv` 里的旧基底，按多个 prior 占比各出一份作品。

用户 2026-09-13 的要求：
  「等所有实验跑完，生成替换 `v49_rt_prior20.csv` 里面重跑基底的作品，
    以及 `_prior` 占比为 0、12.5 的也要有一份」
  → 即：**新基底 × (1−w) + prior2 × w**，其中 w ∈ {0, 0.125, 0.20}，共三份。

背景：
  `v49_rt_prior20.csv` = 重跑基底 ×0.80 + prior2 ×0.20 → LB **0.914498**（当前最高）。
  重跑基底 = `0.65·rank(cliph_rerun) + 0.20·rank(bigg_rerun) + 0.15·rank(s384r)`，反解真实 AUC ≈0.9133。
  反解还给出：先验真实 AUC ≈0.652、**最优先验权重 w\\* ≈0.13**（所以 w=0.30 时掉了 0.0053）。

  ⚠️ **w=0.20 与 w=0.125 的差别极小**（模型预测 +0.0009）——它们主要是"万一 w\\* 估计偏了"的对冲，
     以及"纯基底"（w=0）用来**验证基底推断**（预期 ≈0.9133）。

配方写在本文件顶部的 `SLOTS` 里（**改配方就改这里**）：
  每个槽 = (槽在基底里的权重, [(分量 csv, 该分量在槽内的权重), ...])，槽内权重和必须为 1。

用法：
    python build_final.py            # 生成 + 自检 + 打印台账
    python build_final.py --dry      # 只自检，不写文件
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent

# ============================ 配方（改这里） ============================
# 结构：{配方名: {槽名: (槽权重, [(分量 csv, 槽内权重), ...])}}，槽内权重和必须为 1，槽权重和必须为 1。
#
# 2026-09-13 17:10 的实测依据：
#   cliph300k : CLIP-H + 302k **纯监督**，vs cliph_rerun 相关 0.7785、探针 val 0.99305
#   gpp       : SigLIP2 + GenImage++ 纯监督，vs cliph_rerun 0.7914、**有实测 LB 0.9054**
#   cliph_f2  : CLIP-H + Tiny **31k 纯监督**（只有 1/10 数据），vs cliph_rerun 0.8012 ← 与 302k 同档！
#   ⭐ 三者的共同点：**都没有教师软标签**。而带软标签的（e2 0.9315 / d1 0.9517）全部作废。
#      → **独立性来自"纯监督"，不来自"数据规模"**（此结论由 f2 vs cliph300k 直接对照得出）。
#   互相关：cliph300k-gpp 0.7403、gpp-f2 0.6964、cliph300k-f2 0.8414 → 三者彼此也相当独立。
RECIPES = {
    # 基线：cliph300k(0.15) + gpp(0.20) 注入主槽
    "c3gpp": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 0.65),
                         ("img_cliph300k.csv",   0.15),
                         ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20, [("img_bigg_rerun.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 2026-09-13 18:00 用户要求：「先生成基底：0.65 cliph300k + 0.20 bigg_rerun + 0.15 s384r」，
    #   = **只把 cliph 槽整槽换成 300k 版**，bigg/s384 两槽保持重跑版不动。
    #   ⚠️ 一个必须知道的性质：`cliph300k` 的分数**极度饱和**（42.6% > 0.99、28.2% < 0.01，中间段只剩 17.9%），
    #      整槽替换（而不是小比例混入）会把大片图打成并列，把排序权交给只占 35% 的另两个槽。
    #      `cliph_rerun` 是罕见的饱满分布（99.5% 落在 0.05~0.95）—— 这是换成整槽后**唯一的实质风险**。
    #   支持换的依据：`cliph300k` 的探针 val = **0.99305**（全场最高，同协议 seed=20260925），
    #      且与 `cliph_rerun` 相关仅 **0.7785**。
    "c300k": {
        "cliph": (0.65, [("img_cliph300k.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigg_rerun.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 2026-09-13 18:40 用户拍板：**A + C 都做**（现改名为 v63 系列，v62 让给「换 bigg_rerun2」那套）
    #   C（实际候选）= 只换 cliph + s384 两槽，**bigg 槽保持（重训版）bigg_rerun2 不动**
    #   ⚠️ 2026-09-13 21:00 实测后改用 **25% 那个 checkpoint 的 CSV**（`img_s384_str300k_c25.csv`）：
    #      同协议 val 25% = 0.723350 → 50% = **0.666800（回落 5.7 点，在退化）**，
    #      所以 25% 就是「300k 版」最好的一版；等它训完只会更差，没有理由等。
    "c300k_cs": {
        "cliph": (0.65, [("img_cliph300k.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.15, [("img_s384_str300k_c25.csv", 1.00)]),
    },
    # ⭐ 2026-09-13 18:45 用户的 v62 系列：
    #   `v61_c300k_*` 的三个分量里，`img_cliph300k.csv`（有完整日志+备份包）与
    #   `img_bigg_s384_retrain.csv`（有完整 158 行日志）都可复现，
    #   **只有 `img_bigg_rerun.csv` 的日志残缺（690 B，只到 ep0 step800/7200）**
    #   → 用新机上按严格相同配置重训出的 **`img_bigg_rerun2.csv`** 替换它。
    #   其余一切不变（cliph 槽仍是 cliph300k，s384 槽仍是 s384r），所以 v62 与 v61 只差这一个分量。
    "c300k_b2": {
        "cliph": (0.65, [("img_cliph300k.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⏳ 占位：**三槽全换 300k 版**（用户说的"等 bigg300k、s384_str300k 完成了再替换"）。
    #   现状（2026-09-13 17:45）：
    #     · cliph 槽 → `img_cliph300k.csv`                        ✅ 已有
    #     · s384  槽 → `img_s384_str300k.csv`                     ⏳ 续训中（≈23:45 完，`train_stretch.py` 出 `img_{TAG}.csv`）
    #     · bigg  槽 → **没有"跑完的" bigg300k**（`bigG_crop300k` 已被杀在 75%，我判断它过拟合，没在续训）
    #                  → 只能用 50% 的 checkpoint `img_bigG_crop300k_ckpt_37849.csv`（同协议 val **0.7945**，
    #                    而要替换的 `bigg_rerun` 是 **0.9750** → 差 18 点）
    #   ⚠️ 这个配方预期**大幅掉分**。跑它只是为了把"逐槽换 300k 版"这条线做完整、留档，
    #      **不要拿它当提交候选**（理由与门槛见 AGENTS.md ㉖/㉘）。
    "c300k_x3": {
        "cliph": (0.65, [("img_cliph300k.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigG_crop300k_ckpt_37849.csv", 1.00)]),
        "s384":  (0.15, [("img_s384_str300k_c25.csv", 1.00)]),
    },
    # ⭐ 2026-09-13 20:00 双实例 ensemble：bigg 槽 = 0.5·rank(bigg_rerun2) + 0.5·rank(bigg_rerun3)
    #   两个成员**都在本环境重训、都带完整日志**，SEED 分别是 20261004 / 20261005。
    #   ⚠️ 必须用**不同的 SEED**：㉛ 的 B0 自检已证实本流水线在"同机同库"下是**确定性**的
    #      （同 SEED 再跑 20 步得到逐位相同的 step-0 loss），所以同 seed 的两个成员会完全一样、ensemble 无效。
    #   成员间相关性（纯 seed 漂移，同环境）要等 bigg_rerun3 出来才算得出来 —— 那个数才是干净的可复现性指标。
    "c300k_ens": {
        "cliph": (0.65, [("img_cliph300k.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐ 2026-09-13 21:15 用户要求：`0.65 cliph300k + 0.20 bigg300k + 0.15 s384r`，prior ∈ {0, 0.125, 0.20}
    #   = "cliph 与 bigg 两槽都换 300k 版、s384 槽留重跑版"的中间态（介于 v61 与 v63_x3 之间）。
    #   ⚠️ `bigg300k` 用的是它**50% 的 checkpoint**（`img_bigG_crop300k_ckpt_37849.csv`）——
    #      它是 val 峰值（0.7348 → 0.7945 → 0.7120），也是唯一有 20000 行 CSV 的那一版。
    #   🔴 **预期大幅掉分**：bigg 槽的真实 AUC 从 ~0.8745 掉到 ~0.7945，结构模型算基底 ≈0.9011（−1.22 分）。
    #      出它主要用于把"两槽同时换 300k"这条线补齐 + 用 LB 验证一次模型预测。
    "c300k_b3k": {
        "cliph": (0.65, [("img_cliph300k.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigG_crop300k_ckpt_37849.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-13 23:50 用户要求：**`bigg_rerun` 从所有配方里踢掉**（它的日志只剩 800/7200 步），
    #     一律换成同配方重训的 `bigg_rerun2`（SEED=20261004）/ `bigg_rerun3`（SEED=20261005）——
    #     两者都**权重 + 完整日志**齐备。实测 b2 与 b3 彼此 ρ=0.8357、与 b1 ρ≈0.82，
    #     且 v62(b2) vs v61(b1) 的最终文件 ρ=0.9917 → **换掉对分数几乎无影响，但资料就齐了**。
    #     `c3gpp` 的注入结构完全不变（cliph 槽 0.65/0.15/0.20），只换 bigg 那一格。
    "c3gpp_b2": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 0.65),
                         ("img_cliph300k.csv",   0.15),
                         ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "c3gpp_b3": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 0.65),
                         ("img_cliph300k.csv",   0.15),
                         ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20, [("img_bigg_rerun3.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 双实例 ensemble（b2+b3）：ρ=0.8357 → 槽内 d-prime ×1.0437，是"免费的方差缩减"（预期最终 +0.0015）
    "c3gpp_b23": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 0.65),
                         ("img_cliph300k.csv",   0.15),
                         ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 四源注入（多一路 f2）+ 双实例 ensemble
    "c3gpp_f2_b23": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 0.40),
                         ("img_cliph300k.csv",   0.20),
                         ("img_gpp_siglip2.csv", 0.25),
                         ("img_cliph_f2.csv",    0.15)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 00:25 用户要求：「`v64_c300k_ens_prior125` 把里面的 `cliph300k` 换成 `cliph_rerun`」
    #   = **回到重跑基底的 cliph 槽（cliph_rerun），只保留 bigg 槽的双实例 ensemble**。
    #     等价于「原基底（v49/v57 那套，LB 已验证 0.914498）+ 免费方差缩减」——
    #     是**最保守**的一版：主槽仍是被 LB 验证过 65% 权重的 cliph_rerun，只把 bigg 那一格换成 b2+b3 平均。
    #   预期：基底 0.9133 → 0.9152（+0.0019，来自 ρ=0.8357 的 ensemble）→ 最终（prior .125）≈ **0.9160**。
    #   ⚠️ 与 `c300k_ens`（v64）的区别只有一个槽：那一格是 `cliph300k` vs `cliph_rerun`。
    #   两个分量都有完整训练日志 + 权重在本地，所以这版**全链路可复现**。
    "rt_ens": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 加第三个纯监督注入源 f2（模型算的槽级最优：f2≈0.90 时 0.40/0.20/0.25/0.15）
    "c3gpp_f2": {
        "cliph": (0.65, [("img_cliph_rerun.csv", 0.40),
                         ("img_cliph300k.csv",   0.20),
                         ("img_gpp_siglip2.csv", 0.25),
                         ("img_cliph_f2.csv",    0.15)]),
        "bigg":  (0.20, [("img_bigg_rerun.csv", 1.00)]),
        "s384":  (0.15, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 用户要求：「把 `v67_rt_ens_prior125` 里 `cliph_rerun` 的权重提高一下，prior 20% 保持不变」。
    #   做法：**只动槽权重**，尾部（bigg:s384 = 20:15 = 4:3）按比例收缩，prior 固定 0.20。
    #   动机（有道理）：`corr(重跑基底, cliph_rerun) = 0.9758` —— 基底几乎就是 cliph_rerun，
    #     而尾部那两个（`bigg_rerun2/3` + `s384r`）都是 **Tiny 28.8k 同一份数据**训的（互相相关只 0.746），
    #     即 35% 的权重实际只贡献"一个骨干的一种视角"。
    #   ⚠️ 历史反向证据（必须知道）：历史基底的同类扫描里，把主分量从 65% 提到 70% 是**掉分**的 ——
    #     `b20_s15`(anchor 65%) = 0.908114 → `b20_s10`(anchor 70%) = 0.907320（**−0.00079**）。
    #     → 预期 |Δ| ≲ 0.001，且历史符号偏负。这一轴在**重跑基底**上从未测过，所以值得试，但别期待大增益。
    "rt_ens_c70": {
        "cliph": (0.70, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20 * 0.30 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.30 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "rt_ens_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "rt_ens_c80": {
        "cliph": (0.80, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20 * 0.20 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.20 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 对照组：主槽仍用**有 LB 证据的** `bigg_rerun`(b1)，只提 cliph 权重（排除 ensemble 这个变量）
    "rt_c70": {
        "cliph": (0.70, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20 * 0.30 / 0.35, [("img_bigg_rerun.csv", 1.00)]),
        "s384":  (0.15 * 0.30 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 01:05 **LB 实测回来后的延伸**：
    #   `v68_c75_ens_prior200.csv`（cliph **0.75**） = **0.916175** —— 破了历史最高 0.914498（**+0.001677**）。
    #   → **"提高主槽权重"这个方向被 LB 证实有效**，我此前引用的历史反证（历史基底 65%→70% 掉 0.00079）
    #     在**重跑基底**上不成立。§㉞ 的"新分量必须够强"是**加法**的判据，
    #     而**减法**（把权重集中到已被验证的最强分量）不受它约束 —— 这两件事我之前混在一起了。
    #   ⚠️ 仍未解耦：c75 相对 v49 同时改了 cliph 权重（.65→.75）和 bigg 槽（b1→ens b2/b3）。
    #   → 现在需要**定位峰值**：c70 / c80 两个点已经在手，加上 c85 / c90 / c100 把曲线右端补全。
    "rt_ens_c85": {
        "cliph": (0.85, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20 * 0.15 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.15 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "rt_ens_c90": {
        "cliph": (0.90, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.20 * 0.10 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.10 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # cliph 1.0 = **基底退化成纯 `cliph_rerun`**（尾部两个槽权重为 0）。
    # 用来回答"尾部到底是在帮忙还是拖后腿"这个最根本的问题：
    #   c100 若 ≥ c85 → 尾部**一直是负担**，历史那 35% 权重全是在扣分；
    #   c100 若明显低于 c75/c85 → 存在一个内部最优，尾部有其价值。
    "rt_ens_c100": {
        "cliph": (1.00, [("img_cliph_rerun.csv", 1.00)]),
        "bigg":  (0.00, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.00, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 01:30 用户选定「两个都做」：**主槽 seed 集成（4路）** 与 **再加换骨干那一份（5路）**
    #   4路 = `cliph_rerun` + 3 个同配方换 seed 的学生（20261101/02/03），各 1/4。
    #     实测两两 ρ：cliph_rerun-s1 = 0.96042、cliph_rerun-s2 = 0.96032、**s1-s2 = 0.96543**（两个新 seed 彼此更近）
    #     → 簇内平均 ρ ≈ 0.963 → K=4 噪声 std 缩减 **1.40%**（极限 K=∞ 也只有 1.85%）。
    #   5路 = 上面 4 份 + `img_bigGd384mix.csv`（**bigG 骨干 + 同一套转导数据**），各 1/5。
    #     bigGd384mix 与 cliph_rerun 的 ρ = **0.9223**（比 seed 之间的 0.963 独立得多）→ 5路噪声 std 缩减 **2.27%**。
    #     ⚠️ 它有两个硬伤：唯一值只有 **517**（cliph_rerun 是 1880），且伪AUC 0.965 < 0.973（更弱）。
    #     所以 5 路是"用一点模糊度换更多独立性"，收益与风险都在 0.001 量级 —— 只能靠 LB 判。
    #   ⚠️ 两者都**只动主槽内部**：bigg 槽（ens b2/b3）、s384 槽、prior 全部不变，
    #     所以与已知的 `v68_c75_ens_prior200` = 0.916175 **只差"主槽是否集成"这一个变量**。
    "seed4_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.25), ("img_cliph_s20261101.csv", 0.25),
                         ("img_cliph_s20261102.csv", 0.25), ("img_cliph_s20261103.csv", 0.25)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "seed4_c85": {
        "cliph": (0.85, [("img_cliph_rerun.csv", 0.25), ("img_cliph_s20261101.csv", 0.25),
                         ("img_cliph_s20261102.csv", 0.25), ("img_cliph_s20261103.csv", 0.25)]),
        "bigg":  (0.20 * 0.15 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.15 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "seed4_c100": {
        "cliph": (1.00, [("img_cliph_rerun.csv", 0.25), ("img_cliph_s20261101.csv", 0.25),
                         ("img_cliph_s20261102.csv", 0.25), ("img_cliph_s20261103.csv", 0.25)]),
        "bigg":  (0.00, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.00, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "seed5_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.20), ("img_cliph_s20261101.csv", 0.20),
                         ("img_cliph_s20261102.csv", 0.20), ("img_cliph_s20261103.csv", 0.20),
                         ("img_bigGd384mix.csv", 0.20)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "seed5_c85": {
        "cliph": (0.85, [("img_cliph_rerun.csv", 0.20), ("img_cliph_s20261101.csv", 0.20),
                         ("img_cliph_s20261102.csv", 0.20), ("img_cliph_s20261103.csv", 0.20),
                         ("img_bigGd384mix.csv", 0.20)]),
        "bigg":  (0.20 * 0.15 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.15 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "seed5_c100": {
        "cliph": (1.00, [("img_cliph_rerun.csv", 0.20), ("img_cliph_s20261101.csv", 0.20),
                         ("img_cliph_s20261102.csv", 0.20), ("img_cliph_s20261103.csv", 0.20),
                         ("img_bigGd384mix.csv", 0.20)]),
        "bigg":  (0.00, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.00, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 03:15 **分辨率线第一个产出**：`r_zoom448`（SHORT=448 CROP=224 = 放大 1.75×）
    #   实测：ρ vs cliph_rerun = **0.93379**（比 seed 对的 0.960 独立，比 bigGd384mix 的 0.922 稍差）
    #         ρ vs bigg_rerun = **0.72358**、vs s384r = **0.70155**
    #           ← 而 cliph_rerun 对这两个是 0.860 / 0.817 ⇒ **放大让模型看到了基线没看到的东西**
    #   ⚠️ 伪AUC 0.95315 < cliph_rerun 0.97262 —— **但那是混淆的**：伪AUC 是对着 v49 中位数切分算的，
    #      而 cliph_rerun 占 v49 的 65%，越像 v49 分越高。**不能据此说 r_zoom448 更弱。**
    #   ⇒ 三档从保守到激进，全部 c=0.75 / w=0.20，**与已知的 0.916175 只差主槽内部**：
    #      zoom15  = 主槽 0.85·cliph_rerun + 0.15·r_zoom448   ← 按 `align` 的教训先小权重试（§1.6）
    #      zoom50  = 主槽 0.50 / 0.50                          ← 当"更独立的 seed 集成"用（ρ=.934 < .960）
    #      zoom100 = 主槽整个换成 r_zoom448                     ← 直接回答"放大到底行不行"
    "zoom15_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.85), ("img_r_zoom448.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "zoom50_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.50), ("img_r_zoom448.csv", 0.50)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "zoom100_c75": {
        "cliph": (0.75, [("img_r_zoom448.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 10:50 **放大线的第三档 `r_z896`（SHORT=896 = 3.50×）—— 目前最独立的分量**
    #
    #   实测 ρ vs cliph_rerun，**随放大倍率单调下降**（这是今晚最干净的一条发现）：
    #     SHORT=256 (1.00×) cliph_rerun   ρ = 1.0000  ← 基准
    #     SHORT=448 (1.75×) r_zoom448     ρ = **0.9338**
    #     SHORT=512 (2.00×) r_z512        ρ = **0.9220**
    #     SHORT=896 (3.50×) r_z896        ρ = **0.8602**  ← 已经逼近纯监督族（0.78~0.80）
    #
    #   ⇒ **这修正了 §㉓ 的表述**：独立性不只能靠"换数据 / 换骨干 / 改监督方式"，
    #      **只改裁剪尺度**（同一份数据、同一个教师、同一套 LoRA、同一个 epoch 数）也能造出来，
    #      而且一个点只要 **21 分钟**（r_z512 实测 06:19:02→06:40:15，r_z896 06:40:15→07:01:30）。
    #
    #   ⚠️ 但"更独立"到底是"看到不同证据"还是"更弱更吵"？做了一次判别（10:52）：
    #      拿一个**完全无关的外部信号** `size_prior2`（图像尺寸先验，与图像内容无关）当标尺：
    #        ρ(size_prior2, z896) = **0.1500**  ≈  ρ(size_prior2, z512) = 0.1502 ≈ zoom448 = 0.1506
    #      → **三档与这个正交信号的关联完全一样**，说明 z896 并没丢掉"任务信号"；
    #        而它与其**它模型**的相关整体低 0.03~0.06 ⇒ 更像"用了不同的证据"，而不是"变成噪声"。
    #      ⚠️ 但 0.15 这个数值本身太小，判别力有限，**这个结论不硬**，最终仍要看 LB。
    #
    #   ⚠️ 风险类比：`v64_c300k_ens_prior125.csv` = **0.871805**（cliph300k 进主槽 −0.0427）
    #      证明"低 ρ 不等于有用"。但 z896 与 cliph300k **有一个关键区别**：
    #      z896 是 cliph_rerun 的**同配方单变量版**（只动 SHORT），其收敛损失与 cliph_rerun 同档
    #      （r_c336 末损 0.5605 vs cliph 系 0.5613），而 cliph300k 是换了 302k 数据集的另一个模型。
    #      ⇒ z896 的"单独实力"理应接近 cliph_rerun，这正是 §㉞ 要求的"新分量自己就要够强"。
    #      但这是**推断、不是实测**，所以仍然从小到大排，且必须逐档可回退。
    "z896_15_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.85), ("img_r_z896.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "z896_30_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.70), ("img_r_z896.csv", 0.30)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "z896_50_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.50), ("img_r_z896.csv", 0.50)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "z896_100_c75": {
        "cliph": (0.75, [("img_r_z896.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-14 12:20 **放大轴的中间档 `r_z512`（SHORT=512 = 2.00×）**
    #
    #   为什么要补中间档 —— 12:15 拿齐四档后，这条轴的真面目是**一条单调权衡曲线**：
    #
    #   | 跑 | 放大 | ep3 收敛损失 | ρ vs cliph_rerun |
    #   |---|---|---|---|
    #   | cliph 系种子 | 1.00× | **0.5613** | 0.9604（种子间） |
    #   | `r_zoom448` | 1.75× | 0.5918 | 0.9338 |
    #   | **`r_z512`** | **2.00×** | **0.6020** | **0.9220** |
    #   | `r_z896` | 3.50× | **0.6479** | **0.8602** |
    #
    #   ⇒ **放大是拿「拟合质量」换「独立性」**：放大越多 → 越拟合不上教师的目标（损失越高）
    #     → 排序与基准越不同（ρ 越低）。这条关系**四档全单调**，非常干净。
    #   ⇒ 对照：分辨率轴（`r_c336` 0.5605 / `r_c378` 0.5620）**损失不掉、ρ 也不降** ——
    #     它既没买到独立、也没付出代价，是一根**没有信息的轴**。
    #
    #   ⚠️ 因此**不该只押最独立的那一端**：`r_z896` 的 ρ 最低（0.8602）但它也是**最弱**的
    #     （损失 0.6479，比基准差 15%）。而 §㉞ 的判据是"新分量自己必须够强"，
    #     `v64_c300k`（−0.0427）正是"拿独立性换质量"换亏了的先例。
    #   ⇒ 所以池子必须**在曲线上均匀铺开**：0.15 / 0.30 / 0.50 三档 × {448, 512, 896}。
    #     （448 的对应档是早先建的 `zoom15/zoom50/zoom100_c75`，896 是 `z896_*`，这里补 512。）
    "z512_15_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.85), ("img_r_z512.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "z512_30_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.70), ("img_r_z512.csv", 0.30)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "z512_50_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.50), ("img_r_z512.csv", 0.50)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # `r_c336`（SHORT=384 / CROP=336 = 24×24，需 pos-embed 插值）ρ = 0.9649 —— **更相关**，
    # 说明"改模型分辨率"与"改裁剪尺度"是**两根不同的轴**：前者把模型变得更像基准，后者才制造独立。
    "c336_30_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.70), ("img_r_c336.csv", 0.30)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐⭐ 2026-09-15 01:25 **保留转导 + 掺入独立源 —— 这是"主槽注入"第一次被正确实现**
    #
    #   为什么之前那个 `v64_c300k_ens_prior125`（= 0.871805，−0.0444）不能算"主槽注入失败"：
    #     `c300k_ens` 的配方是 **主槽 100% = `img_cliph300k.csv`**，也就是**整个替换**。
    #     而 `cliph300k` 的 train.log 头写着：`n=301996 real=113177 fake=188819`
    #     + 「只用官方样例的训练划分 800 张，val=200 张」 ⇒ **它是纯监督，没有任何测试集软标签**。
    #     反观 `cliph_rerun`：它是**转导蒸馏学生**，全部价值就在于**复刻教师在测试集上的排序**。
    #   ⇒ **v64 把转导从主槽里彻底拿掉了**，那 0.871805 是"失去转导"的代价，
    #     **不是"C300k 这个独立源没用"的证据**。§㉑ 的模型预测（+0.004~+0.007）**从未被检验过**。
    #
    #   本组配方全部 = `rt_ens_c75` 的骨架（c=0.75 / w=0.20，冠军同款），**只动主槽内部**：
    #     · `cliph_rerun` 保持主导（转导不能丢），把**独立源**按小比例掺进去
    #     · 独立源三个，互相关 0.70~0.84（彼此也独立）：`cliph300k` ρ=.7785 / `gpp` ρ=.7914 / `cliph_f2` ρ=.8012
    #   ⚠️ 风险仍在（§㉔）："独立但弱"没用，`gpp` 实测单独 0.9054、`cliph300k` 单独未知。
    #     所以从 20% 起、逐档可回退，且**一次只测一个变量**。
    "c3_20_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.80), ("img_cliph300k.csv", 0.20)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "c3_35_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.65), ("img_cliph300k.csv", 0.35)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # §㉑ 的正态混合模型算出的两源最优（cliph300k 0.15 + gpp 0.20）
    "c3gpp_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.65), ("img_cliph300k.csv", 0.15), ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # §㉕ 的四源最优（0.40 cliph_rerun + 0.20 cliph300k + 0.25 gpp + 0.15 f2）—— 最激进，只作对照
    "c3gppf2_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.40), ("img_cliph300k.csv", 0.20),
                         ("img_gpp_siglip2.csv", 0.25), ("img_cliph_f2.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-15 01:10 **`cliph_e2` 当主分量 —— 一个从未被用过、且机理上应该更强的分量**
    #
    #   复盘时发现的疏漏：`cliph_e2` / `e1` / `d1` / `d2` / `d3` 这五个 CSV **从未进过任何配方**
    #   （41 个配方一次都没引用，只在注释里被提过）。当时把它们排除的理由是
    #   §㉓ 那句话：「本质是同一家族，不给基底带来新信息，**不作为注入源**」——
    #   但那是**针对"小比例注入"**的判断，**没有回答"它能不能当主分量本身"**。
    #
    #   `cliph_e2` = CLIP-H **+ 测试集 20000 软标签（teacher=anchor）+ Tiny 31369 硬标签**
    #   与 `cliph_rerun` 的**唯一区别就是那个 EXTRA_INDEX**：
    #     `cliph_rerun` 的全部"真标签"只有 **1000 张** sample（w=4.0），其余全靠拟合教师；
    #     `cliph_e2` 多了 **31,369 张真 real/fake 硬标签**（16369 real + 15000 fake）—— **31 倍**。
    #   ⇒ 机理上：`cliph_e2` 应该学到**真正的检测特征**，而不是只做教师的复读机；
    #     同时它仍带测试集软标签，**转导那一层的好处一点没丢**。
    #
    #   ⚠️ 但要诚实说清它的两面性：
    #     · ρ(e2, cliph_rerun) = **0.9315** ⇒ 这是个**实质性改动**（不是微调），不是 0.99 的换汤不换药
    #     · 但同一份数据的一部分（Tiny）已被证明**没能把 ρ 拉低多少**（e1 11949 → 0.9472；e2 31369 → 0.9315），
    #       说明**教师软标签仍然主导**。所以它**可能更强，也可能只是略有不同**，无法先验判定。
    #     · 我们**没有 e2 的 LB 数据**，也没有它的干净 val（cliph 族的 sample val 是泄漏的，§1.6-②）。
    #   ⇒ 结论：这是一个**便宜的、机理明确的高信息量探针**，值得占用一次提交，但不要指望它单独就 +0.02。
    #
    #   配方与 `rt_ens_c75` **逐字相同**，只把主槽那个分量换掉 ⇒ 单变量、可直接归因。
    "e2_c75": {
        "cliph": (0.75, [("img_cliph_e2.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 主槽 = cliph_rerun 与 e2 各半 —— 若 e2 单独略弱但信息互补，这个会优于两者
    "e2_half_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.50), ("img_cliph_e2.csv", 0.50)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 主槽 = cliph_rerun 0.85 + e2 0.15（最保守的试法，按 `align` 教训先小比例）
    "e2_15_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.85), ("img_cliph_e2.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐ 2026-09-14 05:45 **DINOv3-7B 注入**（实测 ρ vs cliph_rerun = **0.9223**）
    #
    #   放进哪一档？——先说清它**不是**"独立注入源"：
    #     §㉓ 认定的独立源是**纯监督族**：cliph300k .7785 / gpp .7914 / cliph_f2 .8012；
    #     而 DINO 是 **0.9223**，属于"同门兄弟"那一档（cliph_d1 .9517 / cliph_e2 .9315 /
    #     种子之间 .9601）。它换掉了骨干（7B / RoPE / 4096 维），ρ 也只从 ~.95 降到 .9223
    #     —— **这又一次证实 §㉓：ρ 由"有没有拿那 20000 张测试集去拟合教师"决定，不由骨干决定。**
    #
    #   ⇒ 所以它**不能**照搬 cliph300k 那种"小比例注入主槽"的逻辑，那条路已被 LB 证伪：
    #      `v64_c300k_ens_prior125.csv` = **0.871805**（cliph300k 进主槽，**−0.0427**）。
    #      它的正确用法是**稀释/替换主分量**，与种子集成同一个机理：ρ 越低，平均后噪声降得越多。
    #      （参考：种子之间 ρ=.9601 → K=4 只降 1.51% 噪声；DINO ρ=.9223 与 cliph 平均，
    #        两路平均就能降 1−√((1+.9223)/2) = **1.94%** —— 比 4 个种子加起来还多。）
    #
    #   ⚠️ 但有一个**本地可见、且指向负面**的信号，必须先说清楚：
    #      **同一损失函数、同一数据、同一 SHORT/CROP/BS/EPOCHS 下**，收敛损失
    #        DINOv3-7B(r16) = **0.5882**   vs   cliph 系种子(CLIP-H r32) = **0.5613**
    #      高 0.027，而且 DINO 的 ep2→ep3 只降 0.0076（**已走平**），
    #      同期 CLIP-H 降了 0.028 —— **DINO 拟合这些目标拟合得更差。**
    #      损失 ≠ 排序质量，所以这不是判决；但在 §1.6「本地指标不能预测 LB」的前提下，
    #      它是**唯一两边可比、且指向上不利**的证据 ⇒ **必须从小比例开始，不许一次放大。**
    #
    #   三档从保守到激进，全部 c=0.75 / w=0.20，**与已知的 0.916175 只差主槽内部一个变量**：
    "dino_r15_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.85), ("img_dino7b.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "dino_r30_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.70), ("img_dino7b.csv", 0.30)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "dino_r50_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.50), ("img_dino7b.csv", 0.50)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "dino_r100_c75": {
        "cliph": (0.75, [("img_dino7b.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-15 02:00 **重蒸产物（新教师）当主槽** —— 用户点名的第 ② 件事「重蒸产物的 w 重新最优化」。
    #   `img_cliph_rt2.csv`   = 教师换成重跑基底后的 4 epoch 学生（实测 ρ vs cliph_rerun = 0.9759）
    #   `img_cliph_rt2e8.csv` = 同配方 8 epoch（训练 loss ep7 = **0.5468** < cliph_rerun 的 0.5613）
    #   动机：cliph_rerun 的教师是**老 anchor**，而重跑基底更强（≈0.9133 vs 0.908114）。
    #   ⚠️ 理论上限：转导学生的 AUC ≤ 教师 AUC。cliph_rerun 已达教师的 99.9%，
    #      所以换教师最多再拿 +0.001 量级 —— **别期待大增益**（§㉓：换教师只会让相关 0.95~0.98，不给独立性）。
    #   ⇒ 与已知冠军 0.916175 **只差主槽内容这一个变量**（bigg 槽 / s384 槽 / prior 全同）。
    "rt2_c75": {
        "cliph": (0.75, [("img_cliph_rt2.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "rt2e8_c75": {
        "cliph": (0.75, [("img_cliph_rt2e8.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "rt2e8_c85": {
        "cliph": (0.85, [("img_cliph_rt2e8.csv", 1.00)]),
        "bigg":  (0.20 * 0.15 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.15 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "rt2e8_c100": {
        "cliph": (1.00, [("img_cliph_rt2e8.csv", 1.00)]),
        "bigg":  (0.00, [("img_bigg_rerun2.csv", 1.00)]),
        "s384":  (0.00, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # **重蒸族 seed 平均**（教师 = 重跑基底，4 个成员：rt2 + 3 个 rt2 seed）。
    # 与 seed4_c75 是**同一机制换一族**：老族（教师=anchor）与 新族（教师=重跑基底）各来一次。
    "rt2seed4_c75": {
        "cliph": (0.75, [("img_cliph_rt2.csv", 0.25), ("img_cliph_rt2_s20261101.csv", 0.25),
                         ("img_cliph_rt2_s20261102.csv", 0.25), ("img_cliph_rt2_s20261103.csv", 0.25)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # **八路总集成**：老族 4 个 seed + 新族 4 个成员（两族都在"同配方换 seed"的同一机制内）
    "all8_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.125), ("img_cliph_s20261101.csv", 0.125),
                         ("img_cliph_s20261102.csv", 0.125), ("img_cliph_s20261103.csv", 0.125),
                         ("img_cliph_rt2.csv", 0.125), ("img_cliph_rt2_s20261101.csv", 0.125),
                         ("img_cliph_rt2_s20261102.csv", 0.125), ("img_cliph_rt2_s20261103.csv", 0.125)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐ 2026-09-15 **多裁剪 TTA 产物当主槽**。
    #   动机：`cliph_rerun.py` 训练用 `RandomCrop`、推理用 `CenterCrop` —— **训练/推理预处理错配**。
    #   TTA 对多个裁剪位置各打一次分再平均，是零训练的标准修法（用户反复强调"训练与推理必须一致"）。
    #   ⚠️ `center1` 位必须逐图重现 `img_cliph_rerun.csv` —— 脚本内建了这条自检。
    "tta_c75": {
        "cliph": (0.75, [("img_cliph_rerun_tta_cross5.csv", 1.00)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "tta_c85": {
        "cliph": (0.85, [("img_cliph_rerun_tta_cross5.csv", 1.00)]),
        "bigg":  (0.20 * 0.15 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.15 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # **TTA + seed 平均**：两个正交的降方差手段叠加（TTA 修预处理错配，seed 平均修初始化噪声）
    "ttaseed5_c75": {
        "cliph": (0.75, [("img_cliph_rerun_tta_cross5.csv", 0.20), ("img_cliph_s20261101.csv", 0.20),
                         ("img_cliph_s20261102.csv", 0.20), ("img_cliph_s20261103.csv", 0.20),
                         ("img_cliph_rerun.csv", 0.20)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐⭐ 2026-09-15 02:20 **只注入 `gpp`（不掺 cliph300k）** —— 用户质疑 cliph300k 太弱，一算果然。
    #
    #   用显式高斯 rank 混合模型（`blend_calc.py`）算出来的结论：
    #     · cliph300k 单独 AUC ≈ **0.826** —— 由 v64/v67 的干净单变量 LB 差（−0.0416）**反解**得到，
    #       而它在 sample 上的探针 val 是 0.9931（**虚高 17 点**，因为初赛 sample 与复赛测试集分布差异大）。
    #       ⇒ 注入 α=0.20：主槽 **−0.0059**、总分 **−0.0035**。**它是净亏的。**
    #     · `gpp` 单独 AUC = **0.9054（LB 实测）**，与主分量 ρ 只 0.7914
    #       ⇒ 注入 α=0.20：主槽 **+0.0060**、总分 **+0.0036**。**唯一真正为正的注入源。**
    #     · 原来的 `c3gpp`（0.65/0.15/0.20）里，cliph300k 把 gpp 的收益几乎全抵掉：只有 +0.0007。
    #   ⚠️ **风险披露（必须知道）**：gpp 的唯一 LB 证据是 `v32_b20_g15`（把 15% 尾部槽换成 gpp）**掉了 0.0027**；
    #      且那条 0.9054 是**早期测试集**上测的，复赛测试集更难、分布可能已变。
    #      ⇒ 所以给 15/20/25/30 四档，**先看低档**，别一次压到 0.35（模型说越大越好，但模型在 align 上错过一次）。
    "gpp15_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.85), ("img_gpp_siglip2.csv", 0.15)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "gpp20_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.80), ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "gpp25_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.75), ("img_gpp_siglip2.csv", 0.25)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "gpp30_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.70), ("img_gpp_siglip2.csv", 0.30)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 两个**独立机制**叠加：gpp 注入（加独立强分量）+ 4 seed 平均（降方差）。
    # 模型预测主槽 ≈ +0.0093；但两个机制各自都有不确定性，**这是期望最高、方差也最大的一版**。
    "gpp20_seed4_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.20), ("img_cliph_s20261101.csv", 0.20),
                         ("img_cliph_s20261102.csv", 0.20), ("img_cliph_s20261103.csv", 0.20),
                         ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # ⭐⭐⭐ 2026-09-15 03:00 **LB 实测回来后的加码**（用户报分）：
    #     v73_seed4_c75_prior200  = **0.917360**（旧冠军 0.916175 → **+0.001185**）
    #     v100_gpp20_seed4        = **0.918887**（→ 再 **+0.001527**）**新冠军**
    #   ⇒ **两条机制都被 LB 证实**：seed 平均 + gpp 注入。后者尤其重要 —— 它**打破了
    #     "加独立分量必掉分"的 0/4 历史记录**，区别在于 `gpp` 有**实测强度 0.9054**，而
    #     `cliph300k` 反解出来只有 **0.826**（见 `blend_calc.py`）。**强度不够的独立分量是负贡献。**
    #   ⇒ **校准出传导系数 k ≈ 0.326**（主槽 AUC 增益 → 总分增益；此前我按 0.60 估，**高估一倍**）：
    #        seed4 : 主槽 +0.0033 → 总分 +0.001185 （k=0.364）
    #        +gpp20: 主槽 +0.0053 → 总分 +0.001527 （k=0.288）
    #   ⇒ 用 k=0.326 重扫 α（`calib.py`）：α=0.30→+0.00045、**α=0.35→+0.00056（峰值）**、
    #        α=0.45→+0.00056、α=0.55→+0.00036（回落）。**增益已很小，这是这条线的尾巴。**
    "gpp35_seed4_c75": {
        "cliph": (0.75, [("img_cliph_rerun.csv", 0.1625), ("img_cliph_s20261101.csv", 0.1625),
                         ("img_cliph_s20261102.csv", 0.1625), ("img_cliph_s20261103.csv", 0.1625),
                         ("img_gpp_siglip2.csv", 0.35)]),
        "bigg":  (0.20 * 0.25 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.25 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    # 抬高 cliph 槽权重：主槽变强后 25% 尾部的相对价值下降 ⇒ 最优 c 应**右移**。
    # 旧槽上实测 c=0.75(0.916175) > c=0.85(0.915787) > c=0.65(0.914498)，0.75~0.85 很平；
    # 槽内换成"4 seed + gpp"之后这一点**从未测过**。
    "gpp20_seed4_c80": {
        "cliph": (0.80, [("img_cliph_rerun.csv", 0.20), ("img_cliph_s20261101.csv", 0.20),
                         ("img_cliph_s20261102.csv", 0.20), ("img_cliph_s20261103.csv", 0.20),
                         ("img_gpp_siglip2.csv", 0.20)]),
        "bigg":  (0.20 * 0.20 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.20 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
    "gpp35_seed4_c80": {
        "cliph": (0.80, [("img_cliph_rerun.csv", 0.1625), ("img_cliph_s20261101.csv", 0.1625),
                         ("img_cliph_s20261102.csv", 0.1625), ("img_cliph_s20261103.csv", 0.1625),
                         ("img_gpp_siglip2.csv", 0.35)]),
        "bigg":  (0.20 * 0.20 / 0.35, [("img_bigg_rerun2.csv", 0.50), ("img_bigg_rerun3.csv", 0.50)]),
        "s384":  (0.15 * 0.20 / 0.35, [("img_bigg_s384_retrain.csv", 1.00)]),
    },
}
DEFAULT_RECIPE = "c3gpp"
PRIOR_W = [0.0, 0.125, 0.20]
# ======================================================================

PRIOR = "size_prior2.csv"
REF = "v49_rt_prior20.csv"      # 已知 LB 0.914498
REF_W = 0.20                    # 它对应的 prior 权重
# 旧基底（用于自检 + 打印对比）
OLD_BASE = [("img_cliph_rerun.csv", 0.65), ("img_bigg_rerun.csv", 0.20),
            ("img_bigg_s384_retrain.csv", 0.15)]


def L(fn):
    d = pd.read_csv(HERE / fn, dtype={"id": str})
    assert not d.id.duplicated().any(), f"{fn} 有重复 id"
    assert len(d) == 20000, f"{fn} 行数 {len(d)} != 20000"
    return d.set_index("id")["score"]


def rk(s, ids):
    return pd.Series(rankdata(s.loc[ids], method="average") / (len(ids) + 1.0), index=ids)


def md5(p):
    return hashlib.md5(p.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="只自检，不写文件")
    ap.add_argument("--recipe", default=DEFAULT_RECIPE, choices=list(RECIPES),
                    help=f"用哪套配方（默认 {DEFAULT_RECIPE}）")
    ap.add_argument("--prefix", default=None, help="输出文件名前缀（默认 v60_<配方名>）")
    ap.add_argument("--prior-w", default=None,
                    help="覆盖 prior 权重，逗号分隔（如 0.17,0.25,0.30）。默认用 PRIOR_W=[0,0.125,0.20]")
    args = ap.parse_args()
    SLOTS = RECIPES[args.recipe]
    prefix = args.prefix or f"v60_{args.recipe}"

    # ---- 配方合法性 ----
    tot = 0.0
    for k, (sw, comps) in SLOTS.items():
        s = sum(w for _, w in comps)
        assert abs(s - 1.0) < 1e-9, f"槽 {k} 内权重和 {s} != 1"
        tot += sw
    assert abs(tot - 1.0) < 1e-9, f"槽权重和 {tot} != 1"

    cl = L(SLOTS["cliph"][1][0][0])
    ids = list(cl.index)
    pr = L(PRIOR)

    # ---- 自检：旧基底 + prior@0.20 必须复现 v49_rt_prior20 ----
    old_base = sum(w * rk(L(f), ids) for f, w in OLD_BASE)
    chk = (1 - REF_W) * old_base + REF_W * rk(pr, ids)
    sp = spearmanr(chk, L(REF).loc[ids]).statistic
    print(f"[自检] 旧基底 + prior@{REF_W}  vs {REF}  corr = {sp:.6f}  （应 ≈1.000000）")
    assert sp > 0.999999, "自检失败：构造器或输入有问题，不要用它出提交文件"

    # ---- 新基底 ----
    base = sum(sw * sum(cw * rk(L(f), ids) for f, cw in comps) for sw, comps in SLOTS.values())
    print()
    print("[新基底配方]")
    for k, (sw, comps) in SLOTS.items():
        for f, cw in comps:
            print(f"    {sw * cw:6.4f}  {f:34s}  （槽 {k} {sw:.2f} × 槽内 {cw:.2f}）")
    print(f"    新基底 vs 旧基底 rank corr = {spearmanr(base, old_base).statistic:+.4f}")

    # ---- 逐份生成 ----
    print()
    rows = []
    w_list = ([float(x) for x in args.prior_w.split(",") if x.strip()]
              if args.prior_w else PRIOR_W)
    for w in w_list:
        z = (1 - w) * base + w * rk(pr, ids)
        out = pd.DataFrame({"id": ids,
                            "score": rankdata(z.to_numpy(), method="average") / (len(z) + 1.0)})
        name = f"{prefix}_prior{int(round(w * 1000)):03d}.csv"
        if not args.dry:
            out.to_csv(HERE / name, index=False, float_format="%.10f")
        # 与当前最高分文件的相关 + 自检
        sp_best = spearmanr(z, L(REF).loc[ids]).statistic
        tag = "纯新基底" if w == 0 else f"新基底×{1-w:.3f} + prior×{w:.3f}"
        rows.append((name, tag, sp_best, "" if args.dry else md5(HERE / name)))
        print(f"  [{'dry' if args.dry else '写出'}] {name:32s} {tag:34s} vs v49best corr={sp_best:.4f}")

    if args.dry:
        return
    print()
    print("| 文件 | 配方 | 与 v49_rt_prior20 的 corr | md5 |")
    print("|---|---|---|---|")
    for name, tag, sp_best, h in rows:
        print(f"| `{name}` | {tag} | {sp_best:.4f} | `{h}` |")
    print()
    print("请逐个跑： " + " && ".join(f"python check_submission.py {r[0]}" for r in rows))


if __name__ == "__main__":
    main()
