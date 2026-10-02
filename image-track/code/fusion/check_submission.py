#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_submission.py —— 提交前的统一校验器（可复用）。

用法：python check_submission.py v57_rt_prior150.csv [更多文件...]
不给参数则校验最近生成的一批候选。

检查项（都是踩过坑的）：
  1. 恰好两列 id,score；20000 行
  2. id 集合与顺序和参照文件（LB 已验证过的 baycup_v31f_b20_s15.csv）逐行一致
     —— 顺序不一致会让 rank 混合按行位置错配（build_mix.py 是按行位置融合的）
  3. 无空值、无重复 id
  4. score ∈ [0,1] 且不是常数（常数值 AUC=0.5）
  5. 打印 md5（提交后可与远端核对）
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
REF = "baycup_v31f_b20_s15.csv"          # LB 实测 0.908114，id 顺序的基准

DEFAULT = ["v57_rt_prior100.csv", "v57_rt_prior125.csv",
           "v57_rt_prior150.csv", "v57_rt_base_only.csv"]


def check(fname):
    p = HERE / fname
    if not p.exists():
        print(f"[{fname}] 不存在")
        return False
    ref = pd.read_csv(HERE / REF, dtype={"id": str})
    c = pd.read_csv(p, dtype={"id": str})
    b = p.read_bytes()
    md5 = hashlib.md5(b).hexdigest()

    issues = []
    if list(c.columns) != ["id", "score"]:
        issues.append(f"列名应为 ['id','score']，实际 {list(c.columns)}")
    if len(c) != 20000:
        issues.append(f"行数应为 20000，实际 {len(c)}")
    if set(ref.id) != set(c.id):
        issues.append("id 集合与参照不一致")
    elif ref.id.tolist() != c.id.tolist():
        issues.append("id 顺序与参照不一致（会导致按行位置融合时错配！）")
    if int(c.isna().sum().sum()) != 0:
        issues.append(f"存在空值 {int(c.isna().sum().sum())}")
    if int(c.id.duplicated().sum()) != 0:
        issues.append(f"id 重复 {int(c.id.duplicated().sum())}")
    if c.score.min() < 0 or c.score.max() > 1:
        issues.append(f"score 越界 [{c.score.min()}, {c.score.max()}]")
    if c.score.nunique() < 1000:
        issues.append(f"score 只有 {c.score.nunique()} 个不同值，疑似常数/退化")

    status = "通过" if not issues else "失败"
    print(f"[{fname}]  {status}   行={len(c)}  "
          f"范围=[{c.score.min():.6f}, {c.score.max():.6f}]  "
          f"唯一值={c.score.nunique()}  {len(b):,}B")
    print(f"    md5 = {md5}")
    for it in issues:
        print(f"    !! {it}")
    return not issues


def main():
    files = sys.argv[1:] or DEFAULT
    ok = all(check(f) for f in files)
    print()
    print("全部通过 ✅" if ok else "存在失败项 ❌")


if __name__ == "__main__":
    main()
