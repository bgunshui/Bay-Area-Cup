#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""check_submission.py —— 提交前的统一校验器（可复用）。

用法：
    python check_submission.py <提交文件.csv> [更多文件...]
    python check_submission.py submission/v100_gpp20_seed4_c75_prior200.csv
不给参数则校验 DEFAULT 里列出的候选。

检查项（都是踩过坑的）：
  1. 恰好两列 id,score；20000 行
  2. id 集合与顺序和参照文件逐行一致
     —— 顺序不一致会让 rank 混合按行位置错配（build_mix.py 是按行位置融合的）
  3. 无空值、无重复 id
  4. score ∈ [0,1] 且不是常数（常数值 AUC=0.5）
  5. 打印 md5（提交后可与远端核对）

路径解析（本仓库版相对冻结交付包有改动，见下）：
  · 命令行给的路径依次按 `cwd` → 脚本所在目录 → 逐级向上 4 层（每层再试 `<层>/` 与 `<层>/submission/`）
    查找。所以在「交付包内 CSV 平铺在包根」和「GitHub 仓库内 CSV 在 image-track/submission/」
    两种布局下都能用；找不到时会把试过的路径全部列出来（不静默）。
  · **id 顺序参照**优先用历史文件 `baycup_v31f_b20_s15.csv`（LB 0.908114），
    它不在本仓库里时自动改用仓库内的 LB 提交件 `v100_gpp20_seed4_c75_prior200.csv`
    （两者 id 顺序相同，都来自同一份测试集清单）。实际用了哪个会打印出来。

退出码：全部通过 0；有任一文件失败 1（便于脚本化调用）。
"""
import hashlib
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")
HERE = Path(__file__).resolve().parent
REF = "baycup_v31f_b20_s15.csv"                 # 历史参照（LB 0.908114）；仅用于对齐 id 顺序
LB_SUB = "v100_gpp20_seed4_c75_prior200.csv"    # 仓库内一定有的 LB 提交件，可作等效参照

DEFAULT = ["v57_rt_prior100.csv", "v57_rt_prior125.csv",
           "v57_rt_prior150.csv", "v57_rt_base_only.csv"]

_REF_FILE = None


def _resolve(fname):
    """把命令行给的路径解析成真实文件；找不到返回 None。"""
    p = Path(fname)
    if p.is_absolute():
        return p if p.is_file() else None
    cands = [Path.cwd() / p]
    base = HERE
    for _ in range(4):                          # code/fusion → code → image-track → 仓库根 → …
        cands += [base / p, base / "submission" / p.name]
        if base.parent == base:
            break
        base = base.parent
    for c in cands:
        if c.is_file():
            return c
    return None


def ref_file():
    """id 顺序的参照文件（历史文件优先，其次仓库内的 LB 提交件）。"""
    global _REF_FILE
    if _REF_FILE is None:
        for nm in (REF, LB_SUB):
            f = _resolve(nm)
            if f:
                _REF_FILE = f
                break
        if _REF_FILE is None:
            sys.exit("❌ 找不到 id 顺序参照文件（试过 "
                     f"{REF} 与 {LB_SUB}），请把它们之一放到当前目录或 submission/ 下。")
    return _REF_FILE


def check(fname):
    p = _resolve(fname)
    if p is None:
        print(f"[{fname}] 不存在（已按 cwd / 脚本目录 / 逐级向上查找）")
        return False
    rp = ref_file()
    ref = pd.read_csv(rp, dtype={"id": str})
    c = pd.read_csv(p, dtype={"id": str})
    b = p.read_bytes()
    md5 = hashlib.md5(b).hexdigest()

    issues = []
    if list(c.columns) != ["id", "score"]:
        issues.append(f"列名应为 ['id','score']，实际 {list(c.columns)}")
    if len(c) != 20000:
        issues.append(f"行数应为 20000，实际 {len(c)}")
    # 列名不对时，下面依赖 id/score 的检查要跳过 —— 否则畸形文件会抛 traceback，
    # 而不是干净地报"失败"（本仓库版修复；原先会 AttributeError）。
    if "id" in c.columns:
        if set(ref.id) != set(c.id):
            issues.append("id 集合与参照不一致")
        elif ref.id.tolist() != c.id.tolist():
            issues.append("id 顺序与参照不一致（会导致按行位置融合时错配！）")
        if int(c.id.duplicated().sum()) != 0:
            issues.append(f"id 重复 {int(c.id.duplicated().sum())}")
        if int(c.id.isna().sum()) != 0:
            issues.append(f"id 存在空值 {int(c.id.isna().sum())}")
    if "score" in c.columns:
        if int(c.score.isna().sum()) != 0:
            issues.append(f"score 存在空值 {int(c.score.isna().sum())}")
        try:
            if float(c.score.min()) < 0 or float(c.score.max()) > 1:
                issues.append(f"score 越界 [{c.score.min()}, {c.score.max()}]")
        except (TypeError, ValueError):
            issues.append("score 不是数值列")
        if c.score.nunique() < 1000:
            issues.append(f"score 只有 {c.score.nunique()} 个不同值，疑似常数/退化")

    status = "通过" if not issues else "失败"
    rng = (f"范围=[{c.score.min():.6f}, {c.score.max():.6f}]  " if "score" in c.columns else "")
    uniq = (f"唯一值={c.score.nunique()}  " if "score" in c.columns else "")
    print(f"[{fname}]  {status}   行={len(c)}  {rng}{uniq}{len(b):,}B")
    print(f"    md5 = {md5}")
    for it in issues:
        print(f"    !! {it}")
    return not issues


def main():
    files = sys.argv[1:]
    if not files:
        files = [f for f in DEFAULT if _resolve(f)]
        if not files:
            files = [LB_SUB]
            print("（未给参数，且 DEFAULT 里那批候选不在本目录）→ 改为校验仓库内的 LB 提交件")
    print(f"id 顺序参照： {ref_file()}")
    # ⚠️ 必须"先全部检查再汇总"：原先是 all(check(f) for f in files)，
    #    而 all() 会在第一个 False 处**短路** → 多文件时后面的文件根本不会被检查。
    results = [check(f) for f in files]
    ok = all(results)
    print()
    print(f"全部通过 ✅（{len(results)} 个文件）" if ok
          else f"存在失败项 ❌（{results.count(False)}/{len(results)} 个失败）")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
