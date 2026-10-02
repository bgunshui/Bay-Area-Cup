#!/usr/bin/env python3
"""Phase 4: hierarchical rank fusion of all branches -> final submissions.

Level 0: per-branch score (OOF + replay rank), grouped into 4 families:
         public / llm / ft / robust
Level 1: within-family constrained weight grid (fast, 3-8 members each)
Level 2: family-level grid (step 0.05) maximizing OOF AUC (rank-based AUC)
Level 3: near-duplicate equalization + retrieval override + rank finalize
Outputs: submit_main.csv (OOF-optimal), submit_robust_prior.csv (public-heavy)
"""
import glob
import itertools
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.feature_extraction.text import TfidfVectorizer

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
OUT_SUB = BASE / "文本赛道" / "submissions"

try:
    import lightgbm as lgb
except Exception:
    lgb = None
from xgboost import XGBClassifier  # noqa: E402


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def window_feats(mat):
    f = np.asarray(mat, dtype=np.float64)
    out = []
    for i in range(f.shape[0]):
        v = f[i][~np.isnan(f[i])]
        if v.size == 0:
            v = np.asarray([0.5])
        out.append([v.mean(), v.max(), v.min(), np.quantile(v, 0.75), v.size,
                    v[0], v[-1], v.std() if v.size > 1 else 0.0])
    return np.asarray(out, dtype=np.float64)


def score_public(tag, sample_mat, replay_mat, folds, y):
    X = window_feats(sample_mat)
    Xt = window_feats(replay_mat)
    oof = np.zeros(len(y), dtype=np.float64)
    tests = []
    for f in range(5):
        tr = np.flatnonzero(folds != f)
        va = np.flatnonzero(folds == f)
        if lgb is not None:
            m = lgb.LGBMClassifier(objective="binary", n_estimators=300, learning_rate=0.03,
                                   num_leaves=7, max_depth=3, min_child_samples=35,
                                   reg_alpha=0.5, reg_lambda=2.0, random_state=20260827 + f,
                                   n_jobs=8, verbosity=-1)
        else:
            m = XGBClassifier(n_estimators=300, learning_rate=0.03, max_depth=3,
                              reg_alpha=0.5, reg_lambda=2.0, random_state=20260827 + f,
                              n_jobs=8, eval_metric="logloss")
        m.fit(X[tr], y[tr])
        oof[va] = m.predict_proba(X[va])[:, 1]
        tests.append(m.predict_proba(Xt)[:, 1])
    return oof, np.mean(tests, axis=0)


def roc_auc_fast(y, scores):
    """Rank-based AUC (Mann-Whitney)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    # average ties
    tied = np.sort(scores)
    uniq, counts = np.unique(tied, return_counts=True)
    if (counts > 1).any():
        start = 0
        for u, c in zip(uniq, counts, strict=True):
            if c > 1:
                ranks[(scores == u)] = (start + 1 + start + c) / 2.0
            start += c
    pos = np.flatnonzero(y == 1)
    s = ranks[pos].sum()
    n1 = len(pos)
    n0 = len(y) - n1
    return (s - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def _compositions(units, k):
    """Yield all k-tuples of non-negative ints summing to units (C(units+k-1, k-1) of them)."""
    if k == 1:
        yield (units,)
        return
    for first in range(units + 1):
        for rest in _compositions(units - first, k - 1):
            yield (first,) + rest


def family_blend(names, oof_mat, rep_mat, step, y, caps=None, floors=None):
    """Search within-family weights (composition grid) maximizing OOF AUC.

    caps:   optional dict {name: max_weight}
    floors: optional dict {name: min_weight} (e.g. keep replay-proven members alive)
    """
    K = len(names)
    if K == 1:
        return np.asarray([1.0])
    units = int(round(1 / step))
    best = None
    for combo in _compositions(units, K):
        w = np.asarray(combo, dtype=np.float64) / units
        if caps:
            ok = True
            for i, nm in enumerate(names):
                if w[i] > caps.get(nm, 1.0) + 1e-9:
                    ok = False
                    break
            if not ok:
                continue
        if floors:
            ok = True
            for i, nm in enumerate(names):
                if w[i] < floors.get(nm, 0.0) - 1e-9:
                    ok = False
                    break
            if not ok:
                continue
        sc = oof_mat @ w
        auc = roc_auc_fast(y, sc)
        key = (auc, -int((w > 0).sum()))
        if best is None or key > best[0]:
            best = (key, w)
    return best[1]


def main():
    OUT_SUB.mkdir(parents=True, exist_ok=True)
    y = np.load(RUNS / "labels.npy").astype(np.int64)
    fold_path = RUNS / "folds.npy"
    if not fold_path.exists():
        sys.path.insert(0, "/data/湾区杯/文本赛道/repro")
        from wanqu_first_submission.folds import make_folds, text_near_duplicate_groups

        folds = make_folds(y, text_near_duplicate_groups(
            pd.read_csv(glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0],
                        encoding="utf-8-sig").text.fillna("").astype(str).tolist()), 5, 20260823)
        np.save(fold_path, folds)
    else:
        folds = np.load(fold_path)
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    ids = test.id.astype(str).tolist()
    n = len(y)
    assert len(ids) == 20000

    branches = {}
    report = {"branch_auc": {}}

    for tag in sorted(p.name[:-len("_sample.npy")] for p in RUNS.glob("*_sample.npy")):
        rp = RUNS / f"{tag}_replay.npy"
        if not rp.exists():
            continue
        sm = np.load(RUNS / f"{tag}_sample.npy")
        rm = np.load(rp)
        if sm.ndim == 2 and sm.shape[1] == 2:
            sm = sm[:, 1:2]
            rm = rm[:, 1:2]
        if sm.ndim == 1:
            sm = sm[:, None]
            rm = rm[:, None]
        if sm.shape[0] != n or rm.shape[0] != len(ids):
            continue
        oof, rep = score_public(tag, sm, rm, folds, y)
        branches[tag] = (oof, rep)
        report["branch_auc"][tag] = float(roc_auc_fast(y, oof))
        print("branch", tag, round(report["branch_auc"][tag], 6), flush=True)

    p = BASE / "文本赛道" / "repro" / "artifacts" / "text" / "qwen_pair_v2" / "predictions.npz"
    if p.exists():
        with np.load(p) as z:
            branches["qwen_pair"] = (z["oof"].astype(np.float64), z["final_raw"].astype(np.float64))
        report["branch_auc"]["qwen_pair"] = float(roc_auc_fast(y, branches["qwen_pair"][0]))
        print("branch qwen_pair", round(report["branch_auc"]["qwen_pair"], 6), flush=True)

    robust_extra = None
    p = BASE / "文本赛道" / "repro" / "artifacts" / "text" / "robust_blend_v23" / "predictions.npz"
    if not p.exists():
        p = BASE / "文本赛道" / "repro" / "artifacts" / "text" / "robust_v2" / "predictions.npz"
    if p.exists():
        with np.load(p) as z:
            branches["robust"] = (z["oof"].astype(np.float64), z["raw_final_blend"].astype(np.float64))
            robust_extra = {"strength": z["retrieval_strength"], "nearest": z["nearest_index"]}
        report["branch_auc"]["robust"] = float(roc_auc_fast(y, branches["robust"][0]))
        print("branch robust", round(report["branch_auc"]["robust"], 6), flush=True)

    for tag in sorted(q.name[:-len("_oof.npy")] for q in RUNS.glob("*_oof.npy")):
        rp = RUNS / f"{tag}_replay.npy"
        if "_fold" in tag or tag in ("q14b_lik", "q14b_all") or not rp.exists():
            continue
        # v1 14B teacher is saturated (36-46% tied) — exclude, use v2 (logit-diff) only
        if tag == "lora14b_official":
            print("skip saturated v1 14B teacher:", tag, flush=True)
            continue
        oof = np.load(RUNS / f"{tag}_oof.npy")
        rep = np.load(rp)
        branches[tag] = (oof, rep)
        report["branch_auc"][tag] = float(roc_auc_fast(y, oof))
        print("branch", tag, round(report["branch_auc"][tag], 6), flush=True)
    for tag in ("q14b_lik", "q14b_all"):
        op, rp = RUNS / f"{tag}_oof.npy", RUNS / f"{tag}_replay.npy"
        if op.exists() and rp.exists():
            branches[tag] = (np.load(op), np.load(rp))
            report["branch_auc"][tag] = float(roc_auc_fast(y, np.load(op)))
            print("branch", tag, round(report["branch_auc"][tag], 6), flush=True)

    assert len(branches) >= 3
    names = list(branches)
    oof_mat = np.column_stack([rank01(branches[k][0]) for k in names])
    rep_mat = np.column_stack([rank01(branches[k][1]) for k in names])
    report["corr"] = pd.DataFrame(oof_mat, columns=names).corr(method="spearman").round(4).to_dict()

    def family(name):
        if name == "robust":
            return "robust"
        if ("qwen_pair" in name or name.startswith("q14b")
                or name.startswith("q3h") or name.startswith("q7b")
                or name.startswith("q4b") or name.startswith("nll_fp")
                or name.startswith("q15b") or name.startswith("q3b_lik")):
            return "llm"
        if "macbert" in name or "lora" in name or name.startswith("pseudo"):
            return "ft"
        return "public"

    fams = ["public", "llm", "ft", "robust"]
    fam_members = {f: [k for k in names if family(k) == f] for f in fams}
    print("families", {f: fam_members[f] for f in fams}, flush=True)

    fam_oof = {}
    fam_rep = {}
    fam_w = {}
    for f in fams:
        members = fam_members[f]
        if not members:
            continue
        idx = [names.index(k) for k in members]
        # 14B teacher saturates (36-46% samples tied at softmax 1.0) -> cap as supplement
        caps = {"lora14b_official": 0.15} if f == "ft" else None
        # replay-proven members must not be zeroed by pure OOF optimization:
        # give minimum weights to 7B champion (0.9615 official replay) & new 7B student
        floors = None
        if f == "ft" and "--ft-fix" in sys.argv:
            floors = {}
            # v2-era core members must not be zeroed by pure OOF:
            # keep new r2 students + v2 14B teacher + replay-proven 7B champion at >=25% each
            for nm in ("lora7b_official", "lora7b_distill_r2", "lora1.5b_distill_r2", "lora14b_official_v2"):
                if nm in members:
                    floors[nm] = 0.2  # >=20% each; 20% left for optimizer/others
        w = family_blend(members, oof_mat[:, idx], rep_mat[:, idx], 0.25, y, caps, floors)
        fam_oof[f] = oof_mat[:, idx] @ w
        fam_rep[f] = rep_mat[:, idx] @ w
        fam_w[f] = dict(zip(members, np.round(w, 3)))
        print("family", f, "auc", round(float(roc_auc_fast(y, fam_oof[f])), 6),
              fam_w[f], flush=True)

    active = [f for f in fams if f in fam_oof]
    F = len(active)
    if F == 1:
        best_w = np.ones(1)
        fams_used = active
    else:
        units = 20
        candidates = []
        # optimized: enumerate simplex
        for combo in itertools.product(range(units + 1), repeat=F - 1):
            s = sum(combo)
            if s > units:
                continue
            w = np.asarray([*combo, units - s], dtype=np.float64) / units
            for idx_f, f in enumerate(active):
                if f == "robust" and w[idx_f] > 0.35:
                    break
                if f == "llm" and w[idx_f] > 0.40:
                    break
                if f == "ft" and w[idx_f] > 0.50:
                    break
                if f == "public" and w[idx_f] > 0.05:
                    break
            else:
                candidates.append(w)
        best = None
        for w in candidates:
            sc = fam_oof_fast = sum((fam_oof[f] * w[i] for i, f in enumerate(active)), np.zeros(n))
            auc = roc_auc_fast(y, sc)
            key = (auc, -int((w > 0).sum()))
            if best is None or key > best[0]:
                best = (key, w)
        best_w = best[1]
        fams_used = active
    print("family weights", dict(zip(fams_used, np.round(best_w, 3))), flush=True)

    oof_final = sum((fam_oof[f] * best_w[i] for i, f in enumerate(fams_used)), np.zeros(n))
    rep_final = sum((fam_rep[f] * best_w[i] for i, f in enumerate(fams_used)), np.zeros(len(rep_mat)))

    def _dup_groups():
        texts = test.text.fillna("").astype(str).tolist()
        vec = TfidfVectorizer(analyzer="char", ngram_range=(3, 4), min_df=2, max_features=40000,
                              sublinear_tf=True, dtype=np.float32)
        M = vec.fit_transform(texts)
        sim = (M @ M.T).toarray()
        groups = np.arange(len(texts))
        for i in range(len(texts)):
            near = np.flatnonzero(sim[i] >= 0.96)
            near = near[near >= i]
            if near.size > 1:
                groups[near] = groups[near[0]]
        uniq = sorted(set(groups.tolist()))
        remap = {g: i for i, g in enumerate(uniq)}
        groups = np.asarray([remap[g] for g in groups])
        return groups

    dup_groups = _dup_groups()

    def finalize(rep_score, label, groups=dup_groups):
        # near-duplicate equalization (groups precomputed once)
        for g in np.unique(groups):
            idx = np.flatnonzero(groups == g)
            if len(idx) > 1:
                rep_score[idx] = rep_score[idx].mean()
        if robust_extra is not None:
            strength = np.asarray(robust_extra["strength"], dtype=np.float64)
            nearest = np.asarray(robust_extra["nearest"], dtype=np.int64)
            targets = np.where(y[nearest] == 1, 0.9995, 0.0005)
            rep_score = (1.0 - strength) * rep_score + strength * targets
        final = rank01(rep_score)
        out = OUT_SUB / f"{label}.csv"
        pd.DataFrame({"id": ids, "score": final}).to_csv(out, index=False)
        print(label, "oof", round(float(roc_auc_fast(y, oof_final)), 6),
              "unique", int(np.unique(final).size), "->", out, flush=True)
        return out

    out_main = finalize(rep_final.copy(), "submit_main")
    # no-public candidates (public detectors weak on replay per official feedback)
    for label, wfix in (
        ("submit_nopub_1", {"public": 0.0, "llm": 0.35, "ft": 0.40, "robust": 0.25}),
        ("submit_nopub_2", {"public": 0.0, "llm": 0.25, "ft": 0.50, "robust": 0.25}),
        ("submit_nopub_3", {"public": 0.0, "llm": 0.30, "ft": 0.30, "robust": 0.40}),
    ):
        rep_w = sum((fam_rep[f] * wfix.get(f, 0.0) for f in fams_used if wfix.get(f, 0.0) > 0),
                    np.zeros(len(rep_mat)))
        finalize(rep_w, label)
    # robust-prior candidate: force public family weight up
    if "public" in fam_oof:
        w = {f: 0.0 for f in fams}
        w["public"] = 0.8
        w["llm"] = 0.1
        w["ft"] = 0.1
        if "robust" in fam_oof:
            w["robust"] = 0.1
            w["public"] = 0.7
        rep_rob = sum((fam_rep[f] * w.get(f, 0.0) for f in fams_used if w.get(f, 0.0) > 0), np.zeros(len(rep_mat)))
        out_rob = finalize(rep_rob, "submit_robust_prior")
    else:
        out_rob = None

    np.savez(RUNS / "fusion_mat.npz", names=np.asarray(names),
             oof_mat=oof_mat, rep_mat=rep_mat)
    report["family_weights_main"] = dict(zip(fams_used, np.round(best_w, 3)))
    report["family_member_weights"] = fam_w
    report["files"] = {"main": str(out_main), "robust_prior": str(out_rob)}
    with open(RUNS / "p4_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("P4 DONE", flush=True)


if __name__ == "__main__":
    main()
