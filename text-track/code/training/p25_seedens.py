#!/usr/bin/env python3
"""Phase 25: 7B multi-seed ensemble branch + probes.
branch = rank01(mean(rank(official), rank(s2), rank(s3))); saved as lora7b_seedens_*.
"""
import glob
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score

RUNS = "/data/湾区杯/文本赛道/runs"
SUB = "/data/湾区杯/文本赛道/submissions"


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def main():
    y = np.load(RUNS + "/labels.npy").astype(np.int64)
    parts = ["lora7b_official", "lora7b_official_s2", "lora7b_official_s3"]
    oofs, reps = [], []
    for p in parts:
        try:
            oofs.append(rank01(np.load(RUNS + f"/{p}_oof.npy")))
            reps.append(rank01(np.load(RUNS + f"/{p}_replay.npy")))
            print("loaded", p, flush=True)
        except Exception as exc:
            print("skip", p, repr(exc)[:100], flush=True)
    if len(oofs) < 2:
        raise RuntimeError("need at least 2 seeds")
    oof = np.mean(oofs, axis=0)
    rep = np.mean(reps, axis=0)
    np.save(RUNS + "/lora7b_seedens_oof.npy", oof)
    np.save(RUNS + "/lora7b_seedens_replay.npy", rank01(rep))
    print("seedens OOF AUC", round(float(roc_auc_score(y, oof)), 6), "seeds", len(oofs), flush=True)

    # probes with seedens as ft
    z = np.load(RUNS + "/fusion_mat.npz")
    names = [str(n) for n in z["names"]]
    rep_all = z["rep_mat"]
    report = json.load(open(RUNS + "/p4_report.json", encoding="utf-8"))
    member_w = report["family_member_weights"]
    fam = {}
    for fname in ("llm", "robust"):
        v = np.zeros(rep_all.shape[0], dtype=np.float64)
        for n, w in member_w.get(fname, {}).items():
            if w > 0 and n in names:
                v += w * rep_all[:, names.index(n)]
        fam[fname] = rank01(v)
    test = pd.read_csv("/data/湾区杯/replay_data/文本赛道/text_test.csv", encoding="utf-8-sig")
    ids = test.id.astype(str).tolist()
    from scipy.stats import rankdata as _rd

    seedens = _rd(rep, method="average") / (len(rep) + 1.0)
    combos = {
        "probe_seedens_1": {"llm": 0.30, "ft": 0.60, "rob": 0.10},
        "probe_seedens_2": {"llm": 0.25, "ft": 0.65, "rob": 0.10},
        "probe_seedens_3": {"llm": 0.30, "ft": 0.55, "rob": 0.15},
    }
    for name, w in combos.items():
        s = w["llm"] * fam["llm"] + w["ft"] * seedens + w["rob"] * fam["robust"]
        s = rank01(s)
        pd.DataFrame({"id": ids, "score": s}).to_csv(SUB + "/" + name + ".csv", index=False)
        print("wrote", name, "unique", int(np.unique(s).size), flush=True)
    print("P25 DONE", flush=True)


if __name__ == "__main__":
    main()
