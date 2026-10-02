#!/usr/bin/env python3
"""Phase 26: seedens2 = clean r2 + s2 + s3 (drop the eviction-degraded original)."""
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
    parts = ["lora7b_official_r2", "lora7b_official_s2", "lora7b_official_s3"]
    oofs, reps = [], []
    for p in parts:
        oofs.append(rank01(np.load(RUNS + f"/{p}_oof.npy")))
        reps.append(rank01(np.load(RUNS + f"/{p}_replay.npy")))
        print("loaded", p, "oof", round(float(roc_auc_score(y, np.load(RUNS + f"/{p}_oof.npy"))), 6), flush=True)
    oof = np.mean(oofs, axis=0)
    rep = rank01(np.mean(reps, axis=0))
    np.save(RUNS + "/lora7b_seedens2_oof.npy", oof)
    np.save(RUNS + "/lora7b_seedens2_replay.npy", rep)
    print("seedens2 OOF AUC", round(float(roc_auc_score(y, oof)), 6), flush=True)

    z = np.load(RUNS + "/fusion_mat.npz")
    names = [str(n) for n in z["names"]]
    rep_all = z["rep_mat"]
    import json

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
    combos = {
        "probe_seedens2_1": {"llm": 0.30, "ft": 0.60, "rob": 0.10},
        "probe_seedens2_2": {"llm": 0.25, "ft": 0.65, "rob": 0.10},
        "probe_seedens2_3": {"llm": 0.30, "ft": 0.55, "rob": 0.15},
    }
    for name, w in combos.items():
        s = w["llm"] * fam["llm"] + w["ft"] * rep + w["rob"] * fam["robust"]
        s = rank01(s)
        pd.DataFrame({"id": ids, "score": s}).to_csv(SUB + "/" + name + ".csv", index=False)
        print("wrote", name, "unique", int(np.unique(s).size), flush=True)
    print("P26 DONE", flush=True)


if __name__ == "__main__":
    main()
