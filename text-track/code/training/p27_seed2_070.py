#!/usr/bin/env python3
"""Phase 27: seedens2-7B at 0.70 weight variants."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import rankdata

RUNS = "/data/湾区杯/文本赛道/runs"
SUB = "/data/湾区杯/文本赛道/submissions"


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def main():
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
    if "lora7b_seedens2" in names:
        seed2 = rank01(rep_all[:, names.index("lora7b_seedens2")])
    else:  # recompute from components
        seed2 = rank01(np.mean([rank01(np.load(RUNS + f"/{p}_replay.npy"))
                                for p in ("lora7b_official_r2", "lora7b_official_s2", "lora7b_official_s3")], axis=0))
    print("seedens2 available", flush=True)

    test = pd.read_csv("/data/湾区杯/replay_data/文本赛道/text_test.csv", encoding="utf-8-sig")
    ids = test.id.astype(str).tolist()
    combos = {
        "probe_seedens2_070a": {"llm": 0.20, "ft": 0.70, "rob": 0.10},
        "probe_seedens2_070b": {"llm": 0.25, "ft": 0.70, "rob": 0.05},
        "probe_seedens2_070c": {"llm": 0.15, "ft": 0.70, "rob": 0.15},
        "probe_seedens2_070d": {"llm": 0.20, "ft": 0.70, "rob": 0.10},
        "probe_seedens2_065": {"llm": 0.25, "ft": 0.65, "rob": 0.10},
    }
    for name, w in combos.items():
        s = w["llm"] * fam["llm"] + w["ft"] * seed2 + w["rob"] * fam["robust"]
        s = rank01(s)
        pd.DataFrame({"id": ids, "score": s}).to_csv(SUB + "/" + name + ".csv", index=False)
        print("wrote", name, "unique", int(np.unique(s).size), flush=True)
    print("P27 DONE", flush=True)


if __name__ == "__main__":
    main()
