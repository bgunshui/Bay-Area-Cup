#!/usr/bin/env python3
"""Phase 17: 7B-heavier probes + KNN smoothing of probe_ft7b_1."""
import json
import sys

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.neighbors import NearestNeighbors

RUNS = "/data/湾区杯/文本赛道/runs"
SUB = "/data/湾区杯/文本赛道/submissions"


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def main():
    z = np.load(RUNS + "/fusion_mat.npz")
    names = [str(n) for n in z["names"]]
    rep = z["rep_mat"]
    report = json.load(open(RUNS + "/p4_report.json", encoding="utf-8"))
    member_w = report["family_member_weights"]
    fam = {}
    for fname in ("llm", "robust"):
        v = np.zeros(rep.shape[0], dtype=np.float64)
        for n, w in member_w.get(fname, {}).items():
            if w > 0 and n in names:
                v += w * rep[:, names.index(n)]
        fam[fname] = rank01(v)
    lora7b = rank01(rep[:, names.index("lora7b_official")])
    # old ft family (without 7B)
    ft_old = np.zeros(rep.shape[0], dtype=np.float64)
    for n, w in member_w.get("ft", {}).items():
        if w > 0 and n in names and n != "lora7b_official":
            ft_old += w * rep[:, names.index(n)]
    ft_old = rank01(ft_old)

    def ft_hyb(alpha):
        return rank01(alpha * lora7b + (1 - alpha) * ft_old)

    test = pd.read_csv("/data/湾区杯/replay_data/文本赛道/text_test.csv", encoding="utf-8-sig")
    ids = test.id.astype(str).tolist()
    combos = {
        "probe_ft7b2_a": {"a": 0.85, "llm": 0.30, "ft": 0.55, "rob": 0.15},
        "probe_ft7b2_b": {"a": 1.00, "llm": 0.30, "ft": 0.60, "rob": 0.10},
        "probe_ft7b2_c": {"a": 0.85, "llm": 0.35, "ft": 0.55, "rob": 0.10},
        "probe_ft7b2_d": {"a": 1.00, "llm": 0.25, "ft": 0.60, "rob": 0.15},
        "probe_ft7b2_e": {"a": 0.90, "llm": 0.30, "ft": 0.60, "rob": 0.10},
    }
    for name, w in combos.items():
        s = w["llm"] * fam["llm"] + w["ft"] * ft_hyb(w["a"]) + w["rob"] * fam["robust"]
        s = rank01(s)
        pd.DataFrame({"id": ids, "score": s}).to_csv(SUB + "/" + name + ".csv", index=False)
        print("wrote", name, "unique", int(np.unique(s).size), flush=True)

    # KNN smoothing of probe_ft7b_1
    try:
        emb = np.load(RUNS + "/q7b_hidden_test.npz")["mean"].astype(np.float32)
        nn = NearestNeighbors(n_neighbors=17, metric="cosine", n_jobs=-1).fit(emb)
        dist, idx = nn.kneighbors(emb)
        sim = 1.0 - np.clip(dist, 0, 1)
        base = rank01(pd.read_csv(SUB + "/probe_ft7b_1.csv").score.to_numpy(np.float64))

        def smooth(scores, k, tau, iters, gamma=0.9):
            s = scores.copy()
            for _ in range(iters):
                nb = idx[:, 1:k + 1]
                w = np.maximum(sim[:, 1:k + 1] - tau, 0)
                wsum = w.sum(axis=1, keepdims=True)
                navg = (w * s[nb]).sum(axis=1) / np.maximum(wsum, 1e-6)[:, 0]
                agree = np.abs(s[nb] - s[:, None]).mean(axis=1)
                conf = (w.sum(axis=1) / np.maximum(wsum, 1e-6)[:, 0] * (wsum[:, 0] > 0)) * (1.0 - agree)
                pull = np.clip(gamma * conf, 0, 0.5)
                s = s + pull * (navg - s)
            return rank01(s)

        for k, tau, iters in ((16, 0.85, 2), (16, 0.90, 1), (8, 0.80, 2)):
            s = smooth(base, k, tau, iters)
            pd.DataFrame({"id": ids, "score": s}).to_csv(
                SUB + f"/knn_ft7b1_k{k}_t{int(tau * 100)}.csv", index=False)
            print(f"wrote knn_ft7b1_k{k}_t{int(tau * 100)}.csv", flush=True)
    except Exception as exc:
        print("knn fail", repr(exc)[:200], flush=True)
    print("P17 DONE", flush=True)


if __name__ == "__main__":
    main()
