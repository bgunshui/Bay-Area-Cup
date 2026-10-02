#!/usr/bin/env python3
"""Phase 44b: generator-trace NLL fingerprint — SPLIT STAGES (avoid HAMI cross-model mem leak).
Usage: python3 p44b_genfp.py --stage 15b | --stage 7b | --stage lgb
 15b: compute 1.5B NLL stats on official+replay -> genfp_f15_{official,replay}.npy
 7b : compute 7B NLL stats (batch 2)          -> genfp_f7_{official,replay}.npy
 lgb: build features + 5-fold LGB              -> genfp_oof.npy / genfp_replay.npy
"""
import argparse, glob, sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path("/data/湾区杯")
RUNS = BASE / "文本赛道" / "runs"
INIT = BASE / "initial_text"
MODELS = BASE / "文本赛道" / "models"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_texts():
    sample_csv = glob.glob(str(INIT / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_csv, encoding="utf-8-sig")
    y = train.label.astype(int).to_numpy()
    official = train.text.fillna("").astype(str).tolist()
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    replay = test.text.fillna("").astype(str).tolist()
    return y, official, replay

def nll_stats(model, tokenizer, texts, tag, dev, bs, maxlen=512):
    model.eval()
    rows = []
    with torch.inference_mode():
        for s in range(0, len(texts), bs):
            enc = tokenizer(texts[s:s+bs], padding=True, truncation=True, max_length=maxlen,
                            return_tensors="pt")
            ids = enc["input_ids"].to(dev); attn = enc["attention_mask"].to(dev)
            logits = model(ids, attention_mask=attn).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            tgt = ids[:, 1:]
            per = -logp.gather(2, tgt[:, :, None]).squeeze(-1)
            m = attn[:, 1:].bool()
            for i in range(per.shape[0]):
                v = per[i][m[i]]
                if v.numel() == 0:
                    rows.append([np.nan]*7)
                else:
                    rows.append([v.mean().item(), v.std().item(),
                                 float(torch.quantile(v, 0.9).item()), float(v.numel()),
                                 float(torch.quantile(v, 0.25).item()),
                                 float(torch.quantile(v, 0.75).item()), v.max().item()])
            if (s // bs) % 40 == 0:
                print(tag, "progress", s, flush=True)
    return np.asarray(rows, dtype=np.float64)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["15b", "7b", "lgb"])
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--bs15", type=int, default=8)
    ap.add_argument("--bs7", type=int, default=2)
    args = ap.parse_args()
    dev = torch.device(args.device)
    y, official, replay = load_texts()

    if args.stage in ("15b", "7b"):
        if args.stage == "15b":
            mp, tag, bs = MODELS / "Qwen2.5-1.5B-Instruct", "15b", args.bs15
        else:
            mp, tag, bs = MODELS / "Qwen2.5-7B-Instruct", "7b", args.bs7
        tok = AutoTokenizer.from_pretrained(str(mp), trust_remote_code=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(str(mp), trust_remote_code=True,
                                                     torch_dtype=torch.bfloat16,
                                                     attn_implementation="eager").to(dev)
        print(tag, "loaded", flush=True)
        Fo = nll_stats(model, tok, official, tag, dev, bs)
        np.save(RUNS / f"genfp_f{tag}_official.npy", Fo)
        Fr = nll_stats(model, tok, replay, tag, dev, bs)
        np.save(RUNS / f"genfp_f{tag}_replay.npy", Fr)
        print(tag, "DONE", Fo.shape, Fr.shape, flush=True)
        return

    # lgb stage
    y, official, replay = load_texts()
    F15o = np.load(RUNS / "genfp_f15b_official.npy"); F15r = np.load(RUNS / "genfp_f15b_replay.npy")
    F7o = np.load(RUNS / "genfp_f7b_official.npy");   F7r = np.load(RUNS / "genfp_f7b_replay.npy")
    def feats(f15, f7):
        d = f15[:, :6] - f7[:, :6]
        ratio = f15[:, 0] / np.maximum(f7[:, 0], 1e-6)
        return np.hstack([f15[:, :6], f7[:, :6], d, ratio[:, None], f15[:, 6:7], f7[:, 6:7]])
    Xo, Xr = feats(F15o, F7o), feats(F15r, F7r)
    print("feats", Xo.shape, flush=True)
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score
    folds = np.load(RUNS / "folds.npy")
    oof = np.zeros(len(y)); tests = []
    for f in range(5):
        tr = np.flatnonzero(folds != f); va = np.flatnonzero(folds == f)
        m = lgb.LGBMClassifier(objective="binary", n_estimators=500, learning_rate=0.03,
                               num_leaves=7, max_depth=3, min_child_samples=35,
                               reg_alpha=0.5, reg_lambda=2.0, random_state=20260909+f,
                               n_jobs=8, verbosity=-1)
        m.fit(Xo[tr], y[tr])
        oof[va] = m.predict_proba(Xo[va])[:, 1]
        tests.append(m.predict_proba(Xr)[:, 1])
    rep = np.mean(tests, axis=0)
    auc = roc_auc_score(y, oof)
    print("genfp OOF AUC", round(auc, 6), flush=True)
    np.save(RUNS / "genfp_oof.npy", oof)
    np.save(RUNS / "genfp_replay.npy", rep)
    print("GENFP LGB DONE", flush=True)

if __name__ == "__main__":
    main()
