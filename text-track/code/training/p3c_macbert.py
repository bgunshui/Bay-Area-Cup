#!/usr/bin/env python3
"""Phase 3c: MacBERT fine-tune with our folds (SEED 20260823) + replay inference.

Usage:
  python3 p3c_macbert.py                   # official + HC3/synth external
  python3 p3c_macbert.py --adversarial runs/rewrite_*.jsonl  # plus rewrite data
"""
import argparse
import glob
import json
import math
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import torch

# torch<2.6 bin-loading workaround (chinese-macbert-base ships pytorch_model.bin)
_orig_torch_load = torch.load


def _patched_load(*a, **k):
    k["weights_only"] = False
    return _orig_torch_load(*a, **k)


torch.load = _patched_load
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import transformers as _trf
import transformers.modeling_utils as _tmu
import transformers.utils.import_utils as _tiu
_tiu.check_torch_load_is_safe = lambda: None
_tmu.check_torch_load_is_safe = lambda: None
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, get_linear_schedule_with_warmup)

BASE = Path("/data/湾区杯")
OUT = BASE / "文本赛道" / "runs"
MODELS = BASE / "文本赛道" / "models"
DEV = torch.device("cuda:0")
SEED = 20260823

sys.path.insert(0, "/data/湾区杯/文本赛道/repro")
from wanqu_first_submission.folds import make_folds, text_near_duplicate_groups  # noqa: E402
from wanqu_first_submission.io_utils import seed_everything  # noqa: E402


def rank01(x):
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def select_chunks(token_ids, window, maximum):
    if not token_ids:
        return [[]]
    if len(token_ids) <= window:
        return [token_ids]
    if maximum == 1:
        return [token_ids[:window]]
    last = len(token_ids) - window
    starts = {0, last // 2, last}
    candidates = range(0, last + 1, max(128, window // 2))
    ranked = sorted(candidates, key=lambda s: len(set(token_ids[s:s + window])), reverse=True)
    for s in ranked:
        starts.add(s)
        if len(starts) >= min(maximum, math.ceil(len(token_ids) / window)):
            break
    chosen = sorted(starts)[:maximum]
    return [token_ids[s:s + window] for s in chosen]


class ChunkDataset(Dataset):
    def __init__(self, documents, indices, labels=None):
        self.rows = []
        for doc_idx in indices:
            lab = None if labels is None else int(labels[doc_idx])
            for chunk in documents[int(doc_idx)]:
                self.rows.append((chunk, int(doc_idx), lab))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        chunk, doc_idx, lab = self.rows[i]
        item = {"input_ids": [101, *chunk, 102], "doc_idx": doc_idx}
        if lab is not None:
            item["labels"] = lab
        return item


def make_collator(tokenizer):
    base = DataCollatorWithPadding(tokenizer, pad_to_multiple_of=8)

    def collate(features):
        doc_idx = torch.tensor([r.pop("doc_idx") for r in features], dtype=torch.long)
        batch = base(features)
        batch["doc_idx"] = doc_idx
        return batch

    return collate


def aggregate(doc_indices, scores, rows):
    grouped = [[] for _ in range(rows)]
    for idx, score in zip(doc_indices, scores, strict=True):
        grouped[int(idx)].append(float(score))
    result = np.zeros(rows, dtype=np.float64)
    for idx, values in enumerate(grouped):
        if not values:
            result[idx] = 0.5
        else:
            arr = np.asarray(values)
            result[idx] = 0.75 * arr.mean() + 0.25 * arr.max()
    return result


def predict_documents(model, tokenizer, documents, indices, device, batch_size=64):
    dataset = ChunkDataset(documents, indices)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                        pin_memory=True, collate_fn=make_collator(tokenizer))
    model.eval()
    dlist, slist = [], []
    with torch.inference_mode():
        for batch in loader:
            doc_idx = batch.pop("doc_idx").numpy()
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(**batch).logits
                score = torch.softmax(logits.float(), dim=-1)[:, 1]
            dlist.append(doc_idx)
            slist.append(score.cpu().numpy())
    doc_indices = np.concatenate(dlist)
    chunk_scores = np.concatenate(slist)
    remap = {int(orig): pos for pos, orig in enumerate(indices)}
    remapped = np.asarray([remap[int(i)] for i in doc_indices], dtype=np.int64)
    return aggregate(remapped, chunk_scores, len(indices))


def tokenize_docs(tokenizer, texts, max_chunks):
    docs = []
    for text in texts:
        ids = tokenizer.encode(str(text or ""), add_special_tokens=False)
        docs.append(select_chunks(ids, window=510, maximum=max_chunks))
    return docs


def load_external_rows(limit=10000):
    from wanqu_first_submission.paths import ROOT

    rows, counts = [], {0: 0, 1: 0}
    hc3 = ROOT / "external_data" / "hc3_chinese" / "all.jsonl"
    if hc3.exists():
        with hc3.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                for v in item.get("human_answers", []):
                    if v and counts[0] < limit:
                        rows.append({"text": v, "label": 0}); counts[0] += 1
                for v in item.get("chatgpt_answers", []):
                    if v and counts[1] < limit:
                        rows.append({"text": v, "label": 1}); counts[1] += 1
                if min(counts.values()) >= limit:
                    break
    qs = ROOT / "external_data" / "qwen_synthetic.jsonl"
    if qs.exists():
        with qs.open(encoding="utf-8") as f:
            for line in f:
                item = json.loads(line)
                if item.get("text"):
                    rows.append(item); counts[1] += 1
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame = frame.drop_duplicates("text").sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    return frame


def load_adversarial(paths):
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    rows.append({"text": item["text"], "label": item["label"]})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adversarial", nargs="*", default=[])
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--tag", default="macbert_c")
    args = ap.parse_args()
    seed_everything(SEED)

    sample_path = glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(BASE / "replay_data" / "文本赛道" / "text_test.csv", encoding="utf-8-sig")
    official_texts = train.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    test_texts = test.text.fillna("").astype(str).tolist()
    folds = np.load(OUT / "folds.npy")
    assert len(folds) == len(y)

    tokenizer = AutoTokenizer.from_pretrained(str(MODELS / "chinese-macbert-base"), local_files_only=True)
    docs_official = tokenize_docs(tokenizer, official_texts, 4)
    docs_test = tokenize_docs(tokenizer, test_texts, 8)

    external = load_external_rows()
    adv = load_adversarial(args.adversarial) if args.adversarial else None
    tag = args.tag + ("_adv" if adv is not None else "")

    def new_model():
        return AutoModelForSequenceClassification.from_pretrained(
            str(MODELS / "chinese-macbert-base"), num_labels=2, local_files_only=True,
            ignore_mismatched_sizes=True).to(DEV)

    # external pretrain (official + ext + adv), 1 epoch
    ext_texts = external.text.fillna("").astype(str).tolist()
    if adv is not None:
        ext_texts = ext_texts + adv.text.fillna("").astype(str).tolist()
        external = pd.concat([external, adv], ignore_index=True)
    ext_labels = external.label.astype(int).to_numpy()
    ext_docs = tokenize_docs(tokenizer, ext_texts, 1)
    ext_ds = ChunkDataset(ext_docs, np.arange(len(ext_texts)), ext_labels)
    ext_loader = DataLoader(ext_ds, batch_size=48, shuffle=True, num_workers=4,
                            pin_memory=True, collate_fn=make_collator(tokenizer))
    model = new_model()
    opt = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, max(1, len(ext_loader) // 10), len(ext_loader))
    model.train()
    for batch in ext_loader:
        batch.pop("doc_idx", None)
        batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    ext_dir = OUT / "macbert_ext_pretrained"
    model.save_pretrained(str(ext_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(ext_dir))
    del model
    torch.cuda.empty_cache()
    print("ext pretrain done rows", len(ext_texts), flush=True)

    oof = np.zeros(len(official_texts), dtype=np.float64)
    replay_folds = []
    t0 = time.time()
    for f in range(5):
        tr = np.flatnonzero(folds != f)
        va = np.flatnonzero(folds == f)
        ds = ChunkDataset(docs_official, tr, y)
        loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=4, pin_memory=True,
                            collate_fn=make_collator(tokenizer))
        model = AutoModelForSequenceClassification.from_pretrained(
            str(ext_dir), num_labels=2, local_files_only=True, ignore_mismatched_sizes=True).to(DEV)
        opt = AdamW(model.parameters(), lr=1.5e-5, weight_decay=0.02)
        sched = get_linear_schedule_with_warmup(opt, max(1, len(loader) * args.epochs // 10),
                                                len(loader) * args.epochs)
        best_auc, best_state = -1.0, None
        for epoch in range(args.epochs):
            model.train()
            losses = []
            for batch in loader:
                batch.pop("doc_idx", None)
                batch = {k: v.to(DEV, non_blocking=True) for k, v in batch.items()}
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(**batch).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            vscore = predict_documents(model, tokenizer, docs_official, va, DEV)
            auc = float(roc_auc_score(y[va], vscore))
            print(f"  fold {f} ep {epoch} loss {np.mean(losses):.4f} auc {auc:.5f} "
                  f"t {time.time()-t0:.0f}s", flush=True)
            if auc > best_auc:
                best_auc = auc
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        oof[va] = predict_documents(model, tokenizer, docs_official, va, DEV)
        replay_folds.append(predict_documents(model, tokenizer, docs_test,
                                              np.arange(len(test_texts)), DEV))
        del model
        torch.cuda.empty_cache()

    replay = np.mean(replay_folds, axis=0)
    np.save(OUT / f"{tag}_oof.npy", oof)
    np.save(OUT / f"{tag}_replay.npy", rank01(replay))
    print(tag, "OOF AUC", round(float(roc_auc_score(y, oof)), 6), flush=True)
    pd.DataFrame({"id": test.id.astype(str), "score": rank01(replay)}).to_csv(
        OUT / f"{tag}_candidate.csv", index=False)
    with open(OUT / f"{tag}_report.json", "w", encoding="utf-8") as fh:
        json.dump({"oof_auc": float(roc_auc_score(y, oof)),
                   "adv_rows": int(len(adv)) if adv is not None else 0,
                   "ext_rows": int(len(external))}, fh, ensure_ascii=False, indent=2)
    print("P3c DONE", flush=True)


if __name__ == "__main__":
    main()
