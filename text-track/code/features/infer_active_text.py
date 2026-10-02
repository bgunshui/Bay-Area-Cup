#!/usr/bin/env python3
from __future__ import annotations

import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from transformers import AutoModelForSequenceClassification, AutoTokenizer


BASE = Path(os.environ.get("AIGC_BASE", "/data/湾区杯"))
OUT = BASE / "active_text"
OUT.mkdir(parents=True, exist_ok=True)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def run_classifier(model_dir: Path, texts: list[str], batch_size: int, tag: str) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir, local_files_only=True, torch_dtype=torch.float16
    )
    model.to(DEVICE).eval()
    values: list[float] = []
    started = time.time()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        encoded = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {key: value.to(DEVICE) for key, value in encoded.items()}
        with torch.inference_mode():
            probabilities = torch.softmax(model(**encoded).logits.float(), dim=1)[:, 1]
        values.extend(probabilities.cpu().numpy().tolist())
        if start == 0 or start + len(batch) == len(texts) or start % (batch_size * 100) == 0:
            print(tag, start + len(batch), "/", len(texts), "sec", round(time.time() - started, 1), flush=True)
    del model, tok
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()
    return np.asarray(values, dtype=np.float64)


def main() -> None:
    sample_path = Path(glob.glob(str(BASE / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0])
    test_path = BASE / "replay_data" / "文本赛道" / "text_test.csv"
    train = pd.read_csv(sample_path, encoding="utf-8-sig")
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    train_text = train["text"].fillna("").astype(str).tolist()
    test_text = test["text"].fillna("").astype(str).tolist()
    ids = test["id"].astype(str).tolist()
    print("rows", len(train), len(test), "device", DEVICE, flush=True)

    models = [
        ("anx", BASE / "active_models" / "anx", 32),
        ("paperpass", BASE / "active_models" / "paperpass", 16),
    ]
    for tag, model_dir, batch_size in models:
        train_score = run_classifier(model_dir, train_text, batch_size, tag + "_train")
        test_score = run_classifier(model_dir, test_text, batch_size, tag + "_test")
        np.save(OUT / f"{tag}_sample.npy", train_score)
        np.save(OUT / f"{tag}_replay.npy", test_score)
        pd.DataFrame({"id": ids, "score": rank01(test_score)}).to_csv(
            OUT / f"text_{tag}.csv", index=False
        )
        print(tag, "sample_auc_raw", float(np.corrcoef(train_score, train["label"].to_numpy())[0, 1]), flush=True)

    # Mokawa's ONNX export is handled separately because it is not a transformers classifier.
    meta = {
        "sample_path": str(sample_path),
        "test_path": str(test_path),
        "train_rows": len(train),
        "test_rows": len(test),
        "device": str(DEVICE),
        "models": [x[0] for x in models],
    }
    (OUT / "run_meta.json").write_text(json.dumps(meta, ensure_ascii=True, indent=2), encoding="utf-8")
    print("DONE", OUT, flush=True)


if __name__ == "__main__":
    main()
