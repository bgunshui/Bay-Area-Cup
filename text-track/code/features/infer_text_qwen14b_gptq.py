from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from auto_gptq import AutoGPTQForCausalLM
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer


def rank01(x: np.ndarray) -> np.ndarray:
    return rankdata(np.asarray(x, dtype=np.float64), method="average") / (len(x) + 1.0)


def extract(items: list[str], tokenizer, backbone, device: torch.device, cache: Path, batch_size: int, max_length: int) -> dict[str, np.ndarray]:
    names = ("mean", "first", "last")
    part_dir = Path(str(cache) + ".parts")
    part_dir.mkdir(parents=True, exist_ok=True)
    arrays = {n: [] for n in names}
    started = time.time()
    for pstart in range(0, len(items), 250):
        pend = min(pstart + 250, len(items))
        part_path = part_dir / f"part_{pstart:07d}_{pend:07d}.npz"
        if part_path.exists():
            loaded = np.load(part_path)
            if all(loaded[n].shape[0] == pend - pstart for n in names):
                for n in names:
                    arrays[n].append(loaded[n].astype(np.float32))
                print(f"reuse {pstart}:{pend}", flush=True)
                continue
        chunks = {n: [] for n in names}
        for start in range(pstart, pend, batch_size):
            batch = items[start:min(start + batch_size, pend)]
            tokenized = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            tokenized = {k: v.to(device) for k, v in tokenized.items()}
            with torch.inference_mode():
                outputs = backbone(**tokenized, output_hidden_states=True, use_cache=False, return_dict=True)
                hidden = outputs.hidden_states[-1]
                mask = tokenized["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                mean = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
                first = hidden[:, 0, :]
                last_idx = tokenized["attention_mask"].sum(1).clamp_min(1) - 1
                last = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_idx, :]
            chunks["mean"].append(torch.nn.functional.normalize(mean.float(), dim=1).cpu().numpy().astype(np.float32))
            chunks["first"].append(torch.nn.functional.normalize(first.float(), dim=1).cpu().numpy().astype(np.float32))
            chunks["last"].append(torch.nn.functional.normalize(last.float(), dim=1).cpu().numpy().astype(np.float32))
        for n in names:
            arrays[n].append(np.concatenate(chunks[n], axis=0))
        np.savez(part_path, **{n: arrays[n][-1] for n in names})
        print(f"saved {pstart}:{pend} elapsed={time.time() - started:.1f}s", flush=True)
    result = {n: np.concatenate(arrays[n], axis=0) for n in names}
    np.savez(cache, **result)
    print("saved", cache, {n: result[n].shape for n in names}, flush=True)
    return result


def load_or_extract(items, tokenizer, backbone, device, cache, batch_size, max_length):
    cache = Path(cache)
    if cache.exists():
        z = np.load(cache)
        if all(z[n].shape[0] == len(items) for n in ("mean", "first", "last")):
            print("reuse", cache, flush=True)
            return {n: np.asarray(z[n], dtype=np.float32) for n in ("mean", "first", "last")}
    return extract(items, tokenizer, backbone, device, cache, batch_size, max_length)


def oof_select(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=20260824)
    best = (0.0, -1.0)
    for c in (0.003, 0.01, 0.03, 0.1, 0.3, 1.0):
        oof = np.zeros(len(y), dtype=np.float64)
        for tr, va in cv.split(x, y):
            clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=c, max_iter=1000, class_weight="balanced", solver="liblinear"))
            clf.fit(x[tr], y[tr])
            oof[va] = clf.decision_function(x[va])
        auc = roc_auc_score(y, oof)
        print("oof", c, f"{auc:.6f}", flush=True)
        if auc > best[1]:
            best = (c, auc)
    return best


def aligned(path: Path, test: pd.DataFrame) -> np.ndarray | None:
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if list(frame.columns) != ["id", "score"] or frame["id"].astype(str).tolist() != test["id"].astype(str).tolist():
        print("skip misaligned", path, flush=True)
        return None
    values = frame.score.to_numpy(dtype=np.float64)
    return rank01(values) if np.isfinite(values).all() else None


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--base", default="/data/湾区杯")
    p.add_argument("--out", default="/data/湾区杯/text_qwen14b_gptq")
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--max-length", type=int, default=256)
    args = p.parse_args()
    base, out = Path(args.base), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sample_path = glob.glob(str(base / "initial_text" / "**" / "text_sample_data.csv"), recursive=True)[0]
    test_path = base / "replay_data" / "文本赛道" / "text_test.csv"
    train, test = pd.read_csv(sample_path, encoding="utf-8-sig"), pd.read_csv(test_path, encoding="utf-8-sig")
    train_text = train.text.fillna("").astype(str).tolist()
    test_text = test.text.fillna("").astype(str).tolist()
    y = train.label.astype(int).to_numpy()
    print("rows", len(train), len(test), flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoGPTQForCausalLM.from_quantized(args.model, device="cuda:0", use_safetensors=True, use_triton=False,
                                               inject_fused_attention=False, inject_fused_mlp=False,
                                               disable_exllama=True, trust_remote_code=True, low_cpu_mem_usage=True)
    model.eval()
    # Only the transformer backbone is needed for embeddings; avoid lm_head memory.
    wrapper = model.model
    if hasattr(wrapper, "lm_head"):
        wrapper.lm_head = None
    backbone = wrapper.model
    backbone.eval()
    device = next(backbone.parameters()).device
    print("device", device, flush=True)
    train_z = load_or_extract(train_text, tokenizer, backbone, device, out / "train_features.npz", args.batch_size, args.max_length)
    test_z = load_or_extract(test_text, tokenizer, backbone, device, out / "test_features.npz", args.batch_size, args.max_length)
    del model
    torch.cuda.empty_cache()

    heads: dict[str, np.ndarray] = {}
    for name in ("mean", "first", "last"):
        best_c, best_auc = oof_select(train_z[name], y)
        clf = make_pipeline(StandardScaler(with_mean=False), LogisticRegression(C=best_c, max_iter=1500, class_weight="balanced", solver="liblinear"))
        clf.fit(train_z[name], y)
        heads[name] = rank01(clf.decision_function(test_z[name]))
        pd.DataFrame({"id": test.id.astype(str), "score": heads[name]}).to_csv(out / f"text_qwen14b_gptq_{name}.csv", index=False)
        print("head", name, "C", best_c, "oof", best_auc, flush=True)
    q = np.mean(list(heads.values()), axis=0)
    candidates = {"text_qwen14b_gptq_mean.csv": heads["mean"], "text_qwen14b_gptq_first.csv": heads["first"],
                  "text_qwen14b_gptq_last.csv": heads["last"], "text_qwen14b_gptq_equal.csv": q}
    for path_name in ("text_mokawa.csv", "text_submit.csv", "text_ensemble_a72_p03_m15_t10.csv"):
        other = aligned(base / path_name, test)
        if other is not None:
            for w in (0.8, 0.9, 0.95):
                candidates[f"text_qwen14b_gptq_equal_{Path(path_name).stem}_{int(w * 100)}.csv"] = w * q + (1 - w) * other
    for name, score in candidates.items():
        pd.DataFrame({"id": test.id.astype(str), "score": score}).to_csv(base / name, index=False)
    with open(out / "run_meta.json", "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "train_rows": len(train), "test_rows": len(test), "max_length": args.max_length,
                   "device": str(device), "test_path": str(test_path)}, f, ensure_ascii=True, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
