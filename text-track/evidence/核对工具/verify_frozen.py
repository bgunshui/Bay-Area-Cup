"""Independently check frozen text predictions using only the standard library.

The upstream scripts, pickle files, service, and models are not executed.
Raw competition IDs/text are not written into this tool's results.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import struct
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone, timedelta
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
TRACK = REPO / "text-track"
OUTPUT = TRACK / "evidence" / "验证输出" / "代码事实核对_2026-10-02.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_npy(path: Path) -> tuple[list, dict]:
    with path.open("rb") as stream:
        if stream.read(6) != b"\x93NUMPY":
            raise ValueError(f"Invalid NPY magic: {path.name}")
        version = tuple(stream.read(2))
        if version == (1, 0):
            size = struct.unpack("<H", stream.read(2))[0]
        elif version in ((2, 0), (3, 0)):
            size = struct.unpack("<I", stream.read(4))[0]
        else:
            raise ValueError(f"Unsupported NPY version {version}")
        header = ast.literal_eval(stream.read(size).decode("utf-8" if version == (3, 0) else "latin1"))
        shape = header["shape"]
        if len(shape) != 1 or header["fortran_order"]:
            raise ValueError(f"Expected one-dimensional C-order array: {path.name}")
        formats = {"<f8": "<d", "<f4": "<f", "<i8": "<q", "<i4": "<i", "|u1": "B"}
        if header["descr"] not in formats:
            raise ValueError(f"Unsupported or object dtype: {header['descr']}")
        fmt = formats[header["descr"]]
        payload = stream.read()
        if len(payload) != shape[0] * struct.calcsize(fmt):
            raise ValueError(f"Truncated NPY payload: {path.name}")
        data = [item[0] for item in struct.iter_unpack(fmt, payload)]
        if not all(math.isfinite(value) for value in data):
            raise ValueError(f"Non-finite frozen values: {path.name}")
        return data, {"dtype": header["descr"], "shape": list(shape), "bytes": path.stat().st_size,
                      "sha256": digest(path)}


def ranks(values: list) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    output = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2
        for index in order[start:end]:
            output[index] = rank
        start = end
    return output


def rank01(values: list) -> list[float]:
    return [rank / (len(values) + 1.0) for rank in ranks(values)]


def sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


def comparison(left: list, right: list) -> dict:
    if len(left) != len(right):
        raise ValueError("Comparison lengths differ")
    rl, rr = ranks(left), ranks(right)
    ml, mr = sum(rl) / len(rl), sum(rr) / len(rr)
    covariance = sum((a - ml) * (b - mr) for a, b in zip(rl, rr))
    variance = math.sqrt(sum((a - ml) ** 2 for a in rl) * sum((b - mr) ** 2 for b in rr))
    return {"rows": len(left), "spearman": covariance / variance,
            "all_ranks_equal": rl == rr,
            "rows_with_rank_difference": sum(a != b for a, b in zip(rl, rr)),
            "max_absolute_score_difference": max(abs(a - b) for a, b in zip(left, right)),
            "mean_absolute_score_difference": sum(abs(a - b) for a, b in zip(left, right)) / len(left),
            "left_unique_scores": len(set(left)), "right_unique_scores": len(set(right))}


def inversion_count(reference: list, other: list) -> dict:
    # Strict discordance excludes ties in either vector. Equal-reference groups
    # are queried before any member is inserted into the Fenwick tree.
    if len(reference) != len(other):
        raise ValueError("Pair comparison lengths differ")
    order = sorted(range(len(reference)), key=reference.__getitem__)
    coordinates = {value: i + 1 for i, value in enumerate(sorted(set(other)))}
    tree = [0] * (len(coordinates) + 1)
    total, seen = 0, 0
    def prefix(pos):
        count = 0
        while pos > 0:
            count += tree[pos]
            pos -= pos & -pos
        return count
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and reference[order[end]] == reference[order[start]]:
            end += 1
        for index in order[start:end]:
            total += seen - prefix(coordinates[other[index]])
        for index in order[start:end]:
            pos = coordinates[other[index]]
            while pos < len(tree):
                tree[pos] += 1
                pos += pos & -pos
        seen += end - start
        start = end
    def tied_pairs(values):
        return sum(count * (count - 1) // 2 for count in Counter(values).values())
    both_tied = tied_pairs(list(zip(reference, other)))
    reference_only_tied = tied_pairs(reference) - both_tied
    other_only_tied = tied_pairs(other) - both_tied
    pair_total = len(reference) * (len(reference) - 1) // 2
    changed = total + reference_only_tied + other_only_tied
    weighted_changes = total + 0.5 * (reference_only_tied + other_only_tied)
    return {"status": "VERIFIED_WITH_TIES", "strict_discordant_pairs": total,
            "reference_only_tied_pairs": reference_only_tied,
            "other_only_tied_pairs": other_only_tied, "both_tied_pairs": both_tied,
            "changed_order_or_tie_pairs": changed, "all_pairs": pair_total,
            "fraction_of_all_pairs": changed / pair_total if pair_total else 0.0,
            "weighted_changes_for_conditional_auc_bound": weighted_changes,
            "auc_bound_if_exactly_10000_positive_and_10000_negative": weighted_changes / 100000000,
            "auc_bound_condition": "Test labels/class counts are absent. Bound assumes 10000 positives/10000 negatives and counts all affected pairs as cross-class; it is not an observed AUC change."}


def check_pair_counter() -> None:
    # Independent exhaustive enumeration on a small example with ties.
    a, b = [0, 0, 1, 1, 2, 3, 3, 4], [3, 1, 1, 0, 2, 2, 4, 4]
    strict = tie_a = tie_b = tie_both = 0
    for i in range(len(a)):
        for j in range(i):
            da, db = a[i] - a[j], b[i] - b[j]
            strict += da * db < 0
            tie_a += da == 0 and db != 0
            tie_b += da != 0 and db == 0
            tie_both += da == 0 and db == 0
    actual = inversion_count(a, b)
    assert (actual["strict_discordant_pairs"], actual["reference_only_tied_pairs"],
            actual["other_only_tied_pairs"], actual["both_tied_pairs"]) == (strict, tie_a, tie_b, tie_both)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    check_pair_counter()
    arrays, inventory = {}, {}
    for path in sorted((TRACK / "evidence" / "离线预测").glob("*.npy")):
        values, meta = read_npy(path)
        arrays[path.stem] = values
        inventory[path.name] = meta
    submission = TRACK / "submission" / "submit_p4_90_7b2_10.csv"
    with submission.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ["id", "score"]:
            raise ValueError("Unexpected submission columns")
        rows = list(reader)
    submitted = [float(row["score"]) for row in rows]
    if len(rows) != 20000 or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Unexpected test row count or duplicate IDs")
    if not all(math.isfinite(value) for value in submitted):
        raise ValueError("Non-finite submission score")

    p4 = rank01(arrays["14b_p4_replay_rank01"])
    r3a = rank01(arrays["7b_r3a_replay_raw_mean"])
    r3b = rank01(arrays["7b_r3b_replay_raw_mean"])
    documented = rank01([0.90 * a + 0.10 * ((b + c) / 2.0) for a, b, c in zip(p4, r3a, r3b)])
    online = [0.90 * sigmoid(a) + 0.10 * sigmoid((b + c) / 2.0)
              for a, b, c in zip(arrays["14b_p4_replay_raw_mean"], arrays["7b_r3a_replay_raw_mean"], arrays["7b_r3b_replay_raw_mean"])]
    frozen_offline = arrays["fusion_offline_submitted"]
    frozen_online = arrays["fusion_online_image"]
    results = {
        "audit_time_asia_shanghai": datetime.now(timezone(timedelta(hours=8))).isoformat(timespec="seconds"),
        "repository_url": "https://github.com/bgunshui/Bay-Area-Cup",
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "method": "Independent Python-standard-library NPY parsing, average ranks, fusion and CSV comparison; no upstream code or pickle/model executed.",
        "array_inventory": inventory,
        "submission": {"rows": len(rows), "unique_ids": len({row["id"] for row in rows}),
                       "sha256": digest(submission), "score_min": min(submitted), "score_max": max(submitted)},
        "readme_quick_formula_vs_csv": comparison(documented, submitted),
        "frozen_offline_array_vs_csv": comparison(frozen_offline, submitted),
        "computed_online_formula_vs_frozen_online": comparison(online, frozen_online),
        "frozen_online_vs_submission": comparison(frozen_online, submitted),
        "online_vs_submission_pair_order_changes": inversion_count(submitted, frozen_online),
        "frozen_offline_vs_online_pair_order_changes": inversion_count(frozen_offline, frozen_online),
        "official_example_labels": {"rows": len(arrays["官方样例_labels"]), "counts": dict(Counter(arrays["官方样例_labels"]))},
        "official_example_folds": {"rows": len(arrays["官方样例_folds"]), "counts": dict(Counter(arrays["官方样例_folds"]))},
        "limitations": ["No competition test labels: official AUC cannot be recalculated.",
                        "Prediction row-to-text alignment cannot be independently checked without competition source texts.",
                        "No model weights, new Docker archive, GPU tests or training executed.",
                        "Using frozen final output as input validates its consistency with CSV, not the preceding training/fusion pipeline."]
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: results[key] for key in ["commit", "readme_quick_formula_vs_csv", "frozen_offline_array_vs_csv",
                "computed_online_formula_vs_frozen_online", "frozen_online_vs_submission", "online_vs_submission_pair_order_changes"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
