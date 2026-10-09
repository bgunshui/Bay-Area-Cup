"""Recompute image-component Spearman correlations from a fixed Git commit.

Standard library only. All CSV bytes come from git show; the repository is
neither checked out nor modified. The only output is the requested JSON record.

Usage:
    python defense/图表/重算分量相关性.py
    python defense/图表/重算分量相关性.py --repo PATH --out defense/图表/RECORD.json

Relative output paths are resolved from --repo; recorded paths are repository-relative.
"""

from argparse import ArgumentParser
from collections import Counter
import csv
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import platform
import subprocess
import sys


COMMIT = "c11fa520f2f73d705e83cb3f40a6a214f8cc7f31"
NAMES = [
    "img_cliph_rerun",
    "img_cliph_s20261101",
    "img_cliph_s20261102",
    "img_cliph_s20261103",
    "img_gpp_siglip2",
    "img_bigg_s384_retrain",
    "img_bigg_rerun2",
    "img_bigg_rerun3",
]


def git(repo, *args):
    result = subprocess.run(
        ["git", "--no-replace-objects", *args], cwd=repo,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
    )
    return result.stdout


def twice_average_rank(scores):
    """Return 2 * average one-based rank, exactly as integers.

    A tied group occupying sorted zero-based positions i..j-1 has average
    one-based rank (i+j+1)/2. Decimal parsing keeps CSV ties exact.
    """
    order = sorted(range(len(scores)), key=scores.__getitem__)
    ranks = [0] * len(scores)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and scores[order[j]] == scores[order[i]]:
            j += 1
        for k in range(i, j):
            ranks[order[k]] = i + j + 1
        i = j
    return ranks


def centered_twice_ranks(ranks):
    center = len(ranks) + 1
    assert sum(ranks) == len(ranks) * center
    return [rank - center for rank in ranks]


def spearman(left, right):
    """Pearson correlation of average ranks, with exact integer sums."""
    assert len(left) == len(right)
    covariance = sum(a * b for a, b in zip(left, right))
    variance_left = sum(a * a for a in left)
    variance_right = sum(b * b for b in right)
    if not variance_left or not variance_right:
        raise ValueError("A constant component has undefined Spearman rho")
    return covariance / math.sqrt(variance_left * variance_right)


def recompute(repo):
    resolved = git(repo, "rev-parse", f"{COMMIT}^{{commit}}").decode().strip()
    assert resolved == COMMIT
    sources, centered = [], {}
    reference_ids = None
    for name in NAMES:
        path = f"image-track/submission/{name}.csv"
        raw = git(repo, "show", f"{COMMIT}:{path}")
        blob_oid = git(repo, "rev-parse", f"{COMMIT}:{path}").decode().strip()
        recomputed_blob_oid = hashlib.sha1(
            b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw
        ).hexdigest()
        assert blob_oid == recomputed_blob_oid
        rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
        ids = [row["id"] for row in rows]
        scores = [Decimal(row["score"]) for row in rows]
        assert len(rows) == 20000 and len(set(ids)) == 20000
        assert all(score.is_finite() for score in scores)
        if reference_ids is None:
            reference_ids = ids
        assert ids == reference_ids, f"ID/order mismatch: {path}"
        counts = Counter(scores)
        centered[name] = centered_twice_ranks(twice_average_rank(scores))
        last_change = git(
            repo, "log", "-1", "--format=%H", COMMIT, "--", path
        ).decode().strip()
        sources.append({
            "component": name,
            "repository_relative_path": path,
            "commit_sha": COMMIT,
            "git_object_spec": f"{COMMIT}:{path}",
            "git_blob_oid_sha1": blob_oid,
            "git_blob_oid_recomputed_from_raw_bytes": recomputed_blob_oid,
            "raw_file_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_file_bytes": len(raw),
            "last_path_change_commit_sha": last_change,
            "row_count": len(rows),
            "unique_id_count": len(set(ids)),
            "ids_equal_reference_in_same_order": ids == reference_ids,
            "id_sequence_sha256": hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest(),
            "unique_score_count": len(counts),
            "tied_score_group_count": sum(1 for count in counts.values() if count > 1),
        })

    def rho(left, right):
        return spearman(centered[left], centered[right])

    family_pairs = []
    for i in range(4):
        for j in range(i + 1, 4):
            family_pairs.append({"left": NAMES[i], "right": NAMES[j],
                                 "spearman_rho": rho(NAMES[i], NAMES[j])})
    family_mean = math.fsum(pair["spearman_rho"] for pair in family_pairs) / 6
    main_siglip = rho(NAMES[0], "img_gpp_siglip2")
    main_bigg384 = rho(NAMES[0], "img_bigg_s384_retrain")
    bigg224 = rho("img_bigg_rerun2", "img_bigg_rerun3")
    # Check the precise independently computed values that informed slide 6.
    expected = [0.9600891535596247, 0.7913757727208824,
                0.7091440666347133, 0.8356879344005202]
    actual = [family_mean, main_siglip, main_bigg384, bigg224]
    assert all(abs(a - e) < 1e-12 for a, e in zip(actual, expected))
    display = [format(value, ".4f") for value in actual[:3]]
    assert display == ["0.9601", "0.7914", "0.7091"]
    return {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit_sha": COMMIT,
        "repository_root": ".",
        "source_access": "git show fixed_commit:path; no checkout/fetch or repository writes",
        "runtime": {"python": platform.python_version(), "platform": platform.platform(),
                    "dependencies": "Python standard library and Git only"},
        "id_alignment": {
            "reference_component": NAMES[0],
            "reference_csv_path": sources[0]["repository_relative_path"],
            "row_count_per_component": 20000,
            "unique_id_count_per_component": 20000,
            "all_eight_components_same_ids_in_same_order": True,
            "id_sequence_hash_encoding": "UTF-8 IDs joined by LF, without trailing LF",
            "id_sequence_sha256": sources[0]["id_sequence_sha256"],
            "first_five_ids": reference_ids[:5],
            "last_five_ids": reference_ids[-5:],
        },
        "method": {
            "score_parsing": "Decimal of each original CSV score string; no rescaling or filtering",
            "tie_handling": "average one-based rank; sort ascending by score",
            "rank_algorithm": "group at zero-based sorted positions i..j-1 gets twice-rank i+j+1",
            "correlation": "Pearson correlation of the two average-rank vectors (Spearman rho)",
            "numerical_algorithm": "Subtract N+1 from twice-ranks; sum covariance and squared deviations as exact Python integers; final division/sqrt uses binary64",
            "family_mean": "Unweighted arithmetic mean of six distinct CLIP-H pairwise correlations; math.fsum",
            "sample_scope": "All 20000 target IDs for each frozen component output",
            "slide_display_precision": "Four decimal places",
        },
        "inputs": sources,
        "results": {
            "cliph_four_component_six_pair_mean": {
                "spearman_rho": family_mean, "display_value": display[0],
                "pair_count": 6, "pairwise_results": family_pairs,
            },
            "cliph_main_vs_siglip2": {
                "left": NAMES[0], "right": "img_gpp_siglip2",
                "spearman_rho": main_siglip, "display_value": display[1],
            },
            "cliph_main_vs_bigg384": {
                "left": NAMES[0], "right": "img_bigg_s384_retrain",
                "spearman_rho": main_bigg384, "display_value": display[2],
            },
            "bigg224_two_seed_pair": {
                "left": "img_bigg_rerun2", "right": "img_bigg_rerun3",
                "spearman_rho": bigg224, "display_value": format(bigg224, ".4f"),
                "shown_on_slide_6": False,
            },
        },
        "limitations": [
            "These are component-versus-component correlations, not service-versus-submission reproduction correlations.",
            "No class labels enter this computation: correlation is not AUC or an attack/generalization measurement.",
            "Low correlation indicates output-ranking diversity and does not by itself establish a fusion gain.",
        ],
        "recompute_script": "defense/图表/重算分量相关性.py",
    }


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path,
                        default=Path(__file__).resolve().parents[2],
                        help="Git repository root; defaults to the repository containing this script")
    parser.add_argument("--out", type=Path,
                        default=Path("defense/图表/图像分量相关性数据.json"),
                        help="Output JSON path, relative to --repo by default")
    args = parser.parse_args()
    record = recompute(args.repo)
    output = args.repo.resolve() / args.out
    output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        print(output.resolve().relative_to(args.repo.resolve()).as_posix())
    except ValueError:
        print(output.name)
    for name, result in record["results"].items():
        print(f"{name}: {result['spearman_rho']:.17g} -> {result['display_value']}")


if __name__ == "__main__":
    main()
