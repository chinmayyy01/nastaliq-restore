"""Comparison statistics across restoration variants.

Loads evaluation JSONs produced by ``eval_restore_ocr.py`` (which now stores
per-sample predictions + damage fraction) and reports, for each variant:

    * corpus CER (the headline number)
    * bootstrap 95% confidence interval on that CER
    * pairwise paired-bootstrap p-values between variants

Because every variant is evaluated on the *same* validation samples, the
paired test is far more sensitive than comparing summary means.

Usage:
    python experiments/compare_stats.py \
        baseline=results/restore_ocr_eval_hard.json \
        ritl=results/restore_ocr_eval_hard_ocr.json \
        flow=results/restore_ocr_eval_flow.json \
        flow_ritl=results/restore_ocr_eval_flow_ocr.json
"""

import argparse
import json
import os
import sys

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from train_baseline import levenshtein  # noqa: E402


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dists(recs, key):
    d = np.array([levenshtein(r[key], r["ref"]) for r in recs], dtype=np.float64)
    L = np.array([max(1, len(r["ref"])) for r in recs], dtype=np.float64)
    return d, L


def corpus_cer(d, L, idx=None):
    if idx is None:
        return d.sum() / L.sum()
    return d[idx].sum() / L[idx].sum()


def bootstrap_ci(d, L, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    N = len(d)
    vals = np.array([corpus_cer(d, L, rng.integers(0, N, N)) for _ in range(n)])
    return corpus_cer(d, L), np.percentile(vals, 2.5), np.percentile(vals, 97.5)


def paired_pvalue(dA, LA, dB, LB, n=4000, seed=0):
    """Two-sided paired-bootstrap p-value for CER(A) != CER(B)."""
    rng = np.random.default_rng(seed)
    N = len(dA)
    diffs = np.array([corpus_cer(dA, LA, rng.integers(0, N, N)) -
                      corpus_cer(dB, LB, rng.integers(0, N, N)) for _ in range(n)])
    return 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()), diffs.mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("variants", nargs="+", help="name=path/to/eval.json")
    parser.add_argument("--key", default="restored", choices=["restored", "damaged", "clean"],
                        help="which prediction to score")
    parser.add_argument("--boot", type=int, default=2000)
    args = parser.parse_args()

    data = {}
    for spec in args.variants:
        name, path = spec.split("=", 1)
        recs = _load(path)["per_sample"]
        d, L = _dists(recs, args.key)
        data[name] = (d, L, recs)

    ref_names = sorted(data, key=lambda k: corpus_cer(*data[k][:2]))
    print(f"\nScoring key: '{args.key}'   (n={len(data[ref_names[0]][0])})\n")
    print(f"{'variant':<22} {'CER':>8} {'95% CI':>18}")
    for name in ref_names:
        cer, lo, hi = bootstrap_ci(data[name][0], data[name][1], args.boot)
        print(f"{name:<22} {cer:>8.4f}  [{lo:.4f}, {hi:.4f}]")

    print("\nPaired-bootstrap p-values (two-sided; <0.05 = significant):")
    names = ref_names
    print(" " * 22 + "".join(f"{n:>12}" for n in names))
    for a in names:
        row = f"{a:<22}"
        for b in names:
            if a == b:
                row += f"{'--':>12}"
            else:
                p, _ = paired_pvalue(data[a][0], data[a][1], data[b][0], data[b][1])
                row += f"{p:>12.3f}"
        print(row)


if __name__ == "__main__":
    main()
