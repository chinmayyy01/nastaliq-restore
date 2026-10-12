"""Aggregate CER across seeds from a sweep of eval JSONs (Modal volume output).

Each eval JSON (from ``eval_restore_ocr.py``) holds per-sample predictions, so
we recompute corpus CER and report mean +/- std across seeds per configuration.

Usage:
    python experiments/aggregate_seeds.py --dir results/stats/sweep
"""

import argparse
import os
import statistics as st
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

from train_baseline import levenshtein  # noqa: E402
import json  # noqa: E402


def cer(path, key="restored"):
    with open(path, encoding="utf-8") as f:
        recs = json.load(f)["per_sample"]
    d = sum(levenshtein(r[key], r["ref"]) for r in recs)
    L = sum(max(1, len(r["ref"])) for r in recs)
    return d / L


GROUPS = {
    "baseline U-Net": ["sw_baseline_s%d_eval.json" % s for s in (0, 1, 2)],
    "U-Net + RITL (l=0.1)": ["sw_ritl_s%d_eval.json" % s for s in (0, 1, 2)],
    "flow": ["sw_flow_s%d_eval.json" % s for s in (0, 1, 2)],
    "flow + RITL (l=0.1)": ["sw_flowritl_s%d_eval.json" % s for s in (0, 1, 2)],
}
LAMBDA_SWEEP = [("0.03", "sw_ritl_l0p03_s0_eval.json"), ("0.1", "sw_ritl_s0_eval.json"),
                ("0.3", "sw_ritl_l0p3_s0_eval.json"), ("1.0", "sw_ritl_l1p0_s0_eval.json")]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=os.path.join(_REPO_ROOT, "results/stats/sweep"))
    parser.add_argument("--key", default="restored")
    args = parser.parse_args()

    print(f"{'config':<24} {'mean CER':>9} {'std':>8}   per-seed")
    for name, files in GROUPS.items():
        vals = [cer(os.path.join(args.dir, f), args.key) for f in files]
        seeds = ", ".join(f"{v:.4f}" for v in vals)
        print(f"{name:<24} {st.mean(vals):>9.4f} {st.pstdev(vals):>8.4f}   {seeds}")

    print("\nlambda_ocr sweep (U-Net + RITL, seed 0):")
    for lam, f in LAMBDA_SWEEP:
        print(f"  lambda={lam:<5} CER {cer(os.path.join(args.dir, f), args.key):.4f}")

    ref = os.path.join(args.dir, GROUPS["baseline U-Net"][0])
    print(f"\nreference: damaged {cer(ref, 'damaged'):.4f} | clean {cer(ref, 'clean'):.4f}")


if __name__ == "__main__":
    main()
