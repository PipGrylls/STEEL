#!/usr/bin/env python3
"""Compare two ``RunParam_*`` output trees file by file.

Used for the three-way validation (py-as-is / py-corrected / rs-steel).
Loads every ``.npy`` present in both trees, reports shape agreement and,
where shapes match, the maximum and median absolute and fractional
deviation.

Two comparison modes matter and they are *not* interchangeable:

* **deterministic** — abundance-matching and star-formation scatter
  disabled on both sides. The implementations then run the same
  arithmetic on the same grid and agreement should be at floating-point
  level. This is the strong numerical-fidelity claim.
* **stochastic** — scatter on. py-steel draws from NumPy's Mersenne
  Twister (plus GSL's taus generator inside ``Functions_c``), rs-steel
  from ``rand``'s ChaCha; the two can never agree element-wise, only in
  ensemble statistics. Fractional deviations of order the per-bin
  Monte-Carlo noise are expected and are *not* a port defect.

This script does not know which mode produced its inputs — it reports
numbers, and the caller states the mode. Mixing them into a single
"agreement" figure would conflate numerical fidelity with Monte-Carlo
noise, which is exactly the distinction the validation exists to make.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_tree(root: Path) -> dict[str, np.ndarray]:
    if not root.is_dir():
        raise SystemExit(f"not a directory: {root}")
    out = {}
    for path in sorted(root.glob("*.npy")):
        try:
            out[path.name] = np.load(path, allow_pickle=False)
        except ValueError as exc:  # object arrays, ragged saves
            print(f"  ! skipping {path.name}: {exc}", file=sys.stderr)
    return out


def deviations(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    """Absolute and fractional deviation summaries.

    NaNs are compared as values: a cell that is NaN in both is agreement
    (``Total_StarFormation`` is NaN wherever no satellite contributed,
    and that pattern is itself a result worth matching), a cell NaN in
    only one is a mismatch and is counted separately rather than being
    quietly dropped by ``nanmax``.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    nan_a, nan_b = np.isnan(a), np.isnan(b)
    both_nan = nan_a & nan_b
    only_one_nan = int(np.count_nonzero(nan_a ^ nan_b))

    finite = ~(nan_a | nan_b)
    if not finite.any():
        return {
            "n": int(a.size),
            "n_nan_mismatch": only_one_nan,
            "n_both_nan": int(np.count_nonzero(both_nan)),
            "max_abs": 0.0,
            "median_abs": 0.0,
            "max_frac": 0.0,
            "median_frac": 0.0,
            "p90_frac": 0.0,
        }

    av, bv = a[finite], b[finite]
    abs_dev = np.abs(av - bv)
    # Fractional deviation relative to the larger magnitude, so a bin
    # that is zero on one side and non-zero on the other reads as 1.0
    # rather than inf.
    scale = np.maximum(np.abs(av), np.abs(bv))
    frac_dev = np.where(scale > 0, abs_dev / np.where(scale > 0, scale, 1.0), 0.0)
    return {
        "n": int(a.size),
        "n_nan_mismatch": only_one_nan,
        "n_both_nan": int(np.count_nonzero(both_nan)),
        "max_abs": float(abs_dev.max()),
        "median_abs": float(np.median(abs_dev)),
        "max_frac": float(frac_dev.max()),
        "median_frac": float(np.median(frac_dev)),
        # The maximum is dominated by the extreme high-mass tail, where
        # a bin holds at most one satellite in either run and a
        # discreteness difference reads as a 100% deviation. p90 says
        # what the bulk of the distribution does.
        "p90_frac": float(np.percentile(frac_dev, 90)),
    }


def compare(left_root: Path, right_root: Path) -> dict:
    left, right = load_tree(left_root), load_tree(right_root)
    names = sorted(set(left) | set(right))
    report = {
        "left": str(left_root),
        "right": str(right_root),
        "only_in_left": sorted(set(left) - set(right)),
        "only_in_right": sorted(set(right) - set(left)),
        "shape_mismatch": {},
        "compared": {},
    }
    for name in names:
        if name not in left or name not in right:
            continue
        a, b = left[name], right[name]
        if a.shape != b.shape:
            report["shape_mismatch"][name] = {"left": list(a.shape), "right": list(b.shape)}
            continue
        report["compared"][name] = deviations(a, b)
    return report


def print_report(report: dict, tolerance: float | None) -> int:
    print(f"left : {report['left']}")
    print(f"right: {report['right']}")
    print()

    if report["only_in_left"]:
        print(f"Only in left ({len(report['only_in_left'])}):")
        for n in report["only_in_left"]:
            print(f"  {n}")
    if report["only_in_right"]:
        print(f"Only in right ({len(report['only_in_right'])}):")
        for n in report["only_in_right"]:
            print(f"  {n}")
    if report["shape_mismatch"]:
        print(f"Shape mismatches ({len(report['shape_mismatch'])}):")
        for n, s in sorted(report["shape_mismatch"].items()):
            print(f"  {n}: {tuple(s['left'])} vs {tuple(s['right'])}")
    print()

    rows = sorted(report["compared"].items(), key=lambda kv: -kv[1]["max_frac"])
    width = max((len(n) for n, _ in rows), default=10)
    print(f"{'file'.ljust(width)}  {'max|d|':>12}  {'med|d|':>12}  {'max frac':>10}  {'med frac':>10}  nan!")
    for name, d in rows:
        print(
            f"{name.ljust(width)}  {d['max_abs']:12.4e}  {d['median_abs']:12.4e}  "
            f"{d['max_frac']:10.4f}  {d['median_frac']:10.4f}  {d['n_nan_mismatch']}"
        )

    status = 0
    if report["shape_mismatch"]:
        status = 1
    if tolerance is not None:
        over = [n for n, d in rows if d["max_frac"] > tolerance]
        if over:
            print(f"\n{len(over)} file(s) exceed the {tolerance} fractional tolerance:")
            for n in over:
                print(f"  {n}")
            status = 1
        else:
            print(f"\nAll {len(rows)} compared files within {tolerance} fractional tolerance.")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("left", type=Path, help="a RunParam_* directory")
    parser.add_argument("right", type=Path, help="another RunParam_* directory")
    parser.add_argument(
        "--tolerance",
        type=float,
        default=None,
        help="fail if any file's max fractional deviation exceeds this",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the report as JSON")
    args = parser.parse_args(argv)

    report = compare(args.left, args.right)
    status = print_report(report, args.tolerance)
    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
