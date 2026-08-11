#!/usr/bin/env python3
"""Run all three implementations on one configuration and compare them.

The three legs:

* **py-as-is**       -- ``STEEL.py`` byte-for-byte unmodified, on ``env/py-asis``
* **py-corrected**   -- the same with ``docs/PORT_CORRECTIONS.md`` applied
* **rs-steel**       -- the Rust port

``py-as-is`` lives on a different branch from the other two (it *is* the
unmodified code, and must stay that way), so this script expects a
worktree for it and will tell you how to make one if it is missing.

Two modes, and they answer different questions:

``--mode deterministic``
    Scatter off everywhere (``STEEL_SCATTER=0`` /
    ``[run] scatter = false``). Both implementations then evaluate the
    same arithmetic on the same grid and should agree to floating point,
    modulo the deliberate corrections. This is the strong
    numerical-fidelity claim. Note py-as-is cannot run in this mode --
    it has no such switch (``GetGasMass`` scatters unconditionally), so
    only py-corrected and rs-steel are compared.

``--mode stochastic``
    Scatter on. The three draw from unrelated generators -- NumPy's
    Mersenne Twister, GSL's taus, and ``rand``'s ChaCha -- so they can
    never agree element-wise, only in ensemble statistics. Run with
    several seeds and compare per-bin means.

Quoting a single blended "agreement" number across both modes would
conflate port fidelity with Monte-Carlo noise, which is exactly the
distinction this exists to draw.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_runs import compare, deviations  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
ASIS_BRANCH = "origin/PipGrylls"

# Each leg gets a mass-accretion-history table built with the cosmology
# its own code asks for. `Get_HM_History` keys its cache on
# `<min><max><bin><h>` and *not* on the cosmology, so without this the
# two Python legs silently share whichever table happened to be written
# first -- which would hide correction 16 completely and make the
# comparison meaningless. See `make_mah_table.py`.
LEG_COSMOLOGY = {"py-as-is": "pipgrylls", "py-corrected": "corrected"}
HUBBLE = 0.6774


def run_dir_for(run_tuple: str) -> str:
    return "RunParam_" + "".join(f"{f}_" for f in run_tuple.split(","))


def mah_table_name(grid) -> str:
    """`Get_HM_History`'s cache filename for this grid."""
    return "{}{}{}{}.dat".format(grid[0], grid[1], grid[2], HUBBLE)


def ensure_mah_table(root: Path, grid, cosmology: str) -> None:
    """Build `root`'s MAH table under `cosmology` if it isn't there.

    Also drops the two *derived* caches, which are cosmology-dependent
    and keyed without it:

    * ``SHMFs_Entering_*.npy`` (`STEEL.py:143`) is built from
      ``AvaHaloMass``, i.e. straight out of the MAH table, but its key is
      grid + h + array shapes;
    * ``hmf_fun.pkl`` (`Functions.Make_HMF_Interp`) has no key at all.

    Leaving either in place silently mixes one cosmology's accretion
    histories with another's subhalo mass function -- the same trap as
    the MAH table itself, one level down.
    """
    out = root / "Data" / "Model" / "Input" / mah_table_name(grid)
    stamp = out.with_suffix(".cosmology")
    if out.exists() and stamp.exists() and stamp.read_text().strip() == cosmology:
        return
    print(f"building {out} ({cosmology})", file=sys.stderr)
    for stale in (out.parent).glob("SHMFs_Entering_*.npy"):
        print(f"  dropping derived cache {stale.name}", file=sys.stderr)
        stale.unlink()
    for name in ("hmf_fun.pkl",):
        stale = out.parent / name
        if stale.exists():
            print(f"  dropping derived cache {name}", file=sys.stderr)
            stale.unlink()
    subprocess.run(
        [str(REPO_ROOT / "env" / "py-legacy" / "bin" / "python"),
         str(REPO_ROOT / "Scripts" / "Validation" / "make_mah_table.py"),
         "--cosmology", cosmology,
         "--halo-min", str(grid[0]), "--halo-max", str(grid[1]), "--halo-bin", str(grid[2]),
         "--out", str(out)],
        check=True, capture_output=True)
    stamp.write_text(cosmology + "\n")


def run_python(interpreter: Path, root: Path, run_tuple: str, grid, scatter: bool, seed: int) -> Path:
    env = dict(os.environ, STEEL_SEED=str(seed))
    if not scatter:
        env["STEEL_SCATTER"] = "0"
    cmd = [
        str(interpreter), str(REPO_ROOT / "Scripts" / "Validation" / "run_py_steel.py"),
        "--root", str(root),
        # Explicit: the driver defaults to *its own* repo root's venv,
        # and the py-as-is worktree has no env/ of its own -- both legs
        # share the one interpreter.
        "--python", str(interpreter),
        "--halo-min", str(grid[0]), "--halo-max", str(grid[1]), "--halo-bin", str(grid[2]),
        "--run", run_tuple,
    ]
    started = time.time()
    proc = subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"py-steel failed in {root}:\n{proc.stderr[-2000:]}")
    elapsed = time.time() - started
    out = root / "Data" / "Model" / "Output" / "RunFiles" / run_dir_for(run_tuple)
    if not out.is_dir():
        raise SystemExit(f"expected output at {out}, which does not exist")
    return out, elapsed


def run_rust(runfile: Path, out_root: Path, grid, scatter: bool, seed: int) -> Path:
    binary = REPO_ROOT / "rust" / "target" / "release" / "steel"
    if not binary.exists():
        raise SystemExit(f"{binary} not found -- run `cargo build --release --manifest-path rust/Cargo.toml`")

    text = runfile.read_text()
    lines = [
        l for l in text.splitlines()
        if not l.startswith(("log_m_min", "log_m_max", "log_m_bin", "scatter", "rng_seed"))
    ]
    lines += [
        f"log_m_min = {grid[0]}", f"log_m_max = {grid[1]}", f"log_m_bin = {grid[2]}",
        f"scatter = {str(scatter).lower()}", f"rng_seed = {seed}",
    ]
    patched = out_root / "runfile.toml"
    out_root.mkdir(parents=True, exist_ok=True)
    patched.write_text("\n".join(lines) + "\n")

    started = time.time()
    proc = subprocess.run([str(binary), str(patched), str(out_root)], capture_output=True, text=True)
    if proc.returncode != 0:
        raise SystemExit(f"rs-steel failed:\n{proc.stderr[-2000:]}")
    elapsed = time.time() - started
    dirs = [d for d in out_root.iterdir() if d.is_dir() and d.name.startswith("RunParam_")]
    if len(dirs) != 1:
        raise SystemExit(f"expected exactly one RunParam_ dir under {out_root}, got {dirs}")
    return dirs[0], elapsed


def ensemble(paths: list[Path]) -> dict[str, np.ndarray]:
    """Per-bin mean over several seeds' output trees."""
    acc: dict[str, list[np.ndarray]] = {}
    for p in paths:
        for f in sorted(p.glob("*.npy")):
            try:
                acc.setdefault(f.name, []).append(np.load(f, allow_pickle=False))
            except ValueError:
                pass
    return {
        k: np.nanmean(np.stack(v), axis=0)
        for k, v in acc.items()
        if len({a.shape for a in v}) == 1
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["deterministic", "stochastic"], default="deterministic")
    parser.add_argument("--run", default="1.0,True,True,True,G19_DPL,G19_SE",
                        help="STEEL.py run tuple")
    parser.add_argument("--runfile", type=Path,
                        default=REPO_ROOT / "rust" / "runfiles" / "published" / "p2-dpl-sf-strip.toml",
                        help="matching rs-steel runfile")
    parser.add_argument("--grid", nargs=3, type=float, default=[11.0, 12.6, 0.5],
                        metavar=("MIN", "MAX", "BIN"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[1],
                        help="stochastic mode: seeds to average over")
    parser.add_argument("--asis-worktree", type=Path, default=REPO_ROOT.parent / "STEEL-pipgrylls")
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "steel-three-way")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    stochastic = args.mode == "stochastic"
    seeds = args.seeds if stochastic else [args.seeds[0]]
    args.work.mkdir(parents=True, exist_ok=True)

    legs: dict[str, list[Path]] = {}
    timings: dict[str, float] = {}

    # --- py-corrected (this worktree) ---
    ensure_mah_table(REPO_ROOT, args.grid, LEG_COSMOLOGY["py-corrected"])
    corrected = []
    for seed in seeds:
        out, elapsed = run_python(
            REPO_ROOT / "env" / "py-asis" / "bin" / "python", REPO_ROOT,
            args.run, args.grid, stochastic, seed)
        dest = args.work / f"py-corrected-{seed}"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(out, dest)
        corrected.append(dest)
        timings["py-corrected"] = elapsed
    legs["py-corrected"] = corrected

    # --- rs-steel ---
    rust = []
    for seed in seeds:
        out, elapsed = run_rust(args.runfile, args.work / f"rs-raw-{seed}", args.grid, stochastic, seed)
        dest = args.work / f"rs-steel-{seed}"
        shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(out, dest)
        rust.append(dest)
        timings["rs-steel"] = elapsed
    legs["rs-steel"] = rust

    # --- py-as-is, only meaningful with scatter on ---
    if stochastic:
        if not (args.asis_worktree / "STEEL.py").exists():
            print(
                f"\nNo py-as-is worktree at {args.asis_worktree}. Create one with:\n"
                f"    git worktree add --detach {args.asis_worktree} {ASIS_BRANCH}\n"
                f"    cd {args.asis_worktree}/Functions && "
                f"{REPO_ROOT}/env/py-asis/bin/python Setup.py build_ext --inplace\n"
                "Skipping the py-as-is leg.\n", file=sys.stderr)
        else:
            ensure_mah_table(args.asis_worktree, args.grid, LEG_COSMOLOGY["py-as-is"])
            asis = []
            for seed in seeds:
                out, elapsed = run_python(
                    REPO_ROOT / "env" / "py-asis" / "bin" / "python", args.asis_worktree,
                    args.run, args.grid, True, seed)
                dest = args.work / f"py-asis-{seed}"
                shutil.rmtree(dest, ignore_errors=True)
                shutil.copytree(out, dest)
                asis.append(dest)
                timings["py-as-is"] = elapsed
            legs["py-as-is"] = asis
    else:
        print("py-as-is has no scatter switch (GetGasMass scatters unconditionally),\n"
              "so it cannot run in deterministic mode. Comparing py-corrected vs rs-steel.\n",
              file=sys.stderr)

    print(f"mode: {args.mode}   run: {args.run}   grid: {args.grid}   seeds: {seeds}")
    print("wall clock: " + ", ".join(f"{k} {v:.1f}s" for k, v in sorted(timings.items())))
    print()

    report = {"mode": args.mode, "run": args.run, "grid": args.grid, "seeds": seeds,
              "timings": timings, "comparisons": {}}

    if stochastic:
        means = {name: ensemble(paths) for name, paths in legs.items()}
        base = "py-as-is" if "py-as-is" in means else "py-corrected"
        for name in means:
            if name == base:
                continue
            rows = {}
            for f in sorted(set(means[base]) & set(means[name])):
                if means[base][f].shape == means[name][f].shape:
                    rows[f] = deviations(means[base][f], means[name][f])
            report["comparisons"][f"{base} vs {name}"] = rows
            print(f"=== ensemble means, {base} vs {name} ({len(seeds)} seed(s)) ===")
            _print_rows(rows)
    else:
        rep = compare(legs["py-corrected"][0], legs["rs-steel"][0])
        report["comparisons"]["py-corrected vs rs-steel"] = rep["compared"]
        report["shape_mismatch"] = rep["shape_mismatch"]
        report["only_in_left"] = rep["only_in_left"]
        report["only_in_right"] = rep["only_in_right"]
        print("=== deterministic: py-corrected vs rs-steel, per bin ===")
        if rep["shape_mismatch"]:
            print("shape mismatches (see Scripts/Validation/README.md):")
            for n, sh in sorted(rep["shape_mismatch"].items()):
                print(f"  {n}: {tuple(sh['left'])} vs {tuple(sh['right'])}")
            print()
        _print_rows(rep["compared"])
        cum = cumulative_rows(legs["py-corrected"][0], legs["rs-steel"][0])
        report["cumulative"] = cum
        _print_cumulative(cum)

    if args.json:
        args.json.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.json}", file=sys.stderr)
    return 0


# Arrays whose last axis is the satellite stellar-mass grid. In
# deterministic mode a per-bin comparison of these is dominated by
# bin-edge crossing rather than by any real disagreement: with scatter
# off every realization of a given (redshift, host, subhalo) bin lands on
# the *same* stellar mass, so the distribution is a delta function and
# the residual ~0.01 dex halo-mass difference between the two cosmology
# implementations can move an entire bin's weight to its neighbour. That
# shows up as a 100% deviation in two bins while the physics agrees.
#
# The reverse-cumulative distribution ("number above this mass") is the
# quantity the papers actually plot integrals of, and it is insensitive
# to a value crossing a bin edge. It is the right deterministic-mode
# metric; the per-bin table is kept alongside it for completeness.
MASS_AXIS_ARRAYS = (
    "Figure3_AnalyticalModel_SMF.npy",
    "Figure10_AnalyticalModel_SMF.npy",
    "SMFhz_AnalyticalModel_SMF_Highz.npy",
    "Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy",
    "Sat_SMHM_Sat_SMHM.npy",
    "Sat_SMHM_Sat_SMHM_Host.npy",
    "Mergers_Accretion_History.npy",
    "Pair_Frac_Pair_Frac.npy",
    "z_infall.npy",
)


def cumulative_rows(left: Path, right: Path) -> dict:
    """Reverse-cumulative agreement along the stellar-mass axis."""
    rows = {}
    for name in MASS_AXIS_ARRAYS:
        lp, rp = left / name, right / name
        if not (lp.exists() and rp.exists()):
            continue
        a, b = np.load(lp), np.load(rp)
        if a.shape != b.shape:
            continue
        ca = np.flip(np.nancumsum(np.flip(a, axis=-1), axis=-1), axis=-1)
        cb = np.flip(np.nancumsum(np.flip(b, axis=-1), axis=-1), axis=-1)
        # Only where there is something to compare: an empty tail is not
        # evidence of agreement.
        mask = (ca != 0) | (cb != 0)
        if not mask.any():
            continue
        d = deviations(ca[mask], cb[mask])
        d["integral_ratio"] = float(np.nansum(b) / np.nansum(a)) if np.nansum(a) else float("nan")
        rows[name] = d
    return rows


def _print_cumulative(rows: dict) -> None:
    if not rows:
        return
    width = max(len(n) for n in rows)
    print()
    print("=== reverse-cumulative along the stellar-mass axis ===")
    print(f"{'file'.ljust(width)}  {'med frac':>10}  {'p90 frac':>10}  {'max frac':>10}  {'integral ratio':>14}")
    for name, d in sorted(rows.items(), key=lambda kv: -kv[1]["p90_frac"]):
        print(f"{name.ljust(width)}  {d['median_frac']:10.6f}  {d['p90_frac']:10.6f}  "
              f"{d['max_frac']:10.6f}  {d['integral_ratio']:14.6f}")
    print("  max frac is set by the extreme high-mass tail, where a bin holds at most one")
    print("  satellite in either run; med/p90 describe the populated part of the function.")


def _print_rows(rows: dict) -> None:
    if not rows:
        print("  (nothing comparable)")
        return
    ordered = sorted(rows.items(), key=lambda kv: -kv[1]["max_frac"])
    width = max(len(n) for n, _ in ordered)
    print(f"{'file'.ljust(width)}  {'max|d|':>12}  {'max frac':>10}  {'med frac':>10}")
    for name, d in ordered:
        print(f"{name.ljust(width)}  {d['max_abs']:12.4e}  {d['max_frac']:10.6f}  {d['median_frac']:10.6f}")


if __name__ == "__main__":
    raise SystemExit(main())
