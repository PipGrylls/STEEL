#!/usr/bin/env python3
"""Run ``STEEL.py`` non-interactively on a chosen grid.

``STEEL.py`` is a script, not a library: its halo-mass grid, redshift
grid and subhalo mass function are all built at module import time, and
the run list is a hardcoded ``Tdyn_Factors`` literal guarded by a
blocking ``input()`` prompt.  There is no import-and-configure path.

Rather than edit the file (the whole point of the ``py-asis``
environment is that ``STEEL.py`` stays byte-for-byte unmodified), this
writes a *generated copy* next to the original with three textual
substitutions applied, and runs that.  The copy sits in the repository
root so every relative path inside it (``./Data/Model/Input/``,
``Functions/OtherModels/VDB13/``) still resolves.

Substitutions, all anchored on exact source lines:

* the ``AnalyticHaloMass_min``/``_max`` and ``AnalyticHaloBin``
  assignments, so a reduced grid can be run in seconds instead of
  ~45 minutes;
* the ``Tdyn_Factors`` block, replaced by the requested run tuples;
* the ``input()`` confirmation prompt, replaced by an unconditional
  proceed.

Nothing else is touched, so the physics executed is exactly the
committed code's.
"""

from __future__ import annotations

import argparse
import ast
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

# Anchors in STEEL.py. Each is matched exactly once; a miss is a hard
# error rather than a silent no-op, because a silently-unpatched grid
# would run for 45 minutes and produce output for the wrong config.
GRID_RE = re.compile(
    r"^AnalyticHaloMass_min = (?P<min>[0-9.]+); AnalyticHaloMass_max = (?P<max>[0-9.]+)"
    r"(?P<trailer>[^\n]*)$",
    re.MULTILINE,
)
BIN_RE = re.compile(r"^(?P<indent>\s+)AnalyticHaloBin = 0\.1$", re.MULTILINE)
# The run list runs from `Tdyn_Factors = []` to the `msg = ...` line.
# Matching it as "one or more `Tdyn_Factors +=` lines" was enough on
# `master`; `PipGrylls` interleaves triple-quoted multi-line blocks and
# blank lines among the commented-out entries, so anchor on the
# terminator instead and take everything in between.
TDYN_RE = re.compile(
    r"^    Tdyn_Factors = \[\]\n.*?(?=^    msg = 'About to run')",
    re.MULTILINE | re.DOTALL,
)
INPUT_RE = re.compile(
    r"^    msg = 'About to run' \+ str\(Tdyn_Factors\)\n"
    r"    shall = input\(.*\)\.lower\(\) != 'y'\n",
    re.MULTILINE,
)
# `PipGrylls` ships with the multiprocessing pool commented out and a
# single-run `OneRealization(Tdyn_Factors[0])` in its place -- fine for
# the one-run-at-a-time debugging it was left mid-session on, wrong for
# a driver that takes a list of runs. Restore the pool.
POOL_RE = re.compile(
    r"^    #For runnning single runs without multiprocessing bugs\n"
    r"    OneRealization\(Tdyn_Factors\[0\]\)\n"
    r"    \n"
    r"    #run ecah instance on a seperate core\n"
    r"    #pool = multiprocessing\.Pool\(processes = len\(Tdyn_Factors\)\)\n"
    r"    #PoolReturn = pool\.map\(OneRealization, Tdyn_Factors\)\n"
    r"    #pool\.close\(\)\n"
    r"    #print\(PoolReturn\)",
    re.MULTILINE,
)
POOL_REPLACEMENT = (
    "    pool = multiprocessing.Pool(processes = len(Tdyn_Factors))\n"
    "    PoolReturn = pool.map(OneRealization, Tdyn_Factors)\n"
    "    pool.close()\n"
    "    print(PoolReturn)"
)


def _sub_once(pattern: re.Pattern[str], replacement: str, text: str, what: str) -> str:
    new, n = pattern.subn(replacement, text)
    if n != 1:
        raise SystemExit(
            f"run_py_steel: expected exactly one match for {what} in STEEL.py, found {n}. "
            "STEEL.py has changed and this driver needs updating."
        )
    return new


def build_patched_source(
    source: str, halo_min: float, halo_max: float, halo_bin: float, runs: list[tuple]
) -> str:
    source = _sub_once(
        GRID_RE,
        rf"AnalyticHaloMass_min = {halo_min}; AnalyticHaloMass_max = {halo_max}\g<trailer>",
        source,
        "the halo mass range",
    )
    source = _sub_once(BIN_RE, rf"\g<indent>AnalyticHaloBin = {halo_bin}", source, "AnalyticHaloBin")
    runs_literal = "".join(f"    Tdyn_Factors += [{run!r}]\n" for run in runs)
    source = _sub_once(TDYN_RE, "    Tdyn_Factors = []\n" + runs_literal, source, "Tdyn_Factors")
    source = _sub_once(INPUT_RE, "    shall = False\n", source, "the input() prompt")
    source = _sub_once(POOL_RE, POOL_REPLACEMENT, source, "the multiprocessing pool")
    return source


def parse_run(spec: str) -> tuple:
    """Parse ``1.0,False,False,True,G19_DPL,Moster`` into the tuple
    ``STEEL.py`` expects: (Tdyn factor str, Stripping, SF, z_evo,
    SFR model, abundance-matching preset)."""
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            f"expected 6 comma-separated fields (factor,stripping,sf,z_evo,sfr,abnmtch), got {len(parts)}: {spec}"
        )
    factor, stripping, sf, z_evo, sfr, abnmtch = parts
    def as_bool(s: str) -> bool:
        if s not in ("True", "False"):
            raise argparse.ArgumentTypeError(f"expected True or False, got {s!r}")
        return s == "True"
    return (factor, as_bool(stripping), as_bool(sf), as_bool(z_evo), sfr, abnmtch)


def run_param_dir(run: tuple) -> str:
    """The output directory `Functions.py`'s SaveData_* family builds."""
    return "RunParam_" + "".join(f"{field}_" for field in run)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--python",
        default=str(REPO_ROOT / "env" / "py-asis" / "bin" / "python"),
        help="interpreter to run STEEL.py with (default: the py-asis venv)",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=REPO_ROOT,
        help="repository (or worktree) whose STEEL.py to run; defaults to this "
             "script's own. The py-as-is leg is a detached worktree at "
             "origin/PipGrylls, which predates Scripts/Validation entirely, so "
             "the driver has to be this copy pointed at that tree.",
    )
    parser.add_argument("--halo-min", type=float, default=11.0)
    parser.add_argument("--halo-max", type=float, default=16.6)
    parser.add_argument("--halo-bin", type=float, default=0.1)
    parser.add_argument(
        "--run",
        type=parse_run,
        action="append",
        required=True,
        metavar="FACTOR,STRIPPING,SF,Z_EVO,SFR,ABNMTCH",
        help="a run tuple; repeatable (STEEL.py runs them in parallel processes)",
    )
    parser.add_argument(
        "--keep-patched",
        action="store_true",
        help="leave the generated STEEL_generated.py in place for inspection",
    )
    args = parser.parse_args(argv)

    root = args.root.resolve()
    steel_py = root / "STEEL.py"
    patched = root / "STEEL_generated.py"
    source = steel_py.read_text()
    patched.write_text(
        build_patched_source(source, args.halo_min, args.halo_max, args.halo_bin, args.run)
    )

    # STEEL.py's PrepareToSave calls `mkdir` without -p and fails
    # noisily if the directory exists; make sure the parent is there and
    # let it handle the leaves.
    (root / "Data" / "Model" / "Output" / "RunFiles").mkdir(parents=True, exist_ok=True)
    (root / "Data" / "Model" / "Input").mkdir(parents=True, exist_ok=True)

    print(
        f"grid: log M = [{args.halo_min}, {args.halo_max}) step {args.halo_bin}; "
        f"{len(args.run)} run(s)",
        file=sys.stderr,
    )
    started = time.time()
    try:
        proc = subprocess.run([args.python, str(patched)], cwd=root)
    finally:
        if not args.keep_patched:
            patched.unlink(missing_ok=True)
    elapsed = time.time() - started
    print(f"STEEL.py finished in {elapsed:.1f}s (exit {proc.returncode})", file=sys.stderr)
    for run in args.run:
        print(run_param_dir(run))
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
