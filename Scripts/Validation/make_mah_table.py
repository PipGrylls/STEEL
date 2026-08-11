#!/usr/bin/env python3
"""Build the cached halo mass-accretion-history table with an *explicit*
cosmology, rather than whichever one the calling branch happens to
hardcode.

``Functions.py::Get_HM_History`` caches its van den Bosch (2014) MAH grid
in ``Data/Model/Input/<min><max><bin><h>.dat``, regenerating it only when
that file is absent. **The cache key does not include the cosmology.**
That is fine while the cosmology is a constant, and it stops being fine
the moment two branches disagree about it -- which is exactly what
happened between ``master`` and ``PipGrylls``:

===============  =======  =====  =======  ======  ==========
Passed to getPWGH  Omega_0      h  sigma_8   nspec  Omega_b h^2
===============  =======  =====  =======  ======  ==========
``master``         0.307  0.678    0.823   0.96      0.02298
``PipGrylls``      0.309  0.677    0.816   1.000     0.022
===============  =======  =====  =======  ======  ==========

``master`` hardcodes the five literals; ``PipGrylls`` interpolates them
from the run's own COLOSSUS cosmology -- except ``nspec``, which it
hardcodes to ``1`` (Harrison-Zel'dovich) where Planck15 has
``n_s = 0.9667``. The net effect on the MAHs is up to **0.08 dex**, and
roughly seven eighths of that is the ``nspec`` slip alone: with
``n_s = 0.9667`` the same switch moves the MAHs by at most 0.011 dex.

Because the cache key ignores all of this, checking out the other branch
and re-running silently reuses the wrong table. This script makes the
choice explicit and writes the table to a named output path, so each leg
of the three-way validation gets a table generated with the cosmology its
own code asks for.

It also sidesteps two things that stop ``PipGrylls``'s ``Halogrowth``
from running anywhere but the machine it was written on: ``getPWGH.f``
opens its output under a hardcoded ``/data/pg1g15/STEEL/...`` prefix, and
the Python then reads it back through a path containing a literal ``*``
that ``np.loadtxt`` does not glob. Neither is physics; both are worked
around here by running ``getPWGH`` in a scratch directory of our own.

Usage::

    Scripts/Validation/make_mah_table.py --cosmology pipgrylls \\
        --halo-min 11.0 --halo-max 16.6 --halo-bin 0.1 \\
        --out Data/Model/Input/11.016.60.10.6774.dat
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
VDB13 = REPO_ROOT / "Functions" / "OtherModels" / "VDB13"

# (Omega_0, h, sigma_8, nspec, Omega_b h^2) exactly as each branch's
# `Halogrowth` formats them into getPWGH's stdin.
COSMOLOGIES = {
    "master": (0.307, 0.678, 0.823, 0.96, 0.02298),
    "pipgrylls": (0.309, 0.677, 0.816, 1.000, 0.022),
    # PipGrylls' intent -- COLOSSUS planck15 throughout, including its
    # actual spectral index instead of the hardcoded 1.
    "corrected": (0.309, 0.677, 0.816, 0.967, 0.022),
}

INPUT_TEMPLATE = (
    "    %.3f                                        ! Omega_0\n"
    "    %.3f                                        ! h (= H_0/100)\n"
    "    %.3f                                        ! sigma8\n"
    "    %.3f                                         ! nspec\n"
    "    %.3f                                      ! Omega_b_h2\n"
    "    %.1E                                         ! M_0  (h^{-1} Msun)\n"
    "    0.0                                          ! z_0\n"
    "    1                                            ! median (0) or averages (1)\n"
    "    %s.dat                                       !Output File\n"
)


def build_getpwgh(workdir: Path) -> Path:
    """Compile getPWGH into ``workdir`` and return the binary's path.

    Compiled fresh rather than reusing the committed binary: the
    committed one is a 2019 build of a source file that has since been
    edited on some branches, and a stale binary would silently
    contradict the source this script quotes.
    """
    for name in ("getPWGH.f", "cosmo_sub.f", "quadpack.f", "paramfile.h", "Makefile.mke"):
        shutil.copy(VDB13 / name, workdir / name)

    # Undo the `fileplace` prefix if this checkout carries it, so the
    # output lands in `workdir` instead of a path that does not exist.
    src = (workdir / "getPWGH.f").read_text()
    src = src.replace(
        '      fileplace = "/data/pg1g15/STEEL/Functions/OtherModels/VDB13/"\n'
        "      OPEN(10,file=fileplace//outfile,status='UNKNOWN')",
        "      OPEN(10,file=outfile,status='UNKNOWN')",
    ).replace("      CHARACTER fileplace*47\n\n\n", "")
    (workdir / "getPWGH.f").write_text(src)

    subprocess.run(
        ["make", "-f", "Makefile.mke"], cwd=workdir, check=True, capture_output=True
    )
    return workdir / "getPWGH"


def one_halo(binary: Path, workdir: Path, params: tuple, log_m_h: float) -> tuple:
    """Run getPWGH for a single halo mass, returning ``(z, log10 M(z))``."""
    tag = f"mah_{log_m_h:.6f}"
    (workdir / f"{tag}.in").write_text(INPUT_TEMPLATE % (*params, 10**log_m_h, tag))
    with open(workdir / f"{tag}.in") as stdin:
        subprocess.run([str(binary)], cwd=workdir, stdin=stdin, check=True, capture_output=True)
    table = np.loadtxt(workdir / f"{tag}.dat")
    (workdir / f"{tag}.in").unlink()
    (workdir / f"{tag}.dat").unlink()
    # Column layout, from getPWGH.f's WRITE(10,74): index, z, t,
    # log10 M(z)/M0, Vmax ratio, Vmax/Vvir, concentration, accretion rate.
    return table[:, 1], table[:, 3] + log_m_h


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cosmology", choices=sorted(COSMOLOGIES), required=True)
    ap.add_argument("--halo-min", type=float, default=11.0)
    ap.add_argument("--halo-max", type=float, default=16.6)
    ap.add_argument("--halo-bin", type=float, default=0.1)
    # COLOSSUS planck15's h to full precision. Deliberately *not*
    # `COSMOLOGIES[...][1]`: that is the `%.3f`-rounded value getPWGH is
    # handed, whereas the halo grid and the cache filename both use the
    # unrounded one, so reusing the rounded value would shift every grid
    # point by 6e-5 dex and change the filename.
    ap.add_argument("--h", type=float, default=0.6774)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args(argv)

    params = COSMOLOGIES[args.cosmology]
    h = args.h

    # `STEEL.py` builds its grid in Mvir h^-1, so the same log10(h) shift
    # applies here.
    grid = np.arange(args.halo_min + np.log10(h), args.halo_max + np.log10(h), args.halo_bin)
    print(f"{args.cosmology}: {len(grid)} halo masses, "
          f"Omega_0={params[0]} h={params[1]} sigma_8={params[2]} nspec={params[3]} Omega_b h^2={params[4]}",
          file=sys.stderr)

    with tempfile.TemporaryDirectory() as tmp:
        workdir = Path(tmp)
        binary = build_getpwgh(workdir)
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            results = list(pool.map(lambda m: one_halo(binary, workdir, params, m), grid))

    columns = [results[0][0]] + [m for _, m in results]
    table = np.column_stack(columns)
    # `Get_HM_History` sorts by the z=0 row; the grid is already
    # ascending, but sort anyway so the output is byte-comparable with a
    # table the Python itself produced.
    table = table[:, np.argsort(table[0])]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.out, table)
    print(f"wrote {args.out} {table.shape}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
