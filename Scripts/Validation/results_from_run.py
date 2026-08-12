"""Reproduce several results figures from ONE already-completed
deterministic three-way run (py-corrected vs rs-steel), rather than
running a fresh simulation per figure. Companion to results_figure3.py
(Paper 2 Fig. 3) and paper2_figures.py (Figs. 6 & 7).

Currently covers:

* Paper 2 Figure 5 -- the fraction of z=0.1-observed satellites
  accreted by a given redshift, from the z_infall.npy accumulator.
* A pair-fraction-vs-redshift plot in the style of Paper 3's figures,
  using the exact Return_PF_Plot logic from
  Scripts/CentralPostprocessing.py (PORT-FIX H1 applies here) rather
  than a reimplementation -- run against both legs' real Pair_Frac
  output via a throwaway OutputFolder redirect.

py-as-is is intentionally absent: this reuses the deterministic run
built in results_figure3.py's companion invocation of three_way.py,
and py-as-is has no deterministic mode (A7).

Usage (from repo root, env/py-legacy or env/py-asis active):
    python Scripts/Validation/results_from_run.py \
        --py-corrected /path/to/py-corrected-1 \
        --rs-steel /path/to/rs-steel-1 \
        --outdir Figures/PortValidation
"""
import argparse
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LEG_STYLE = {
    "py-corrected": dict(color="black", lw=1.8, ls="-", zorder=3, label="py-corrected"),
    "rs-steel": dict(color="crimson", lw=1.4, ls="--", dashes=(4, 2), zorder=4, label="rs-steel"),
}


def plot_accretion_redshift(py_corrected, rs_steel, outdir):
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for name, run_dir in [("py-corrected", py_corrected), ("rs-steel", rs_steel)]:
        z = np.load(os.path.join(run_dir, "z_infall_z.npy"))
        zi = np.load(os.path.join(run_dir, "z_infall.npy"))
        total = np.nansum(zi, axis=1)
        order = np.argsort(z)
        zs, ts = z[order], total[order]
        cum = np.cumsum(ts) / np.sum(ts)
        ax.plot(zs, cum, **LEG_STYLE[name])
        if name == "py-corrected":
            f05 = cum[np.searchsorted(zs, 0.5)]
            f01 = cum[np.searchsorted(zs, 0.1)]
    ax.axhline(0.5, color="0.75", lw=0.8, ls=":")
    ax.set_xlabel(r"redshift of accretion, $z_\mathrm{infall}$")
    ax.set_ylabel(r"fraction of $z=0.1$ satellites accreted by $z_\mathrm{infall}$")
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Paper 2 Fig. 5 -- accretion-redshift distribution (G19)", fontsize=10)
    fig.text(0.15, 0.8, f"f(z<0.5) = {f05:.2f}\nf(z<0.1) = {f01:.2f}", fontsize=9, color="0.35")
    fig.tight_layout()
    out = os.path.join(outdir, "Paper2_Fig5_AccretionRedshift.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote:", out)


def _redirect_output_folder(run_dir, run_tuple, scratch_root):
    """CentralPostprocessing's LoadData_* functions read from
    Functions.OutputFolder + 'RunParam_<tuple>/'. The three-way runner
    copies each leg's output to a directory named by seed, not by
    RunParam, so this builds a throwaway OutputFolder containing a
    correctly-named symlink to the real data instead of copying it."""
    dir_name = "RunParam_" + "".join(f"{p}_" for p in run_tuple)
    fake_root = os.path.join(scratch_root, os.path.basename(run_dir) + "-outputfolder")
    os.makedirs(fake_root, exist_ok=True)
    link = os.path.join(fake_root, dir_name)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link) if os.path.islink(link) else shutil.rmtree(link)
    os.symlink(os.path.abspath(run_dir), link)
    return fake_root + "/"


def plot_pair_fraction(py_corrected, rs_steel, scratch_root, outdir):
    sys.path.insert(0, REPO_ROOT)
    from Scripts import CentralPostprocessing as CP

    run_tuple = ("1.0", True, True, True, "G19_DPL", "G19_SE")

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for name, run_dir in [("py-corrected", py_corrected), ("rs-steel", rs_steel)]:
        CP.F.OutputFolder = _redirect_output_folder(run_dir, run_tuple, scratch_root)
        d = CP.PairFractionData(run_tuple)
        z, pf, _, _ = d.Return_PF_Plot(d.SMF_interp, Parent_Cut=11, UpperLimit=True)
        pf = np.asarray(pf, dtype=float)
        z = np.asarray(z, dtype=float)
        mask = np.isfinite(pf) & (z <= 3)
        ax.plot(z[mask], pf[mask], **LEG_STYLE[name])

    ax.set_xlabel("redshift")
    ax.set_ylabel("pair fraction")
    ax.set_xlim(0, 3)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        "Paper 3 style -- pair fraction vs. redshift (G19), Return_PF_Plot post-H1-fix",
        fontsize=9.5,
    )
    fig.tight_layout()
    out = os.path.join(outdir, "Paper3_PairFraction_vs_z.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", required=True)
    ap.add_argument("--rs-steel", required=True)
    ap.add_argument("--outdir", default="Figures/PortValidation")
    ap.add_argument("--scratch", default="/tmp/steel-results-from-run")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.scratch, exist_ok=True)

    plot_accretion_redshift(args.py_corrected, args.rs_steel, args.outdir)
    plot_pair_fraction(args.py_corrected, args.rs_steel, args.scratch, args.outdir)


if __name__ == "__main__":
    main()
