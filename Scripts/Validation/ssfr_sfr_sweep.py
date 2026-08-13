"""Reproduce Paper 1 Fig. 13's content: satellite sSFR distributions
in 3 stellar-mass bins, overlaid across SFR models (one colour per
model, solid/dashed for py-corrected/rs-steel), from the same
Satellite_sSFR histogram ssfr_plot.py uses.

Must run in STOCHASTIC mode (scatter on, several seeds averaged), for
the same reason documented at length in ssfr_plot.py's docstring:
without the model's 0.3 dex Monte Carlo SFR scatter, satellites in a
given mass bin collapse onto a handful of discrete SFR trajectories,
and once enough hit the gas-depletion sSFR floor (exactly 1e-12/yr,
Functions_c.pyx ~L186-217) the whole bin piles onto that one value
instead of showing a real distribution. This was found and fixed for
Paper2_Fig9_sSFR.png on 2026-08-13 (user-caught: "looks nothing like
paper 2 result") and the same defect was flagged as likely present
here too, since this script reads from the same deterministic-mode
Satellite_sSFR array and had the same masked-line plotting bug.

Usage, one or more seed directories per (model, leg):
    python Scripts/Validation/ssfr_sfr_sweep.py \
        --run "T16:/path/t16/py-corrected-1,.../py-corrected-2,...:/path/t16/rs-steel-1,...:/path/t16/rs-steel-2,..." \
        --out Figures/PortValidation/Paper1_Fig13_sSFRSweep.png

Simpler: pass --run label:py_glob:rs_glob where py_glob/rs_glob are
comma-separated seed directories (typically the py-corrected-*/
rs-steel-* output of Scripts/Validation/three_way.py --mode
stochastic --seeds 1 2 3 ...).
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4C72B0", "#DD8452"]


def ensemble_row(run_dirs, target_mass):
    rows = []
    ssfr_range = None
    for run_dir in run_dirs:
        sat_mass = np.load(os.path.join(run_dir, "sSFR_Surviving_Sat_SMF_MassRange.npy"))
        ssfr_range = np.load(os.path.join(run_dir, "sSFR_Range.npy"))
        data = np.load(os.path.join(run_dir, "Satellite_sSFR.npy"))
        i = int(np.searchsorted(sat_mass, target_mass))
        rows.append(data[i])
    return ssfr_range, np.nanmean(np.stack(rows), axis=0), sat_mass[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                     help="label:py_dir1,py_dir2,...:rs_dir1,rs_dir2,... (one or more seed dirs per leg)")
    ap.add_argument("--mass-bins", type=float, nargs="+", default=[9.5, 10.5, 11.0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, len(args.mass_bins), figsize=(4.2 * len(args.mass_bins), 4.6), sharey=True)

    for ci, spec in enumerate(args.run):
        label, py_dirs, rs_dirs = spec.split(":", 2)
        py_dirs = py_dirs.split(",")
        rs_dirs = rs_dirs.split(",")
        color = COLORS[ci % len(COLORS)]
        for run_dirs, ls, lw in [(py_dirs, "-", 1.8), (rs_dirs, "--", 1.3)]:
            for ax, target in zip(axes, args.mass_bins):
                ssfr_range, row, actual_mass = ensemble_row(run_dirs, target)
                kwargs = dict(color=color, lw=lw, ls=ls)
                if ls == "--":
                    kwargs["dashes"] = (4, 2)
                lbl = label if ls == "-" else None
                ax.step(ssfr_range, row, where="mid", label=lbl, **kwargs)
                ax.set_title(rf"$\log M_*\approx{actual_mass:.1f}$", fontsize=10)
                ax.set_xlabel(r"$\log_{10}\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
                ax.set_xlim(-13, -9)

    axes[0].set_ylabel(r"$N\ [\mathrm{Mpc}^{-3}\,\mathrm{h}^3]$")
    axes[0].legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none",
                   fontsize=8.5, title="solid=py-corrected\ndashed=rs-steel")
    n_seeds = len(args.run[0].split(":", 2)[1].split(","))
    fig.suptitle(f"Paper 1 Fig. 13 style -- satellite sSFR, SFR-model sweep (G18), "
                 f"stochastic, {n_seeds}-seed ensemble", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
