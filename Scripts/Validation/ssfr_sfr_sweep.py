"""Reproduce Paper 2 Fig. 13's content: satellite sSFR distributions
in 3 stellar-mass bins, overlaid across SFR models (one colour per
model, solid/dashed for py-corrected/rs-steel), from the same
Satellite_sSFR histogram ssfr_plot.py uses. The paper's second model
is CE (continuity); no CE run was built here, so this substitutes
G19_DPL. Same deterministic-mode small-number-statistics caveat as
ssfr_plot.py's module docstring applies to the sparse bins.

Usage:
    python Scripts/Validation/ssfr_sfr_sweep.py \
        --run "T16:/path/t16/py-corrected-1:/path/t16/rs-steel-1" \
        --run "G19_DPL:/path/dpl/py-corrected-1:/path/dpl/rs-steel-1" \
        --out Figures/PortValidation/Paper2_Fig13_sSFRSweep.png
"""
import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = ["#4C72B0", "#DD8452"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label:py_dir:rs_dir")
    ap.add_argument("--mass-bins", type=float, nargs="+", default=[9.5, 10.5, 11.0])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, len(args.mass_bins), figsize=(4.2 * len(args.mass_bins), 4.6), sharey=True)

    for ci, spec in enumerate(args.run):
        label, py_dir, rs_dir = spec.split(":", 2)
        color = COLORS[ci % len(COLORS)]
        for run_dir, ls, lw in [(py_dir, "-", 1.8), (rs_dir, "--", 1.3)]:
            sat_mass = np.load(os.path.join(run_dir, "sSFR_Surviving_Sat_SMF_MassRange.npy"))
            ssfr_range = np.load(os.path.join(run_dir, "sSFR_Range.npy"))
            data = np.load(os.path.join(run_dir, "Satellite_sSFR.npy"))
            for ax, target in zip(axes, args.mass_bins):
                i = int(np.searchsorted(sat_mass, target))
                row = data[i]
                n = min(len(ssfr_range), len(row))
                mask = row[:n] > 0
                kwargs = dict(color=color, lw=lw, ls=ls)
                if ls == "--":
                    kwargs["dashes"] = (4, 2)
                lbl = label if ls == "-" else None
                ax.plot(ssfr_range[:n][mask], row[:n][mask], label=lbl, **kwargs)
                ax.set_title(rf"$\log M_*\approx{sat_mass[i]:.1f}$", fontsize=10)
                ax.set_xlabel(r"$\log_{10}\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")

    axes[0].set_ylabel(r"$N\ [\mathrm{Mpc}^{-3}\,\mathrm{h}^3]$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5, title="solid=py-corrected\ndashed=rs-steel")
    fig.suptitle("Paper 2 Fig. 13 style -- satellite sSFR, SFR-model sweep (G19)", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    print("wrote:", args.out)

if __name__ == "__main__":
    main()
