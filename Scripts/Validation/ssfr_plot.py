"""Reproduce the satellite side of Paper 1 Fig. 9: sSFR distributions
in a few stellar-mass bins, from the Satellite_sSFR 2D histogram
(satellite mass x sSFR) both implementations already produce at the
reference (z~0.1) epoch. The published figure also shows a
post-processed central-galaxy sSFR distribution (dashed black line);
that side needs CentralPostprocessing analysis this script doesn't
run, so this is satellites only.

Deterministic mode collapses each mass bin's satellites onto a handful
of discrete sSFR values (no scatter to smear them across bins), so
this inherits the same small-number-statistics sensitivity already
documented for other deterministic-mode outputs in docs/VALIDATION.md
(median agreement is exact -- most bins are identically zero -- but a
populated bin holding one or two satellites can disagree by order-1
between implementations from a hairline difference in which bin edge
a value falls on either side of). Not a sign of a port defect on its
own; see the same caveat for Figure3's p90 column.

Usage:
    python Scripts/Validation/ssfr_plot.py \
        --py-corrected /path/to/py-corrected-1 \
        --rs-steel /path/to/rs-steel-1 \
        --out Figures/PortValidation/Paper1_Fig9_sSFR.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LEG_STYLE = {
    "py-corrected": dict(color="black", lw=1.8, ls="-", zorder=3, label="py-corrected"),
    "rs-steel": dict(color="crimson", lw=1.4, ls="--", dashes=(4, 2), zorder=4, label="rs-steel"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", required=True)
    ap.add_argument("--rs-steel", required=True)
    ap.add_argument("--mass-bins", type=float, nargs="+", default=[9.5, 10.5, 11.0])
    ap.add_argument("--out", default="Figures/PortValidation/Paper1_Fig9_sSFR.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, len(args.mass_bins), figsize=(4.2 * len(args.mass_bins), 4.6), sharey=True)

    for name, run_dir in [("py-corrected", args.py_corrected), ("rs-steel", args.rs_steel)]:
        sat_mass = np.load(os.path.join(run_dir, "sSFR_Surviving_Sat_SMF_MassRange.npy"))
        ssfr_range = np.load(os.path.join(run_dir, "sSFR_Range.npy"))
        data = np.load(os.path.join(run_dir, "Satellite_sSFR.npy"))  # (sat_mass, sSFR)

        for ax, target in zip(axes, args.mass_bins):
            i = int(np.searchsorted(sat_mass, target))
            row = data[i]
            n = min(len(ssfr_range), len(row))
            mask = row[:n] > 0
            ax.plot(ssfr_range[:n][mask], row[:n][mask], **LEG_STYLE[name])
            ax.set_title(rf"$\log M_*\approx{sat_mass[i]:.1f}$", fontsize=10)
            ax.set_xlabel(r"$\log_{10}\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")

    axes[0].set_ylabel(r"$N\ [\mathrm{Mpc}^{-3}\,\mathrm{h}^3]$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    fig.suptitle(
        "Paper 1 Fig. 9 style -- satellite sSFR distribution (G19), deterministic\n"
        "sparse bins are noisy (small-number statistics, see module docstring), not a port defect",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
