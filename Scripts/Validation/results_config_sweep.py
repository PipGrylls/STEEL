"""Overlay the satellite SMF across multiple STEEL configs -- Paper 2
Fig. 14's "frozen / SF only / SF+stripping" comparison (here: frozen
vs. SF+stripping, from two already-completed deterministic runs) and
Fig. 10's "f_tdyn = 0.5/1.0/2.5" comparison, whichever runs are passed.
Each config gets its own colour; line style (solid/dashed) encodes
py-corrected vs. rs-steel, so both "does the port agree" and "does the
config change the result" read at once.

Usage:
    python Scripts/Validation/results_config_sweep.py \
        --run frozen:/path/to/frozen/py-corrected-1:/path/to/frozen/rs-steel-1 \
        --run "sf+strip:/path/.../py-corrected-1:/path/.../rs-steel-1" \
        --out Figures/PortValidation/Paper2_Fig14_ConfigSweep.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def reverse_cumulative(smf):
    return np.flip(np.nancumsum(np.flip(smf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True,
                     help="label:py_corrected_dir:rs_steel_dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cumulative", action="store_true",
                     help="plot reverse-cumulative N(>M*) instead of differential phi")
    ap.add_argument("--title", default="Paper 2 style -- satellite SMF config sweep (G19)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, spec in enumerate(args.run):
        label, py_dir, rs_dir = spec.split(":", 2)
        color = COLORS[i % len(COLORS)]
        for leg_dir, ls, lw, z in [(py_dir, "-", 1.8, 3), (rs_dir, "--", 1.4, 4)]:
            mass = np.load(os.path.join(leg_dir, "Figure3_Surviving_Sat_SMF_MassRange.npy"))
            smf = np.load(os.path.join(leg_dir, "Figure3_AnalyticalModel_SMF.npy"))
            y = reverse_cumulative(smf) if args.cumulative else smf
            mask = y > 0
            kwargs = dict(color=color, ls=ls, lw=lw, zorder=z, label=label if ls == "-" else None)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            ax.plot(mass[mask], np.log10(y[mask]), **kwargs)

    ax.set_xlabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    ylabel = r"$\log_{10} N(>M_*)\ [\mathrm{Mpc}^{-3}]$" if args.cumulative else r"$\log_{10}\phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$"
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right", frameon=False, fontsize=9, title="solid=py-corrected, dashed=rs-steel")
    ax.set_title(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
