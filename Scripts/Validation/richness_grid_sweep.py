"""Reproduce the real 3x2-panel structure shared by Paper 1 (arXiv
1812.00015) Figs. 9, 11, and 15: satellite number density (top row)
and fractional distribution (bottom row, eq. 16: F(dMh) = N(>x)|dMh /
N(>x)) vs. parent halo mass, for 3 increasing stellar-mass cuts
(columns), overlaid across several STEEL configs. One colour per
config, solid/dashed for py-corrected/rs-steel, same convention as
richness_config_sweep.py (which only builds a single top-row panel).

Usage:
    python Scripts/Validation/richness_grid_sweep.py \
        --run "f_tdyn=1.0+evo:/path/py-corrected-1:/path/rs-steel-1" \
        --run "f_tdyn=inf+evo:/path/.../py-corrected-1:/path/.../rs-steel-1" \
        --sm-cuts 10.0 10.5 11.0 \
        --out Figures/PortValidation/Paper1_Fig9_Grid.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label:py_corrected_dir:rs_steel_dir")
    ap.add_argument("--sm-cuts", type=float, nargs=3, default=[10.0, 10.5, 11.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="satellite distributions, config sweep (G19)")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), sharex="col")

    for i, spec in enumerate(args.run):
        label, py_dir, rs_dir = spec.split(":", 2)
        color = COLORS[i % len(COLORS)]
        for leg_dir, ls, lw, zorder in [(py_dir, "-", 1.8, 3), (rs_dir, "--", 1.4, 4)]:
            host_mass = np.load(os.path.join(leg_dir, "Raw_Richness_AvaHaloMass.npy"))[0]
            sat_mass = np.load(os.path.join(leg_dir, "Raw_Richness_Surviving_Sat_SMF_MassRange.npy"))
            cube = np.load(os.path.join(leg_dir, "Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy"))
            for col, cut in enumerate(args.sm_cuts):
                cut_bin = np.searchsorted(sat_mass, cut)
                n_above = np.nansum(cube[0, :, cut_bin:], axis=1)
                mask = n_above > 0
                kwargs = dict(color=color, ls=ls, lw=lw, zorder=zorder, label=label if ls == "-" else None)
                if ls == "--":
                    kwargs["dashes"] = (4, 2)
                axes[0, col].plot(host_mass[mask], np.log10(n_above[mask]), **kwargs)

                total = np.nansum(n_above)
                frac = n_above / total if total > 0 else n_above
                axes[1, col].plot(host_mass[mask], frac[mask], **{**kwargs, "label": None})

    for col, cut in enumerate(args.sm_cuts):
        axes[0, col].set_title(rf"$M_{{*,\mathrm{{sat}}}}>10^{{{cut:g}}}$", fontsize=10)
        axes[1, col].set_xlabel(r"$\log_{10} M_{h,\mathrm{cent}}\ [\mathrm{M}_\odot]$")
    axes[0, 0].set_ylabel(r"$\log_{10}\phi\ [\mathrm{Mpc}^{-3}]$")
    axes[1, 0].set_ylabel("Fraction")
    axes[0, 0].legend(loc="upper right", frameon=False, fontsize=8, title="solid=py-corrected, dashed=rs-steel")
    fig.suptitle(args.title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
