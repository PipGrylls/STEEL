"""Reproduce Paper 2 Fig. 15's content (also Fig. 9/11's shape):
satellite number density above a stellar mass cut vs. parent halo
mass at z~0.1, overlaid across multiple STEEL configs -- one colour
per config, solid/dashed for py-corrected/rs-steel, same convention
as results_config_sweep.py. Reuses already-completed runs.

Usage:
    python Scripts/Validation/richness_config_sweep.py \
        --run frozen:/path/frozen/py-corrected-1:/path/frozen/rs-steel-1 \
        --run "SF+stripping:/path/.../py-corrected-1:/path/.../rs-steel-1" \
        --out Figures/PortValidation/Paper2_Fig15_ConfigSweep.png
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
    ap.add_argument("--sm-cut", type=float, default=10.0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, spec in enumerate(args.run):
        label, py_dir, rs_dir = spec.split(":", 2)
        color = COLORS[i % len(COLORS)]
        for leg_dir, ls, lw, z in [(py_dir, "-", 1.8, 3), (rs_dir, "--", 1.4, 4)]:
            z_arr = np.load(os.path.join(leg_dir, "Raw_Richness_Highz_z.npy"))
            host_mass = np.load(os.path.join(leg_dir, "Raw_Richness_AvaHaloMass.npy"))
            sat_mass = np.load(os.path.join(leg_dir, "Raw_Richness_Surviving_Sat_SMF_MassRange.npy"))
            cube = np.load(os.path.join(leg_dir, "Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy"))
            cut_bin = np.searchsorted(sat_mass, args.sm_cut)
            n_above = np.nansum(cube[0, :, cut_bin:], axis=1)
            mask = n_above > 0
            kwargs = dict(color=color, ls=ls, lw=lw, zorder=z, label=label if ls == "-" else None)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            ax.plot(host_mass[0][mask], np.log10(n_above[mask]), **kwargs)

    ax.set_xlabel(r"$\log_{10} M_{h,\mathrm{cent}}\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$\log_{10} N(>10^{%.0f}\,\mathrm{M}_\odot)\ [\mathrm{Mpc}^{-3}]$" % args.sm_cut)
    ax.legend(loc="upper right", frameon=False, fontsize=9, title="solid=py-corrected, dashed=rs-steel")
    ax.set_title("Paper 2 Fig. 15 style -- satellites per parent halo, config sweep (G19)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
