"""Reproduce Paper 1 Fig. 4: merger rate per Gyr at fixed halo mass, as
a function of redshift, for several present-day (z~0.1) halo masses.

Mergers_Accretion_History[z_bin, j, sat_mass] accumulates, per code
comment ("N dex-1 per halo"), the number of satellites merging into
host-mass-bin j's growth track at accretion redshift z_bin (j indexes
a fixed z~0.1-anchored halo growth history throughout the array, not a
redshift-dependent grid position -- confirmed against
Mergers_AvaHaloMass, which changes smoothly with z at fixed j).
Summing over the satellite-mass axis gives total mergers (all mass
ratios, not just major mergers -- unlike a strict Fakhouri+2010
comparison) per halo per accretion-redshift bin; dividing by the
step's cosmic-time width converts this to a rate per Gyr.

No Fakhouri+2010 analytic-fit band is drawn (external-data overlay,
out of this reproduction's scope, same as SDSS/Illustris/Mundy
elsewhere).

Usage:
    python Scripts/Validation/merger_rate_plot.py \
        --py-corrected /path/to/py-corrected-1 \
        --rs-steel /path/to/rs-steel-1 \
        --out Figures/PortValidation/Paper1_Fig4_MergerRate.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from colossus.cosmology import cosmology

cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()

LEG_STYLE = {
    "py-corrected": dict(color="black", lw=1.8, ls="-", zorder=3, label="py-corrected"),
    "rs-steel": dict(color="crimson", lw=1.4, ls="--", dashes=(4, 2), zorder=4, label="rs-steel"),
}
COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", required=True)
    ap.add_argument("--rs-steel", required=True)
    ap.add_argument("--targets", type=float, nargs="+", default=[11.0, 12.0, 13.0, 14.0])
    ap.add_argument("--out", default="Figures/PortValidation/Paper2_Fig4_MergerRate.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for name, run_dir, ls in [("py-corrected", args.py_corrected, "-"), ("rs-steel", args.rs_steel, "--")]:
        z = np.load(os.path.join(run_dir, "Mergers_z.npy"))
        avh = np.load(os.path.join(run_dir, "Mergers_AvaHaloMass.npy"))
        mah = np.load(os.path.join(run_dir, "Mergers_Accretion_History.npy"))
        t = Cosmo.age(z)
        dt = np.abs(np.diff(t))
        dt = np.append(dt, dt[-1])
        merger_count = np.nansum(mah, axis=-1)  # (z, host)

        for i, target in enumerate(args.targets):
            j = int(np.searchsorted(avh[0], target))
            rate = merger_count[:, j] / dt
            mask = rate > 0
            kwargs = dict(color=COLORS[i % len(COLORS)], lw=1.8 if ls == "-" else 1.3, ls=ls)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            label = rf"$\log M_{{h}}(z{{=}}0.1){{=}}{avh[0, j]:.1f}$" if ls == "-" else None
            ax.plot(z[mask], rate[mask], label=label, **kwargs)

    ax.set_xlabel("redshift")
    ax.set_ylabel(r"merger rate $[\mathrm{Gyr}^{-1}]$ per halo (all mass ratios)")
    ax.set_yscale("log")
    ax.legend(loc="upper left", frameon=False, fontsize=9, title="solid=py-corrected, dashed=rs-steel")
    ax.set_title("Paper 2 Fig. 4 style -- merger rate vs. redshift, fixed halo mass (G19)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
