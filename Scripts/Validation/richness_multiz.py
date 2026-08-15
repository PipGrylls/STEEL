"""Reproduce Paper 1 Fig. 5: satellite number density above a stellar
mass cut, as a function of parent halo mass, at several redshifts
(colour-coded, matching the paper's own convention) -- from the
Raw_Richness_Surviving_Sat_SMF_Weighting_highz accumulator, a full
(z, host, sat_mass) cube (unlike Figure10_AnalyticalModel_SMF, which
only has a single z=0.1 slice saved). Reuses an already-completed run,
no new simulation needed. One mass cut (M*>10^10, matching
results_from_run.py::plot_satellite_distribution's default), not
Paper 1's full comparison to SDSS/Wang+2016/Wen&Han+2018/Illustris.

Usage:
    python Scripts/Validation/richness_multiz.py \
        --py-corrected /path/to/py-corrected-1 \
        --rs-steel /path/to/rs-steel-1 \
        --out Figures/PortValidation/Paper1_Fig5_SatelliteDistribution_MultiZ.png
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", required=True)
    ap.add_argument("--rs-steel", required=True)
    ap.add_argument("--out", default="Figures/PortValidation/Paper2_Fig5_SatelliteDistribution_MultiZ.png")
    ap.add_argument("--sm-cut", type=float, default=10.0)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    targets = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0]
    cmap = matplotlib.colormaps.get_cmap("viridis").resampled(len(targets))

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for name, run_dir, ls, lw in [("py-corrected", args.py_corrected, "-", 1.8), ("rs-steel", args.rs_steel, "--", 1.3)]:
        z = np.load(os.path.join(run_dir, "Raw_Richness_Highz_z.npy"))
        host_mass = np.load(os.path.join(run_dir, "Raw_Richness_AvaHaloMass.npy"))
        sat_mass = np.load(os.path.join(run_dir, "Raw_Richness_Surviving_Sat_SMF_MassRange.npy"))
        cube = np.load(os.path.join(run_dir, "Raw_Richness_Surviving_Sat_SMF_Weighting_highz.npy"))
        cut_bin = np.searchsorted(sat_mass, args.sm_cut)

        for i, target in enumerate(targets):
            zi = np.searchsorted(z, target)
            n_above = np.nansum(cube[zi, :, cut_bin:], axis=1)
            mask = n_above > 0
            kwargs = dict(color=cmap(i), lw=lw, ls=ls)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            label = f"z={z[zi]:.1f}" if ls == "-" else None
            ax.plot(host_mass[zi][mask], np.log10(n_above[mask]), label=label, **kwargs)

    ax.set_xlabel(r"$\log_{10} M_{h,\mathrm{cent}}\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$\log_{10} N(>10^{%.0f}\,\mathrm{M}_\odot)\ [\mathrm{Mpc}^{-3}]$" % args.sm_cut)
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=2, title="solid=py-corrected, dashed=rs-steel")
    ax.set_title("Paper 2 Fig. 5 style -- satellites per parent halo, multi-z (G19)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
