"""Reproduce Paper 2 Fig. 3 (total surviving subhalo mass function,
"total USSHMF") at f_tdyn=1.0, z~0.1 -- the HMF-integrated counterpart
to Fig. 2's one-parent-halo USSHMF (see ushmf_plot.py). No analytic
total-USHMF reference line (needs a second HMF-weighted dn_dlnX
integral not built here), so this is the f_tdyn=1.0 total-USSHMF line
only, not the full USHMF-vs-three-f_tdyn panel.

py-corrected writes this array as Data/Model/Output/Other/SubHaloes/
Surviving_Subhalos<f_tdyn>.dat (whitespace-separated, first row/column
are the subhalo-mass grid and redshift grid respectively) rather than
into the RunParam_ tree, so its path is passed explicitly.

Usage:
    python Scripts/Validation/total_ushmf_plot.py \
        --py-dat Data/Model/Output/Other/SubHaloes/Surviving_Subhalos1.0.dat \
        --rs-dir /path/to/rs-steel-1 \
        --out Figures/PortValidation/Paper2_Fig3_TotalUSSHMF.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-dat", required=True)
    ap.add_argument("--rs-dir", required=True)
    ap.add_argument("--out", default="Figures/PortValidation/Paper1_Fig3_TotalUSSHMF.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    raw = np.loadtxt(args.py_dat)
    mass_py = raw[0, 1:]
    data_py = raw[1:, 1:]  # (z, subhalo)

    mass_rs = np.load(os.path.join(args.rs_dir, "MultiEpoch_SatHaloMass.npy"))
    data_rs = np.load(os.path.join(args.rs_dir, "Surviving_Subhalos.npy"))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    y_py, y_rs = data_py[0], data_rs[0]
    mask_py, mask_rs = y_py > 0, y_rs > 0
    ax.plot(mass_py[mask_py], np.log10(y_py[mask_py]), color="black", lw=1.8, label="py-corrected")
    ax.plot(mass_rs[mask_rs], np.log10(y_rs[mask_rs]), color="crimson", lw=1.4, ls="--", dashes=(4, 2), label="rs-steel")
    ax.set_xlabel(r"$\log_{10} M_{h,\mathrm{sat}}\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$\log_{10}$ total USSHMF $[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(r"Paper 1 Fig. 3 style -- total surviving SHMF, $f_{t\mathrm{dyn}}=1.0$, z=0.1 (G19)", fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
