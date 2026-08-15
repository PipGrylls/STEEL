"""Reproduce Paper 2 Fig. 2 (surviving/unevolved subhalo mass function,
"USSHMF") for one parent halo mass and f_tdyn=1.0, from the
Surviving_Subhalos_ByParent accumulator both implementations already
produce -- no analytic USHMF reference line (that needs a second,
un-weighted curve from dn_dlnX not built here), so this is the single
f_tdyn=1.0 "USSHMF" line, not the full USHMF-vs-three-USSHMF panel.

py-corrected writes this array to Data/Model/Output/Other/SubHaloes/
(outside the RunParam_ tree LoadData_* reads, and outside what the
three-way runner's run_python() copies out) rather than the standard
output tree, so its path is passed explicitly here rather than
inferred from a run directory the way the other results_*.py scripts
do.

Usage:
    python Scripts/Validation/ushmf_plot.py \
        --py-subhalo-mass /path/py-corrected-1/MultiEpoch_SatHaloMass.npy \
        --py-ushmf Data/Model/Output/Other/SubHaloes/Surviving_Subhalos_ByParent1.0.npy \
        --py-avahalomass /path/py-corrected-1/Figure3_AvaHaloMass.npy \
        --rs-dir /path/rs-steel-1 \
        --target-log-mh 12.80 \
        --out Figures/PortValidation/Paper2_Fig2_USSHMF.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-subhalo-mass", required=True)
    ap.add_argument("--py-ushmf", required=True)
    ap.add_argument("--py-avahalomass", required=True)
    ap.add_argument("--rs-dir", required=True)
    ap.add_argument("--target-log-mh", type=float, default=12.80)
    ap.add_argument("--out", default="Figures/PortValidation/Paper1_Fig2_USSHMF.png")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sub_mass_py = np.load(args.py_subhalo_mass)
    sub_mass_rs = np.load(os.path.join(args.rs_dir, "MultiEpoch_SatHaloMass.npy"))
    ushmf_py = np.load(args.py_ushmf)  # (z, host, subhalo)
    ushmf_rs = np.load(os.path.join(args.rs_dir, "Surviving_Subhalos_ByParent.npy"))

    avh = np.load(args.py_avahalomass)  # (z, host)
    host_bin = int(np.searchsorted(avh[0], args.target_log_mh))

    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    y_py = ushmf_py[0, host_bin, :]
    y_rs = ushmf_rs[0, host_bin, :]
    mask_py, mask_rs = y_py > 0, y_rs > 0
    ax.plot(sub_mass_py[mask_py], np.log10(y_py[mask_py]), color="black", lw=1.8, label="py-corrected")
    ax.plot(sub_mass_rs[mask_rs], np.log10(y_rs[mask_rs]), color="crimson", lw=1.4, ls="--", dashes=(4, 2), label="rs-steel")
    ax.set_xlabel(r"$\log_{10} M_{h,\mathrm{sat}}\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$\log_{10}$ USSHMF $[\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        rf"Paper 1 Fig. 2 style -- surviving SHMF, $\log M_{{h,\mathrm{{cent}}}}={avh[0][host_bin]:.2f}$, "
        r"$f_{t\mathrm{dyn}}=1.0$ (G19)",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
