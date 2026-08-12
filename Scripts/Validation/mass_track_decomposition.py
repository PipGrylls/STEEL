"""Reproduce Paper 2 (arXiv 1910.08417, MNRAS 491) Fig. 6/8's real
3-row structure for 3 central mass tracks (log M*,cen(z=0.1) = 11.0,
11.5, 12.0): top row total mass (total/accretion/SFH), middle row
fractional contribution to growth since z=3, bottom row instantaneous
growth-rate ratio. Built entirely from mass_tracks.py's existing dump
output (log_sm_am = "Total", log_sm_insitu = "SFH") -- "Accretion" is
their difference in linear mass, `10**log_sm_am - 10**log_sm_insitu`,
since the abundance-matched track already represents the *total*
stellar mass budget (accretion implicitly included) while the in-situ
track isolates the star-formation-only budget. No new run needed
beyond the extra target-mass track dumps already in mass_tracks.py.

Usage:
    python Scripts/Validation/mass_track_decomposition.py \
        --corrected-glob ".../corrected_fig6_*.csv" \
        --rust-glob ".../rust_fig6_*.csv" \
        --title "Paper 2 Fig. 6 style -- mass track decomposition (PyMorph)" \
        --out Figures/PortValidation/Paper2_Fig6_MassTrackDecomposition.png
"""
import argparse
import glob
import os

import numpy as np


def load_track(path):
    z, log_mh, log_sm_am, log_sm_insitu = [], [], [], []
    with open(path) as f:
        next(f)
        for line in f:
            zi, lm, la, li = line.strip().split(",")
            z.append(float(zi)); log_mh.append(float(lm))
            log_sm_am.append(float(la)); log_sm_insitu.append(float(li))
    order = np.argsort(z)[::-1]  # z descending (early times first)
    z = np.asarray(z)[order]
    log_sm_am = np.asarray(log_sm_am)[order]
    log_sm_insitu = np.asarray(log_sm_insitu)[order]
    return z, log_sm_am, log_sm_insitu


def target_from_path(path):
    base = os.path.basename(path)
    return float(base.rsplit("_", 1)[1].replace(".csv", ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected-glob", required=True)
    ap.add_argument("--rust-glob", required=True)
    ap.add_argument("--title", default="Paper 2 Fig. 6 style -- mass track decomposition (PyMorph)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corrected_files = sorted(glob.glob(args.corrected_glob), key=target_from_path)
    rust_files = sorted(glob.glob(args.rust_glob), key=target_from_path)
    targets = [target_from_path(f) for f in corrected_files]
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, axes = plt.subplots(3, len(targets), figsize=(4.6 * len(targets), 10), sharex="col")

    for col, (target, cfile, rfile) in enumerate(zip(targets, corrected_files, rust_files)):
        color = colors[col % len(colors)]
        for (z, sm_am, sm_insitu), ls, lw, leg in [
            (load_track(cfile), "-", 1.8, "py-corrected"),
            (load_track(rfile), "--", 1.3, "rs-steel"),
        ]:
            # Accretion = Total - SFH in linear mass; near the track's
            # start (z=3) these are equal by construction
            # (mass_tracks.py sets log_sm_insitu[0] = log_sm_am[0]), so
            # the difference is ~0 there -- mask those points out
            # rather than floor them, which would otherwise draw a
            # misleading cliff to log10(mass)=0.
            accretion_lin = 10 ** sm_am - 10 ** sm_insitu
            positive = accretion_lin > 10.0
            log_accretion = np.full_like(accretion_lin, np.nan)
            log_accretion[positive] = np.log10(accretion_lin[positive])

            kwargs = dict(color=color, lw=lw, ls=ls)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            axes[0, col].plot(z, sm_am, **{**kwargs, "label": "Total" if leg == "py-corrected" else None})
            axes[0, col].plot(z, log_accretion, **{**kwargs, "alpha": 0.55,
                               "label": "Accretion" if leg == "py-corrected" else None})
            axes[0, col].plot(z, sm_insitu, **{**kwargs, "alpha": 0.3,
                               "label": "SFH" if leg == "py-corrected" else None})

            m_tot_3 = 10 ** sm_am[0]
            m_tot_0 = 10 ** sm_am[-1]
            denom = max(m_tot_0 - m_tot_3, 1.0)
            frac_acc = (accretion_lin - accretion_lin[0]) / denom
            frac_sfh = (10 ** sm_insitu - 10 ** sm_insitu[0]) / denom
            axes[1, col].plot(z, frac_acc, **kwargs)
            axes[1, col].plot(z, frac_sfh, **{**kwargs, "alpha": 0.4})

            dt = np.diff(z)
            mdot_cen = np.diff(10 ** sm_am) / dt
            mdot_acc = np.diff(accretion_lin) / dt
            mdot_sfh = np.diff(10 ** sm_insitu) / dt
            zc = 0.5 * (z[:-1] + z[1:])
            safe = np.abs(mdot_cen) > 0
            axes[2, col].plot(zc[safe], (mdot_acc[safe] / mdot_cen[safe]), **kwargs)
            axes[2, col].plot(zc[safe], (mdot_sfh[safe] / mdot_cen[safe]), **{**kwargs, "alpha": 0.4})

        axes[0, col].set_title(rf"$M_{{*,\mathrm{{cen}}}}(z{{=}}0)=10^{{{target:g}}}\,M_\odot$", fontsize=10)
        axes[2, col].set_xlabel("redshift")
        axes[2, col].set_xscale("log")
        axes[2, col].set_xlim(0.1, 3)
        axes[2, col].axhline(1, color="0.7", lw=0.7)

    axes[0, 0].set_ylabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    axes[1, 0].set_ylabel("Fractional contribution\nsince z=3")
    axes[2, 0].set_ylabel(r"$\dot M_X / \dot M_\mathrm{cen}$")
    axes[0, 0].legend(loc="lower left", frameon=False, fontsize=8)
    fig.suptitle(args.title + "\n(solid=py-corrected, dashed=rs-steel)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
