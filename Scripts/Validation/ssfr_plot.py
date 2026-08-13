"""Reproduce the satellite side of Paper 2 (arXiv 1910.08417, MNRAS
491) Fig. 9: sSFR distributions in the paper's exact 3 mass ranges
(10-10.5, 10.5-11.3, 11.3-12.5), from the Satellite_sSFR 2D histogram
(satellite mass x sSFR) both implementations already produce, summed
over the satellite-mass axis within each range (not a single
nearest-bin snapshot). The published figure also shows a
post-processed central-galaxy sSFR distribution (dashed black line)
from the dynamical-quenching model; that side needs
CentralPostprocessing analysis this script doesn't run, so this is
satellites only.

Must run in STOCHASTIC mode (scatter on, several seeds averaged), not
deterministic. This isn't optional the way it is for the other
figures in this validation: the real Fig. 9 (see the published PDF)
is a smooth, markedly bimodal distribution -- a quenched peak near
sSFR ~ 1e-12/yr and a star-forming peak near sSFR ~ 1e-10/yr -- built
from many Monte Carlo satellites with Gaussian scatter on SFR
(Functions_c.pyx's `apply scatter to SFR`, 0.3 dex). In deterministic
mode every satellite in a given mass/host bin follows the *same*
SFR trajectory, so once enough of them hit the gas-depletion floor
(SFR clamped to sSFR=1e-12/yr exactly, Functions_c.pyx ~L186-217)
essentially the entire population piles onto that one discrete bin
-- not a port defect, just the wrong mode for a figure whose shape
depends on scatter to exist at all. (Confirmed by comparison: a
deterministic-mode run of this exact config puts ~80% of one mass
bin's satellites in the single sSFR=-12.0 bin with the next-largest
bin 3 orders of magnitude smaller; an 8-seed stochastic ensemble of
the same config recovers the expected two-peak shape.)

Usage (from repo root), one or more seed directories per leg:
    python Scripts/Validation/ssfr_plot.py \
        --py-corrected /path/to/py-corrected-1 /path/to/py-corrected-2 ... \
        --rs-steel /path/to/rs-steel-1 /path/to/rs-steel-2 ... \
        --out Figures/PortValidation/Paper2_Fig9_sSFR.png

Typically produced via Scripts/Validation/three_way.py --mode
stochastic --seeds 1 2 3 4 5 6 7 8 --run 1.0,True,True,True,CE,G19_SE
--runfile rust/runfiles/published/p2-evo-sf-strip.toml, then pointing
this script at the resulting py-corrected-*/rs-steel-* directories.
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


MASS_RANGES = [(10.0, 10.5), (10.5, 11.3), (11.3, 12.5)]


def load_ensemble_row(run_dirs, lo, hi):
    """Mean sSFR histogram (summed over the satellite-mass range) across seeds."""
    rows = []
    ssfr_range = sat_mass = None
    for run_dir in run_dirs:
        sat_mass = np.load(os.path.join(run_dir, "sSFR_Surviving_Sat_SMF_MassRange.npy"))
        ssfr_range = np.load(os.path.join(run_dir, "sSFR_Range.npy"))
        data = np.load(os.path.join(run_dir, "Satellite_sSFR.npy"))  # (sat_mass, sSFR)
        i_lo = int(np.searchsorted(sat_mass, lo))
        i_hi = int(np.searchsorted(sat_mass, hi))
        rows.append(np.nansum(data[i_lo:i_hi], axis=0))
    return ssfr_range, np.nanmean(np.stack(rows), axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", nargs="+", required=True, help="one or more seed directories")
    ap.add_argument("--rs-steel", nargs="+", required=True, help="one or more seed directories")
    ap.add_argument("--mass-ranges", type=float, nargs="+", default=None,
                     help="flat list lo1 hi1 lo2 hi2 ...; defaults to the paper's 10-10.5/10.5-11.3/11.3-12.5")
    ap.add_argument("--out", default="Figures/PortValidation/Paper2_Fig9_sSFR.png")
    args = ap.parse_args()

    if args.mass_ranges:
        flat = args.mass_ranges
        ranges = [(flat[i], flat[i + 1]) for i in range(0, len(flat), 2)]
    else:
        ranges = MASS_RANGES

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, axes = plt.subplots(1, len(ranges), figsize=(4.2 * len(ranges), 4.6), sharey=True)

    for name, run_dirs in [("py-corrected", args.py_corrected), ("rs-steel", args.rs_steel)]:
        for ax, (lo, hi) in zip(axes, ranges):
            ssfr_range, row = load_ensemble_row(run_dirs, lo, hi)
            bin_width = ssfr_range[1] - ssfr_range[0]
            # Step histogram, not a connect-the-nonzero-dots line: with
            # scatter on this is a real, mostly-populated histogram, but
            # the zero bins between sparse tails still need to read as
            # zero rather than be skipped over by a straight line.
            ax.step(ssfr_range, row, where="mid", **LEG_STYLE[name])
            ax.set_title(rf"${lo:g}<\log M_{{*,\mathrm{{sat}}}}<{hi:g}$", fontsize=10)
            ax.set_xlabel(r"$\log_{10}\mathrm{sSFR}\ [\mathrm{yr}^{-1}]$")
            ax.set_xlim(-13, -9)

    axes[0].set_ylabel(r"$N\ [\mathrm{Mpc}^{-3}\,\mathrm{h}^3]$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
    n_seeds = len(args.py_corrected)
    fig.suptitle(
        f"Paper 2 Fig. 9 style -- satellite sSFR distribution (G19), "
        f"stochastic, {n_seeds}-seed ensemble mean",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
