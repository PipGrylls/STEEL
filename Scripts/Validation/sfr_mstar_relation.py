"""Reproduce Paper 2 (arXiv 1910.08417, MNRAS 491) Fig. 7/10's content:
the star-forming main-sequence
SFR-M* relation at several redshifts, built from the same central mass
tracks mass_tracks.py computes (one track per target z=0 stellar mass,
in-situ-only SFR history, G19_DPL). Each track's in-situ mass at a
given z pairs with the SFR the model assigns it there -- since
Starformation_Centrals/CentralEvolution never quenches these tracks
(see mass_tracks.py's module docstring), that SFR is always the
G19_DPL closed-form value evaluated at that track's (log_sm_insitu, z),
transcribed here identically to mass_tracks.py's dump() loop and to
DoublePowerLawSfr::central() on the Rust side. Sweeping several target
masses at fixed z traces out the main-sequence relation as an actual
simulation-track output (path-dependent through each halo's growth
history), not just the bare closed-form function evaluated on an
arbitrary M* grid.

py-as-is omitted for the same reason as mass_tracks.py (G3: Halogrowth
needs a compiled getPWGH side-channel it doesn't provide standalone).

Usage:
    python Scripts/Validation/sfr_mstar_relation.py \
        --corrected-glob "/tmp/sfr_ms/corrected_*.csv" \
        --rust-glob "/tmp/sfr_ms/rust_*.csv" \
        --redshifts 0.0 1.0 2.0 \
        --out Figures/PortValidation/Paper1_Fig7_SFR_Mstar.png
"""
import argparse
import glob
import os

import numpy as np


def dpl_sfr(log_sm, z):
    m_n = 10.65 + 0.33 * z - 0.08 * z**2
    norm = 10 ** (0.69 + 0.71 * z - 0.088 * z**2)
    alpha = 1.0 - 0.022 * z + 0.009 * z**2
    beta = 1.8 - 1.0 * z + 0.1 * z**2
    return 2 * norm / (10 ** (-alpha * (log_sm - m_n)) + 10 ** (beta * (log_sm - m_n)))


def track_point_at_z(csv_path, target_z):
    z, log_sm = [], []
    with open(csv_path) as f:
        next(f)
        for line in f:
            zi, _log_mh, _log_sm_am, log_sm_insitu = line.strip().split(",")
            z.append(float(zi))
            log_sm.append(float(log_sm_insitu))
    z = np.asarray(z)
    log_sm = np.asarray(log_sm)
    i = int(np.argmin(np.abs(z - target_z)))
    return log_sm[i], z[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corrected-glob", required=True)
    ap.add_argument("--rust-glob", required=True)
    ap.add_argument("--redshifts", type=float, nargs="+", default=[0.0, 1.0, 2.0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="Paper 2 Fig. 7 style -- SFR-M* main sequence from central tracks (G19_DPL)")
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]

    corrected_files = sorted(glob.glob(args.corrected_glob))
    rust_files = sorted(glob.glob(args.rust_glob))
    assert corrected_files and rust_files, "no track CSVs matched"

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for zi_idx, target_z in enumerate(args.redshifts):
        color = colors[zi_idx % len(colors)]
        for files, ls, lw, label in [
            (corrected_files, "-", 1.8, "py-corrected"),
            (rust_files, "--", 1.3, "rs-steel"),
        ]:
            pts = [track_point_at_z(f, target_z) for f in files]
            pts = [(sm, z) for sm, z in pts if abs(z - target_z) < 0.3]
            if not pts:
                continue
            pts.sort()
            log_sm = np.array([p[0] for p in pts])
            log_sfr = np.log10(dpl_sfr(log_sm, target_z))
            kwargs = dict(color=color, lw=lw, ls=ls)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            lbl = f"z={target_z:g}" if ls == "-" else None
            ax.plot(log_sm, log_sfr, marker="o", ms=3, label=lbl, **kwargs)

    ax.set_xlabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$\log_{10}\mathrm{SFR}\ [\mathrm{M}_\odot\,\mathrm{yr}^{-1}]$")
    ax.legend(loc="upper left", frameon=False, fontsize=9, title="solid=py-corrected\ndashed=rs-steel")
    ax.set_title(args.title, fontsize=10)
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
