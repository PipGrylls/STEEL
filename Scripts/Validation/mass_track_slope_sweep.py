"""Reproduce Paper 3 Fig. 7 (and the underlying comparison Fig. 3
discusses): central-galaxy mass tracks across several values of the
HMevo preset's high-mass-slope-evolution parameter (gamma11) -- the
family Paper 3 uses to show how the SMHM relation's evolution shapes
the accretion-vs-SFR mass budget. Reuses mass_tracks.py's dump/combine
machinery (same z=0 anchor, same py-as-is-absent caveat, same G3
reason) at one target mass, one line per gamma11, abundance-matching
only (the in-situ-SFR-only strand is nearly gamma-independent at this
target mass and clutters the comparison).

Usage:
    python Scripts/Validation/mass_track_slope_sweep.py \
        --run "gamma=0.1:/path/g0.1_corrected.csv:/path/g0.1_rust.csv" \
        --run "gamma=0.2:/path/g0.2_corrected.csv:/path/g0.2_rust.csv" \
        --out Figures/PortValidation/Paper3_Fig7_SlopeEvolutionSweep.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label:corrected_csv:rust_csv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="11.5")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    for i, spec in enumerate(args.run):
        label, corrected_csv, rust_csv = spec.split(":", 2)
        color = COLORS[i % len(COLORS)]
        for csv, ls, lw, z in [(corrected_csv, "-", 1.8, 3), (rust_csv, "--", 1.4, 4)]:
            df = pd.read_csv(csv).sort_values("z", ascending=False)
            kwargs = dict(color=color, ls=ls, lw=lw, zorder=z, label=label if ls == "-" else None)
            if ls == "--":
                kwargs["dashes"] = (4, 2)
            ax.plot(df.z, df.log_sm_am, **kwargs)

    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$ (abundance matching)")
    ax.invert_xaxis()
    ax.legend(
        loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=9,
        title="solid=py-corrected\ndashed=rs-steel",
    )
    ax.set_title(
        f"Paper 3 Fig. 7 style -- halo mass track, HMevo slope-evolution sweep\n"
        f"target log M*(z=0)={args.target}, py-as-is absent (G3)",
        fontsize=9.5,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
