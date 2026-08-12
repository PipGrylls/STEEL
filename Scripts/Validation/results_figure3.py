"""Reproduce Paper 2 Figure 3 (the local, z=0.1 satellite stellar mass
function) for the three-way port validation: py-corrected vs rs-steel,
deterministic mode, published grid -- the strong numerical-fidelity
claim from docs/VALIDATION.md Sec. 1, turned into a picture instead of
a table.

py-as-is has no deterministic mode (GetGasMass scatters
unconditionally -- PORT-FIX A7), so it cannot appear on this panel on
equal terms with the other two; it is not shown here rather than mixed
in on a different footing. See Scripts/Validation/paper2_figures.py
for figures where all three legs are directly comparable.

Two panels: the differential SMF (log Phi vs log M*, matching the
paper's own axes) and the reverse-cumulative N(>M*) (insensitive to
bin-edge sliding, the metric docs/VALIDATION.md actually quotes
agreement numbers for -- see Scripts/Validation/three_way.py
::cumulative_rows for why the per-bin view is not the right
deterministic-mode metric on its own).

Usage (from repo root):
    python Scripts/Validation/results_figure3.py \
        --py-corrected /path/to/py-corrected-1 \
        --rs-steel /path/to/rs-steel-1 \
        --out Figures/PortValidation/Paper2_Fig3_SatelliteSMF.png
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_smf(run_dir):
    smf = np.load(os.path.join(run_dir, "Figure3_AnalyticalModel_SMF.npy"))
    mass = np.load(os.path.join(run_dir, "Figure3_Surviving_Sat_SMF_MassRange.npy"))
    return np.asarray(mass, dtype=float), np.asarray(smf, dtype=float)


def reverse_cumulative(smf):
    return np.flip(np.nancumsum(np.flip(smf)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", required=True)
    ap.add_argument("--rs-steel", required=True)
    ap.add_argument("--out", default="Figures/PortValidation/Paper2_Fig3_SatelliteSMF.png")
    args = ap.parse_args()

    mass_c, smf_c = load_smf(args.py_corrected)
    mass_r, smf_r = load_smf(args.rs_steel)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    LEG_STYLE = {
        "py-corrected": dict(color="black", lw=1.8, ls="-", zorder=3, label="py-corrected"),
        "rs-steel": dict(color="crimson", lw=1.4, ls="--", dashes=(4, 2), zorder=4, label="rs-steel"),
    }

    fig, (ax_diff, ax_cum) = plt.subplots(1, 2, figsize=(11.0, 5.0))

    mask_c = smf_c > 0
    mask_r = smf_r > 0
    ax_diff.plot(mass_c[mask_c], np.log10(smf_c[mask_c]), **LEG_STYLE["py-corrected"])
    ax_diff.plot(mass_r[mask_r], np.log10(smf_r[mask_r]), **LEG_STYLE["rs-steel"])
    ax_diff.set_xlabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    ax_diff.set_ylabel(r"$\log_{10}\phi\ [\mathrm{Mpc}^{-3}\,\mathrm{dex}^{-1}]$")
    ax_diff.set_title("differential", fontsize=10)
    ax_diff.legend(loc="upper right", frameon=False, fontsize=9)

    cum_c = reverse_cumulative(smf_c)
    cum_r = reverse_cumulative(smf_r)
    mask = (cum_c > 0) | (cum_r > 0)
    ax_cum.plot(mass_c[mask], np.log10(np.where(cum_c[mask] > 0, cum_c[mask], np.nan)), **LEG_STYLE["py-corrected"])
    ax_cum.plot(mass_r[mask], np.log10(np.where(cum_r[mask] > 0, cum_r[mask], np.nan)), **LEG_STYLE["rs-steel"])
    ax_cum.set_xlabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    ax_cum.set_ylabel(r"$\log_{10} N(>M_*)\ [\mathrm{Mpc}^{-3}]$")
    ax_cum.set_title("reverse-cumulative", fontsize=10)

    fig.suptitle(
        "Paper 2 Fig. 3 -- local (z=0.1) satellite SMF (G19), deterministic, published grid",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(args.out, dpi=200)
    plt.close(fig)

    finite = mask_c & mask_r & (mass_c == mass_r) if mass_c.shape == mass_r.shape else mask
    integral_ratio = float(np.nansum(smf_r) / np.nansum(smf_c)) if np.nansum(smf_c) else float("nan")
    print(f"integral ratio (rs-steel / py-corrected): {integral_ratio:.4f}")
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
