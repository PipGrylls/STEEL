"""Reproduce Paper 2 (arXiv 1910.08417, MNRAS 491) Fig. 11: predicted
fraction of ellipticals vs. stellar mass at 3 redshifts, from the
cumulative fraction of central mass tracks that have had a major
merger (satellite/central stellar mass ratio > 0.25) since z=3.

This was previously marked infeasible based on `STEEL.py`'s
`P_Elliptical` array, which is genuinely dead (allocated, never
written). The real computation described in the paper's Section 5.3
is a *post-processing* one over central mass tracks and the merger
accretion history -- unrelated to that array -- using exactly the
`Mergers_Accretion_History` cube merger_rate_plot.py already reads for
Paper 1 Fig. 4, integrated differently.

Method (reconstructed from the paper text; the paper doesn't give the
exact formula, so this is a documented interpretation, not a verified
match):
  1. For each z=0.1-anchored halo growth track `j`
     (`Mergers_AvaHaloMass[z_bin, j]`), the central's stellar mass at
     each step is `SMHM(AvaHaloMass[z_bin, j], z_bin)`, evaluated live
     via `Functions.DarkMatterToStellarMass` (same call pattern as
     mass_tracks.py) rather than reconstructed from saved output --
     nothing in the saved arrays records the central's own stellar
     mass at each step, only the satellites'.
  2. `Mergers_Accretion_History[z_bin, j, :]` gives the merging
     satellite stellar-mass histogram (a *count* for that step, not
     the per-Gyr rate Fig. 4 derives from the same cube); summing the
     bins above `0.25 * central_mass(z_bin)` gives the expected number
     of major mergers in that step.
  3. Summing over z_bin from z=3 down to the redshift of interest
     gives the expected number of major mergers N_major(z) so far.
  4. Converting an expected count to "fraction with >=1 major merger"
     needs a distributional assumption the paper doesn't spell out;
     this script uses the Poisson form f = 1 - exp(-N_major), the
     natural choice for STEEL's expectation-value statistical
     accretion history.

Both legs' avh (halo-mass) arrays are converted to central stellar
mass via the same live Python SMHM call -- SMHM evaluation isn't the
object of the py-vs-rust comparison here, the merger accretion history
(which does differ, and is read from each leg's own output) is.

Usage:
    python Scripts/Validation/elliptical_fraction_plot.py \
        --py-corrected /path/to/py-corrected-1 \
        --rs-steel /path/to/rs-steel-1 \
        --repo-root . \
        --out Figures/PortValidation/Paper2_Fig11_EllipticalFraction.png
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

MASS_RATIO_CUT = 0.25
TARGET_REDSHIFTS = [0.1, 1.0, 2.0]
Z_COLOR = {0.1: "#4C72B0", 1.0: "#DD8452", 2.0: "#55A868"}
Z_STYLE = {0.1: "-", 1.0: "--", 2.0: "-."}


def _import_functions(repo_root):
    repo_root = os.path.abspath(repo_root)
    for mod in list(sys.modules):
        if mod == "Functions" or mod.startswith("Functions."):
            del sys.modules[mod]
    sys.path.insert(0, repo_root)
    try:
        from Functions import Functions as F
    finally:
        sys.path.remove(repo_root)
    return F


def _abn_mtch_g18():
    return {
        "Behroozi13": False, "Behroozi18": False, "B18c": False, "B18t": False,
        "G18": True, "G18_notSE": False, "G19_SE": False, "G19_cMod": False,
        "Lorenzo18": False, "Moster": False, "Moster10": False, "RP17": False,
        "Illustris": False, "z_Evo": True, "Scatter": 0.15,
        "Override_0": False, "Override_z": False,
        "Override": {"M10": 11.95, "SHMnorm10": 0.032, "beta10": 1.61, "gamma10": 0.54,
                      "M11": 0.4, "SHMnorm11": -0.02, "beta11": -0.6, "gamma11": -0.1},
        "PFT": False, "HMevo": False, "HMevo_param": None,
    }


def elliptical_fraction(run_dir, F, params, target_z, n_targets=10):
    z = np.load(os.path.join(run_dir, "Mergers_z.npy"))
    avh = np.load(os.path.join(run_dir, "Mergers_AvaHaloMass.npy"))
    sat_mass = np.load(os.path.join(run_dir, "Mergers_Surviving_Sat_SMF_MassRange.npy"))
    mah = np.load(os.path.join(run_dir, "Mergers_Accretion_History.npy"))

    order = np.argsort(z)[::-1]  # z descending (early times first)
    z, avh, mah = z[order], avh[order], mah[order]

    n_steps, n_j = avh.shape
    central_sm = np.full((n_steps, n_j), np.nan)
    for i in range(n_steps):
        central_sm[i] = F.DarkMatterToStellarMass(avh[i], float(z[i]), {"AbnMtch": params}, ScatterOn=False)

    target_i = int(np.clip(np.searchsorted(-z, -target_z), 0, n_steps - 1))
    n_major = np.zeros(n_j)
    for i in range(n_steps):
        if z[i] < target_z:
            break
        cm = central_sm[i]
        valid = np.isfinite(cm)
        cut = MASS_RATIO_CUT * (10.0 ** cm[valid])
        cut_bins = np.searchsorted(10.0 ** sat_mass, cut)
        for k, jidx in enumerate(np.nonzero(valid)[0]):
            n_major[jidx] += np.nansum(mah[i, jidx, cut_bins[k]:])

    log_sm_target = central_sm[target_i]
    frac = 1.0 - np.exp(-n_major)
    mask = np.isfinite(log_sm_target)
    log_sm_target, frac = log_sm_target[mask], frac[mask]
    order2 = np.argsort(log_sm_target)
    log_sm_target, frac = log_sm_target[order2], frac[order2]

    lo, hi = np.nanpercentile(log_sm_target, [2, 98])
    bin_edges = np.linspace(lo, hi, n_targets + 1)
    out_x, out_y = [], []
    for b in range(n_targets):
        sel = (log_sm_target >= bin_edges[b]) & (log_sm_target < bin_edges[b + 1])
        if sel.sum() == 0:
            continue
        out_x.append(0.5 * (bin_edges[b] + bin_edges[b + 1]))
        out_y.append(np.nanmean(frac[sel]))
    return np.array(out_x), np.array(out_y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--py-corrected", required=True)
    ap.add_argument("--rs-steel", required=True)
    ap.add_argument("--repo-root", default=".")
    ap.add_argument("--out", default="Figures/PortValidation/Paper2_Fig11_EllipticalFraction.png")
    args = ap.parse_args()

    F = _import_functions(args.repo_root)
    params = _abn_mtch_g18()

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for target_z in TARGET_REDSHIFTS:
        color = Z_COLOR[target_z]
        for run_dir, lw, ls_extra, label in [
            (args.py_corrected, 1.8, "", f"z={target_z:g}"),
            (args.rs_steel, 1.3, "dashed", None),
        ]:
            x, y = elliptical_fraction(run_dir, F, params, target_z)
            if ls_extra == "dashed":
                ax.plot(x, y, color=color, lw=lw, ls="--", dashes=(4, 2))
            else:
                ax.plot(x, y, color=color, lw=lw, ls=Z_STYLE[target_z], label=label)

    ax.set_xlabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$f_\mathrm{elliptical}$")
    ax.set_ylim(0, 1.0)
    ax.legend(loc="upper left", frameon=False, fontsize=9,
               title="solid=py-corrected, dashed=rs-steel")
    ax.set_title(
        "Paper 2 Fig. 11 style -- elliptical fraction from major-merger history (G18)\n"
        "(f = 1-exp(-N_major), an interpretation -- see script docstring)",
        fontsize=9,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=200)
    plt.close(fig)
    print("wrote:", args.out)


if __name__ == "__main__":
    main()
