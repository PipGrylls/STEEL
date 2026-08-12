"""Reproduce the "mass track" figures shared by Paper 1 (Figs. 6 & 8)
and Paper 3 (Fig. 7): for a target z=0 central stellar mass, the
halo's van den Bosch (2014) growth history back to z=3, the
abundance-matching mass M*_AM(z) = SMHM(Mh(z), z) along that track, and
the in-situ-only stellar mass from evolving Starformation_Centrals
with zero external accretion (quenching disabled throughout, so this
isolates the star-formation strand of the published 3-line
decomposition -- not the full accretion-vs-SFR-vs-total figure, which
needs the per-track merger accumulation this script doesn't build).
See rust/steel-postprocess/examples/dump_mass_tracks.rs for the Rust
side and its doc comment for the units/quenching-convention notes that
apply here too (Starformation_Centrals uses the same
`t_quench < t[i] || i==0` condition as CentralEvolution::evolve), and
for why z=0 rather than the papers' z=0.1 (Functions.py::Halogrowth
has no z0 parameter -- it is hardcoded to z0=0).

py-as-is cannot run this at all: Halogrowth shells out to a compiled
getPWGH with a hardcoded machine-specific output path (PORT-FIX G3),
so it only works through Scripts/Validation/make_mah_table.py's
fresh-compile-into-a-scratch-dir workaround, not a bare Halogrowth()
call like this script makes. py-as-is is omitted from this figure for
that reason, not A7.

Two modes, same split as the other Scripts/Validation dump/combine
pairs (py-as-is and py-corrected need different interpreters):

  --dump: compute one leg's track, write a CSV matching
    dump_mass_tracks.rs's columns (z,log_mh,log_sm_am,log_sm_insitu).
  --combine: read all three legs' CSVs and plot.
"""
import argparse
import os
import sys

import numpy as np


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


def _abn_mtch(gamma11=None, preset="g19_se"):
    """gamma11=None, preset="g19_se": G19_SE (PyMorph), the default
    used elsewhere in this session's reproductions. preset="g19_cmod":
    G19_cMod (cmodel/de Vaucouleurs), Paper 2 Fig. 8's variant of
    Fig. 6. gamma11=<float>: the HMevo preset (Paper 3's
    high-mass-slope-evolution family, matching MosterFormSmhm::hmevo
    on the Rust side) -- z=0.1 base parameters M10=11.91,
    SHMnorm10=0.029, beta10=2.09, gamma10=0.64, M11=0.518,
    SHMnorm11=-0.018, beta11=-1.031, with gamma11 as given."""
    is_hmevo = gamma11 is not None
    return {
        "Behroozi13": False, "Behroozi18": False, "B18c": False, "B18t": False,
        "G18": False, "G18_notSE": False,
        "G19_SE": (not is_hmevo) and preset == "g19_se",
        "G19_cMod": (not is_hmevo) and preset == "g19_cmod",
        "Lorenzo18": False, "Moster": False, "Moster10": False, "RP17": False,
        "Illustris": False, "z_Evo": True, "Scatter": 0.15,
        "Override_0": False, "Override_z": False,
        "Override": {
            "M10": 11.95, "SHMnorm10": 0.032, "beta10": 1.61, "gamma10": 0.54,
            "M11": 0.4, "SHMnorm11": -0.014, "beta11": -2, "gamma11": 0.08,
        },
        "PFT": False, "M_PFT1": False, "M_PFT2": False, "M_PFT3": False,
        "N_PFT1": False, "N_PFT2": False, "N_PFT3": False,
        "b_PFT1": False, "b_PFT2": False, "b_PFT3": False,
        "g_PFT1": False, "g_PFT2": False, "g_PFT3": False, "g_PFT4": False,
        "HMevo": is_hmevo, "HMevo_param": gamma11,
    }


def _invert_smhm(F, params, target_log_sm, z=0.0):
    lo, hi = 9.0, 17.0
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        val = float(F.DarkMatterToStellarMass(np.array([mid]), z, params, ScatterOn=False)[0])
        if val - target_log_sm < 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def dump(repo_root, target_log_sm, out_csv, gamma11=None, preset="g19_se"):
    F = _import_functions(repo_root)
    params = {"AbnMtch": _abn_mtch(gamma11, preset)}

    log_dm_z01 = _invert_smhm(F, params, target_log_sm)
    z_track, log_mh_track = F.Halogrowth(log_dm_z01)
    z_track = np.asarray(z_track, dtype=float)
    log_mh_track = np.asarray(log_mh_track, dtype=float)

    # Descending z (highest redshift / earliest time first), matching
    # the forward-in-time integration below (log_sm_insitu[0] is the
    # track's starting point, at its highest redshift).
    order = np.argsort(z_track)[::-1]
    z_track, log_mh_track = z_track[order], log_mh_track[order]
    keep = z_track <= 3.0
    z_track, log_mh_track = z_track[keep], log_mh_track[keep]

    log_sm_am = np.array([
        float(F.DarkMatterToStellarMass(np.array([lm]), zi, params, ScatterOn=False)[0])
        for lm, zi in zip(log_mh_track, z_track)
    ])

    t = F.Cosmo.age(z_track)
    dt = np.diff(t)
    dt = np.append(dt, dt[-1])

    log_sm_insitu = np.zeros_like(z_track)
    log_sm_insitu[0] = log_sm_am[0]
    sfh = np.zeros_like(z_track)
    t_quench = t[0] - 1.0  # below the track's first age: never quenches
    sfr = 0.0
    for i in range(len(z_track)):
        if t_quench < t[i] or i == 0:
            # G19_DPL centrals -- Functions_c.pyx::Starformation_Centrals,
            # SFR_Model_int==6 branch, transcribed exactly (matches
            # DoublePowerLawSfr::central() on the Rust side).
            sm = log_sm_insitu[i]
            zi = z_track[i]
            m_n = 10.65 + 0.33 * zi - 0.08 * zi**2
            norm = 10 ** (0.69 + 0.71 * zi - 0.088 * zi**2)
            alpha = 1.0 - 0.022 * zi + 0.009 * zi**2
            beta = 1.8 - 1.0 * zi + 0.1 * zi**2
            m_per_y = 2 * norm / (10 ** (-alpha * (sm - m_n)) + 10 ** (beta * (sm - m_n)))
            sfr = m_per_y
        sfh[i] = sfr * dt[i] * 1e9
        gmlr = 0.0
        if 0 < i < len(z_track) - 1:
            for j in range(i):
                f1 = 1 - 0.05 * np.log(np.abs(t[j] - t[i]) * 1e9 / 1.4e6 + 1)
                f2 = 1 - 0.05 * np.log(np.abs(t[j] - t[i + 1]) * 1e9 / 1.4e6 + 1)
                gmlr += abs(sfh[j] * (f1 - f2)) / (abs(t[i] - t[i + 1]) * 1e9)
        m_dot = sfr - gmlr
        if i < len(z_track) - 1:
            log_sm_insitu[i + 1] = np.log10(10**log_sm_insitu[i] + m_dot * dt[i] * 1e9)

    with open(out_csv, "w") as f:
        f.write("z,log_mh,log_sm_am,log_sm_insitu\n")
        for i in range(len(z_track)):
            f.write(f"{z_track[i]:.6f},{log_mh_track[i]:.6f},{log_sm_am[i]:.6f},{log_sm_insitu[i]:.6f}\n")


def combine_and_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(args.outdir, exist_ok=True)
    LEG_STYLE = {
        "py-as-is": dict(color="0.55", lw=4.0, ls="-", zorder=1, label="py-as-is (published)"),
        "py-corrected": dict(color="black", lw=1.6, ls="-", zorder=3, label="py-corrected"),
        "rs-steel": dict(color="crimson", lw=1.4, ls="--", dashes=(4, 2), zorder=4, label="rs-steel"),
    }
    legs = {"py-corrected": pd.read_csv(args.corrected), "rs-steel": pd.read_csv(args.rust)}
    if args.asis:
        legs["py-as-is"] = pd.read_csv(args.asis)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for leg, df in legs.items():
        style = LEG_STYLE[leg]
        df = df.sort_values("z", ascending=False)
        kwargs = {k: v for k, v in style.items() if k != "label"}
        ax.plot(df.z, df.log_sm_am, **kwargs, label=f"{style['label']} (AM)")
        ax.plot(df.z, df.log_sm_insitu, **{**kwargs, "alpha": 0.55}, label=f"{style['label']} (in-situ SFR only)")
    ax.set_xlabel("redshift")
    ax.set_ylabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    ax.invert_xaxis()
    ax.legend(loc="lower left", frameon=False, fontsize=8.5)
    title = (
        f"Paper 1 Figs. 6/8, Paper 3 Fig. 7 style -- mass track (G19, target log M*(z=0)={args.target})"
    )
    if "py-as-is" not in legs:
        title += "\npy-as-is absent: Halogrowth cannot run standalone there (G3)"
    ax.set_title(title, fontsize=9.5)
    fig.tight_layout()
    out = os.path.join(args.outdir, "MassTrack.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--repo-root", required=True)
    d.add_argument("--target", type=float, required=True)
    d.add_argument("--gamma11", type=float, default=None, help="HMevo preset high-mass-slope-evolution param; omit for G19_SE/G19_cMod")
    d.add_argument("--preset", default="g19_se", choices=["g19_se", "g19_cmod"], help="SMHM preset when gamma11 is omitted")
    d.add_argument("--out", required=True)

    c = sub.add_parser("combine")
    c.add_argument("--asis", default=None, help="omit if py-as-is can't run this leg (e.g. G3)")
    c.add_argument("--corrected", required=True)
    c.add_argument("--rust", required=True)
    c.add_argument("--target", required=True)
    c.add_argument("--outdir", default="Figures/PortValidation")

    args = ap.parse_args()
    if args.mode == "dump":
        dump(args.repo_root, args.target, args.out, gamma11=args.gamma11, preset=args.preset)
    else:
        combine_and_plot(args)


if __name__ == "__main__":
    main()
