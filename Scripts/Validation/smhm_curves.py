"""Reproduce the SMHM-relation panels shared across all three papers:
Paper 1 Fig. 3 (left), Paper 2 Fig. 4 (left), Paper 3 Figs. 4-6
(left panels) -- the G19_SE (PyMorph) and G19_cMod (cmodel) relations
at z=0.1 and z=2.0. Pure model function, no simulation run, so all
three implementations are directly comparable with no Monte Carlo
noise -- same rationale as paper2_figures.py's Figures 6 & 7.

Two modes, same split as paper2_figures.py and for the same reason
(py-as-is and py-corrected need different Python interpreters):

  --dump: compute one leg's curves, write a CSV matching
    dump_smhm_curves.rs's columns (model,z,log_dm,log_sm).
  --combine: read all three legs' CSVs and plot.

Driven end to end by Scripts/Validation/run_smhm_curves.sh.
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


def _abn_mtch(preset):
    return {
        "Behroozi13": False, "Behroozi18": False, "B18c": False, "B18t": False,
        "G18": False, "G18_notSE": False, "G19_SE": preset == "G19_SE",
        "G19_cMod": preset == "G19_cMod",
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
        "HMevo": False, "HMevo_param": None,
    }


def dump(repo_root, out_csv):
    F = _import_functions(repo_root)
    log_dm = np.arange(10.5, 15.0 + 1e-9, 0.05)
    with open(out_csv, "w") as f:
        f.write("model,z,log_dm,log_sm\n")
        for preset in ("G19_SE", "G19_cMod"):
            params = {"AbnMtch": _abn_mtch(preset)}
            for z in (0.1, 2.0):
                log_sm = F.DarkMatterToStellarMass(log_dm, z, params, ScatterOn=False)
                for dm, sm in zip(log_dm, log_sm):
                    f.write(f"{preset},{z},{dm:.3f},{sm:.6f}\n")


def combine_and_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(args.outdir, exist_ok=True)

    LEG_COLOR = {"py-as-is": "0.55", "py-corrected": "black", "rs-steel": "crimson"}
    LEG_LW = {"py-as-is": 4.0, "py-corrected": 1.6, "rs-steel": 1.4}
    LEG_ZORDER = {"py-as-is": 1, "py-corrected": 3, "rs-steel": 4}
    Z_STYLE = {0.1: dict(linestyle="-"), 2.0: dict(linestyle="--", dashes=(4, 2))}

    legs = {
        "py-as-is": pd.read_csv(args.asis),
        "py-corrected": pd.read_csv(args.corrected),
        "rs-steel": pd.read_csv(args.rust),
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    for ax, preset in zip(axes, ("G19_SE", "G19_cMod")):
        for leg, df in legs.items():
            for z in (0.1, 2.0):
                sub = df[(df.model == preset) & np.isclose(df.z, z)].sort_values("log_dm")
                label = f"{leg} (z={z})" if z == 0.1 else None
                ax.plot(
                    sub.log_dm, sub.log_sm, label=label,
                    color=LEG_COLOR[leg], lw=LEG_LW[leg], zorder=LEG_ZORDER[leg],
                    **Z_STYLE[z],
                )
        ax.set_xlabel(r"$\log_{10} M_h\ [\mathrm{M}_\odot]$")
        ax.set_title(("PyMorph (G19_SE)" if preset == "G19_SE" else "cmodel (G19_cMod)"), fontsize=10)
    axes[0].set_ylabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.text(0.5, 0.02, "solid: z=0.1   dashed: z=2.0", ha="center", fontsize=9, color="0.4")
    fig.suptitle("Paper 1 Fig. 3 / Paper 2 Fig. 4 / Paper 3 Figs. 4-6 -- SMHM relation", fontsize=10.5)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    out = os.path.join(args.outdir, "SMHM_Relation.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    print("wrote:", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--repo-root", required=True)
    d.add_argument("--out", required=True)

    c = sub.add_parser("combine")
    c.add_argument("--asis", required=True)
    c.add_argument("--corrected", required=True)
    c.add_argument("--rust", required=True)
    c.add_argument("--outdir", default="Figures/PortValidation")

    args = ap.parse_args()
    if args.mode == "dump":
        dump(args.repo_root, args.out)
    else:
        combine_and_plot(args)


if __name__ == "__main__":
    main()
