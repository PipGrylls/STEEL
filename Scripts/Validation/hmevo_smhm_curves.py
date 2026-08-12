"""Reproduce Paper 3 Fig. 6's left panel: the SMHM relation under the
HMevo preset (Paper 3's high-mass-slope-evolution family) at z=0.1 and
z=2.0, for several gamma11 values. Companion to smhm_curves.py (the
G19_SE/G19_cMod version feeding Paper 1 Fig. 3 / Paper 2 Fig. 4 /
Paper 3 Figs. 4-5); same dump/combine split for the same reason
(py-as-is and py-corrected need different interpreters).

Usage:
    python Scripts/Validation/hmevo_smhm_curves.py dump \
        --repo-root . --out corrected.csv
    python Scripts/Validation/hmevo_smhm_curves.py combine \
        --corrected corrected.csv --rust rust.csv \
        --outdir Figures/PortValidation
"""
import argparse
import os
import sys

import numpy as np

GAMMA11_VALUES = [0.1, 0.2, 0.5]


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


def _abn_mtch(gamma11):
    return {
        "Behroozi13": False, "Behroozi18": False, "B18c": False, "B18t": False,
        "G18": False, "G18_notSE": False, "G19_SE": False, "G19_cMod": False,
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
        "HMevo": True, "HMevo_param": gamma11,
    }


def dump(repo_root, out_csv):
    F = _import_functions(repo_root)
    log_dm = np.arange(10.5, 15.0 + 1e-9, 0.05)
    with open(out_csv, "w") as f:
        f.write("gamma11,z,log_dm,log_sm\n")
        for gamma11 in GAMMA11_VALUES:
            params = {"AbnMtch": _abn_mtch(gamma11)}
            for z in (0.1, 2.0):
                log_sm = F.DarkMatterToStellarMass(log_dm, z, params, ScatterOn=False)
                for dm, sm in zip(log_dm, log_sm):
                    f.write(f"{gamma11},{z},{dm:.3f},{sm:.6f}\n")


def combine_and_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(args.outdir, exist_ok=True)
    COLORS = ["#4C72B0", "#DD8452", "#55A868"]
    corrected = pd.read_csv(args.corrected)
    rust = pd.read_csv(args.rust)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    for ax, z in zip(axes, (0.1, 2.0)):
        for i, gamma11 in enumerate(GAMMA11_VALUES):
            color = COLORS[i % len(COLORS)]
            for df, lw, ls, dashes in [(corrected, 1.8, "-", None), (rust, 1.3, "--", (4, 2))]:
                sub = df[(df.gamma11 == gamma11) & np.isclose(df.z, z)].sort_values("log_dm")
                kwargs = dict(color=color, lw=lw, ls=ls)
                if dashes:
                    kwargs["dashes"] = dashes
                label = rf"$\gamma_{{11}}={gamma11}$" if ls == "-" else None
                ax.plot(sub.log_dm, sub.log_sm, label=label, **kwargs)
        ax.set_xlabel(r"$\log_{10} M_h\ [\mathrm{M}_\odot]$")
        ax.set_title(f"z={z}", fontsize=10)
    axes[0].set_ylabel(r"$\log_{10} M_*\ [\mathrm{M}_\odot]$")
    axes[0].legend(loc="upper left", frameon=False, fontsize=9, title="solid=py-corrected\ndashed=rs-steel")
    fig.suptitle("Paper 3 Fig. 6 style -- HMevo SMHM relation", fontsize=10.5)
    fig.tight_layout()
    out = os.path.join(args.outdir, "Paper3_Fig6_HMevoSMHM.png")
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
