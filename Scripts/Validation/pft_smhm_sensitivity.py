"""Reproduce Paper 3 Fig. 3's content: the SMHM-parameter sensitivity
sweep from Table 2 of arXiv:2001.06017 -- each Moster-form coefficient
(M, N, beta, gamma) perturbed one at a time off the G19_SE (PyMorph)
baseline, both its z=0.1 value (top row) and its z-evolution term
(bottom row), via `Functions.py`'s `AbnMtch['PFT']` branch and its
`*_PFT*` flags. This is a [fn] figure -- no simulation run, just the
SMHM function evaluated over a mass grid -- so unlike most figures in
this reproduction it needs no run directory, only Functions.py itself.

Note: the deltas used here match Functions.py's PFT branch exactly
(M_PFT1/N_PFT1/b_PFT1/g_PFT1, and the _PFT2/_PFT3 z-evolution pairs),
not Paper 3's printed Table 2, which states N's z=0.1 delta as +0.04;
the code computes +0.004. The code is the validation target, not the
paper text -- see rust/steel-plugins/examples/dump_pft_smhm.rs's
matching comment.

Usage:
    python Scripts/Validation/pft_smhm_sensitivity.py dump \
        --repo-root . --out corrected.csv
    python Scripts/Validation/pft_smhm_sensitivity.py combine \
        --corrected corrected.csv --rust rust.csv \
        --outdir Figures/PortValidation
"""
import argparse
import os
import sys

import numpy as np

PANELS = ["M", "N", "beta", "gamma"]
# (z0.1 PFT flag, z-evo-plus PFT flag, z-evo-minus PFT flag)
PFT_FLAGS = {
    "M": ("M_PFT1", "M_PFT2", "M_PFT3"),
    "N": ("N_PFT1", "N_PFT2", "N_PFT3"),
    "beta": ("b_PFT1", "b_PFT2", "b_PFT3"),
    "gamma": ("g_PFT1", "g_PFT2", "g_PFT3"),
}


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


ALL_PFT_FLAGS = [f for flags in PFT_FLAGS.values() for f in flags] + ["g_PFT4"]


def _abn_mtch(active_flag=None):
    d = {
        "Behroozi13": False, "Behroozi18": False, "B18c": False, "B18t": False,
        "G18": False, "G18_notSE": False, "G19_SE": False, "G19_cMod": False,
        "Lorenzo18": False, "Moster": False, "Moster10": False, "RP17": False,
        "Illustris": False, "z_Evo": True, "Scatter": 0.15,
        "Override_0": False, "Override_z": False,
        "Override": {
            "M10": 11.95, "SHMnorm10": 0.032, "beta10": 1.61, "gamma10": 0.54,
            "M11": 0.4, "SHMnorm11": -0.014, "beta11": -2, "gamma11": 0.08,
        },
        "PFT": True,
        "HMevo": False, "HMevo_param": None,
    }
    for f in ALL_PFT_FLAGS:
        d[f] = False
    if active_flag is not None:
        d[active_flag] = True
    return d


def dump(repo_root, out_csv):
    F = _import_functions(repo_root)
    log_dm = np.arange(10.5, 15.0 + 1e-9, 0.05)
    with open(out_csv, "w") as f:
        f.write("panel,variant,z,log_dm,log_sm\n")
        for panel in PANELS:
            z01_flag, zplus_flag, zminus_flag = PFT_FLAGS[panel]
            for variant, flag in [
                ("baseline", None), ("alt_z0.1", z01_flag),
                ("zevo_plus", zplus_flag), ("zevo_minus", zminus_flag),
            ]:
                params = {"AbnMtch": _abn_mtch(flag)}
                for z in (0.1, 2.0):
                    log_sm = F.DarkMatterToStellarMass(log_dm, z, params, ScatterOn=False)
                    for dm, sm in zip(log_dm, log_sm):
                        f.write(f"{panel},{variant},{z},{dm:.3f},{sm:.6f}\n")


def combine_and_plot(args):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    os.makedirs(args.outdir, exist_ok=True)
    corrected = pd.read_csv(args.corrected)
    rust = pd.read_csv(args.rust)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8), sharex=True, sharey=True)
    row_specs = [
        ("z0.1_row", ["baseline", "alt_z0.1"], {"baseline": "0.4", "alt_z0.1": "#DD8452"}),
        ("zevo_row", ["baseline", "zevo_plus", "zevo_minus"],
         {"baseline": "0.4", "zevo_plus": "#C44E52", "zevo_minus": "#55A868"}),
    ]

    for row_i, (row, variants, colors) in enumerate(row_specs):
        for col_i, panel in enumerate(PANELS):
            ax = axes[row_i, col_i]
            for variant in variants:
                color = colors[variant]
                for df, lw, ls, dashes in [(corrected, 1.8, "-", None), (rust, 1.3, "--", (4, 2))]:
                    sub = df[(df.panel == panel) & (df.variant == variant)]
                    for z, zls in [(0.1, ":"), (2.0, "-")]:
                        zz = sub[np.isclose(sub.z, z)].sort_values("log_dm")
                        if zz.empty:
                            continue
                        final_ls = ls if zls == "-" else zls
                        kwargs = dict(color=color, lw=lw, ls=final_ls)
                        if final_ls == "--" and dashes:
                            kwargs["dashes"] = dashes
                        ax.plot(zz.log_dm, zz.log_sm, **kwargs)
            ax.set_title(f"{panel} ({'z=0.1 altered' if row_i == 0 else 'z-evolution altered'})", fontsize=9)
            if row_i == 1:
                ax.set_xlabel(r"$\log_{10} M_h$")
            if col_i == 0:
                ax.set_ylabel(r"$\log_{10} M_*$")

    fig.legend(
        handles=[
            plt.Line2D([], [], color="0.4", lw=1.8, label="baseline (py-corrected)"),
            plt.Line2D([], [], color="#DD8452", lw=1.8, label="z=0.1 altered"),
            plt.Line2D([], [], color="#C44E52", lw=1.8, label="z-evo +"),
            plt.Line2D([], [], color="#55A868", lw=1.8, label="z-evo -"),
            plt.Line2D([], [], color="k", lw=1.8, ls=":", label="z=0.1"),
            plt.Line2D([], [], color="k", lw=1.8, ls="-", label="z=2.0"),
            plt.Line2D([], [], color="k", lw=1.3, ls="--", label="rs-steel (dashed)"),
        ],
        loc="lower center", ncol=4, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.0),
    )
    fig.suptitle(
        "Paper 3 Fig. 3 style -- SMHM parameter sensitivity (Table 2 deltas, PFT branch)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.09, 1, 0.95])
    out = os.path.join(args.outdir, "Paper3_Fig3_PFTSensitivity.png")
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
