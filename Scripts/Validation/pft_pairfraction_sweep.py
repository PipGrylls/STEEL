"""Reproduce the actual scientific result of Paper 3 (arXiv 2001.06017)
Figure 3: pair fraction vs. redshift for each of the 13 single-coefficient
SMHM perturbations (Table 2 / AbnMtch['PFT']), grouped into 4 panels
(M, N, beta, gamma), each showing the PyMorph reference plus its
z=0.1-altered and z-evolution +/- variants -- 4 lines per panel,
matching the real figure's center-panel content. This is the piece
Paper3_Fig3_PFTSensitivity.png (the outer SMHM-curve panels only) was
missing: the actual pair-fraction-vs-z result, which needs a full
SF+stripping run per variant (14 total: reference + 13 PFT flags),
not just a closed-form SMHM evaluation.

Uses the exact Return_PF_Plot logic from Scripts/CentralPostprocessing.py
(PORT-FIX H1 applies), same pattern as results_from_run.py's
plot_pair_fraction, via a throwaway OutputFolder redirect per run.

Usage:
    python Scripts/Validation/pft_pairfraction_sweep.py \
        --work-root /tmp/steel-three-way \
        --outdir Figures/PortValidation
"""
import argparse
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PANELS = {
    "M": ("M_PFT1", "M_PFT2", "M_PFT3"),
    "N": ("N_PFT1", "N_PFT2", "N_PFT3"),
    "beta": ("b_PFT1", "b_PFT2", "b_PFT3"),
    "gamma": ("g_PFT1", "g_PFT2", "g_PFT3"),
}
VARIANT_COLOR = {"z0.1_alt": "#4C72B0", "zevo_plus": "#55A868", "zevo_minus": "#C44E52"}
VARIANT_LABEL = {"z0.1_alt": "{p}_0.1,alt", "zevo_plus": "{p}_z,+", "zevo_minus": "{p}_z,-"}


def _redirect_output_folder(run_dir, run_tuple, scratch_root):
    dir_name = "RunParam_" + "".join(f"{p}_" for p in run_tuple)
    fake_root = os.path.join(scratch_root, os.path.basename(run_dir) + "-outputfolder")
    os.makedirs(fake_root, exist_ok=True)
    link = os.path.join(fake_root, dir_name)
    if os.path.islink(link) or os.path.exists(link):
        os.remove(link) if os.path.islink(link) else shutil.rmtree(link)
    os.symlink(os.path.abspath(run_dir), link)
    return fake_root + "/"


def pair_fraction_vs_z(run_dir, run_tuple, scratch_root, CP):
    CP.F.OutputFolder = _redirect_output_folder(run_dir, run_tuple, scratch_root)
    d = CP.PairFractionData(run_tuple)
    z, pf, _, _ = d.Return_PF_Plot(d.SMF_interp, Parent_Cut=11, UpperLimit=True)
    z = np.asarray(z, dtype=float)
    pf = np.asarray(pf, dtype=float)
    mask = np.isfinite(pf) & (z <= 3.5)
    return z[mask], pf[mask]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work-root", required=True, help="dir containing three_way_pft_<name>/{py-corrected,rs-steel}-1")
    ap.add_argument("--outdir", default="Figures/PortValidation")
    args = ap.parse_args()

    sys.path.insert(0, REPO_ROOT)
    from Scripts import CentralPostprocessing as CP

    os.makedirs(args.outdir, exist_ok=True)
    scratch = os.path.join(args.outdir, ".pft_scratch")
    os.makedirs(scratch, exist_ok=True)

    def run_dirs(name):
        base = os.path.join(args.work_root, f"three_way_pft_{name}")
        return os.path.join(base, "py-corrected-1"), os.path.join(base, "rs-steel-1")

    ref_py, ref_rs = run_dirs("reference")
    ref_tuple = ("1.0", True, True, True, "CE", "G19_SE")

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), sharey=True)
    for ax, (panel, flags) in zip(axes, PANELS.items()):
        for leg_dir, leg_tuple, ls, lw, ref_kwargs in [
            (ref_py, ref_tuple, "-", 1.8, dict(color="0.35", label="PyMorph (reference)")),
        ]:
            z, pf = pair_fraction_vs_z(leg_dir, leg_tuple, scratch, CP)
            ax.plot(z, pf, ls=ls, lw=lw, **ref_kwargs)
        z, pf = pair_fraction_vs_z(ref_rs, ref_tuple, scratch, CP)
        ax.plot(z, pf, color="0.35", lw=1.3, ls="--", dashes=(4, 2))

        for variant, flag in zip(("z0.1_alt", "zevo_plus", "zevo_minus"), flags):
            color = VARIANT_COLOR[variant]
            py_dir, rs_dir = run_dirs(flag)
            variant_tuple = ("1.0", True, True, True, "CE", flag)
            z, pf = pair_fraction_vs_z(py_dir, variant_tuple, scratch, CP)
            ax.plot(z, pf, color=color, lw=1.8, ls="-", label=VARIANT_LABEL[variant].format(p=panel))
            z, pf = pair_fraction_vs_z(rs_dir, variant_tuple, scratch, CP)
            ax.plot(z, pf, color=color, lw=1.3, ls="--", dashes=(4, 2))

        ax.set_title(panel, fontsize=11)
        ax.set_xlabel("redshift")
        ax.set_xlim(0, 3.5)
        ax.legend(loc="upper right", frameon=True, framealpha=0.9, edgecolor="none", fontsize=7.5)

    axes[0].set_ylabel("pair fraction")
    fig.suptitle(
        "Paper 3 Fig. 3 style -- pair fraction vs z, SMHM parameter sensitivity (solid=py-corrected, dashed=rs-steel)",
        fontsize=10.5,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(args.outdir, "Paper3_Fig3_PairFractionSensitivity.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    shutil.rmtree(scratch, ignore_errors=True)
    print("wrote:", out)


if __name__ == "__main__":
    main()
