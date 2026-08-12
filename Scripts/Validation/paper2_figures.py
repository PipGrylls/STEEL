"""Reproduce Paper 2 Figures 6 and 7 (arXiv 1812.00015 / MNRAS 483,
2506) for the three-way port validation: py-as-is, py-corrected, and
rs-steel overlaid on one axes per figure.

Both figures plot pure model *functions* -- no simulation run, no
scatter, no observational data -- which is exactly what makes them a
clean fidelity check: every point is directly comparable between the
three implementations with no Monte Carlo noise to average away.

Figure 6: the Wetzel+2013/Fillingham+2016 quenching-delay time-scale
tau_q(M*,sat) for three example host masses.
Figure 7: the McCavana+2012/Boylan-Kolchin+2008 dynamical-friction
merging time-scale vs. subhalo mass (left) and satellite stellar mass
via the G19_SE SMHM relation at z=1.5 (right), for three example host
masses.

py-as-is and py-corrected need *different* Python interpreters
(env/py-asis is 3.10, matching the compiled Functions_c in the
detached origin/PipGrylls worktree; env/py-legacy is 3.11, matching
this checkout) -- so this script runs in two modes:

  --dump: compute one leg's curves with whatever interpreter invokes
    it, write a CSV in the same format `rust/steel-plugins/examples/
    dump_quenching.rs` / `dump_merger_time.rs` produce.
  --combine: read all three legs' CSVs (2 python + 1 rust) and plot.

Driven end to end by `Scripts/Validation/run_paper2_figures.sh`.
"""
import argparse
import inspect
import os
import sys
import textwrap

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


HOST_MASSES_FIG6 = [10.0, 12.5, 15.0]
HOST_MASSES_FIG7 = [12.0, 13.0, 14.0]


def dump_quenching(repo_root, out_csv):
    """tau_delay(log_sm, host_mass) computed by literally executing the
    Wetzel+2013/Fillingham+2016 block from `Functions.py::StarFormation`
    as committed in `repo_root` -- not a reimplementation. Isolates
    that block (it depends only on SM_Sat and AvaHaloMass[0], both
    supplied here) via source extraction + exec, since the block's
    locals (Tau_d) are not otherwise returned by the function."""
    F = _import_functions(repo_root)
    src = inspect.getsource(F.StarFormation)
    # index() lands on the marker text itself, stripping the leading
    # indentation of that line; back up to the start of the line so
    # dedent() sees a consistent common prefix across the whole block.
    start = src.rindex("\n", 0, src.index("#Quenching, Wetzel")) + 1
    end = src.index("T_quench = t[0]")
    block = textwrap.dedent(src[start:end])

    # z_infall=0 removes the Cowley+2019 (1+z)^-1.5 scaling both
    # branches apply after the Fillingham block, so tau_delay is
    # directly comparable to the paper's static plot of tau_q(M*).
    log_sm = np.arange(7.0, 12.0 + 1e-9, 0.02)
    with open(out_csv, "w") as f:
        f.write("log_sm,host_mass,tau_delay\n")
        for host in HOST_MASSES_FIG6:
            ns = {
                "np": np, "SM_Sat": log_sm.copy(), "AvaHaloMass": np.array([host]),
                "z_infall": 0.0,
            }
            exec(block, ns)
            for sm, tau_d in zip(log_sm, ns["Tau_d"]):
                f.write(f"{sm:.3f},{host:.2f},{tau_d:.10f}\n")


def dump_merger(repo_root, out_csv):
    """t_merge(log_subhalo_mass, host_mass) at z=1.5 from the real
    `DynamicalFriction` + `DynamicalTime_Fun`, and the same subhalo
    masses converted to satellite stellar mass via the G19_SE SMHM
    relation (`DarkMatterToStellarMass`), both called directly -- these
    are standalone functions, no source-extraction trick needed."""
    F = _import_functions(repo_root)
    z = 1.5
    params = {"AltDynamicalTime": 1, "AltDynamicalTimeB": False}

    abn_mtch = {
        "Behroozi13": False, "Behroozi18": False, "B18c": False, "B18t": False,
        "G18": False, "G18_notSE": False, "G19_SE": True, "G19_cMod": False,
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
    smhm_params = {"AbnMtch": abn_mtch}

    time_to_z0 = float(F.Cosmo.lookbackTime(z) - F.Cosmo.lookbackTime(0))

    with open(out_csv, "w") as f:
        f.write(f"time_to_z0_gyr,{time_to_z0:.10f}\n")
        f.write("log_host_mass,log_subhalo_mass,log_sat_stellar_mass,t_merge_gyr\n")
        for host in HOST_MASSES_FIG7:
            log_sub = np.arange(9.0, host + 1e-9, 0.05)
            t_merge = F.DynamicalFriction(host, log_sub, z, params)
            log_sm = F.DarkMatterToStellarMass(log_sub, z, smhm_params, ScatterOn=False)
            for ls, lm, t in zip(log_sub, log_sm, t_merge):
                f.write(f"{host:.2f},{ls:.3f},{lm:.6f},{t:.10f}\n")


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

    # --- Figure 6 ---
    # One host mass (10), not all three the paper shows: overlaying all
    # three per leg in one leg-colour makes the as-is/corrected
    # divergence invisible, because as-is's single (buggy) cutoff is a
    # strict subset of the pixels the corrected curve's *own* three
    # branches already cover. Mh,cent=10 is also where the found bug
    # (A8) has its largest effect: cutoff 9.0 (as-is) vs 8.0 (fixed).
    fig6_host = 10.0
    q = {
        "py-as-is": pd.read_csv(args.asis_fig6),
        "py-corrected": pd.read_csv(args.corrected_fig6),
        "rs-steel": pd.read_csv(args.rust_fig6).rename(columns={"tau_delay_gyr": "tau_delay"}),
    }
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    for leg, df in q.items():
        style = LEG_STYLE[leg]
        sub = df[np.isclose(df["host_mass"], fig6_host)].sort_values("log_sm")
        ax.plot(sub["log_sm"], sub["tau_delay"], **style)
    ax.set_xlabel(r"$\log_{10} M_{*,\mathrm{sat}}\ [\mathrm{M}_\odot]$")
    ax.set_ylabel(r"$\tau_q$ [Gyr]")
    ax.set_xlim(7, 12)
    ax.set_ylim(0.8, 3.7)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title(
        r"Paper 2 Fig. 6 -- quenching delay time-scale (G19), $\log M_{h,\mathrm{cent}}=10$",
        fontsize=10,
    )
    fig.tight_layout()
    out6 = os.path.join(args.outdir, "Paper2_Fig6_Quenching.png")
    fig.savefig(out6, dpi=200)
    plt.close(fig)

    # --- Figure 7 ---
    # One host mass (13, the middle of the paper's three) -- no known
    # correction touches DynamicalFriction/DynamicalTime_Fun, so unlike
    # Figure 6 this is a clean three-way overlap: the point is fidelity,
    # not a bug story.
    fig7_host = 13.0

    def load_merger(path):
        with open(path) as fh:
            t0 = float(fh.readline().strip().split(",")[1])
        df = pd.read_csv(path, skiprows=1)
        df = df.rename(columns={
            "log_host_mass": "host_mass",
            "log_sat_stellar_mass": "log_sat_mass",
            "t_merge_gyr": "t_merge",
        })
        return df, t0

    m = {
        "py-as-is": load_merger(args.asis_fig7),
        "py-corrected": load_merger(args.corrected_fig7),
        "rs-steel": load_merger(args.rust_fig7),
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.0), sharey=True)
    for leg, (df, _) in m.items():
        style = LEG_STYLE[leg]
        sub = df[np.isclose(df["host_mass"], fig7_host)].sort_values("log_subhalo_mass")
        axes[0].plot(sub["log_subhalo_mass"], sub["t_merge"], **style)
        axes[1].plot(sub["log_sat_mass"], sub["t_merge"], **{k: v for k, v in style.items() if k != "label"})
    for leg, (_, t0) in m.items():
        axes[0].axhline(t0, color=LEG_STYLE[leg]["color"], lw=0.8, ls=":", alpha=0.6)

    axes[0].set_xlabel(r"$\log_{10} M_{h,\mathrm{sat}}\ [\mathrm{M}_\odot]$")
    axes[1].set_xlabel(r"$\log_{10} M_{*,\mathrm{sat}}\ [\mathrm{M}_\odot]$ (G19 SMHM, $z=1.5$)")
    axes[0].set_ylabel(r"$\tau_\mathrm{merge}$ [Gyr]")
    axes[0].set_ylim(0, 14)
    axes[0].set_xlim(11, 13)
    axes[1].set_xlim(4, 11)
    axes[0].legend(loc="upper right", frameon=False, fontsize=9)
    fig.suptitle(
        r"Paper 2 Fig. 7 -- dynamical-friction merging time-scale (G19), $\log M_{h,\mathrm{cent}}=13$",
        fontsize=10,
    )
    fig.tight_layout()
    out7 = os.path.join(args.outdir, "Paper2_Fig7_MergerTimescale.png")
    fig.savefig(out7, dpi=200)
    plt.close(fig)

    t0s = {leg: t0 for leg, (_, t0) in m.items()}
    print("time_to_z0 [Gyr]:", ", ".join(f"{k}={v:.4f}" for k, v in t0s.items()))
    print("wrote:", out6)
    print("wrote:", out7)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="mode", required=True)

    d = sub.add_parser("dump")
    d.add_argument("--figure", choices=["6", "7"], required=True)
    d.add_argument("--repo-root", required=True)
    d.add_argument("--out", required=True)

    c = sub.add_parser("combine")
    c.add_argument("--asis-fig6", required=True)
    c.add_argument("--corrected-fig6", required=True)
    c.add_argument("--rust-fig6", required=True)
    c.add_argument("--asis-fig7", required=True)
    c.add_argument("--corrected-fig7", required=True)
    c.add_argument("--rust-fig7", required=True)
    c.add_argument("--outdir", default="Figures/PortValidation")

    args = ap.parse_args()
    if args.mode == "dump":
        if args.figure == "6":
            dump_quenching(args.repo_root, args.out)
        else:
            dump_merger(args.repo_root, args.out)
    else:
        combine_and_plot(args)


if __name__ == "__main__":
    main()
