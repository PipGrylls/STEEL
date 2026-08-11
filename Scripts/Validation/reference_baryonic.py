#!/usr/bin/env python3
"""Regenerate the Rust ``BaryonicPipeline`` regression baseline from the
Python it is a port of.

``rust/steel-plugins/tests/baryonic_pipeline.rs`` pins the pipeline's
output against ``Functions_c.pyx::Starformation_c``. This script drives
the *actual committed Cython* on the same fixture, so the baseline is a
measurement of the Python rather than a snapshot of the Rust.

Run from the repository root with the ``py-asis`` interpreter:

    env/py-asis/bin/python Scripts/Validation/reference_baryonic.py

Note on the noiseless path: ``Functions.py::GetGasMass`` applies
``np.random.normal(GasMass, 0.2)`` unconditionally, with no
``ScatterOn`` parameter, so the Cython's own ``Scatter_On = 0`` never
produced a fully noiseless trajectory -- the gas ceiling that caps the
star formation rate stayed random. ``--gas-scatter`` reproduces that
behaviour; the default (off) is the genuinely noiseless trajectory the
Rust's single ``scatter`` switch produces.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from colossus.cosmology import cosmology  # noqa: E402

cosmology.setCosmology("planck15")
Cosmo = cosmology.getCurrent()

from Functions import Functions as F  # noqa: E402
from Functions import Functions_c  # noqa: E402

# The fixture in rust/steel-plugins/tests/baryonic_pipeline.rs.
N_STEPS = 11
Z = np.array([1.0 - i * 0.05 for i in range(N_STEPS)])
LOG_SM_INFALL = 10.0
LOG_HOST_MASS = 13.0
LOG_SAT_MASS = 11.5
T_DYN_FRICTION = 3.0
SFR_MODEL = "CE"

PARAMETERS = {
    "AbnMtch": {"z_Evo": True, "Moster": False, "Override_0": False, "Override_z": False,
                "G18": False, "G18_notSE": False, "Scatter": 0.15},
    "AltDynamicalTime": 1,
    "NormRnd": 0.5,
    "SFR_Model": SFR_MODEL,
    "PreProcessing": False,
    "AltDynamicalTimeB": False,
}


def build(stripping: bool, gas_scatter: bool, age_grid=None):
    sm = np.array([LOG_SM_INFALL])

    # Timeline, in `Functions.py::StarFormation`'s own convention:
    # *lookback* time, decreasing with index. This matters. The Cython's
    # quenched branch is
    #     SFR = SFR_tquench * exp(-((T_quench - t[i]) / Tau_f))
    # which only decays if `T_quench > t[i]` once quenched -- true for
    # lookback time with `T_quench = t[0] - Tau_d`, and false (an
    # exponential blow-up) if ages are passed instead. The Rust's
    # `BaryonicPipeline` re-derives the same expression for its
    # increasing-age convention; this script has to use the Python's.
    # Ages come from the Rust's own age(z) when a grid is supplied, so
    # this compares the *baryonic pipeline* rather than re-measuring the
    # cosmology port (which Milestone 2 already validated separately).
    # Negating them gives the decreasing sequence the Cython wants while
    # preserving every |t[j] - t[i]| difference exactly.
    ages = age_grid if age_grid is not None else np.array([Cosmo.age(zi) for zi in Z])
    t = -ages
    d_t = np.abs(np.diff(t))
    d_t = np.append(d_t, d_t[-1])

    # Wetzel+13 quenching, with the Fillingham+16 host dependence and
    # the Cowley+19 redshift scaling -- Functions.py::StarFormation.
    tau_f = -0.5 * sm + 5.7
    tau_f[tau_f <= 0.2] = 0.2
    tau_d = 3.5 - np.exp((sm - 10.8) * 2)
    tau_d[tau_d <= 1.0] = 1.0
    host_dep = np.clip((LOG_HOST_MASS - 15) / 5, 0, 1)
    tau_d[sm < 9 + host_dep] = 2.0
    tau_d = tau_d * np.power(1 + Z[0], -3 / 2)
    tau_f = tau_f * np.power(1 + Z[0], -3 / 2)
    t_quench = t[0] - tau_d  # lookback-time convention, as Functions.py

    # Gas ceiling.
    log_sfr = F.StarFormationRate(sm, Z[0], PARAMETERS, ScatterOn=False)
    gas = 9.22 + 0.81 * log_sfr
    if gas_scatter:
        gas = np.random.normal(gas, 0.2)
    gas = np.minimum(gas, F.GetMaxGasMass(LOG_SAT_MASS))
    max_gas = np.power(10, gas)

    # Cattaneo+2011 stripping factor track.
    if stripping:
        time_fraction = np.clip(np.abs(ages - ages[0]) / T_DYN_FRICTION, 0, 1)
        mh_ms = 10 ** (LOG_HOST_MASS - LOG_SAT_MASS)
        strip = 0.6 ** ((1.428 / (2 * np.pi)) * (mh_ms / np.log(1 + mh_ms)))
        strip_factor = np.log10(strip + (1 - strip) * (1 - time_fraction))
    else:
        strip_factor = np.zeros_like(t)

    m_out, _, _, _ = Functions_c.Starformation_c(
        sm, t, d_t, Z, max_gas, t_quench, tau_f,
        StripFactor=strip_factor, z_infall=Z[0], SFR_Model=SFR_MODEL,
        Stripping=1 if stripping else 0, Scatter_On=0,
    )
    return np.asarray(m_out)[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gas-scatter", action="store_true",
                        help="reproduce Functions.py's unconditional gas-mass scatter")
    parser.add_argument("--age-grid", type=Path, default=None,
                        help="file of age(z) values, one per line, from "
                             "`cargo run --example dump_baryonic` (isolates the "
                             "pipeline from the cosmology port)")
    args = parser.parse_args(argv)

    ages = None
    if args.age_grid:
        ages = np.array([float(l) for l in args.age_grid.read_text().split()])

    for label, stripping in (("unstripped", False), ("stripped", True)):
        track = build(stripping, args.gas_scatter, ages)
        print(f"    // {label}, Scatter_On = 0, gas scatter {'on' if args.gas_scatter else 'off'}")
        print("    let expected = [")
        for v in track:
            print(f"        {v:.10f},")
        print("    ];")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
