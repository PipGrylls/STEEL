"""Slice 1: the empirical ICL ceiling as a bound on stripping strength.

Reproduces the 2026-08-21 result, with one substantive change: the
Gonzalez+07 ceiling is quoted at M500c and STEEL's grid is Mvir, so the
halo-mass axis is now genuinely converted rather than annotated as a
caveat. The conversion is performed by the Rust layer and its steps are
returned for provenance.

The two endpoints also differ in `h_convention` (STEEL's grid is
`per_h`, GZZ07's own record is `h_free`; see the comment in `run()`),
so the mass-definition conversion and the h-convention conversion both
apply. Getting that pairing backwards is exactly the "Msun/h -> h-free"
bug the project's own design doc calls out as a repeat offender.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernel.convert import convert
from kernel.definitions import Definition

GZZ07_CAVEATS = ["not-icl-only", "extraction-abstract"]

_HALO = dict(quantity="m_halo", component="halo", aperture="r500",
             imf="chabrier", cosmology="planck15", z_range=[0.1, 0.1])


def max_allowed_strength(strengths, f_icl, bound):
    """Largest stripping strength keeping `f_icl` at or below `bound`.

    `f_icl` increases monotonically with strength, so linear interpolation
    against the bound is safe. Returns 0.0 when even zero stripping
    exceeds the bound, and the largest tested strength when the bound is
    never reached (i.e. unconstrained by this data).
    """
    if f_icl[0] > bound:
        return 0.0
    if f_icl[-1] <= bound:
        return strengths[-1]
    for i in range(1, len(f_icl)):
        if f_icl[i] > bound:
            s0, s1 = strengths[i - 1], strengths[i]
            f0, f1 = f_icl[i - 1], f_icl[i]
            return s0 + (bound - f0) * (s1 - s0) / (f1 - f0)
    return strengths[-1]


def run(sweep_csv: str, gzz07_value: float, out_png: str) -> dict:
    rows = [r for r in csv.DictReader(open(sweep_csv))
            if r["satellite_sf"].lower() == "false"]

    by_mass: dict[float, list[tuple[float, float]]] = {}
    for r in rows:
        by_mass.setdefault(float(r["log_mh_perh"]), []).append(
            (float(r["strength"]), float(r["f_icl"])))

    # h_convention, not just mass_def, genuinely differs between these two
    # endpoints -- this is a fact about the two sides, not a free choice:
    #   * STEEL's own halo grid is Msun/h internally (docs/model-assumptions.md,
    #     "h convention | Msun/h internally"), matching the sweep CSV's own
    #     `log_mh_perh` column name, so the model side is declared `per_h`.
    #   * The GZZ07 measurement's own definition record (already used
    #     verbatim in research/tests/test_definitions.py and test_store.py,
    #     `gzz07-f-bcg-icl-r500`) declares `h_convention: "h_free"` for its
    #     M500c value, so the measurement side is declared `h_free`.
    # Swapping these (labelling the model `h_free` and the measurement
    # `per_h`) would silently apply the h-factor backwards -- exactly the
    # "Msun/h -> h-free" defect class this apparatus exists to catch
    # (docs/superpowers/specs/2026-08-22-research-apparatus-design.md's
    # motivation table names that bug by name).
    m500c = Definition.from_dict({**_HALO, "mass_def": "M500c",
                                  "h_convention": "h_free"})
    mvir = Definition.from_dict({**_HALO, "mass_def": "Mvir",
                                 "h_convention": "per_h"})

    xs, ys = [], []
    path: list[str] = []
    for log_mh_perh in sorted(by_mass):
        pairs = sorted(by_mass[log_mh_perh])
        strengths = [p[0] for p in pairs]
        f_icl = [p[1] for p in pairs]
        # The ceiling is an M500c measurement; STEEL's axis is Mvir. Convert
        # the axis onto the measurement's definition before comparing.
        log_mh_converted, steps = convert(log_mh_perh, mvir, m500c, z=0.1)
        # This is the same conversion (Mvir -> M500c at fixed z) applied at
        # every grid mass, so the *steps* (the named operations performed)
        # are expected to be identical each time even though the numeric
        # inputs/outputs differ -- the Rust CLI's step labels describe the
        # operation, not the operands. Rather than silently keeping only
        # the last iteration's provenance (which would happen to be
        # correct here purely by luck, and would go stale silently if the
        # CLI ever started emitting value-dependent step descriptions),
        # record the first iteration's steps and assert every subsequent
        # iteration agrees. A mismatch would mean the conversion is not
        # actually the single operation this derivation assumes it is,
        # which is exactly the kind of silent-misconversion bug this
        # apparatus exists to catch.
        if not path:
            path = steps
        elif steps != path:
            raise AssertionError(
                "conversion steps differ across the mass grid: "
                f"{path!r} != {steps!r} at log_mh_perh={log_mh_perh}; "
                "the Mvir->M500c conversion is expected to be the same "
                "operation at every mass")
        xs.append(log_mh_converted)
        ys.append(max_allowed_strength(strengths, f_icl, gzz07_value))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(xs, ys, "-", color="crimson", lw=2.0)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax.annotate("published (Cattaneo+11) baseline", xy=(xs[0], 1.03),
                fontsize=8, color="0.4")
    ax.set_xlabel(r"host halo mass  $\log_{10} M_{500c}$  [$\mathrm{M}_\odot$]")
    ax.set_ylabel("max. stripping strength\nallowed by the ICL ceiling")
    ax.set_title("ICL ceiling as a bound on stripping strength\n"
                 "(halo mass converted to the measurement's own definition)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {"figure": out_png,
            "max_strength_at": dict(zip(xs, ys)),
            "path": path,
            "caveats": GZZ07_CAVEATS}
