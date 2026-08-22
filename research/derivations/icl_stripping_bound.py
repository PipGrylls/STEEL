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

GZZ07_CAVEATS = [
    "not-icl-only",
    "extraction-abstract",
    # GZZ07's own definition record (research/tests/test_definitions.py,
    # research/tests/test_store.py's `gzz07-f-bcg-icl-r500`) declares
    # cosmology="wmap7". This derivation's endpoints below use "planck15"
    # so the mass-conversion CLI (calibrated to Planck15, per
    # docs/model-assumptions.md) can run at all. `kernel.convert` has no
    # cosmology-conversion operation, and inventing one here would be
    # exactly the unverified-physics shortcut this apparatus exists to
    # forbid, so the mismatch is not silently dropped -- it is surfaced as
    # a caveat that propagates into any claim built on this derivation.
    "cosmology-mismatch-wmap7-vs-planck15",
]

_HALO = dict(quantity="m_halo", component="halo", aperture="r500",
             imf="chabrier", cosmology="planck15", z_range=[0.1, 0.1])


def max_allowed_strength(strengths, f_icl, bound):
    """Largest stripping strength keeping `f_icl` at or below `bound`.

    Returns `(value, kind)`. `f_icl` increases monotonically with
    strength, so linear interpolation against the bound is safe whenever
    the tested range actually brackets it -- that is the only case where
    `value` is a genuine bound (`kind == "bound"`).

    When the tested range does *not* bracket the ceiling, `value` is a
    limit, not a bound, and callers must not treat it as one:

    - `kind == "upper_limit"`: even the smallest tested strength already
      exceeds the ceiling. `value` is that smallest tested strength
      (never an invented literal like `0.0` -- if the sweep never tested
      zero, `0.0` was never observed and must not be reported as if it
      had been). The true maximum allowed strength is at or below
      `value`, unresolved by this data.
    - `kind == "lower_limit"`: the ceiling is never reached even at the
      largest tested strength. `value` is that largest tested strength --
      this data is unconstrained above it; the sweep simply stopped
      there, it did not find a real bound.
    """
    if f_icl[0] > bound:
        return strengths[0], "upper_limit"
    if f_icl[-1] <= bound:
        return strengths[-1], "lower_limit"
    for i in range(1, len(f_icl)):
        if f_icl[i] > bound:
            s0, s1 = strengths[i - 1], strengths[i]
            f0, f1 = f_icl[i - 1], f_icl[i]
            return s0 + (bound - f0) * (s1 - s0) / (f1 - f0), "bound"
    return strengths[-1], "lower_limit"


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

    xs, ys, kinds = [], [], []
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
        value, kind = max_allowed_strength(strengths, f_icl, gzz07_value)
        xs.append(log_mh_converted)
        ys.append(value)
        kinds.append(kind)

    fig, ax = plt.subplots(figsize=(7.5, 4.6))

    # A thin grey guide connects every point (bound or censored) so the eye
    # can follow the overall trend, but the *scientific claim* -- "this
    # strength is allowed" -- is only ever drawn for genuine bounds. Censored
    # points (the sweep didn't bracket the ceiling) get a visually distinct
    # marker plus their own legend entry, so a reader can't mistake "the
    # sweep stopped here" for "stripping this strong is allowed here".
    ax.plot(xs, ys, "-", color="0.8", lw=1.0, zorder=1)

    bound_xy = [(x, y) for x, y, k in zip(xs, ys, kinds) if k == "bound"]
    lower_xy = [(x, y) for x, y, k in zip(xs, ys, kinds) if k == "lower_limit"]
    upper_xy = [(x, y) for x, y, k in zip(xs, ys, kinds) if k == "upper_limit"]

    if bound_xy:
        bx, by = zip(*bound_xy)
        ax.plot(bx, by, "-o", color="crimson", lw=2.0, ms=4, zorder=3,
                label="bound (ceiling crossed within the tested range)")
    if lower_xy:
        lx, ly = zip(*lower_xy)
        ax.plot(lx, ly, "^", color="steelblue", ms=6, zorder=3,
                label="lower limit -- unconstrained: ceiling never\n"
                      "reached at the largest tested strength")
    if upper_xy:
        ux, uy = zip(*upper_xy)
        ax.plot(ux, uy, "v", color="darkorange", ms=6, zorder=3,
                label="upper limit -- unconstrained: ceiling already\n"
                      "exceeded at the smallest tested strength")

    ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax.annotate("published (Cattaneo+11) baseline", xy=(xs[0], 1.03),
                fontsize=8, color="0.4")
    ax.set_xlabel(r"host halo mass  $\log_{10} M_{500c}$  [$\mathrm{M}_\odot$]")
    ax.set_ylabel("max. stripping strength\nallowed by the ICL ceiling")
    ax.set_title("ICL ceiling as a bound on stripping strength\n"
                 "(halo mass converted to the measurement's own definition)",
                 fontsize=10)
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    max_strength_at = {x: {"value": y, "kind": k}
                       for x, y, k in zip(xs, ys, kinds)}

    return {"figure": out_png,
            "max_strength_at": max_strength_at,
            "path": path,
            "caveats": list(GZZ07_CAVEATS)}
