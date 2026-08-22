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

**This derivation's comparison is refused, and that refusal is the
result.** `MODEL_FICL` and `GZZ07_FBCGICL` below are the two definitions
the headline claim would have to combine, and `require_comparable`
rejects them on seven fields at once. The derivation does not route
around that: it records the refusal, propagates the blocking fields as
caveats, and any claim built on it is written as `draft` carrying the
referee's `REVISE` verdict. See `run()`'s return value and `record()`.
"""
import csv
import datetime
import hashlib
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernel.convert import convert
from kernel.definitions import IncompatibleDefinitions, Definition, require_comparable

REPO_ROOT = Path(__file__).resolve().parents[2]
DERIVATION_PATH = Path(__file__).resolve()
DERIVATION_NAME = "icl_stripping_bound"

# The sweep this derivation consumes, committed so the result is
# reproducible from a clean clone rather than from a session scratchpad.
SWEEP_CSV = REPO_ROOT / "research" / "data" / "falsification_lowmass.csv"

# GZZ07's measurement as it is recorded in the store
# (`gzz07-f-bcg-icl-r500`; the same dict appears in
# research/tests/test_store.py, written before this derivation existed).
GZZ07_MEASUREMENT_ID = "gzz07-f-bcg-icl-r500"
GZZ07_SOURCE_ID = "arxiv:0705.1726"
GZZ07_FBCGICL = Definition.from_dict(dict(
    quantity="f_bcg_icl", component="bcg+icl", mass_def="M500c",
    aperture="r500", h_convention="h_free", imf="chabrier",
    cosmology="wmap7", z_range=[0.0, 0.13]))

# The model quantity the ceiling would be applied to. Declared honestly,
# field by field, against what `falsification_sweep.rs` actually computes:
#
#   aperture   -- "unknown". The model number is
#                 `out.icl_stripped_mass.column(j).sum()`: a sum over the
#                 whole host with no radial information whatsoever. It was
#                 previously declared "r500", which was simply false. Per
#                 the design, "unknown" *blocks* comparison rather than
#                 silently permitting it, and here it does exactly that --
#                 that is the apparatus working, not a problem to route
#                 around.
#   quantity/  -- "f_icl"/"icl". The model excludes the BCG and excludes
#   component     surviving satellites; GZZ07's f_BCG+ICL includes both.
#                 Different quantities in both numerator and denominator.
#   cosmology  -- "planck15", STEEL's own (docs/model-assumptions.md).
#                 Not "so the CLI can run": `kernel.convert._endpoint`
#                 never transmits cosmology at all, so the old comment
#                 justifying a wmap7 -> planck15 relabel was factually
#                 wrong. The relabel bought nothing and hid a real
#                 mismatch.
#   z_range    -- [0.1, 0.1], the single redshift the sweep was run at.
#                 GZZ07's own range is [0.0, 0.13]; the narrowing is a
#                 real difference and is surfaced, not assumed away.
#   mass_def   -- "Mvir", the definition the hosts are *binned* by. f_icl
#                 is itself dimensionless; this records the binning axis.
MODEL_FICL = Definition.from_dict(dict(
    quantity="f_icl", component="icl", mass_def="Mvir", aperture="unknown",
    h_convention="per_h", imf="chabrier", cosmology="planck15",
    z_range=[0.1, 0.1]))

GZZ07_CAVEATS = [
    "not-icl-only",
    "extraction-abstract",
    # GZZ07's record declares cosmology="wmap7"; STEEL is Planck15
    # (docs/model-assumptions.md). `kernel.convert` has no
    # cosmology-conversion operation and inventing one here would be
    # exactly the unverified-physics shortcut this apparatus exists to
    # forbid, so the mismatch is surfaced rather than relabelled away.
    "cosmology-mismatch-wmap7-vs-planck15",
    # z_range: the sweep is a single redshift, GZZ07's sample spans
    # 0.0-0.13. Previously changed silently.
    "z-range-narrowed-model-0.1-vs-gzz07-0.0-0.13",
]

# Referee findings on the slice-1 claim that are research questions, not
# code defects. They are recorded here so that any claim built on this
# derivation inherits them automatically (gate 4) rather than depending on
# someone restating them by hand.
REFEREE_CAVEATS = [
    # F4, decisive: an empty conversion path between the two quantities.
    "f-icl-vs-f-bcg-icl-different-quantity-numerator-and-denominator",
    # F7: Moster13 sits on the central side of the comparison as well.
    "residual-circularity-moster13-central-curve",
    # F9: the sweep's satellite_sf true/false blocks are byte-identical.
    "satellite-sf-switch-had-no-effect-in-this-sweep",
    # F10: GZZ07's trend runs opposite in sign to the model's.
    "gzz07-mass-trend-opposite-in-sign-to-model",
    # Deferred minor from task 9: f_icl is concave across the 0 -> 0.80
    # gap, so linear interpolation over-estimates the allowed strength.
    "linear-interpolation-across-f-icl-gap-is-conservative",
]

REFEREE_VERDICT = "REVISE"

# Both endpoints of the axis conversion. This re-expresses *STEEL's own*
# halo mass in GZZ07's mass definition and h-convention; it does not
# import GZZ07's number, so the cosmology is Planck15 on both sides and no
# cosmology conversion is implied (there is none, and the CLI could not
# perform one). `aperture` is "unknown" on both sides: a halo mass carries
# no photometric aperture, and its radial extent is entailed by `mass_def`
# -- the field the CLI actually converts. `convert` permits that because
# the two sides agree; `require_comparable` still refuses it, which is why
# the comparison below is blocked and the conversion is not.
_AXIS = dict(quantity="m_halo", component="halo", aperture="unknown",
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


def _require_distinct(converted: list[float], sources: list[float]) -> None:
    """Refuse if two grid masses converted to the same value.

    `max_strength_at` is keyed on the converted mass, so a repeat would
    silently drop a point -- and the headline "14 genuine bounds of 30"
    count is read straight off that mapping. The conversion is monotonic
    in mass, so this should be unreachable; if it ever fires, the
    conversion has stopped being injective and every number downstream is
    suspect, which is worth an exception rather than a quietly shorter
    list.
    """
    if len(set(converted)) == len(converted):
        return
    groups: dict[float, list[float]] = {}
    for x, src in zip(converted, sources):
        groups.setdefault(x, []).append(src)
    clashes = {x: srcs for x, srcs in groups.items() if len(srcs) > 1}
    raise AssertionError(
        "two or more grid masses converted to the same value, which would "
        "collapse silently in max_strength_at "
        f"(converted -> log_mh_perh sources): {clashes}")


def _check_redshift(rows: list[dict], fieldnames, z: float) -> list[str]:
    """Cross-check the conversion redshift against the sweep, or say so.

    `z` used to be hardcoded to 0.1 and never checked against anything.
    Two things are checkable and are now checked:

    * the declared `z_range` on the model endpoints must be exactly
      `(z, z)`, so the redshift the conversion is evaluated at cannot
      drift away from the redshift the definition claims;
    * if the sweep CSV carries a `z` column, every row must agree with it.

    `falsification_sweep.rs` does not emit a `z` column, so for the
    committed sweep the second check cannot run. That is not silently
    ignored: it returns a caveat recording that the model redshift is an
    asserted assumption rather than an observed property of the input.
    """
    declared = tuple(MODEL_FICL.z_range)
    if declared != (z, z):
        raise AssertionError(
            f"conversion redshift z={z} disagrees with the declared model "
            f"z_range {declared}; the definition and the arithmetic must "
            "not drift apart")
    if "z" not in (fieldnames or []):
        return ["model-redshift-not-recorded-in-sweep-csv"]
    bad = sorted({r["z"] for r in rows if float(r["z"]) != z})
    if bad:
        raise AssertionError(
            f"sweep CSV contains redshift(s) {bad} but the conversion is "
            f"evaluated at z={z}")
    return []


def run(sweep_csv: str, gzz07_value: float, out_png: str,
        z: float = 0.1, tools: dict | None = None) -> dict:
    """Compute the bound, draw the figure, and (with `tools`) record it.

    `tools` is the dict of MCP tool callables from
    `mcp_server.server.build_server`. When supplied, the figure is
    registered as a `derivation_run` output and a `draft` claim is written
    (gate 3) -- see `record()`. When omitted the computation is pure and
    needs no database, which is what keeps the arithmetic testable without
    a live Mongo.

    Note the parameter is the tool callables, not a `Store`: the MCP tool
    boundary is the only write path the spec permits, and a derivation is
    held to it just as an agent is.
    """
    with open(sweep_csv, newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = reader.fieldnames
        rows = [r for r in reader if r["satellite_sf"].lower() == "false"]

    caveats = list(GZZ07_CAVEATS) + list(REFEREE_CAVEATS)
    caveats += _check_redshift(rows, fieldnames, z)

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
    m500c = Definition.from_dict({**_AXIS, "mass_def": "M500c",
                                  "h_convention": "h_free"})
    mvir = Definition.from_dict({**_AXIS, "mass_def": "Mvir",
                                 "h_convention": "per_h"})

    xs, ys, kinds, sources = [], [], [], []
    path: list[str] = []
    for log_mh_perh in sorted(by_mass):
        pairs = sorted(by_mass[log_mh_perh])
        strengths = [p[0] for p in pairs]
        f_icl = [p[1] for p in pairs]
        # The ceiling is an M500c measurement; STEEL's axis is Mvir. Convert
        # the axis onto the measurement's definition before comparing.
        log_mh_converted, steps = convert(log_mh_perh, mvir, m500c, z=z)
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
        sources.append(log_mh_perh)

    # `max_strength_at` below is keyed on the *converted* mass, so two grid
    # masses converting to the same value would silently collapse into one
    # entry -- and the headline "14 genuine bounds of 30" count is read off
    # that mapping. Refuse rather than quietly lose a point. The conversion
    # is monotonic in mass, so this can only fire if the input grid itself
    # repeats a mass, which is a defect in the sweep worth hearing about.
    _require_distinct(xs, sources)

    # THE COMPARISON THIS DERIVATION EXISTS TO MAKE, AND ITS REFUSAL.
    #
    # Everything above re-expresses the model's *halo-mass axis* in GZZ07's
    # mass definition. That is a legitimate conversion and it succeeded.
    # It is not the scientific claim. The claim is `f_icl <= 0.40`, which
    # combines the model's f_icl with GZZ07's f_BCG+ICL -- and those two
    # definitions are not comparable. Ask the gate explicitly rather than
    # assuming; a comparison nobody asked permission for is exactly how the
    # M500-vs-Mvir defect shipped three times.
    comparison_refused = None
    try:
        require_comparable(MODEL_FICL, GZZ07_FBCGICL)
    except IncompatibleDefinitions as exc:
        blocking = MODEL_FICL.differences(GZZ07_FBCGICL)
        comparison_refused = {"blocking_fields": blocking, "detail": str(exc)}
        caveats.append("comparison-refused-" + "-".join(blocking))

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

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

    # The figure must not read as a result when the comparison behind it was
    # refused. A reader who sees only the PNG has to see the refusal too --
    # a caveat that lives only in the database is a caveat that drifts.
    if comparison_refused:
        fig.text(0.5, 0.015,
                 "REFUSED (referee verdict " + REFEREE_VERDICT + "): model f_icl "
                 "is not comparable with GZZ07 f_BCG+ICL.\nBlocking definition "
                 "fields: " + ", ".join(comparison_refused["blocking_fields"])
                 + ". Draft only -- do not cite.",
                 ha="center", va="bottom", fontsize=7.5, color="crimson")
        fig.tight_layout(rect=(0, 0.075, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    max_strength_at = {x: {"value": y, "kind": k}
                       for x, y, k in zip(xs, ys, kinds)}
    points = [{"log_mh_perh": s, "log_m_converted": x, "value": y, "kind": k}
              for s, x, y, k in zip(sources, xs, ys, kinds)]

    result = {"figure": out_png,
              "max_strength_at": max_strength_at,
              "points": points,
              "path": path,
              "comparison_refused": comparison_refused,
              "referee_verdict": REFEREE_VERDICT,
              "caveats": caveats}
    if tools is not None:
        result["recorded"] = record(result, tools, sweep_csv)
    return result


# --------------------------------------------------------------------------
# Recording the run. Gate 3: "a figure may only be produced as a
# `derivation_run` output". Everything below exists so that running this
# derivation leaves a record, rather than an orphan PNG whose only
# connection to its inputs is that a human remembers making it.
#
# Every write goes through the MCP tool callables returned by
# `mcp_server.server.build_server`, never through `kernel.store.Store`.
# That is not decoration: the spec makes the tool boundary the only write
# path precisely so an agent cannot script around the gates, and a
# derivation that imported `Store` directly would be demonstrating the
# bypass it is supposed to make impossible.
# --------------------------------------------------------------------------

SWEEP_RUN_ID = "steel-falsification-lowmass-sweep"
DERIVATION_RUN_ID = "derivation-run-icl-stripping-bound"
CLAIM_ID = "claim-icl-ceiling-bounds-stripping-strength"


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _repo_relative(path) -> str:
    """Record paths relative to the repo when possible, so the run record
    means the same thing on another machine."""
    resolved = Path(path).resolve()
    if resolved.is_relative_to(REPO_ROOT):
        return str(resolved.relative_to(REPO_ROOT))
    return str(resolved)


def code_hash() -> str:
    """SHA-256 of this derivation's source -- the `derivation_run` code hash."""
    return _sha256(DERIVATION_PATH)


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _git_state() -> dict:
    """Commit and dirtiness, for gate 5 (a dirty tree may back a draft only)."""
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                          capture_output=True, text=True)
    status = subprocess.run(["git", "status", "--porcelain"], cwd=REPO_ROOT,
                            capture_output=True, text=True)
    if head.returncode != 0 or status.returncode != 0:
        return {"commit": "unknown", "dirty": True}
    return {"commit": head.stdout.strip(), "dirty": bool(status.stdout.strip())}


def _put(store_put, doc: dict) -> str:
    """Write through the MCP tool, turning a refusal into a loud failure.

    The tools return `{"ok": false, "error": ...}` rather than raising, so
    an unchecked call would let a refused write pass for a successful one.
    """
    result = store_put(doc)
    if not result.get("ok"):
        raise RuntimeError(
            f"store refused {doc.get('kind')} {doc.get('_id')!r}: "
            f"{result.get('error')}")
    return result["id"]


def seed_gzz07_measurement(tools: dict) -> str:
    """Write the GZZ07 measurement this derivation reads.

    Stands in for the `data-curator` agent so the slice is reproducible
    from a clean clone. Kept separate from `record()` because a derivation
    producing its own input measurement would be exactly the circularity
    the roster's role split exists to prevent -- this is a seeding step,
    not part of the derivation.
    """
    verified = tools["store_verify_source"](GZZ07_SOURCE_ID, "arxiv-api-resolved")
    if not verified.get("ok"):
        raise RuntimeError(f"source verification refused: {verified.get('error')}")
    return _put(tools["store_put"], {
        "_id": GZZ07_MEASUREMENT_ID,
        "kind": "measurement",
        "definition": {"quantity": GZZ07_FBCGICL.quantity,
                       "component": GZZ07_FBCGICL.component,
                       "mass_def": GZZ07_FBCGICL.mass_def,
                       "aperture": GZZ07_FBCGICL.aperture,
                       "h_convention": GZZ07_FBCGICL.h_convention,
                       "imf": GZZ07_FBCGICL.imf,
                       "cosmology": GZZ07_FBCGICL.cosmology,
                       "z_range": list(GZZ07_FBCGICL.z_range)},
        "payload": {"value": 0.40},
        "source_id": GZZ07_SOURCE_ID,
        "source_snapshot": {"arxiv": "0705.1726",
                            "verified_at": "2026-08-21T00:00:00Z",
                            "verification_method": "arxiv-api-resolved",
                            "extraction": "abstract",
                            "locator": "abstract, sentence 6"},
        "caveats": ["not-icl-only", "extraction-abstract"],
        "created_by": "data-curator",
        "created_at": _now(),
    })


def _inherited_caveats(store_query, input_ids: list[str]) -> list[str]:
    """Gate 4: a claim inherits the union of its inputs' caveats.

    Read back from the store rather than from local variables, so a caveat
    added to an input document by someone else still propagates. Restating
    caveats by hand is the drift this gate exists to stop.
    """
    found = store_query({"_id": {"$in": input_ids}})
    if not found.get("ok"):
        raise RuntimeError(f"store query failed: {found.get('error')}")
    union: list[str] = []
    for doc in found["results"]:
        for c in doc.get("caveats", []):
            if c not in union:
                union.append(c)
    return union


def record(result: dict, tools: dict, sweep_csv: str) -> dict:
    """Write this run's `model_run`, `derivation_run` and `claim`.

    The claim is written `status: "draft"` carrying the referee's `REVISE`
    verdict. It must never be written any other way from here: gate 6 says
    a claim cannot leave draft without a referee verdict, and the verdict
    on this one is a refusal. `status` is not a parameter for that reason.
    """
    store_put, store_query = tools["store_put"], tools["store_query"]
    git = _git_state()

    sweep_id = _put(store_put, {
        "_id": SWEEP_RUN_ID,
        "kind": "model_run",
        "payload": {"generator": "rust/steel-postprocess/examples/falsification_sweep.rs",
                    "csv": _repo_relative(sweep_csv),
                    "csv_sha256": _sha256(sweep_csv),
                    "git_commit": git["commit"],
                    "git_dirty": git["dirty"]},
        # F9: the satellite_sf true/false halves of this CSV are
        # byte-identical, so the "star formation off" control did nothing.
        # Recorded on the input itself, so anything consuming it inherits
        # the caveat whether or not it remembers to restate it.
        "caveats": ["satellite-sf-switch-had-no-effect-in-this-sweep"],
        "created_by": "model-runner",
        "created_at": _now(),
    })

    input_ids = [sweep_id, GZZ07_MEASUREMENT_ID]
    derivation_id = _put(store_put, {
        "_id": DERIVATION_RUN_ID,
        "kind": "derivation_run",
        "payload": {"derivation": f"research/derivations/{DERIVATION_NAME}.py",
                    "code_sha256": code_hash(),
                    "inputs": input_ids,
                    "figure": result["figure"],
                    "conversion_path": result["path"],
                    "points": result["points"],
                    "comparison_refused": result["comparison_refused"],
                    "git_commit": git["commit"],
                    "git_dirty": git["dirty"]},
        "caveats": list(result["caveats"]),
        "created_by": "analyst",
        "created_at": _now(),
    })

    claim_inputs = [derivation_id, *input_ids]
    caveats = _inherited_caveats(store_query, claim_inputs)
    claim_id = _put(store_put, {
        "_id": CLAIM_ID,
        "kind": "claim",
        # Gate 6: draft, because the referee returned REVISE. The verdict
        # is carried on the document, not in a note somewhere.
        "status": "draft",
        "payload": {
            "text": "The empirical ICL ceiling bounds STEEL's stripping "
                    "strength below the published Cattaneo+11 baseline.",
            "referee_verdict": REFEREE_VERDICT,
            "refused": True,
            "comparison_refused": result["comparison_refused"],
            "do_not_cite": "The comparison behind this claim was refused; "
                           "the claim is recorded as evidence of the "
                           "refusal, not as a result.",
        },
        "inputs": claim_inputs,
        "caveats": caveats,
        "created_by": "analyst",
        "created_at": _now(),
    })
    return {"model_run": sweep_id, "derivation_run": derivation_id,
            "claim": claim_id, "caveats": caveats}


def main(uri: str = "mongodb://localhost:27017",
         db: str = "steel_research") -> dict:
    """Run the derivation and record it. Requires Mongo and the Rust CLI."""
    from mcp_server.server import build_server

    tools = build_server(uri, db=db)
    seed_gzz07_measurement(tools)
    out_png = REPO_ROOT / "research" / "figures" / f"{DERIVATION_NAME}.png"
    out_png.parent.mkdir(parents=True, exist_ok=True)
    result = run(str(SWEEP_CSV), gzz07_value=0.40, out_png=str(out_png),
                 tools=tools)
    return {**result["recorded"], "figure": str(out_png), "result": result}


if __name__ == "__main__":
    import json

    print(json.dumps({k: v for k, v in main().items() if k != "result"},
                     indent=2))
