"""Bridge to `steel-harmonise-cli`.

No conversion arithmetic lives here. Python owns definitions and
provenance; every number is converted by the Rust layer, so the formulas
have exactly one tested implementation.
"""
import json
import subprocess
from pathlib import Path

from .definitions import Definition

CLI = Path(__file__).resolve().parents[2] / "rust" / "target" / "release" / "steel-harmonise-cli"

# The only definition fields `steel-harmonise-cli` can actually change.
# These are the three `_endpoint` transmits; everything else in a
# `Definition` is invisible to the CLI.
CONVERTIBLE_FIELDS = ("mass_def", "imf", "h_convention")

# Everything else. The CLI has no operation for any of these -- there is no
# aperture conversion, no cosmology conversion, no redshift-range
# conversion, and changing `quantity` or `component` is a change of
# physical meaning, not of units. Because `_endpoint` never transmits them,
# a difference here used to be dropped silently: the CLI would happily
# convert the mass and return a value that looked authoritative while the
# two endpoints described different things. That is the exact failure class
# this apparatus exists to forbid, so it is a refusal.
NON_CONVERTIBLE_FIELDS = ("quantity", "component", "aperture",
                          "cosmology", "z_range")


class ConversionError(Exception):
    """The conversion was refused or the CLI failed."""


def _endpoint(defn: Definition) -> dict:
    return {"mass_def": defn.mass_def, "imf": defn.imf,
            "h_convention": defn.h_convention}


def _require_convertible(frm: Definition, to: Definition) -> None:
    """Refuse when the two endpoints differ in a field the CLI cannot change.

    Note the deliberate difference from `require_comparable`: this checks
    *inequality only*, not `"unknown"`. Conversion and comparison are
    different questions. Re-expressing a halo mass from Mvir to M500c is a
    valid operation whether or not the aperture of the underlying
    measurement is known, provided it is the *same* on both sides -- the
    conversion does not depend on it. Deciding whether the converted value
    may then be *compared* with something else is `require_comparable`'s
    job, and there `"unknown"` blocks, including against itself. Folding
    the two checks together here would either let real mismatches through
    or make every honestly-`unknown` field unconvertible; keeping them
    separate lets a derivation convert an axis and still be refused at the
    comparison, which is what the design intends.
    """
    offending = [f for f in NON_CONVERTIBLE_FIELDS
                 if getattr(frm, f) != getattr(to, f)]
    if offending:
        detail = "; ".join(
            f"{f}: {getattr(frm, f)!r} != {getattr(to, f)!r}" for f in offending)
        raise ConversionError(
            "steel-harmonise-cli cannot convert " + ", ".join(offending)
            + f" -- these differ between the two endpoints ({detail}). "
            "Only " + ", ".join(CONVERTIBLE_FIELDS) + " are convertible; "
            "a difference in any other field is a change of meaning, not "
            "of units, and must be resolved explicitly rather than "
            "silently ignored")


def _select_op(frm: Definition, to: Definition) -> str:
    """Pick which CLI operation covers the requested conversion.

    The CLI only performs one kind of conversion per call, so the choice
    is driven by which field actually changed:

    - `imf` differs, `mass_def` does not -> `convert_stellar` (IMF offset;
      it also carries any `h_convention` change along for free).
    - `mass_def` differs, `imf` does not -> `convert_mass` (NFW mass
      definition conversion; it also carries any `h_convention` change).
    - Neither differs (e.g. only `h_convention` changes) -> `convert_mass`.
      Verified against the CLI directly: with `mass_def` and `imf` held
      fixed, `convert_mass` and `convert_stellar` produce the identical
      h-convention-adjusted value, so either op is a correct no-op/h-only
      conversion here; `convert_mass` is picked as the default.
    - Both `mass_def` and `imf` differ -> refuse. Each op silently ignores
      the field it doesn't own (confirmed against the CLI: asking
      `convert_mass` to also change `imf` returns the IMF-unconverted
      value, and asking `convert_stellar` to also change `mass_def`
      returns the mass-definition-unconverted value), so picking either
      one would silently drop half the requested conversion. That is
      exactly the failure mode this apparatus exists to prevent, so it is
      a `ConversionError` instead of a guess.
    """
    mass_def_changed = frm.mass_def != to.mass_def
    imf_changed = frm.imf != to.imf
    if mass_def_changed and imf_changed:
        raise ConversionError(
            "cannot change mass_def and imf in a single conversion step "
            f"(from mass_def={frm.mass_def!r} imf={frm.imf!r} "
            f"to mass_def={to.mass_def!r} imf={to.imf!r}); "
            "convert in two steps instead")
    if imf_changed:
        return "convert_stellar"
    return "convert_mass"


def convert(log_m: float, frm: Definition, to: Definition, z: float) -> tuple[float, list[str]]:
    """Convert `log_m` from one definition to another.

    Returns the converted value and the ordered list of steps taken, which
    the caller records as provenance.

    Refuses (rather than silently ignoring) any difference in a field the
    CLI cannot convert -- see `_require_convertible`.
    """
    if not CLI.exists():
        raise ConversionError(
            f"{CLI} not built; run: cargo build --release -p steel-harmonise-cli")
    _require_convertible(frm, to)
    op = _select_op(frm, to)
    req = {"op": op, "log_m": log_m, "z": z,
           "from": _endpoint(frm), "to": _endpoint(to)}
    try:
        proc = subprocess.run([str(CLI)], input=json.dumps(req),
                              capture_output=True, text=True)
    except OSError as exc:
        raise ConversionError(f"failed to run {CLI}: {exc}") from exc
    if proc.returncode != 0:
        raise ConversionError(proc.stderr.strip() or "steel-harmonise-cli failed")
    out = json.loads(proc.stdout)
    return out["log_m"], out["path"]
