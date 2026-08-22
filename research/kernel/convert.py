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


class ConversionError(Exception):
    """The conversion was refused or the CLI failed."""


def _endpoint(defn: Definition) -> dict:
    return {"mass_def": defn.mass_def, "imf": defn.imf,
            "h_convention": defn.h_convention}


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
    """
    if not CLI.exists():
        raise ConversionError(
            f"{CLI} not built; run: cargo build --release -p steel-harmonise-cli")
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
