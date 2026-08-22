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


def convert(log_m: float, frm: Definition, to: Definition, z: float) -> tuple[float, list[str]]:
    """Convert `log_m` from one definition to another.

    Returns the converted value and the ordered list of steps taken, which
    the caller records as provenance.
    """
    if not CLI.exists():
        raise ConversionError(
            f"{CLI} not built; run: cargo build --release -p steel-harmonise-cli")
    op = "convert_mass" if frm.mass_def != to.mass_def or to.quantity.startswith("m_") \
        else "convert_stellar"
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
