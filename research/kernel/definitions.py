"""The comparability fingerprint.

Two quantities may only be combined when every field of their `Definition`
agrees. `"unknown"` is missing information rather than a matching value, so
it never compares -- including against itself. That asymmetry is
deliberate: it turns an unrecorded assumption into a hard stop instead of a
silent pass.
"""
from dataclasses import dataclass, fields
from typing import Any

FIELDS = ("quantity", "component", "mass_def", "aperture",
          "h_convention", "imf", "cosmology", "z_range")

UNKNOWN = "unknown"


class IncompatibleDefinitions(Exception):
    """Raised when two definitions cannot be compared without conversion."""


@dataclass(frozen=True)
class Definition:
    quantity: str
    component: str
    mass_def: str
    aperture: str
    h_convention: str
    imf: str
    cosmology: str
    z_range: tuple

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Definition":
        missing = [f for f in FIELDS if f not in d]
        if missing:
            raise ValueError(f"definition missing required field(s): {', '.join(missing)}")
        z = d["z_range"]
        return cls(**{**{f: d[f] for f in FIELDS if f != "z_range"},
                      "z_range": tuple(z)})

    def differences(self, other: "Definition") -> list[str]:
        """Field names that block comparison, including any `unknown`."""
        out = []
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if mine == UNKNOWN or theirs == UNKNOWN or mine != theirs:
                out.append(f.name)
        return out

    def is_comparable_to(self, other: "Definition") -> bool:
        return not self.differences(other)


def require_comparable(a: Definition, b: Definition) -> None:
    diff = a.differences(b)
    if diff:
        raise IncompatibleDefinitions(
            "cannot compare without explicit conversion; differing or unknown: "
            + ", ".join(diff))
