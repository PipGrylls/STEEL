#!/usr/bin/env python3
"""Expand ``configs/published_runs.toml`` into runnable inputs.

Emits, for every entry in the published-run table:

* a ``.toml`` runfile for ``rust/target/release/steel``, and
* the ``--run`` argument string for
  ``Scripts/Validation/run_py_steel.py``,

so the two implementations are driven from one source of truth rather
than from two hand-maintained lists that can drift apart.

Also expands the Paper 3 ``p3-pft-family`` entry into its fourteen
single-coefficient SMHM perturbations, which exist in the Python only as
a chain of ``if Paramaters['X_PFTn']`` offsets in
``Functions/Functions.py`` and have no named preset on either side.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - the py-asis env is 3.10
    import tomli as tomllib  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
TABLE = REPO_ROOT / "configs" / "published_runs.toml"

# Functions/Functions.py:612-613 -- the base every PFT variant perturbs.
PFT_BASE = {
    "m10": 12.0,
    "shmnorm10": 0.032,
    "beta10": 1.5,
    "gamma10": 0.56,
    "m11": 0.6,
    "shmnorm11": -0.014,
    "beta11": -0.7,
    "gamma11": 0.08,
    "scatter": 0.15,
}

# Functions/Functions.py:614-638 -- one offset each, applied to the base.
PFT_VARIANTS = {
    "M_PFT1": ("m10", -0.25),
    "M_PFT2": ("m11", +0.1),
    "M_PFT3": ("m11", -0.1),
    "N_PFT1": ("shmnorm10", +0.004),
    "N_PFT2": ("shmnorm11", +0.007),
    "N_PFT3": ("shmnorm11", -0.007),
    "b_PFT1": ("beta10", -0.3),
    "b_PFT2": ("beta11", +0.3),
    "b_PFT3": ("beta11", -0.3),
    "g_PFT1": ("gamma10", +0.06),
    "g_PFT2": ("gamma11", +0.2),
    "g_PFT3": ("gamma11", -0.2),
    "g_PFT4": ("gamma10", -0.1),
}

# Python SFR_Model string -> (rust model, rust preset). The `_PP`
# suffix is stripped first; it sets Paramaters['PreProcessing'] rather
# than selecting a different main sequence.
SFR_MAP = {
    "T16": ("tomczak_form", "t16"),
    "CE": ("tomczak_form", "ce"),
    "Illustris": ("tomczak_form", "illustris"),
    "S15": ("schreiber_form", "s15"),
    "S16": ("schreiber_form", "s15"),  # Functions_c's "S16" has S15's coefficients
    "S16CE": ("schreiber_form", "s16ce"),
    "G19_DPL": ("double_power_law", None),
}

# Python AbnMtch key -> (rust model, rust preset).
SMHM_MAP = {
    "Moster": ("moster_form", "moster13"),
    "Moster10": ("moster_form", "moster10"),
    "G18": ("moster_form", "g18"),
    "G18_notSE": ("moster_form", "g18_not_se"),
    "G19_SE": ("moster_form", "g19_se"),
    "G19_cMod": ("moster_form", "g19_c_mod"),
    "Illustris": ("moster_form", "illustris"),
    "Behroozi13": ("behroozi_form", "behroozi13"),
    "B18c": ("behroozi_form", "b18c"),
    "B18t": ("behroozi_form", "b18t"),
    "Lorenzo18": ("behroozi_form", "lorenzo18"),
}


def toml_float(x: float) -> str:
    """TOML spelling of a float, including the infinities."""
    if x == float("inf"):
        return "inf"
    if x == float("-inf"):
        return "-inf"
    return repr(float(x))


def rust_runfile(entry: dict, params: dict | None = None) -> str:
    f_tdyn, stripping, sf, z_evo, sfr_model, abn_mtch = entry["tuple_runnable"]
    factor = float("inf") if str(f_tdyn) == "inf" else float(f_tdyn)

    pre_processing = sfr_model.endswith("_PP")
    sfr_key = sfr_model[:-3] if pre_processing else sfr_model
    if sfr_key not in SFR_MAP:
        raise SystemExit(f"{entry['id']}: unknown SFR_Model {sfr_model!r}")
    sfr_kind, sfr_preset = SFR_MAP[sfr_key]

    lines = [
        f"# {entry['id']} -- generated from configs/published_runs.toml.",
        "# Do not edit by hand; edit the table and re-run",
        "# Scripts/Validation/make_runfiles.py.",
        f"# Published as: {entry.get('tuple_published', entry['tuple_runnable'])}",
    ]
    for fig in entry.get("figures", []):
        lines.append(f"#   figure: {fig}")
    if pre_processing:
        lines.append(
            "# `_PP`: Paramaters['PreProcessing'] = True -- pre-quenches a "
            "mass-dependent prefix of each satellite's realization ensemble at infall."
        )
    lines += [
        "",
        "[merger_time]",
        f"dynamical_time_factor = {toml_float(factor)}",
        "redshift_correction = false",
        "",
        "[smhm]",
    ]

    if params is not None:
        # `legacy_name` keeps each PFT variant in its own output
        # directory; without it all fourteen derive "Override_z" and
        # overwrite each other.
        lines += [
            'model = "moster_form"',
            'preset = "override_z"',
            f"z_evo = {str(z_evo).lower()}",
            f'legacy_name = "{abn_mtch}"',
            "",
            "[smhm.params]",
        ]
        for key in ("m10", "shmnorm10", "beta10", "gamma10", "m11", "shmnorm11", "beta11", "gamma11", "scatter"):
            lines.append(f"{key} = {toml_float(params[key])}")
    else:
        if abn_mtch not in SMHM_MAP:
            raise SystemExit(f"{entry['id']}: unknown AbnMtch {abn_mtch!r}")
        smhm_kind, smhm_preset = SMHM_MAP[abn_mtch]
        lines += [f'model = "{smhm_kind}"', f'preset = "{smhm_preset}"', f"z_evo = {str(z_evo).lower()}"]

    lines += ["", "[sfr]", f'model = "{sfr_kind}"']
    if sfr_preset:
        lines.append(f'preset = "{sfr_preset}"')
    lines += [
        "",
        "[run]",
        "log_m_min = 11.0",
        "log_m_max = 16.6",
        "log_m_bin = 0.1",
        f"star_formation = {str(sf).lower()}",
        f"pre_processing = {str(pre_processing).lower()}",
        f"stellar_stripping = {str(stripping).lower()}",
        "",
    ]
    return "\n".join(lines)


def py_run_arg(entry: dict) -> str:
    fields = entry["tuple_runnable"]
    return ",".join(str(f) if not isinstance(f, bool) else ("True" if f else "False") for f in fields)


def expand_pft() -> list[tuple[str, dict]]:
    out = []
    for name, (key, offset) in PFT_VARIANTS.items():
        params = dict(PFT_BASE)
        params[key] = params[key] + offset
        out.append((name, params))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "rust" / "runfiles" / "published")
    parser.add_argument("--paper", action="append", help="restrict to P1/P2/P3; repeatable")
    parser.add_argument("--list-py", action="store_true", help="print the py-steel --run arguments and exit")
    args = parser.parse_args(argv)

    table = tomllib.loads(TABLE.read_text())
    runs = [r for r in table["run"] if not args.paper or r.get("paper") in args.paper]

    if args.list_py:
        for entry in runs:
            if "tuple_runnable" in entry:
                print(f"{entry['id']}\t{py_run_arg(entry)}")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    written = 0
    for entry in runs:
        if entry["id"] == "p3-pft-family":
            for name, params in expand_pft():
                stub = dict(entry)
                stub["tuple_runnable"] = ["1.0", True, True, True, "CE", name]
                stub["id"] = f"p3-pft-{name}"
                path = args.out / f"p3-pft-{name}.toml"
                path.write_text(rust_runfile(stub, params))
                written += 1
            continue
        if "tuple_runnable" not in entry:
            continue
        path = args.out / f"{entry['id']}.toml"
        path.write_text(rust_runfile(entry))
        written += 1

    print(f"wrote {written} runfiles to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
