import math

import pytest
from kernel.convert import convert, ConversionError
from kernel.definitions import Definition

BASE = dict(quantity="m_halo", component="halo", aperture="r500",
            imf="chabrier", cosmology="planck15", z_range=[0.1, 0.1])


def d(**over):
    return Definition.from_dict({**BASE, "mass_def": "M500c",
                                 "h_convention": "h_free", **over})


def test_mass_definition_conversion_increases_virial_mass():
    log_m, path = convert(14.0, d(), d(mass_def="Mvir"), z=0.1)
    assert 14.0 < log_m < 15.0
    assert any("Mvir" in step for step in path)


def test_unknown_mass_definition_raises():
    with pytest.raises(ConversionError):
        convert(14.0, d(mass_def="unknown"), d(mass_def="Mvir"), z=0.1)


def test_imf_only_difference_routes_to_convert_stellar():
    """An IMF-only change must take the IMF-offset path (`convert_stellar`),
    not silently fall through to a mass-definition conversion that would
    leave the value unchanged (the Finding-1 bug: `m_star` starts with
    `m_`, so a naive "quantity starts with m_" heuristic misroutes this)."""
    log_m, path = convert(14.0, d(imf="salpeter"), d(imf="chabrier"), z=0.1)
    assert log_m == pytest.approx(13.76, abs=1e-9)
    assert any("Salpeter->Chabrier" in step for step in path)


def test_mass_def_and_imf_both_differ_raises():
    """Changing both `mass_def` and `imf` in one call is two conversions
    folded into one; the CLI only does one at a time and silently drops
    whichever field it doesn't own, so this must refuse rather than guess
    which one to apply."""
    with pytest.raises(ConversionError):
        convert(14.0, d(), d(mass_def="Mvir", imf="kroupa"), z=0.1)


@pytest.mark.parametrize("field,value", [
    ("quantity", "m_star"),
    ("component", "icl"),
    ("aperture", "r200"),
    ("cosmology", "wmap7"),
    ("z_range", [0.0, 0.13]),
])
def test_non_convertible_field_difference_is_refused(field, value):
    """`_endpoint` transmits only mass_def/imf/h_convention, so a difference
    in any other field used to be dropped silently and the CLI would return
    an authoritative-looking number for two endpoints that describe
    different things. Each of the five must refuse, naming itself."""
    with pytest.raises(ConversionError, match=field):
        convert(14.0, d(), d(mass_def="Mvir", **{field: value}), z=0.1)


def test_refusal_names_every_offending_field_not_just_the_first():
    """A caller fixing one field at a time would otherwise be led through
    the mismatches one refusal at a time, and could easily stop early."""
    with pytest.raises(ConversionError) as exc:
        convert(14.0, d(), d(mass_def="Mvir", aperture="r200",
                             cosmology="wmap7"), z=0.1)
    assert "aperture" in str(exc.value)
    assert "cosmology" in str(exc.value)


def test_unknown_aperture_on_both_sides_still_converts():
    """Conversion is not comparison. An honestly-`unknown` aperture that is
    the same on both endpoints does not affect a mass-definition
    conversion, so `convert` allows it; `require_comparable` is the gate
    that refuses `unknown`, and it still does (see
    test_definitions.py). Folding the two checks together would make every
    honestly-unknown field unconvertible."""
    log_m, path = convert(14.0, d(aperture="unknown"),
                          d(mass_def="Mvir", aperture="unknown"), z=0.1)
    assert 14.0 < log_m < 15.0
    assert any("Mvir" in step for step in path)


def test_unknown_aperture_on_one_side_only_is_refused():
    """...but an `unknown` facing a known value is still a difference."""
    with pytest.raises(ConversionError, match="aperture"):
        convert(14.0, d(aperture="unknown"), d(mass_def="Mvir"), z=0.1)


def test_missing_cli_binary_raises_with_actionable_message(monkeypatch):
    from pathlib import Path

    import kernel.convert as convert_mod

    monkeypatch.setattr(convert_mod, "CLI", Path("/nonexistent/steel-harmonise-cli"))
    with pytest.raises(ConversionError, match="cargo build --release -p steel-harmonise-cli"):
        convert(14.0, d(), d(mass_def="Mvir"), z=0.1)


def test_agrees_with_colossus():
    """Independent validation -- the conversion is new physics, so it is
    checked against an established implementation, not just itself.

    We use colossus's `dutton14` concentration model, matching the
    DuttonMaccio14 relation our Rust conversion uses, so this is a genuine
    check of the NFW mass-definition arithmetic rather than a comparison
    between two different concentration relations. That lets us hold a
    tight 0.01 dex tolerance instead of a loose one.
    """
    colossus = pytest.importorskip("colossus.halo.mass_defs")
    from colossus.cosmology import cosmology as ccosmo
    ccosmo.setCosmology("planck15")
    from colossus.halo.concentration import concentration as c_of_m

    m500c_per_h = 1e14  # Msun/h
    c500 = c_of_m(m500c_per_h, "500c", 0.1, model="dutton14")
    m_vir_ref, _, _ = colossus.changeMassDefinition(
        m500c_per_h, c500, 0.1, "500c", "vir")

    got, _ = convert(14.0, d(h_convention="per_h"),
                     d(mass_def="Mvir", h_convention="per_h"), z=0.1)
    # Same concentration relation (DuttonMaccio14 / dutton14) on both sides,
    # so this is a real test of the conversion arithmetic, not a comparison
    # between two different concentration relations.
    assert abs(got - math.log10(m_vir_ref)) < 0.01
