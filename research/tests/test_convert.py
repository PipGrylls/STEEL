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
