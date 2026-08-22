import pytest
from kernel.definitions import Definition, IncompatibleDefinitions, require_comparable

GZZ07 = dict(quantity="f_bcg_icl", component="bcg+icl", mass_def="M500c",
             aperture="r500", h_convention="h_free", imf="chabrier",
             cosmology="wmap7", z_range=[0.0, 0.13])


def test_identical_definitions_are_comparable():
    a, b = Definition.from_dict(GZZ07), Definition.from_dict(GZZ07)
    assert a.is_comparable_to(b)
    require_comparable(a, b)


def test_differing_mass_definition_blocks_comparison():
    other = Definition.from_dict({**GZZ07, "mass_def": "Mvir"})
    assert not Definition.from_dict(GZZ07).is_comparable_to(other)
    with pytest.raises(IncompatibleDefinitions, match="mass_def"):
        require_comparable(Definition.from_dict(GZZ07), other)


def test_unknown_never_compares_even_with_itself():
    """`unknown` is missing information, not a value that happens to match."""
    d = Definition.from_dict({**GZZ07, "imf": "unknown"})
    assert not d.is_comparable_to(d)
    with pytest.raises(IncompatibleDefinitions, match="imf"):
        require_comparable(d, d)


def test_component_mismatch_blocks_comparison():
    """BCG+ICL is not ICL-only -- the caveat that went unenforced all session."""
    icl_only = Definition.from_dict({**GZZ07, "component": "icl"})
    with pytest.raises(IncompatibleDefinitions, match="component"):
        require_comparable(Definition.from_dict(GZZ07), icl_only)


def test_missing_field_is_rejected_at_construction():
    incomplete = {k: v for k, v in GZZ07.items() if k != "imf"}
    with pytest.raises(ValueError, match="imf"):
        Definition.from_dict(incomplete)
