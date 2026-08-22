import pytest
from mcp_server_under_test import build_server  # re-exported below

pytestmark = pytest.mark.integration


@pytest.fixture
def tools():
    return build_server("mongodb://localhost:27017", db="steel_research_mcp_test")


def test_put_is_refused_without_a_verified_source(tools):
    """The gate must live server-side, not in a bypassable library."""
    result = tools["store_put"]({"_id": "x", "kind": "measurement",
                                 "definition": {}, "source_id": "arxiv:9999.99999"})
    assert result["ok"] is False
    assert "verified source" in result["error"]


def test_verify_then_put_succeeds(tools):
    tools["store_verify_source"]("arxiv:0705.1726", "arxiv-api-resolved")
    doc = {"_id": "m1", "kind": "measurement",
           "definition": {"quantity": "f_bcg_icl", "component": "bcg+icl",
                          "mass_def": "M500c", "aperture": "r500",
                          "h_convention": "h_free", "imf": "chabrier",
                          "cosmology": "wmap7", "z_range": [0.0, 0.13]},
           "payload": {"value": 0.40}, "source_id": "arxiv:0705.1726",
           "source_snapshot": {"extraction": "abstract"}}
    assert tools["store_put"](doc)["ok"] is True
    assert len(tools["store_query"]({"kind": "measurement"})["results"]) == 1


def test_bad_verification_method_is_refused(tools):
    result = tools["store_verify_source"]("arxiv:1.1", "i-remember-it")
    assert result["ok"] is False


def test_unknown_kind_is_refused_through_the_tool(tools):
    """A typo in `kind` must surface as a clean refusal, not a crash --
    proven through the tool boundary, not just the `Store` class."""
    result = tools["store_put"]({"_id": "x1", "kind": "measurment"})
    assert result["ok"] is False
    assert "kind" in result["error"]


def test_definition_missing_a_field_is_refused_through_the_tool(tools):
    tools["store_verify_source"]("arxiv:0705.1726", "arxiv-api-resolved")
    doc = {"_id": "x2", "kind": "measurement", "source_id": "arxiv:0705.1726",
           "source_snapshot": {"extraction": "abstract"},
           "definition": {"quantity": "f_bcg_icl", "component": "bcg+icl",
                          "mass_def": "M500c", "aperture": "r500",
                          "h_convention": "h_free", "imf": "chabrier",
                          "cosmology": "wmap7"}}  # missing z_range
    result = tools["store_put"](doc)
    assert result["ok"] is False
    assert "z_range" in result["error"]


def test_definition_as_string_is_refused_through_the_tool(tools):
    tools["store_verify_source"]("arxiv:0705.1726", "arxiv-api-resolved")
    doc = {"_id": "x3", "kind": "measurement", "source_id": "arxiv:0705.1726",
           "source_snapshot": {"extraction": "abstract"},
           "definition": "quantity component mass_def aperture h_convention imf cosmology z_range"}
    result = tools["store_put"](doc)
    assert result["ok"] is False
    assert "definition" in result["error"]


def test_bad_extraction_is_refused_through_the_tool(tools):
    tools["store_verify_source"]("arxiv:0705.1726", "arxiv-api-resolved")
    doc = {"_id": "x4", "kind": "measurement", "source_id": "arxiv:0705.1726",
           "definition": {"quantity": "f_bcg_icl", "component": "bcg+icl",
                          "mass_def": "M500c", "aperture": "r500",
                          "h_convention": "h_free", "imf": "chabrier",
                          "cosmology": "wmap7", "z_range": [0.0, 0.13]},
           "source_snapshot": {"extraction": "remembered"}}
    result = tools["store_put"](doc)
    assert result["ok"] is False
    assert "extraction" in result["error"]


def test_missing_id_is_refused_through_the_tool(tools):
    """`put()` keys its write on `doc["_id"]`; without this gate a doc
    lacking `_id` would reach `doc["_id"]` unhandled and raise `KeyError`
    instead of a clean refusal."""
    result = tools["store_put"]({"kind": "source"})
    assert result["ok"] is False
    assert "_id" in result["error"]
