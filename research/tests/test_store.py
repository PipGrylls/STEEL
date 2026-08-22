import pytest
from kernel.store import Store, GateViolation

pytestmark = pytest.mark.integration

GZZ07_DEF = dict(quantity="f_bcg_icl", component="bcg+icl", mass_def="M500c",
                 aperture="r500", h_convention="h_free", imf="chabrier",
                 cosmology="wmap7", z_range=[0.0, 0.13])


@pytest.fixture
def store():
    s = Store("mongodb://localhost:27017", db="steel_research_test")
    s.drop()
    s.ensure_schema()
    return s


def measurement(**over):
    return {"_id": "gzz07-f-bcg-icl-r500", "kind": "measurement",
            "definition": dict(GZZ07_DEF),
            "payload": {"value": 0.40},
            "source_id": "arxiv:0705.1726",
            "source_snapshot": {"arxiv": "0705.1726",
                                "verified_at": "2026-08-21T00:00:00Z",
                                "verification_method": "arxiv-api-resolved",
                                "extraction": "abstract"},
            "caveats": ["not-icl-only"], **over}


def test_measurement_requires_a_verified_source(store):
    """Gate 1 -- the fabricated-citation failure."""
    with pytest.raises(GateViolation, match="verified source"):
        store.put(measurement())  # source not registered yet


def test_measurement_accepted_once_source_verified(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    assert store.put(measurement()) == "gzz07-f-bcg-icl-r500"


def test_blank_definition_field_is_rejected(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement()
    del bad["definition"]["imf"]
    with pytest.raises(GateViolation, match="imf"):
        store.put(bad)


def test_extraction_method_is_mandatory_and_enumerated(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement()
    bad["source_snapshot"]["extraction"] = "remembered"
    with pytest.raises(GateViolation, match="extraction"):
        store.put(bad)


def test_query_returns_definition_and_verification_state(store):
    """A cache hit must carry enough context to judge reuse."""
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    store.put(measurement())
    hits = store.query({"definition.quantity": "f_bcg_icl"})
    assert len(hits) == 1
    assert hits[0]["definition"]["mass_def"] == "M500c"
    assert hits[0]["source_snapshot"]["extraction"] == "abstract"
