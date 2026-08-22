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


def test_misspelled_kind_is_rejected(store):
    """A typo in `kind` must not silently bypass every gate below it."""
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement(kind="measurment")
    with pytest.raises(GateViolation, match="kind"):
        store.put(bad)


def test_missing_kind_is_rejected(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement()
    del bad["kind"]
    with pytest.raises(GateViolation, match="kind"):
        store.put(bad)


def test_known_non_measurement_kind_is_accepted(store):
    """The allowlist must not over-block legitimate non-measurement kinds."""
    doc = {"_id": "q-icl-definition-ambiguity", "kind": "question",
           "payload": {"text": "Is ICL fraction defined w.r.t. M500c or M200c in GZZ07?"}}
    assert store.put(doc) == "q-icl-definition-ambiguity"


def test_definition_as_string_does_not_bypass_the_field_check(store):
    """A string containing every field name must not fake `in` membership."""
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement(definition=(
        "quantity component mass_def aperture h_convention imf cosmology z_range"))
    with pytest.raises(GateViolation, match="definition"):
        store.put(bad)


def test_definition_as_list_does_not_bypass_the_field_check(store):
    """A list of the field names must not fake `in` membership either."""
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement(definition=["quantity", "component", "mass_def", "aperture",
                                  "h_convention", "imf", "cosmology", "z_range"])
    with pytest.raises(GateViolation, match="definition"):
        store.put(bad)


def test_source_snapshot_as_string_is_a_gate_violation_not_a_crash(store):
    """A malformed source_snapshot must be refused cleanly, not raise AttributeError."""
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement(source_snapshot="abstract")
    with pytest.raises(GateViolation, match="source_snapshot"):
        store.put(bad)
