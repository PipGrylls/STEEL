import csv

import pytest

from derivations.icl_stripping_bound import (
    CLAIM_ID, DERIVATION_RUN_ID, GZZ07_FBCGICL, GZZ07_MEASUREMENT_ID,
    MODEL_FICL, REFEREE_VERDICT, SWEEP_CSV, SWEEP_RUN_ID, _require_distinct,
    code_hash, max_allowed_strength, run, seed_gzz07_measurement)


def write_sweep(path, rows=None, header_extra=(), row_extra=()):
    rows = rows or [("False", s, 14.0, f) for s, f in
                    [(0.0, 0.0), (1.0, 0.30), (2.0, 0.50)]]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["satellite_sf", "strength", "log_mh_perh", "log_sm_smhm",
                    "log_accreted", "ratio", "log_icl", "f_icl", *header_extra])
        for sf, s, mh, f in rows:
            w.writerow([sf, s, mh, 11.5, 11.9, 1.0, 11.2, f, *row_extra])
    return str(path)


def test_max_allowed_strength_interpolates_the_crossing():
    """The helper previously copy-pasted between two plotting scripts."""
    strengths = [0.0, 1.0, 2.0]
    f_icl = [0.0, 0.30, 0.50]
    # bound 0.40 sits halfway between strengths 1 and 2
    value, kind = max_allowed_strength(strengths, f_icl, 0.40)
    assert abs(value - 1.5) < 1e-9
    assert kind == "bound"


def test_ceiling_below_every_sample_gives_zero():
    value, kind = max_allowed_strength([0.0, 1.0], [0.5, 0.9], 0.1)
    assert value == 0.0
    assert kind == "upper_limit"


def test_ceiling_below_every_sample_never_invents_an_untested_strength():
    """The smallest *tested* strength is reported, not a hardcoded 0.0 --
    if the sweep never tested 0.0, 0.0 must never appear as if it had
    been. This is the sibling of the zero-invention bug: censoring from
    below must report the real grid value, exactly like censoring from
    above already does (see the lower_limit test)."""
    value, kind = max_allowed_strength([0.5, 1.0], [0.6, 0.9], 0.1)
    assert value == 0.5
    assert kind == "upper_limit"


def test_ceiling_above_every_sample_returns_the_tested_maximum():
    value, kind = max_allowed_strength([0.0, 4.0], [0.0, 0.1], 0.9)
    assert value == 4.0
    assert kind == "lower_limit"


def test_run_applies_the_mass_conversion_and_records_it(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["satellite_sf", "strength", "log_mh_perh", "log_sm_smhm",
                    "log_accreted", "ratio", "log_icl", "f_icl"])
        for s, f in [(0.0, 0.0), (1.0, 0.30), (2.0, 0.50)]:
            w.writerow(["False", s, 14.0, 11.5, 11.9, 1.0, 11.2, f])
    out = run(str(csv_path), gzz07_value=0.40, out_png=str(tmp_path / "f.png"))
    assert any("Mvir" in step for step in out["path"]), \
        "the Mvir->M500c conversion must be recorded, not skipped"
    assert "not-icl-only" in out["caveats"]


def test_run_uses_the_physically_correct_h_convention_pairing(tmp_path):
    """Regression guard for the h-convention fix.

    STEEL's own Mvir grid is `per_h` (docs/model-assumptions.md: "h
    convention | Msun/h internally"); GZZ07's own M500c record is
    `h_free` (research/tests/test_definitions.py,
    research/tests/test_store.py's `gzz07-f-bcg-icl-r500`). Reverting to
    the reversed pairing applies the h-factor backwards, but it still
    makes `any("Mvir" in step for step in out["path"])` true and still
    reports a plausible-looking, still-downward shift -- so that check
    alone cannot catch a regression here. Pin the exact path and the
    numeric result instead: the correct pairing gives 14.0 -> ~13.9507
    dex, versus the reversed pairing's ~13.6167 dex -- a ~0.33 dex
    difference, far larger than any floating-point tolerance.

    (13.6167, not 13.6543: the latter was quoted in the task-9 fix report
    and copied into this docstring. A wrong number in a committed file is
    the specific failure this apparatus exists to forbid, so it is
    corrected here even though no assertion depended on it.)
    """
    csv_path = tmp_path / "sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["satellite_sf", "strength", "log_mh_perh", "log_sm_smhm",
                    "log_accreted", "ratio", "log_icl", "f_icl"])
        for s, f in [(0.0, 0.0), (1.0, 0.30), (2.0, 0.50)]:
            w.writerow(["False", s, 14.0, 11.5, 11.9, 1.0, 11.2, f])
    out = run(str(csv_path), gzz07_value=0.40, out_png=str(tmp_path / "f.png"))
    assert out["path"] == ["Mvir->M500c (DuttonMaccio14, NFW)", "per_h->h_free"]
    (converted_mass,) = out["max_strength_at"].keys()
    assert converted_mass == pytest.approx(13.9507, abs=5e-4)


def test_run_surfaces_the_cosmology_mismatch_as_a_caveat(tmp_path):
    """GZZ07's own record is cosmology="wmap7"; STEEL is "planck15"
    (docs/model-assumptions.md), and `kernel.convert` performs no
    cosmology conversion. That mismatch must be surfaced, not silently
    dropped -- an apparatus built to forbid silent definitional mismatches
    must not commit one in its own flagship derivation.

    (This docstring used to say the endpoints were relabelled to planck15
    "so the CLI can run". That was false -- `_endpoint` never transmits
    cosmology -- and the relabel only hid the mismatch. Both sides now
    declare their true cosmology and the difference blocks the
    comparison.)"""
    csv_path = tmp_path / "sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["satellite_sf", "strength", "log_mh_perh", "log_sm_smhm",
                    "log_accreted", "ratio", "log_icl", "f_icl"])
        for s, f in [(0.0, 0.0), (1.0, 0.30), (2.0, 0.50)]:
            w.writerow(["False", s, 14.0, 11.5, 11.9, 1.0, 11.2, f])
    out = run(str(csv_path), gzz07_value=0.40, out_png=str(tmp_path / "f.png"))
    assert "cosmology-mismatch-wmap7-vs-planck15" in out["caveats"]


def test_run_flags_censored_points_distinctly_from_real_bounds(tmp_path):
    """A censored point (ceiling never reached by the sweep) must not be
    indistinguishable from a genuine interpolated bound in the returned
    data -- that is precisely the ambiguity that made the low-mass half
    of the figure misleading before this fix."""
    csv_path = tmp_path / "sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["satellite_sf", "strength", "log_mh_perh", "log_sm_smhm",
                    "log_accreted", "ratio", "log_icl", "f_icl"])
        # f_icl never reaches the 0.40 ceiling -- must be reported censored.
        for s, f in [(0.0, 0.0), (1.0, 0.05), (2.0, 0.10)]:
            w.writerow(["False", s, 12.0, 11.0, 10.5, 1.0, 9.5, f])
    out = run(str(csv_path), gzz07_value=0.40, out_png=str(tmp_path / "f.png"))
    (point,) = out["max_strength_at"].values()
    assert point["kind"] == "lower_limit"
    assert point["value"] == 2.0


# --------------------------------------------------------------------------
# The refusal. These are the point of the slice: the derivation's headline
# comparison is not permitted, and that has to be visible in its output.
# --------------------------------------------------------------------------

def test_the_headline_comparison_is_refused(tmp_path):
    """model f_icl vs GZZ07 f_BCG+ICL must not be silently allowed."""
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"))
    assert out["comparison_refused"] is not None
    assert out["referee_verdict"] == "REVISE"


def test_the_refusal_names_the_unknown_aperture(tmp_path):
    """The model number is a sum over the whole host with no radial
    information, so its aperture is `unknown` -- and `unknown` is supposed
    to block comparison rather than silently permit it. If this stops
    appearing, either the declaration has been quietly made false again
    (it used to claim "r500") or the gate has been defeated."""
    assert MODEL_FICL.aperture == "unknown"
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"))
    assert "aperture" in out["comparison_refused"]["blocking_fields"]


def test_the_refusal_names_every_incomparable_field(tmp_path):
    """F4's numerator/denominator mismatch shows up as quantity+component;
    the cosmology and z_range differences are the ones this derivation used
    to paper over by relabelling."""
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"))
    blocking = out["comparison_refused"]["blocking_fields"]
    for field in ("quantity", "component", "aperture", "cosmology", "z_range"):
        assert field in blocking


def test_declared_cosmology_is_the_models_own_not_a_relabel():
    """The old comment justified relabelling wmap7 -> planck15 as needed
    "so the mass-conversion CLI can run at all". `convert` never transmits
    cosmology, so that was false and the relabel only hid a real
    mismatch. Both sides now declare their true value."""
    assert MODEL_FICL.cosmology == "planck15"
    assert GZZ07_FBCGICL.cosmology == "wmap7"


def test_z_range_narrowing_is_surfaced_not_silent(tmp_path):
    """GZZ07 spans z 0.0-0.13; the sweep is a single z=0.1. The narrowing
    used to happen with no caveat at all."""
    assert tuple(GZZ07_FBCGICL.z_range) == (0.0, 0.13)
    assert tuple(MODEL_FICL.z_range) == (0.1, 0.1)
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"))
    assert "z-range-narrowed-model-0.1-vs-gzz07-0.0-0.13" in out["caveats"]


def test_out_of_scope_referee_findings_are_carried_as_caveats(tmp_path):
    """F4, F7, F9 and F10 are research questions, not code defects. They
    must ride along on the derivation so a claim inherits them rather than
    depending on someone restating them."""
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"))
    for caveat in ("f-icl-vs-f-bcg-icl-different-quantity-numerator-and-denominator",
                   "residual-circularity-moster13-central-curve",
                   "satellite-sf-switch-had-no-effect-in-this-sweep",
                   "gzz07-mass-trend-opposite-in-sign-to-model"):
        assert caveat in out["caveats"]


# --------------------------------------------------------------------------
# Redshift and the duplicate-collapse guard.
# --------------------------------------------------------------------------

def test_conversion_redshift_must_match_the_declared_z_range(tmp_path):
    """`z` was hardcoded to 0.1 and checked against nothing. It must not be
    able to drift away from the z_range the definition declares."""
    with pytest.raises(AssertionError, match="z_range"):
        run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
            out_png=str(tmp_path / "f.png"), z=0.5)


def test_missing_z_column_is_recorded_as_a_caveat_not_assumed(tmp_path):
    """falsification_sweep.rs emits no `z` column, so the model redshift
    cannot be cross-checked against the input. Say so rather than let the
    hardcoded value pass for a verified one."""
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"))
    assert "model-redshift-not-recorded-in-sweep-csv" in out["caveats"]


def test_z_column_disagreeing_with_the_conversion_raises(tmp_path):
    """If the sweep ever does record its redshift, a mismatch must stop the
    derivation rather than be converted at the wrong z."""
    path = write_sweep(tmp_path / "s.csv", header_extra=("z",), row_extra=(0.7,))
    with pytest.raises(AssertionError, match="0.7"):
        run(path, gzz07_value=0.40, out_png=str(tmp_path / "f.png"))


def test_z_column_agreeing_adds_no_caveat(tmp_path):
    path = write_sweep(tmp_path / "s.csv", header_extra=("z",), row_extra=(0.1,))
    out = run(path, gzz07_value=0.40, out_png=str(tmp_path / "f.png"))
    assert "model-redshift-not-recorded-in-sweep-csv" not in out["caveats"]


def test_duplicate_converted_masses_are_refused_not_collapsed():
    """`max_strength_at` is keyed on the converted mass, and the headline
    "14 genuine bounds of 30" count is read off it, so a repeat would
    silently drop a point instead of failing."""
    with pytest.raises(AssertionError, match="collapse"):
        _require_distinct([13.9, 13.9, 14.2], [14.0, 14.1, 14.4])


def test_distinct_converted_masses_pass():
    _require_distinct([13.9, 14.0, 14.2], [14.0, 14.1, 14.4])


def test_committed_sweep_csv_is_reproducible_from_a_clean_clone():
    """The input used to live only in a git-ignored session scratchpad."""
    assert SWEEP_CSV.exists(), f"{SWEEP_CSV} must be committed"


# --------------------------------------------------------------------------
# Gate 3: the figure is a `derivation_run` output, written through the MCP
# tools. Needs Mongo, so it is marked `integration` like test_store.py.
# --------------------------------------------------------------------------

@pytest.fixture
def tools():
    from mcp_server.server import build_server
    return build_server("mongodb://localhost:27017",
                        db="steel_research_derivation_test")


@pytest.mark.integration
def test_run_records_a_derivation_run_and_a_draft_claim(tmp_path, tools):
    seed_gzz07_measurement(tools)
    png = str(tmp_path / "f.png")
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=png, tools=tools)

    (dr,) = tools["store_query"]({"_id": DERIVATION_RUN_ID})["results"]
    assert dr["kind"] == "derivation_run"
    assert dr["payload"]["code_sha256"] == code_hash()
    assert dr["payload"]["figure"] == png
    assert dr["payload"]["inputs"] == [SWEEP_RUN_ID, GZZ07_MEASUREMENT_ID]
    assert dr["created_at"]

    (claim,) = tools["store_query"]({"_id": CLAIM_ID})["results"]
    assert claim["status"] == "draft"
    assert claim["payload"]["referee_verdict"] == REFEREE_VERDICT == "REVISE"
    assert claim["payload"]["refused"] is True
    assert out["recorded"]["claim"] == CLAIM_ID


@pytest.mark.integration
def test_claim_inherits_the_union_of_its_inputs_caveats(tmp_path, tools):
    """Gate 4. Read back from the store, not restated by hand -- including
    a caveat that lives only on an input document."""
    seed_gzz07_measurement(tools)
    out = run(write_sweep(tmp_path / "s.csv"), gzz07_value=0.40,
              out_png=str(tmp_path / "f.png"), tools=tools)
    caveats = out["recorded"]["caveats"]
    # from the GZZ07 measurement document
    assert "not-icl-only" in caveats
    assert "extraction-abstract" in caveats
    # from the sweep model_run document (F9)
    assert "satellite-sf-switch-had-no-effect-in-this-sweep" in caveats
    # from the derivation itself
    assert any(c.startswith("comparison-refused-") for c in caveats)


@pytest.mark.integration
def test_a_refused_store_write_is_not_mistaken_for_success(tools):
    """The MCP tools return {"ok": false} rather than raising, so an
    unchecked call would let a refused write pass for a successful one."""
    from derivations.icl_stripping_bound import _put
    with pytest.raises(RuntimeError, match="store refused"):
        _put(tools["store_put"], {"_id": "x", "kind": "not-a-real-kind"})
