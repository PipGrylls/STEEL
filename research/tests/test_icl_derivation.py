import csv

import pytest

from derivations.icl_stripping_bound import max_allowed_strength, run


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
        "the M500->Mvir conversion must be recorded, not skipped"
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
    dex, versus the reversed pairing's ~13.6543 dex -- a ~0.3 dex
    difference, far larger than any floating-point tolerance.
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
    """GZZ07's own record is cosmology="wmap7"; this derivation's
    endpoints are "planck15" so the CLI can run, and `kernel.convert`
    performs no cosmology conversion. That mismatch must be surfaced, not
    silently dropped -- an apparatus built to forbid silent definitional
    mismatches must not commit one in its own flagship derivation."""
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
