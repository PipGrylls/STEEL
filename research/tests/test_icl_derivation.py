import csv

from derivations.icl_stripping_bound import max_allowed_strength, run


def test_max_allowed_strength_interpolates_the_crossing():
    """The helper previously copy-pasted between two plotting scripts."""
    strengths = [0.0, 1.0, 2.0]
    f_icl = [0.0, 0.30, 0.50]
    # bound 0.40 sits halfway between strengths 1 and 2
    assert abs(max_allowed_strength(strengths, f_icl, 0.40) - 1.5) < 1e-9


def test_ceiling_below_every_sample_gives_zero():
    assert max_allowed_strength([0.0, 1.0], [0.5, 0.9], 0.1) == 0.0


def test_ceiling_above_every_sample_returns_the_tested_maximum():
    assert max_allowed_strength([0.0, 4.0], [0.0, 0.1], 0.9) == 4.0


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
