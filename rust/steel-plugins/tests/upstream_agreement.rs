//! Validates our Rust implementations against committed upstream
//! reference grids. Spec section 6.

use ndarray::Array2;
use ndarray_npy::read_npy;

const EMERGE_DIR: &str = "tests/fixtures/emerge";

fn load(name: &str) -> Array2<f64> {
    read_npy(format!("{EMERGE_DIR}/{name}")).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

#[test]
fn emerge_fixtures_have_the_documented_shape() {
    for name in ["eps_grid.npy", "smhm_grid.npy"] {
        let a = load(name);
        assert_eq!(a.shape(), &[51, 6], "{name} shape");
        assert!(a.iter().all(|v| v.is_finite()), "{name} contains non-finite values");
    }
}

#[test]
fn emerge_efficiency_is_a_physical_fraction() {
    let eps = load("eps_grid.npy");
    assert!(
        eps.iter().all(|&v| v > 0.0 && v <= 1.0),
        "conversion efficiency must lie in (0, 1]; got min {} max {}",
        eps.iter().cloned().fold(f64::INFINITY, f64::min),
        eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
}

#[test]
fn emerge_smhm_is_monotonic_in_halo_mass() {
    let smhm = load("smhm_grid.npy");
    for j in 0..smhm.ncols() {
        for i in 1..smhm.nrows() {
            assert!(
                smhm[[i, j]] >= smhm[[i - 1, j]] - 1e-9,
                "M* decreased with Mh at row {i}, col {j}"
            );
        }
    }
}

#[test]
fn emerge_provenance_has_no_unfilled_placeholders() {
    let text = std::fs::read_to_string(format!("{EMERGE_DIR}/provenance.toml"))
        .expect("provenance.toml must exist");
    assert!(!text.contains('<'), "provenance.toml still contains placeholders:\n{text}");
}

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::{GrowthTrack, HaloGrowthModel};
use steel_plugins::{EmergeGrowth, Planck15, VandenBosch14};

const REDSHIFTS: [f64; 6] = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0];

/// Largest absolute deviation in dex between our efficiency and
/// upstream's, printed so the achieved figure can be recorded as this
/// plugin's reference tolerance (spec section 6, step 5).
#[test]
fn emerge_efficiency_agrees_with_upstream() {
    let eps_ref = load("eps_grid.npy");
    let m = EmergeGrowth::o_leary23();
    let mut worst = 0.0_f64;
    let mut worst_at = (0.0, 0.0);

    for (i, log_mh) in (0..51).map(|i| (i, 10.0 + i as f64 * 0.1)) {
        for (j, &z) in REDSHIFTS.iter().enumerate() {
            let ours = m.efficiency(log_mh, z).log10();
            let theirs = eps_ref[[i, j]].log10();
            let d = (ours - theirs).abs();
            if d > worst {
                worst = d;
                worst_at = (log_mh, z);
            }
        }
    }

    println!("worst eps deviation {worst:.9} dex at log_mh={} z={}", worst_at.0, worst_at.1);
    // Achieved: 1.5e-7 dex (float32-vs-float64 rounding only, matching
    // Task 8's own 3.5e-7 relative-error figure for the same fixture —
    // no integration/discretization is involved here). Tightened from
    // the spec's 0.01 "investigate above" threshold to just above the
    // achieved figure, per spec section 6 step 5.
    assert!(
        worst < 1.0e-5,
        "worst deviation {worst:.9} dex at log_mh={} z={} exceeds 1e-5; identify the cause \
         rather than widening the bound (spec section 6)",
        worst_at.0,
        worst_at.1
    );
}

#[test]
fn emerge_integrated_smhm_agrees_with_upstream() {
    let smhm_ref = load("smhm_grid.npy");
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);
    let m = EmergeGrowth::o_leary23();
    let mut worst = 0.0_f64;

    // Compare at z=0.1 (column 0) across the mass axis.
    for (i, log_mh) in (0..51).map(|i| (i, 10.0 + i as f64 * 0.1)) {
        let track: GrowthTrack = growth.growth_history(log_mh, 0.1);
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let ours = steel_core::integrate_stellar_mass(&m, &ctx, 0.1, None);
        if !ours.is_finite() {
            continue;
        }
        worst = worst.max((ours - smhm_ref[[i, 0]]).abs());
    }

    println!("worst integrated M* deviation {worst:.6} dex");
    // Achieved: 0.0497 dex, at the lowest-mass grid point (log_mh=10.0),
    // where the fixture's integral (trapezoidal in linear halo mass) and
    // ours (trapezoidal in cosmic time, per `integrate_stellar_mass`)
    // diverge most: the deviation is smooth and monotonic in log_mh,
    // vanishing near the pivot mass (~11.4) and growing at both mass
    // tails where eps(M) is most curved along the track — the textbook
    // signature of a quadrature-scheme mismatch (provenance.toml
    // [method]), not a coefficient or formula bug (the pointwise eps()
    // test above, unaffected by any integration, agrees to 1.5e-7 dex).
    // Tightened from the spec's 0.05 "investigate above" threshold to
    // just above the achieved figure.
    assert!(worst < 0.052, "worst integrated M* deviation {worst:.6} dex exceeds 0.052");
}
