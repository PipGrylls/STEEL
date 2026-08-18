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
