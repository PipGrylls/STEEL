//! Galaxy gas-mass model, a direct port of
//! `Functions.py::GetGasMass`/`GetMaxGasMass`.
//!
//! Named `StewartScaling` to match the plan's naming, but note this is
//! *not* actually the Stewart+2009 scaling relation the code's old,
//! commented-out formula was attributed to (`Functions.py:101-104`,
//! dead code) — the active relation (`9.22 + 0.81 log(SFR)`) is
//! labeled in the source only as a "new relation using the
//! M*-SFR-Mgas proxy", with no paper attribution given. Flagging this
//! so the name isn't read as a citation claim.

use rand::RngCore;
use rand_distr::{Distribution, Normal};

use steel_core::cosmology::Cosmology;
use steel_core::gas::GasMassModel;

pub struct StewartScaling {
    /// dex scatter on the SFR-gas-mass proxy (`Functions.py`'s fixed
    /// `0.2`).
    pub scatter: f64,
    omega_b0: f64,
    omega_m0: f64,
}

impl StewartScaling {
    pub fn from_cosmology(cosmology: &dyn Cosmology) -> Self {
        Self { scatter: 0.2, omega_b0: cosmology.omega_b0(), omega_m0: cosmology.omega_m0() }
    }
}

impl GasMassModel for StewartScaling {
    fn max_gas_mass(&self, log_halo_mass: f64) -> f64 {
        log_halo_mass + (self.omega_b0 / self.omega_m0).log10()
    }

    fn gas_mass(&self, log_sfr: f64, log_halo_mass: f64, rng: Option<&mut dyn RngCore>) -> f64 {
        let mean = 9.22 + 0.81 * log_sfr;
        // `scatter` is a public field; guard against a non-positive or
        // non-finite value rather than letting `Normal::new(..).unwrap()`
        // panic (same class of issue as the SMHM models').
        let draw = match rng {
            Some(rng) if self.scatter > 0.0 && self.scatter.is_finite() => {
                Normal::new(mean, self.scatter).unwrap().sample(rng)
            }
            _ => mean,
        };
        draw.min(self.max_gas_mass(log_halo_mass))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn max_gas_mass_is_below_the_halo_mass() {
        let cosmo = Planck15::new();
        let model = StewartScaling::from_cosmology(&cosmo);
        let max_gas = model.max_gas_mass(12.0);
        assert!(max_gas < 12.0, "max_gas={max_gas} should be below the halo mass (baryon fraction < 1)");
    }

    #[test]
    fn gas_mass_never_exceeds_the_cap() {
        let cosmo = Planck15::new();
        let model = StewartScaling::from_cosmology(&cosmo);
        let mut rng = StdRng::seed_from_u64(7);
        for _ in 0..1000 {
            let g = model.gas_mass(2.0, 10.0, Some(&mut rng)); // deliberately high SFR to try to blow the cap
            assert!(g <= model.max_gas_mass(10.0) + 1e-9, "gas_mass={g} exceeded the cap");
        }
    }

    #[test]
    fn invalid_scatter_is_treated_as_no_scatter_rather_than_panicking() {
        // `scatter` is a public field, so these values are reachable and
        // would otherwise panic inside `Normal::new`.
        let cosmo = Planck15::new();
        let expected_mean = 9.22 - 0.81; // log_sfr = -1.0, well under the cap
        for bad in [0.0, -0.5, f64::NAN] {
            let mut model = StewartScaling::from_cosmology(&cosmo);
            model.scatter = bad;
            let mut rng = StdRng::seed_from_u64(1);
            let got = model.gas_mass(-1.0, 14.0, Some(&mut rng));
            assert!((got - expected_mean).abs() < 1e-12, "scatter={bad} should give the unscattered mean");
        }
    }
}
