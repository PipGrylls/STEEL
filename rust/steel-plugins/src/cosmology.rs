//! Planck15 flat LCDM cosmology (COLOSSUS's `"planck15"` preset), the
//! only [`Cosmology`] implementation shipped today.

use steel_core::cosmology::Cosmology;

use crate::numerics::{cumulative_trapezoid, InterpTable};

/// 1 / (100 km/s/Mpc), in Gyr — the Hubble time for `H0 = 100 h`.
const HUBBLE_TIME_GYR_PER_H: f64 = 9.778127;

/// Flat LCDM cosmology with the Planck 2015 parameters used throughout
/// STEEL (`cosmology.setCosmology("planck15")` in the Python).
pub struct Planck15 {
    h0: f64,
    omega_m0: f64,
    omega_b0: f64,
    omega_r0: f64,
    omega_de0: f64,
    sigma8: f64,
    n_spec: f64,
    /// age(a) in units of the Hubble time (1/H0), tabulated by
    /// cumulative trapezoidal integration and queried by linear
    /// interpolation.
    age_table: InterpTable,
}

impl Planck15 {
    /// Number of points in the internal age(a) integration table.
    /// Comfortably resolves the z <~ 10 range STEEL actually evaluates
    /// (halo/subhalo mass function grids top out around z=6-7).
    const N_GRID: usize = 4000;
    const A_MIN: f64 = 1e-6;

    pub fn new() -> Self {
        let omega_m0 = 0.3089;
        let omega_b0 = 0.0486;
        let h0 = 67.74;
        let sigma8 = 0.8159;
        let n_spec = 0.9667;
        // Radiation density: Omega_gamma0 h^2 = 2.4692e-5 (T_cmb=2.7255K),
        // scaled by (1 + 0.2271 N_eff) with N_eff = 3.046 to include
        // neutrinos. Approximate; refine against COLOSSUS if Milestone 2
        // validation shows it matters at the precision we need.
        let h = h0 / 100.0;
        let omega_gamma0 = 2.4692e-5 / (h * h);
        let omega_r0 = omega_gamma0 * (1.0 + 0.2271 * 3.046);
        let omega_de0 = 1.0 - omega_m0 - omega_r0; // flat

        let age_table = Self::build_age_table(omega_m0, omega_r0, omega_de0);

        Self {
            h0,
            omega_m0,
            omega_b0,
            omega_r0,
            omega_de0,
            sigma8,
            n_spec,
            age_table,
        }
    }

    fn e_of_a(a: f64, omega_m0: f64, omega_r0: f64, omega_de0: f64) -> f64 {
        (omega_m0 * a.powi(-3) + omega_r0 * a.powi(-4) + omega_de0).sqrt()
    }

    fn build_age_table(omega_m0: f64, omega_r0: f64, omega_de0: f64) -> InterpTable {
        let n = Self::N_GRID;
        let a_min = Self::A_MIN;
        let a: Vec<f64> = (0..=n)
            .map(|i| a_min + (1.0 - a_min) * i as f64 / n as f64)
            .collect();
        // Integrand for t(a) H0 = int_0^a da' / (a' E(a')), which is
        // smooth and -> 0 as a' -> 0 (radiation-dominated limit).
        let integrand: Vec<f64> = a
            .iter()
            .map(|&ai| 1.0 / (ai * Self::e_of_a(ai, omega_m0, omega_r0, omega_de0)))
            .collect();
        let mut age_in_hubble_times = cumulative_trapezoid(&a, &integrand);
        // Analytic radiation-domination age at a_min: t H0 = a^2 / (2 sqrt(Omega_r0)),
        // added as the starting offset the cumulative integral is missing
        // (it started at a_min, not a=0).
        let offset = a_min * a_min / (2.0 * omega_r0.sqrt());
        for v in age_in_hubble_times.iter_mut() {
            *v += offset;
        }
        InterpTable::new(a, age_in_hubble_times)
    }

    fn hubble_time_gyr(&self) -> f64 {
        HUBBLE_TIME_GYR_PER_H / self.h()
    }
}

impl Default for Planck15 {
    fn default() -> Self {
        Self::new()
    }
}

impl Cosmology for Planck15 {
    fn h0(&self) -> f64 {
        self.h0
    }

    fn omega_m0(&self) -> f64 {
        self.omega_m0
    }

    fn omega_b0(&self) -> f64 {
        self.omega_b0
    }

    fn omega_de0(&self) -> f64 {
        self.omega_de0
    }

    fn omega_r0(&self) -> f64 {
        self.omega_r0
    }

    fn sigma8(&self) -> f64 {
        self.sigma8
    }

    fn n_spec(&self) -> f64 {
        self.n_spec
    }

    fn e_z(&self, z: f64) -> f64 {
        Self::e_of_a(1.0 / (1.0 + z), self.omega_m0, self.omega_r0, self.omega_de0)
    }

    fn age(&self, z: f64) -> f64 {
        let a = 1.0 / (1.0 + z);
        self.age_table.eval(a) * self.hubble_time_gyr()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use steel_core::cosmology::MassDefinition;

    #[test]
    fn e_z_at_zero_is_one() {
        let cosmo = Planck15::new();
        assert!((cosmo.e_z(0.0) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn age_is_monotonically_decreasing_with_redshift() {
        let cosmo = Planck15::new();
        let ages: Vec<f64> = [0.0, 0.5, 1.0, 2.0, 5.0].iter().map(|&z| cosmo.age(z)).collect();
        for w in ages.windows(2) {
            assert!(w[0] > w[1], "age should decrease with increasing z: {:?}", ages);
        }
    }

    #[test]
    fn age_at_z0_is_close_to_known_planck_value() {
        // Planck15 age of the universe today is ~13.8 Gyr.
        let cosmo = Planck15::new();
        let age0 = cosmo.age(0.0);
        assert!((13.0..14.5).contains(&age0), "age(0) = {age0}, expected ~13.8 Gyr");
    }

    #[test]
    fn delta_vir_matches_bryan_norman_at_z0() {
        let cosmo = Planck15::new();
        // At z=0, Omega_m(0) = Omega_m0 for a flat cosmology.
        let x = cosmo.omega_m0 - 1.0;
        let expected = 18.0 * std::f64::consts::PI.powi(2) + 82.0 * x - 39.0 * x * x;
        assert!((cosmo.delta_vir(0.0) - expected).abs() < 1e-9);
    }

    #[test]
    fn m_to_r_scales_as_cube_root_of_mass() {
        let cosmo = Planck15::new();
        let r1 = cosmo.m_to_r(1e12, 0.0, MassDefinition::Vir);
        let r8 = cosmo.m_to_r(8e12, 0.0, MassDefinition::Vir);
        assert!((r8 / r1 - 2.0).abs() < 1e-9);
    }

    #[test]
    fn m_to_r_is_reasonable_for_milky_way_mass_halo() {
        // A ~1e12 Msun/h halo at z=0 should have a virial radius of a
        // few hundred kpc/h.
        let cosmo = Planck15::new();
        let r = cosmo.m_to_r(1e12, 0.0, MassDefinition::Vir);
        assert!((50.0..500.0).contains(&r), "R_vir = {r} kpc/h, expected O(100s)");
    }
}
