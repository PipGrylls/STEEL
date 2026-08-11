//! Dynamical-friction merger/infall timescale, a direct port of
//! `Functions.py::DynamicalFriction` + `DynamicalTime_Fun`
//! (McCavana 2012 mass-ratio parametrization of the
//! Boylan-Kolchin+2008 formula).

use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::merger_time::MergerTimescaleModel;

/// McCavana 2012 / Boylan-Kolchin+2008 dynamical friction timescale.
///
/// Note on spelling: `Paramaters` in the field docs below is *not* a
/// typo on our side — it is the Python's own misspelled identifier
/// (`Functions.py` uses it 52 times, e.g. `Paramaters['AltDynamicalTime']`
/// at line 473). These backticked strings are exact, grep-able
/// cross-references to the source, so "correcting" them would point at
/// something that does not exist in the codebase.
///
/// The Python's stochastic alternative for the orbital-circularity
/// parameter (`NormalRnd` drawn from `N(0.5, 0.23)`, Khochfar & Burkert
/// 2006) is commented-out dead code (`Functions.py:480-486`) — every
/// real STEEL run uses the fixed `NormalRnd=0.5` "average circular
/// orbit" value, which is what this ports.
pub struct McCavanaBK08 {
    /// Multiplicative factor on the dynamical time (`Paramaters['AltDynamicalTime']`).
    /// `1.0` reproduces the unmodified timescale.
    pub dynamical_time_factor: f64,
    /// Extra `1/(1+z)` redshift correction on the dynamical time
    /// (`Paramaters['AltDynamicalTimeB']`).
    pub redshift_correction: bool,
}

impl McCavanaBK08 {
    pub fn new(dynamical_time_factor: f64, redshift_correction: bool) -> Self {
        Self { dynamical_time_factor, redshift_correction }
    }

    fn dynamical_time_gyr(&self, z: f64, cosmology: &dyn Cosmology) -> f64 {
        let mut t_dyn =
            1.628 / cosmology.h() * (cosmology.delta_vir(z) / 178.0).powf(-0.5) * cosmology.e_z(z).powi(-1);
        t_dyn *= self.dynamical_time_factor;
        if self.redshift_correction {
            t_dyn *= 1.0 / (1.0 + z);
        }
        t_dyn
    }
}

impl Default for McCavanaBK08 {
    fn default() -> Self {
        Self::new(1.0, false)
    }
}

impl MergerTimescaleModel for McCavanaBK08 {
    fn infall_time(
        &self,
        log_host_mass: f64,
        log_sat_mass: f64,
        z: f64,
        cosmology: &dyn Cosmology,
    ) -> f64 {
        const A: f64 = 0.9;
        const B: f64 = 1.0;
        const C: f64 = 0.6;
        const D: f64 = 0.1;
        const NORMAL_RND: f64 = 0.5; // "average" circular-orbit value

        let mass_ratio = 10f64.powf(log_host_mass - log_sat_mass);
        // Virial radius in physical Mpc: m_to_r returns a number in
        // kpc/h (the h-scaled convention used throughout this port —
        // dividing by h*1000 gives the physical Mpc value), matching
        // `Functions.py::DynamicalFriction`'s
        // `M_to_R(...) / (h * 10**3)`.
        let vr_mpc = cosmology.m_to_r(10f64.powf(log_host_mass), z, MassDefinition::Vir)
            / (cosmology.h() * 1000.0);
        let t_dyn = self.dynamical_time_gyr(z, cosmology);

        let orbital_energy =
            (vr_mpc * NORMAL_RND.powf(2.17)) / (1.0 - (1.0 - NORMAL_RND * NORMAL_RND).sqrt());

        let part1 = mass_ratio.powf(B);
        let part2 = (1.0 + mass_ratio).ln();
        let part3 = (C * NORMAL_RND).exp();
        let part4 = (orbital_energy / vr_mpc).powf(D);

        t_dyn * A * (part1 / part2) * part3 * part4
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;

    #[test]
    fn infall_time_is_positive_and_finite() {
        let cosmo = Planck15::new();
        let model = McCavanaBK08::default();
        let t = model.infall_time(14.0, 12.0, 0.0, &cosmo);
        assert!(t > 0.0 && t.is_finite(), "t = {t}");
    }

    #[test]
    fn larger_mass_ratio_gives_longer_infall_time() {
        // A satellite much less massive than its host sinks more slowly.
        let cosmo = Planck15::new();
        let model = McCavanaBK08::default();
        let t_small_ratio = model.infall_time(12.0, 11.9, 0.0, &cosmo); // host/sat ~ 1.26
        let t_large_ratio = model.infall_time(14.0, 11.0, 0.0, &cosmo); // host/sat ~ 1000
        assert!(t_large_ratio > t_small_ratio, "{t_large_ratio} vs {t_small_ratio}");
    }

    #[test]
    fn dynamical_time_factor_scales_linearly() {
        let cosmo = Planck15::new();
        let base = McCavanaBK08::new(1.0, false);
        let doubled = McCavanaBK08::new(2.0, false);
        let t_base = base.infall_time(13.0, 11.0, 0.5, &cosmo);
        let t_doubled = doubled.infall_time(13.0, 11.0, 0.5, &cosmo);
        assert!((t_doubled / t_base - 2.0).abs() < 1e-9);
    }
}
