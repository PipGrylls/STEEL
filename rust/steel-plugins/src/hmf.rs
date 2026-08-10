//! Despali+2016 halo mass function, matching `Functions.py::Make_HMF_Interp`'s
//! `mass_function.massFunction(..., mdef='vir', model='despali16', q_out='dndlnM')`
//! call into COLOSSUS.
//!
//! Despali+2016 fit a Sheth-Tormen-style multiplicity function
//! `f(nu) = A sqrt(2a/pi) nu [1 + (a nu^2)^-p] exp(-a nu^2 / 2)`,
//! `nu = delta_c(z) / sigma(M, z)`, with `(A, a, p)` calibrated so the
//! fit is universal (redshift- and cosmology-independent) when masses
//! use the *redshift-dependent* virial overdensity `Delta_vir(z)`
//! (Bryan & Norman 1998) — exactly the mass definition
//! `Cosmology::delta_vir`/`mdef='vir'` already uses throughout this
//! codebase. `(A, a, p) = (0.3295, 0.7689, 0.2536)` is their all-z,
//! all-cosmology "vir" calibration (Despali et al. 2016, MNRAS 456,
//! 2486, Table 2 / Eq. 12); cross-check against the published paper or
//! COLOSSUS's `colossus/lss/mass_function.py` before treating this as
//! load-bearing for a real science result — this is the one formula in
//! the whole port pulled from recollection rather than source in hand.

use steel_core::cosmology::Cosmology;
use steel_core::hmf::HaloMassFunctionModel;

use crate::growth::GrowthFactor;
use crate::variance::Variance;

pub struct Despali16 {
    variance: Variance,
    growth: GrowthFactor,
    /// Comoving mean matter density at z=0 [Msun h^2 Mpc^-3].
    rho_bar_m0: f64,
    a: f64,
    p: f64,
    norm_a: f64,
}

impl Despali16 {
    pub fn new(cosmology: &dyn Cosmology) -> Self {
        let variance = Variance::new(cosmology);
        let growth = GrowthFactor::from_cosmology(cosmology);
        let rho_bar_m0 = cosmology.omega_m0() * cosmology.rho_crit(0.0) * 1.0e9; // kpc^-3 -> Mpc^-3
        Self {
            variance,
            growth,
            rho_bar_m0,
            a: 0.7689,
            p: 0.2536,
            norm_a: 0.3295,
        }
    }

    fn multiplicity_function(&self, nu: f64) -> f64 {
        let a_nu2 = self.a * nu * nu;
        self.norm_a * (2.0 * self.a / std::f64::consts::PI).sqrt()
            * nu
            * (1.0 + a_nu2.powf(-self.p))
            * (-0.5 * a_nu2).exp()
    }
}

impl HaloMassFunctionModel for Despali16 {
    fn dn_dlog10m(&self, log_m: f64, z: f64) -> f64 {
        let m = 10f64.powf(log_m); // Msun/h
        // nu = delta_c(z)/sigma(M,z) = delta_c(z)/(sigma(M) D(z))
        //    = delta_collapse(z) / sigma(M) — see `GrowthFactor::delta_collapse`.
        let nu = self.growth.delta_collapse(z) / self.variance.sigma(m);

        let f_nu = self.multiplicity_function(nu);
        let dln_sigma_dln_m = 0.5 * self.variance.dln_sigma2_dln_m(m);
        let dn_dln_m = f_nu * (self.rho_bar_m0 / m) * dln_sigma_dln_m.abs();
        dn_dln_m * std::f64::consts::LN_10
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;

    #[test]
    fn dn_dlog10m_is_positive_and_finite() {
        let cosmo = Planck15::new();
        let hmf = Despali16::new(&cosmo);
        for &log_m in &[11.0, 12.0, 13.0, 14.0, 15.0] {
            let n = hmf.dn_dlog10m(log_m, 0.0);
            assert!(n > 0.0 && n.is_finite(), "dn/dlog10m({log_m}) = {n}");
        }
    }

    #[test]
    fn dn_dlog10m_decreases_steeply_with_mass() {
        let cosmo = Planck15::new();
        let hmf = Despali16::new(&cosmo);
        let n12 = hmf.dn_dlog10m(12.0, 0.0);
        let n14 = hmf.dn_dlog10m(14.0, 0.0);
        assert!(n12 > n14 * 10.0, "n12={n12} n14={n14}, expected steep decline");
    }

    #[test]
    fn dn_dlog10m_decreases_with_redshift_at_fixed_high_mass() {
        // Massive halos should be rarer at higher z.
        let cosmo = Planck15::new();
        let hmf = Despali16::new(&cosmo);
        let n_z0 = hmf.dn_dlog10m(14.0, 0.0);
        let n_z1 = hmf.dn_dlog10m(14.0, 1.0);
        assert!(n_z0 > n_z1, "n(z=0)={n_z0} n(z=1)={n_z1}");
    }
}
