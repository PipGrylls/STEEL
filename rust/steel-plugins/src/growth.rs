//! Linear growth factor D(z) and the spherical-collapse threshold,
//! needed by the van den Bosch (2014) halo-growth root-find and by the
//! Despali+2016 halo mass function.

use crate::numerics::{cumulative_trapezoid, InterpTable};
use steel_core::cosmology::Cosmology;

/// The linear growth factor D(a), normalized D(a=1) = 1.
///
/// `cosmo_sub.f::growth_rate` computes the same physical quantity via a
/// closed-form-plus-integral representation (`f2`/`f3`/`ff3` + Romberg
/// quadrature) specific to flat matter+Lambda cosmologies. We instead
/// tabulate the standard growth integral
/// `D(a) ∝ E(a) int_0^a da' / (a' E(a'))^3`
/// (matter + dark energy only, no radiation term — matching the
/// physical content of the Fortran's own formula, which also omits
/// radiation) via cumulative trapezoidal integration, reusing the same
/// pattern `Planck15::age` uses. Numerically equivalent, simpler to get
/// right in Rust than porting the Fortran's own quadrature machinery.
pub struct GrowthFactor {
    omega_m0: f64,
    omega_de0: f64,
    /// D(a) normalized so D(1) = 1, tabulated over a in (0, 1].
    table: InterpTable,
}

impl GrowthFactor {
    const N_GRID: usize = 4000;
    const A_MIN: f64 = 1e-6;

    pub fn new(omega_m0: f64, omega_de0: f64) -> Self {
        let n = Self::N_GRID;
        let a_min = Self::A_MIN;
        let a: Vec<f64> = (0..=n)
            .map(|i| a_min + (1.0 - a_min) * i as f64 / n as f64)
            .collect();
        let e_of_a = |ai: f64| (omega_m0 * ai.powi(-3) + omega_de0).sqrt();
        // Integrand -> 0 as a' -> 0 in matter domination (no singularity,
        // same reasoning as the age(a) integrand in Planck15).
        let integrand: Vec<f64> = a.iter().map(|&ai| 1.0 / (ai * e_of_a(ai)).powi(3)).collect();
        let cum = cumulative_trapezoid(&a, &integrand);
        let unnormalized: Vec<f64> = a.iter().zip(&cum).map(|(&ai, &ci)| e_of_a(ai) * ci).collect();
        let d_at_1 = *unnormalized.last().unwrap();
        let normalized: Vec<f64> = unnormalized.iter().map(|&d| d / d_at_1).collect();

        Self {
            omega_m0,
            omega_de0,
            table: InterpTable::new(a, normalized),
        }
    }

    pub fn from_cosmology(cosmology: &dyn Cosmology) -> Self {
        // Growth theory conventionally omits radiation; fold whatever
        // small Omega_r0 there is into the dark-energy term so the two
        // still sum to 1 (flat).
        Self::new(cosmology.omega_m0(), cosmology.omega_de0() + cosmology.omega_r0())
    }

    /// D(a), a = 1/(1+z), normalized D(1) = 1.
    pub fn d_of_a(&self, a: f64) -> f64 {
        self.table.eval(a)
    }

    /// D(z), normalized D(z=0) = 1.
    pub fn d_of_z(&self, z: f64) -> f64 {
        self.d_of_a(1.0 / (1.0 + z))
    }

    pub fn omega_m0(&self) -> f64 {
        self.omega_m0
    }

    pub fn omega_de0(&self) -> f64 {
        self.omega_de0
    }

    /// Omega_m(z), reconstructed from the cached Omega_m0/Omega_de0
    /// (radiation folded into the dark-energy term at construction —
    /// see [`Self::from_cosmology`]). Self-contained so callers that
    /// only hold a `GrowthFactor` (not a `dyn Cosmology`) can still get
    /// Omega_m(z), e.g. for [`Self::delta_collapse`].
    pub fn omega_m_at_z(&self, z: f64) -> f64 {
        let e_z2 = self.omega_m0 * (1.0 + z).powi(3) + self.omega_de0;
        self.omega_m0 * (1.0 + z).powi(3) / e_z2
    }

    /// `Delta_collapse(z) = delta_c(Omega_m(z)) / D(z)`
    /// (`getPWGH.f`'s main mass-accretion-history loop), the
    /// linear-theory collapse threshold scaled by the growth factor.
    /// Used by both `VandenBosch14` (directly) and `Despali16`
    /// (algebraically: `nu = delta_c(z)/sigma(M,z)
    /// = delta_c(z)/(sigma(M) D(z)) = delta_collapse(z)/sigma(M)`, the
    /// same quantity divided differently).
    pub fn delta_collapse(&self, z: f64) -> f64 {
        delta_c(self.omega_m_at_z(z)) / self.d_of_z(z)
    }
}

/// Critical overdensity for collapse at z=0 in the flat-LCDM fitting
/// form used by `cosmo_sub.f::Delta_c` (Nakamura & Suto 1997 /
/// Navarro-Frenk-White 1997 approximation): `delta_c(z) = 0.15 (12 pi)^(2/3) Omega_m(z)^0.0055`.
///
/// This is a different quantity from [`steel_core::Cosmology::delta_vir`]
/// (the Bryan & Norman 1998 virial *overdensity*, in units of the
/// critical density) — `delta_c` is the linear-theory collapse
/// threshold, only used inside the growth-history root-find and the
/// halo mass function.
pub fn delta_c(omega_m_z: f64) -> f64 {
    0.15 * (12.0 * std::f64::consts::PI).powf(2.0 / 3.0) * omega_m_z.powf(0.0055)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn growth_factor_is_one_at_z_zero() {
        let g = GrowthFactor::new(0.3089, 0.6911);
        assert!((g.d_of_z(0.0) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn growth_factor_decreases_with_redshift() {
        let g = GrowthFactor::new(0.3089, 0.6911);
        let d0 = g.d_of_z(0.0);
        let d1 = g.d_of_z(1.0);
        let d5 = g.d_of_z(5.0);
        assert!(d0 > d1 && d1 > d5, "D(0)={d0} D(1)={d1} D(5)={d5}");
    }

    #[test]
    fn growth_factor_approaches_matter_domination_scaling_at_high_z() {
        // In matter domination D(a) ~ a, so D(z)*(1+z) should tend to a
        // constant at high z.
        let g = GrowthFactor::new(0.3089, 0.6911);
        let r10 = g.d_of_z(10.0) * 11.0;
        let r20 = g.d_of_z(20.0) * 21.0;
        assert!((r10 - r20).abs() / r10 < 0.05, "r10={r10} r20={r20}");
    }

    #[test]
    fn delta_c_matches_flat_value_at_z_zero() {
        // At z=0 with Omega_m(0)=Omega_m0, matches the well known
        // Omega_m0=1 Einstein-de-Sitter value of ~1.686 closely for
        // Omega_m0 close to 1, and is close to that even at Omega_m0~0.3
        // since the exponent 0.0055 is tiny.
        let d = delta_c(0.3089);
        assert!((1.67..1.7).contains(&d), "delta_c(Om=0.3089) = {d}");
    }
}
