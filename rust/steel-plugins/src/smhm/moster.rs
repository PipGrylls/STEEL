//! Moster-form stellar-mass-halo-mass relation, a direct port of
//! `Functions.py::DarkMatterToStellarMass` — which is really one
//! equation (Moster 2010, Eq. 2) shared by eight named presets
//! (`Moster13`, `Moster10`, `G18`, `G18_notSE`, `G19_SE`, `G19_cMod`,
//! `Illustris`, `Override`) that only differ in their eight
//! coefficients, not their functional form.

use rand::RngCore;
use rand_distr::{Distribution, Normal};

use steel_core::smhm::SmhmModel;

/// Which redshift-evolution parametrization applies. The Python's three
/// `if`/`elif`/`else` branches (`Functions.py:526-534`) collapse to two
/// distinct outcomes: the `elif` and `else` arms compute the identical
/// expression, so only `Moster` is actually special-cased.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ZEvo {
    /// No redshift evolution: `zparameter = 0`.
    Fixed,
    /// `zparameter = z/(1+z)` — only the `Moster13`/`Moster10` presets
    /// use this form.
    MosterStyle,
    /// `zparameter = (z-0.1)/(1+z)` — every other preset.
    ShiftedStyle,
}

pub struct MosterFormSmhm {
    pub m10: f64,
    pub shmnorm10: f64,
    pub beta10: f64,
    pub gamma10: f64,
    pub m11: f64,
    pub shmnorm11: f64,
    pub beta11: f64,
    pub gamma11: f64,
    pub scatter: f64,
    pub z_evo: ZEvo,
}

impl MosterFormSmhm {
    fn z_parameter(&self, z: f64) -> f64 {
        match self.z_evo {
            ZEvo::Fixed => 0.0,
            ZEvo::MosterStyle => z / (1.0 + z),
            ZEvo::ShiftedStyle => (z - 0.1) / (1.0 + z),
        }
    }

    /// Moster+2013, `AbnMtch['Moster']` in the Python.
    pub fn moster13(z_evo: bool) -> Self {
        Self {
            m10: 11.590,
            shmnorm10: 0.0351,
            beta10: 1.376,
            gamma10: 0.608,
            m11: 1.195,
            shmnorm11: -0.0247,
            beta11: -0.826,
            gamma11: 0.329,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::MosterStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['Moster10']`.
    pub fn moster10(z_evo: bool) -> Self {
        Self {
            m10: 11.884,
            shmnorm10: 0.28320,
            beta10: 1.057,
            gamma10: 0.556,
            m11: 1.195,
            shmnorm11: -0.0247,
            beta11: -0.826,
            gamma11: 0.329,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['G18']`.
    pub fn g18(z_evo: bool) -> Self {
        Self {
            m10: 11.95,
            shmnorm10: 0.032,
            beta10: 1.61,
            gamma10: 0.54,
            m11: 0.4,
            shmnorm11: -0.02,
            beta11: -0.6,
            gamma11: -0.1,
            scatter: 0.11,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['G18_notSE']`.
    pub fn g18_not_se(z_evo: bool) -> Self {
        Self {
            m10: 11.95,
            shmnorm10: 0.032,
            beta10: 1.61,
            gamma10: 0.62,
            m11: 0.4,
            shmnorm11: -0.02,
            beta11: -0.6,
            gamma11: 0.0,
            scatter: 0.11,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['G19_SE']` — STEEL's default abundance-matching choice.
    pub fn g19_se(z_evo: bool) -> Self {
        Self {
            m10: 12.0,
            shmnorm10: 0.032,
            beta10: 1.5,
            gamma10: 0.56,
            m11: 0.6,
            shmnorm11: -0.014,
            beta11: -0.7,
            gamma11: 0.08,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['G19_cMod']`.
    pub fn g19_c_mod(z_evo: bool) -> Self {
        Self {
            m10: 12.0,
            shmnorm10: 0.032,
            beta10: 1.74,
            gamma10: 0.66,
            m11: 0.4,
            shmnorm11: -0.024,
            beta11: -0.74,
            gamma11: -0.12,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['Illustris']`.
    pub fn illustris(z_evo: bool) -> Self {
        Self {
            m10: 11.8,
            shmnorm10: 0.018,
            beta10: 1.5,
            gamma10: 0.31,
            m11: 0.0,
            shmnorm11: -0.01,
            beta11: 0.0,
            gamma11: -0.12,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['Override_0']`: user-supplied z=0.1 coefficients, with
    /// the high-z evolution terms fixed at the Python's hardcoded
    /// defaults (`Functions.py:570`) rather than user-supplied.
    pub fn override_z0(m10: f64, shmnorm10: f64, beta10: f64, gamma10: f64, scatter: f64, z_evo: bool) -> Self {
        Self {
            m10,
            shmnorm10,
            beta10,
            gamma10,
            m11: 0.4,
            shmnorm11: -0.02,
            beta11: -0.6,
            gamma11: -0.1,
            scatter,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['Override_z']`: fully user-supplied coefficients.
    #[allow(clippy::too_many_arguments)]
    pub fn override_full(
        m10: f64,
        shmnorm10: f64,
        beta10: f64,
        gamma10: f64,
        m11: f64,
        shmnorm11: f64,
        beta11: f64,
        gamma11: f64,
        scatter: f64,
        z_evo: bool,
    ) -> Self {
        Self {
            m10,
            shmnorm10,
            beta10,
            gamma10,
            m11,
            shmnorm11,
            beta11,
            gamma11,
            scatter,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }
}

impl SmhmModel for MosterFormSmhm {
    fn stellar_mass(&self, log_dm: f64, z: f64, rng: Option<&mut dyn RngCore>) -> f64 {
        let zp = self.z_parameter(z);
        let m = self.m10 + self.m11 * zp;
        let n = self.shmnorm10 + self.shmnorm11 * zp;
        let b = self.beta10 + self.beta11 * zp;
        let g = self.gamma10 + self.gamma11 * zp;

        let ratio = 10f64.powf(log_dm - m);
        let sm = 10f64.powf(log_dm) * (2.0 * n / (ratio.powf(-b) + ratio.powf(g)));
        let log_sm = sm.log10();

        match rng {
            Some(r) => {
                let normal = Normal::new(0.0, self.scatter).unwrap();
                log_sm + normal.sample(r)
            }
            None => log_sm,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn g19_se_is_monotonically_increasing_with_halo_mass() {
        let model = MosterFormSmhm::g19_se(true);
        let sm1 = model.stellar_mass(11.0, 0.1, None);
        let sm2 = model.stellar_mass(12.0, 0.1, None);
        let sm3 = model.stellar_mass(13.0, 0.1, None);
        assert!(sm1 < sm2 && sm2 < sm3, "{sm1} {sm2} {sm3}");
    }

    #[test]
    fn g19_se_peaks_near_the_expected_knee() {
        // Sanity check against the thesis's Ch.2 SMHM discussion: the
        // knee of the SMHM relation sits close to M10=12.0 for G19_SE.
        let model = MosterFormSmhm::g19_se(true);
        let sm_at_knee = model.stellar_mass(12.0, 0.1, None);
        assert!((10.5..11.5).contains(&sm_at_knee), "SM(M=12) = {sm_at_knee}");
    }

    #[test]
    fn fixed_z_evo_ignores_redshift() {
        let model = MosterFormSmhm::g19_se(false);
        let sm_z0 = model.stellar_mass(12.0, 0.0, None);
        let sm_z2 = model.stellar_mass(12.0, 2.0, None);
        assert!((sm_z0 - sm_z2).abs() < 1e-12);
    }

    #[test]
    fn scatter_changes_the_result_deterministically_given_a_seed() {
        let model = MosterFormSmhm::g19_se(true);
        let mut rng1 = StdRng::seed_from_u64(42);
        let mut rng2 = StdRng::seed_from_u64(42);
        let sm1 = model.stellar_mass(12.0, 0.1, Some(&mut rng1));
        let sm2 = model.stellar_mass(12.0, 0.1, Some(&mut rng2));
        assert_eq!(sm1, sm2, "same seed should give same scatter draw");

        let noiseless = model.stellar_mass(12.0, 0.1, None);
        assert!((sm1 - noiseless).abs() > 1e-6, "scatter should perturb the result");
    }
}
