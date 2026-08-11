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
    /// `zparameter = z/(1+z)` — only the `Moster13` preset uses this
    /// form. (`Moster10` uses `ShiftedStyle` like everything else:
    /// `Functions.py:526-534` special-cases `Paramaters['Moster']`
    /// alone, and `Moster10` falls through to the `elif`/`else` arms,
    /// which both compute the shifted expression.)
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

    /// `AbnMtch['G19_SE']` — STEEL's default abundance-matching choice,
    /// the PyMorph-calibrated fit of Grylls et al. (2019).
    ///
    /// These are the `PipGrylls`-branch coefficients, i.e. the ones the
    /// papers were run with. `master` still carries the earlier
    /// `12.0, 0.032, 1.5, 0.56 / 0.6, -0.014, -0.7, 0.08`; see
    /// `docs/PORT_CORRECTIONS.md` for why that baseline was wrong.
    pub fn g19_se(z_evo: bool) -> Self {
        Self {
            m10: 11.925,
            shmnorm10: 0.032,
            beta10: 1.639,
            gamma10: 0.532,
            m11: 0.576,
            shmnorm11: -0.014,
            beta11: -0.693,
            gamma11: 0.03,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['G19_cMod']` — the cmodel-photometry counterpart of
    /// [`g19_se`](Self::g19_se). `PipGrylls` values; `master` carries
    /// `12.0, 0.032, 1.74, 0.66 / 0.4, -0.024, -0.74, -0.12`.
    pub fn g19_c_mod(z_evo: bool) -> Self {
        Self {
            m10: 11.91,
            shmnorm10: 0.029,
            beta10: 2.09,
            gamma10: 0.64,
            m11: 0.644,
            shmnorm11: -0.019,
            beta11: -1.422,
            gamma11: -0.043,
            scatter: 0.15,
            z_evo: if z_evo { ZEvo::ShiftedStyle } else { ZEvo::Fixed },
        }
    }

    /// `AbnMtch['PFT']` — the pair-fraction-testing base, identical to
    /// [`g19_se`](Self::g19_se). The thirteen `*_PFT*` flags each nudge
    /// exactly one coefficient off this base; apply them by mutating
    /// the returned struct's public fields (see
    /// `Scripts/Validation/make_runfiles.py`, which generates the
    /// runfiles for all thirteen).
    pub fn pft(z_evo: bool) -> Self {
        Self::g19_se(z_evo)
    }

    /// `AbnMtch['HMevo']` — the `G19_cMod` form with a *free* `gamma11`
    /// (`AbnMtch['HMevo_param']`, parsed by `STEEL.py` out of the last
    /// three characters of the run's model name, e.g. `HMevo_alt_0.3`).
    /// Note the other three high-redshift terms differ from
    /// [`g19_c_mod`](Self::g19_c_mod): `0.518, -0.018, -1.031` rather
    /// than `0.644, -0.019, -1.422`.
    pub fn hmevo(gamma11: f64, z_evo: bool) -> Self {
        Self {
            m10: 11.91,
            shmnorm10: 0.029,
            beta10: 2.09,
            gamma10: 0.64,
            m11: 0.518,
            shmnorm11: -0.018,
            beta11: -1.031,
            gamma11,
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

        // `scatter` is a public field and `override_*` takes it straight
        // from the caller, so a non-positive or non-finite value is
        // reachable through the public API — treat it as "no scatter"
        // rather than letting `Normal::new(..).unwrap()` panic.
        match rng {
            Some(r) if self.scatter > 0.0 && self.scatter.is_finite() => {
                let normal = Normal::new(0.0, self.scatter).unwrap();
                log_sm + normal.sample(r)
            }
            _ => log_sm,
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
        // knee of the SMHM relation sits close to M10=11.925 for G19_SE.
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

    #[test]
    fn invalid_scatter_is_treated_as_no_scatter_rather_than_panicking() {
        // `scatter` is public and `override_*` takes it from the caller,
        // so these values are reachable; `Normal::new` would panic on
        // each of them.
        let noiseless = MosterFormSmhm::g19_se(true).stellar_mass(12.0, 0.1, None);
        for bad in [0.0, -0.5, f64::NAN] {
            let mut model = MosterFormSmhm::g19_se(true);
            model.scatter = bad;
            let mut rng = StdRng::seed_from_u64(1);
            let got = model.stellar_mass(12.0, 0.1, Some(&mut rng));
            assert!((got - noiseless).abs() < 1e-12, "scatter={bad} should give the noiseless value");
        }
    }
}
