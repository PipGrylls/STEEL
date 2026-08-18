//! Behroozi-family stellar-mass-halo-mass relations, a direct port of
//! `Functions.py::DarkMatterToStellarMass_Alt`. Unlike the Moster-form
//! presets (`super::moster`), these really are two distinct functional
//! forms sharing a family name, so they're two enum variants of one
//! equation each, not eight presets of one shared equation.
//!
//! Memoryless in the accretion history: `_ctx` is ignored by design, not omission.

use rand::RngCore;
use rand_distr::{Distribution, Normal};

use steel_core::accretion::AccretionContext;
use steel_core::compat::{Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor};
use steel_core::cosmology::MassDefinition;
use steel_core::smhm::SmhmModel;

/// Behroozi+2018-style coefficients: `e`, `M`, `alpha` each have 4
/// terms (constant, `a`-scaling, `ln(a)`-scaling, `z`-scaling); `beta`,
/// `gamma` have 3 (no `ln(a)` term).
struct B18Coeffs {
    e: [f64; 4],
    m: [f64; 4],
    alpha: [f64; 4],
    beta: [f64; 3],
    gamma: [f64; 3],
    delta: f64,
}

/// Behroozi+2013/Lorenzo+2018-style coefficients.
struct B13Coeffs {
    e: [f64; 4],
    m: [f64; 3],
    alpha: [f64; 2],
    delta: [f64; 3],
    gamma: [f64; 3],
}

enum Form {
    B18(B18Coeffs),
    B13(B13Coeffs),
}

pub struct BehrooziFormSmhm {
    form: Form,
    /// `Functions.py::DarkMatterToStellarMass_Alt`'s `Scatter` default
    /// argument (`0.001`) — small and fixed, unlike the Moster-form
    /// presets' `AbnMtch['Scatter']`-driven value.
    pub scatter: f64,
}

impl BehrooziFormSmhm {
    /// `AbnMtch['B18c']` ("centrals").
    pub fn behroozi18c() -> Self {
        Self {
            form: Form::B18(B18Coeffs {
                e: [-1.340, 0.404, -0.048, 0.133],
                m: [12.027, 2.582, 2.594, -0.409],
                alpha: [1.999, -1.710, -1.393, 0.192],
                beta: [0.502, -0.267, -0.197],
                gamma: [-0.788, -1.947, -0.658],
                delta: 0.340,
            }),
            scatter: 0.001,
        }
    }

    /// `AbnMtch['B18t']` ("true"/all-galaxies).
    pub fn behroozi18t() -> Self {
        Self {
            form: Form::B18(B18Coeffs {
                e: [-1.357, 0.139, -0.230, 0.157],
                m: [11.968, 2.231, 2.359, -0.374],
                alpha: [2.025, -1.365, -1.174, 0.167],
                beta: [0.520, -0.135, -0.161],
                gamma: [-0.729, -1.764, -0.639],
                delta: 0.351,
            }),
            scatter: 0.001,
        }
    }

    /// `AbnMtch['Behroozi13']`.
    pub fn behroozi13() -> Self {
        Self {
            form: Form::B13(B13Coeffs {
                e: [-1.777, -0.006, 0.000, -0.119],
                m: [11.514, -1.793, -0.251],
                alpha: [-1.412, 0.731],
                delta: [3.508, 2.608, -0.043],
                gamma: [0.361, 1.391, 0.279],
            }),
            scatter: 0.001,
        }
    }

    /// `AbnMtch['Lorenzo18']`.
    pub fn lorenzo18() -> Self {
        Self {
            form: Form::B13(B13Coeffs {
                e: [-1.6695, -0.006, 0.000, -0.119],
                m: [11.6097, -1.793, -0.251],
                alpha: [-1.998, 0.731],
                delta: [3.2108, 2.608, -0.043],
                gamma: [0.4222, 1.391, 0.279],
            }),
            scatter: 0.001,
        }
    }

    fn stellar_mass_noiseless(&self, log_dm: f64, z: f64) -> f64 {
        match &self.form {
            Form::B18(c) => Self::b18_stellar_mass(c, log_dm, z),
            Form::B13(c) => Self::b13_stellar_mass(c, log_dm, z),
        }
    }

    fn b18_stellar_mass(c: &B18Coeffs, log_dm: f64, z: f64) -> f64 {
        let a = 1.0 / (1.0 + z);
        let afac = a - 1.0;

        let log10_m = c.m[0] + c.m[1] * afac - c.m[2] * a.ln() + c.m[3] * z;
        let e_ = c.e[0] + c.e[1] * afac - c.e[2] * a.ln() + c.e[3] * z;
        let alpha_ = c.alpha[0] + c.alpha[1] * afac - c.alpha[2] * a.ln() + c.alpha[3] * z;
        let beta_ = c.beta[0] + c.beta[1] * afac + c.beta[2] * z;
        let log10_g = c.gamma[0] + c.gamma[1] * afac + c.gamma[2] * z;

        let x = log_dm - log10_m;
        let gamma_ = 10f64.powf(log10_g);

        let part1 = (10f64.powf(-alpha_ * x) + 10f64.powf(-beta_ * x)).log10();
        let part2 = (-0.5 * (x / c.delta).powi(2)).exp();

        log10_m + (e_ - part1 + gamma_ * part2)
    }

    fn b13_stellar_mass(c: &B13Coeffs, log_dm: f64, z: f64) -> f64 {
        let a = 1.0 / (1.0 + z);
        let afac = a - 1.0;
        let v = (-4.0 * a * a).exp();

        let m_exp = c.m[0] + (c.m[1] * afac + c.m[2] * z) * v;
        let e_exp = c.e[0] + (c.e[1] * afac + c.e[2] * z) * v + c.e[3] * afac;
        let alpha_ = c.alpha[0] + (c.alpha[1] * afac) * v;
        let delta_ = c.delta[0] + (c.delta[1] * afac + c.delta[2] * z) * v;
        let gamma_ = c.gamma[0] + (c.gamma[1] * afac + c.gamma[2] * z) * v;

        let e_ = 10f64.powf(e_exp);
        let m_ = 10f64.powf(m_exp);

        let f = |x: f64| {
            let part1 = (10f64.powf(alpha_ * x) + 1.0).log10();
            let part2 = (1.0 + x.exp()).log10().powf(gamma_);
            let part3 = 1.0 + (10f64.powf(-x)).exp();
            -part1 + delta_ * (part2 / part3)
        };

        let part1 = (e_ * m_).log10();
        let part2 = f((10f64.powf(log_dm) / m_).log10());
        let part3 = f(0.0);

        part1 + part2 - part3
    }
}

impl SmhmModel for BehrooziFormSmhm {
    fn stellar_mass(
        &self,
        log_dm: f64,
        z: f64,
        _ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        let log_sm = self.stellar_mass_noiseless(log_dm, z);
        // `scatter` is a public field, so a non-positive or non-finite
        // value is reachable — treat it as "no scatter" rather than
        // letting `Normal::new(..).unwrap()` panic.
        match rng {
            Some(r) if self.scatter > 0.0 && self.scatter.is_finite() => {
                let normal = Normal::new(0.0, self.scatter).unwrap();
                log_sm + normal.sample(r)
            }
            _ => log_sm,
        }
    }
}

impl DescribedPlugin for BehrooziFormSmhm {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "behroozi_form",
            // Behroozi+2013/2018 presets are calibrated on a Chabrier
            // IMF.
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::PerH,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            // Applies its own log-normal scatter via `self.scatter`,
            // same as `MosterFormSmhm`.
            provides: &[Capability::StellarMass, Capability::Scatter],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::flat_ctx;

    #[test]
    fn b18c_is_monotonically_increasing_with_halo_mass() {
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let model = BehrooziFormSmhm::behroozi18c();
        let sm1 = model.stellar_mass(11.0, 0.1, &ctx, None);
        let sm2 = model.stellar_mass(12.0, 0.1, &ctx, None);
        let sm3 = model.stellar_mass(13.0, 0.1, &ctx, None);
        assert!(sm1 < sm2 && sm2 < sm3, "{sm1} {sm2} {sm3}");
    }

    #[test]
    fn b13_is_monotonically_increasing_with_halo_mass() {
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let model = BehrooziFormSmhm::behroozi13();
        let sm1 = model.stellar_mass(11.0, 0.1, &ctx, None);
        let sm2 = model.stellar_mass(12.0, 0.1, &ctx, None);
        let sm3 = model.stellar_mass(13.0, 0.1, &ctx, None);
        assert!(sm1 < sm2 && sm2 < sm3, "{sm1} {sm2} {sm3}");
    }

    #[test]
    fn lorenzo18_is_monotonically_increasing_with_halo_mass() {
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let model = BehrooziFormSmhm::lorenzo18();
        let sm1 = model.stellar_mass(11.0, 0.1, &ctx, None);
        let sm2 = model.stellar_mass(12.0, 0.1, &ctx, None);
        let sm3 = model.stellar_mass(13.0, 0.1, &ctx, None);
        assert!(sm1 < sm2 && sm2 < sm3, "{sm1} {sm2} {sm3}");
    }

    #[test]
    fn b18_and_b13_give_broadly_similar_stellar_masses_at_the_knee() {
        // Both families are fits to similar central-galaxy SMHM data,
        // so at a ~Milky-Way-mass halo they should agree to within a
        // dex or so even though the functional forms differ.
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let b18 = BehrooziFormSmhm::behroozi18c();
        let b13 = BehrooziFormSmhm::behroozi13();
        let sm_b18 = b18.stellar_mass(12.0, 0.1, &ctx, None);
        let sm_b13 = b13.stellar_mass(12.0, 0.1, &ctx, None);
        assert!((sm_b18 - sm_b13).abs() < 1.0, "b18={sm_b18} b13={sm_b13}");
    }

    #[test]
    fn invalid_scatter_is_treated_as_no_scatter_rather_than_panicking() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;

        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let noiseless = BehrooziFormSmhm::behroozi18c().stellar_mass(12.0, 0.1, &ctx, None);
        for bad in [0.0, -0.5, f64::NAN] {
            let mut model = BehrooziFormSmhm::behroozi18c();
            model.scatter = bad;
            let mut rng = StdRng::seed_from_u64(1);
            let got = model.stellar_mass(12.0, 0.1, &ctx, Some(&mut rng));
            assert!((got - noiseless).abs() < 1e-12, "scatter={bad} should give the noiseless value");
        }
    }
}
