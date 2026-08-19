//! Rodríguez-Puebla et al. (2017) stellar-mass-halo-mass relation, a
//! port of `Functions.py::SHMR_RP17` and its helper `SHMR_func`.
//!
//! This is a third functional family, distinct from both siblings in
//! this module. `super::moster` is the Moster (2010) double power law;
//! `super::behroozi` is the Behroozi+2013/2018 form with its
//! `a`/`ln a`/`z` coefficient expansion. This one is the Behroozi et
//! al. (2010) `g(x)` form with Rodríguez-Puebla's redshift
//! parametrisation on top:
//!
//! ```text
//! g(x)     = -log10(10^(-alpha x) + 1) + delta [log10(1 + e^x)]^gamma / (1 + e^(10^-x))
//! log10 M* = log10 eps + log10 M1 + g(x) - g(0),   x = log10 Mvir - log10 M1
//! ```
//!
//! with each of `alpha`, `delta`, `gamma`, `log10 eps`, `log10 M1`
//! evolving as `c0 + P(c1, c2, z) Q(z)`, where
//! `P(x, y, z) = y z - x z/(1+z)` and `Q(z) = exp(-4/(1+z)^2)`.
//!
//! It reaches STEEL through `AbnMtch['RP17']`, which short-circuits
//! `DarkMatterToStellarMass` before any of the Moster-form presets are
//! consulted. The branch exists only on `PipGrylls` (and its
//! descendants) — `master` has no RP17 at all.
//!
//! **No scatter.** `SHMR_RP17` returns the mean relation and the
//! `ScatterOn`/`Scatter` arguments never reach it, because the early
//! `return SHMR_RP17(z, DM)` sits above the block that would have
//! applied the log-normal draw. `stellar_mass` therefore ignores its
//! `rng` argument; that is faithful to the source, not an omission.
//!
//! Memoryless in the accretion history: `_ctx` is ignored by design, not omission.

use rand::RngCore;

use steel_core::accretion::AccretionContext;
use steel_core::compat::{Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor};
use steel_core::cosmology::MassDefinition;
use steel_core::smhm::SmhmModel;

/// `P(x, y, z) = y z - x z/(1+z)` — the redshift expansion shared by
/// every coefficient. Note the argument order: the *first* coefficient
/// multiplies the `z/(1+z)` term and enters with a minus sign.
fn p(x: f64, y: f64, z: f64) -> f64 {
    y * z - x * z / (1.0 + z)
}

/// `Q(z) = exp(-4/(1+z)^2)` — the damping that switches the redshift
/// terms off at `z = 0` (`Q(0) = e^-4 ~ 0.018`).
fn q(z: f64) -> f64 {
    (-4.0 / ((1.0 + z) * (1.0 + z))).exp()
}

/// The Behroozi et al. (2010) `g(x)` shape function. Spelled to match
/// `Functions.py::SHMR_func`'s `g` term by term, including the
/// `log10(1 + e^x)` written out longhand rather than as a
/// numerically-nicer softplus — the two differ in the last bits and
/// this port is meant to be diffable against the source.
fn g(x: f64, alpha: f64, gamma: f64, delta: f64) -> f64 {
    -(10f64.powf(-alpha * x) + 1.0).log10()
        + delta * (1.0 + x.exp()).log10().powf(gamma) / (1.0 + 10f64.powf(-x).exp())
}

/// `Functions.py::SHMR_func` — the mean relation at fixed coefficients.
/// Public because it is a reusable Behroozi+2010 SHMR in its own right,
/// independent of Rodríguez-Puebla's redshift fit.
pub fn shmr_behroozi10(alpha: f64, delta: f64, gamma: f64, log10_eps: f64, log10_m1: f64, log10_mvir: f64) -> f64 {
    let x = log10_mvir - log10_m1;
    log10_eps + log10_m1 + g(x, alpha, gamma, delta) - g(0.0, alpha, gamma, delta)
}

/// `AbnMtch['RP17']`.
pub struct RodriguezPuebla17;

impl RodriguezPuebla17 {
    /// The five redshift-dependent coefficients at `z`, in the order
    /// `(alpha, delta, gamma, log10 eps, log10 M1)`.
    fn coefficients(z: f64) -> (f64, f64, f64, f64, f64) {
        const AL: (f64, f64, f64) = (1.975, 0.714, 0.042);
        const DE: (f64, f64, f64) = (3.390, -0.472, -0.931);
        const GA: (f64, f64) = (0.498, -0.157);
        const EP: (f64, f64, f64, f64) = (-1.758, 0.110, -0.061, -0.023);
        const M0: (f64, f64, f64) = (11.548, -1.297, -0.026);

        let qz = q(z);
        (
            AL.0 + p(AL.1, AL.2, z) * qz,
            DE.0 + p(DE.1, DE.2, z) * qz,
            GA.0 + p(GA.1, 0.0, z) * qz,
            // `log10 eps` alone carries a second, undamped P term.
            EP.0 + p(EP.1, EP.2, z) * qz + p(EP.3, 0.0, z),
            M0.0 + p(M0.1, M0.2, z) * qz,
        )
    }
}

impl SmhmModel for RodriguezPuebla17 {
    fn stellar_mass(
        &self,
        log_dm: f64,
        z: f64,
        _ctx: &AccretionContext<'_>,
        _rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        let (alpha, delta, gamma, log10_eps, log10_m1) = Self::coefficients(z);
        shmr_behroozi10(alpha, delta, gamma, log10_eps, log10_m1, log_dm)
    }
}

impl DescribedPlugin for RodriguezPuebla17 {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "rodriguez_puebla_form",
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::PerH,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            // Returns the mean relation only; no scatter is applied, so
            // `Scatter` is deliberately omitted, permitting another
            // scatter source in the composition.
            provides: &[Capability::StellarMass],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_support::flat_ctx;

    #[test]
    fn is_monotonically_increasing_with_halo_mass() {
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let model = RodriguezPuebla17;
        let masses: Vec<f64> = (110..=150).map(|i| i as f64 / 10.0).collect();
        for w in masses.windows(2) {
            let (a, b) =
                (model.stellar_mass(w[0], 0.1, &ctx, None), model.stellar_mass(w[1], 0.1, &ctx, None));
            assert!(b > a, "SM({}) = {a} >= SM({}) = {b}", w[0], w[1]);
        }
    }

    #[test]
    fn lands_in_the_right_ballpark_at_the_knee() {
        // The RP17 relation should put a Milky-Way-scale halo at
        // roughly a Milky-Way-scale stellar mass.
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let sm = RodriguezPuebla17.stellar_mass(12.0, 0.1, &ctx, None);
        assert!((10.0..11.2).contains(&sm), "SM(1e12) = {sm}");
    }

    #[test]
    fn redshift_terms_are_damped_but_not_absent_at_z_zero() {
        // Q(0) = e^-4 ~ 0.0183, so z=0 is *near* but not equal to the
        // bare (c0) coefficients; and log10 eps carries an undamped P
        // term that vanishes at z=0 too.
        let (alpha, delta, gamma, log10_eps, log10_m1) = RodriguezPuebla17::coefficients(0.0);
        assert_eq!((alpha, delta, gamma, log10_eps, log10_m1), (1.975, 3.390, 0.498, -1.758, 11.548));
    }

    #[test]
    fn evolves_with_redshift() {
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let model = RodriguezPuebla17;
        let z0 = model.stellar_mass(12.0, 0.1, &ctx, None);
        let z2 = model.stellar_mass(12.0, 2.0, &ctx, None);
        assert!((z0 - z2).abs() > 0.01, "z=0.1 {z0} vs z=2 {z2}");
    }

    #[test]
    fn ignores_the_rng_because_the_python_returns_before_applying_scatter() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        let (track, cosmo) = flat_ctx();
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let model = RodriguezPuebla17;
        let mut rng = StdRng::seed_from_u64(7);
        let scattered = model.stellar_mass(12.0, 0.1, &ctx, Some(&mut rng));
        let plain = model.stellar_mass(12.0, 0.1, &ctx, None);
        assert_eq!(scattered, plain);
    }
}
