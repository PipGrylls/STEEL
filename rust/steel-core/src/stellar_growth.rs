//! Rate-based stellar mass assembly.
//!
//! `SmhmModel` answers "what M* corresponds to this Mh at this z?" as a
//! memoryless map. EMERGE and UniverseMachine instead specify a *rate*:
//! EMERGE as a baryon conversion efficiency applied to the halo
//! accretion rate, UniverseMachine as an SFR drawn from halo properties.
//! For both, M* is the time integral of that rate along the object's
//! growth track.
//!
//! Keeping these separate from `SmhmModel` matters beyond tidiness:
//! EMERGE's efficiency double power law is *algebraically identical* to
//! the Moster form in `steel_plugins::smhm::moster`, but multiplies a
//! rate rather than a mass. Substituting its coefficients into
//! `MosterFormSmhm` would silently produce a wrong SMHM curve.

use rand::RngCore;

use crate::accretion::AccretionContext;
use crate::smhm::SmhmModel;

pub trait StellarGrowthModel: Send + Sync {
    /// log10 dM*/dt \[Msun/yr\] for a halo of mass `log_mh` \[log10
    /// Msun\] at redshift `z`.
    ///
    /// `rng` is present because UniverseMachine draws SFR from a bimodal
    /// PDF: the rate is intrinsically stochastic, not a mean relation
    /// with scatter added afterwards. Models with a deterministic rate
    /// ignore it.
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64;
}

/// Lets a boxed trait object satisfy the same `M: StellarGrowthModel`
/// bound a concrete model does -- needed so `StellarGrowthAsSmhm` can
/// wrap `Box<dyn StellarGrowthModel>` (what `steel_cli::registry`
/// actually has after selecting a model at runtime) without a second,
/// object-specific adapter impl.
impl StellarGrowthModel for Box<dyn StellarGrowthModel> {
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        (**self).stellar_growth_rate(log_mh, z, ctx, rng)
    }
}

/// Call `model.stellar_growth_rate`, reborrowing `rng` rather than
/// consuming it.
///
/// `Option<&mut dyn RngCore>` cannot be reborrowed across repeated calls
/// with `.as_deref_mut()` the way `Option<&mut T>` for a sized `T` can:
/// rustc's borrow checker ties the reborrow's lifetime to the outer
/// `Option`'s full lifetime once it is routed through a trait-object
/// call, so a second call in the same scope is rejected as a second
/// mutable borrow. Matching on the `&mut Option` and re-wrapping the
/// arm's `&mut dyn RngCore` sidesteps that limitation.
fn call_rate(
    model: &dyn StellarGrowthModel,
    log_mh: f64,
    z: f64,
    ctx: &AccretionContext<'_>,
    rng: &mut Option<&mut dyn RngCore>,
) -> f64 {
    match rng {
        Some(r) => model.stellar_growth_rate(log_mh, z, ctx, Some(&mut **r)),
        None => model.stellar_growth_rate(log_mh, z, ctx, None),
    }
}

/// Integrate `model`'s rate along `ctx.own_track` from the track's
/// earliest sample down to `z_end`, returning log10 M*/Msun.
///
/// Trapezoidal in cosmic time. `own_track.z` is increasing into the past
/// (index 0 is the observed epoch), so integration walks the track in
/// reverse. Samples at `z < z_end` are excluded.
///
/// Returns `f64::NEG_INFINITY` when no time elapses (zero mass, whose
/// log is negative infinity) rather than `NaN`, so callers can compare
/// and clamp without special-casing.
pub fn integrate_stellar_mass(
    model: &dyn StellarGrowthModel,
    ctx: &AccretionContext<'_>,
    z_end: f64,
    mut rng: Option<&mut dyn RngCore>,
) -> f64 {
    let t = ctx.own_track;
    debug_assert_eq!(t.z.len(), t.log_mass.len(), "GrowthTrack axes must be equal length");

    // Indices from oldest to youngest, keeping only z >= z_end.
    let idx: Vec<usize> = (0..t.z.len()).rev().filter(|&i| t.z[i] >= z_end).collect();
    if idx.len() < 2 {
        return f64::NEG_INFINITY;
    }

    let mut mass = 0.0_f64; // Msun, linear
    for w in idx.windows(2) {
        let (i0, i1) = (w[0], w[1]); // i0 older, i1 younger
        // age() is in Gyr; rates are per year.
        let dt_yr = (ctx.cosmology.age(t.z[i1]) - ctx.cosmology.age(t.z[i0])) * 1.0e9;
        if dt_yr <= 0.0 {
            continue;
        }
        let r0 = 10f64.powf(call_rate(model, t.log_mass[i0], t.z[i0], ctx, &mut rng));
        let r1 = 10f64.powf(call_rate(model, t.log_mass[i1], t.z[i1], ctx, &mut rng));
        mass += 0.5 * (r0 + r1) * dt_yr;
    }

    if mass <= 0.0 {
        f64::NEG_INFINITY
    } else {
        mass.log10()
    }
}

/// Adapts a rate-based [`StellarGrowthModel`] to the memoryless
/// [`SmhmModel`] interface by integrating its rate over the accretion
/// history `ctx` already carries.
///
/// This is what lets `[stellar_growth]` (EMERGE, UniverseMachine) drive
/// the same orchestrator loop `[smhm]` does
/// (`steel_core::context::Simulation::run`'s single
/// `self.smhm.stellar_mass(...)` call site): the loop always builds an
/// `AccretionContext` whose `own_track` is the object's own halo-mass
/// history evaluated to the redshift being queried, which is exactly
/// what [`integrate_stellar_mass`] consumes. `log_dm` is intentionally
/// unused -- the rate integrator reads the halo mass history from
/// `ctx.own_track` rather than from a single scalar, and that track is
/// already consistent with `log_dm` at the call site (both come from
/// the same halo mass sample).
pub struct StellarGrowthAsSmhm<M> {
    pub model: M,
}

impl<M> StellarGrowthAsSmhm<M> {
    pub fn new(model: M) -> Self {
        Self { model }
    }
}

impl<M: StellarGrowthModel> SmhmModel for StellarGrowthAsSmhm<M> {
    fn stellar_mass(
        &self,
        _log_dm: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        integrate_stellar_mass(&self.model, ctx, z, rng)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::{Cosmology, MassDefinition};
    use crate::halo_growth::GrowthTrack;

    struct StubCosmo;
    impl Cosmology for StubCosmo {
        fn h0(&self) -> f64 { 67.74 }
        fn omega_m0(&self) -> f64 { 0.3089 }
        fn omega_b0(&self) -> f64 { 0.0486 }
        fn omega_de0(&self) -> f64 { 0.6911 }
        fn omega_r0(&self) -> f64 { 0.0 }
        fn sigma8(&self) -> f64 { 0.8159 }
        fn n_spec(&self) -> f64 { 0.9667 }
        fn e_z(&self, z: f64) -> f64 {
            (self.omega_m0() * (1.0 + z).powi(3) + self.omega_de0()).sqrt()
        }
        /// Deliberately linear in (1+z)^-1 so the exact integral of a
        /// constant rate is hand-computable.
        fn age(&self, z: f64) -> f64 { 13.8 / (1.0 + z) }
    }

    /// Constant 1 Msun/yr regardless of mass, redshift, or history.
    struct ConstantRate;
    impl StellarGrowthModel for ConstantRate {
        fn stellar_growth_rate(
            &self,
            _log_mh: f64,
            _z: f64,
            _ctx: &AccretionContext<'_>,
            _rng: Option<&mut dyn rand::RngCore>,
        ) -> f64 {
            0.0 // log10(1.0)
        }
    }

    fn track() -> GrowthTrack {
        // z decreasing into the present is NOT the convention: GrowthTrack
        // is increasing into the past, so index 0 is the observed epoch.
        GrowthTrack { z: vec![0.0, 1.0, 2.0, 3.0], log_mass: vec![12.0, 11.6, 11.2, 10.8] }
    }

    #[test]
    fn constant_rate_integrates_to_rate_times_elapsed_time() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        // Integrate from the track's earliest epoch (z=3) to z=0.
        // age(0) - age(3) = 13.8 - 3.45 = 10.35 Gyr = 1.035e10 yr.
        // At 1 Msun/yr that is 1.035e10 Msun.
        let got = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        let expected = 1.035e10f64.log10();
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn integrating_to_an_earlier_epoch_gives_less_mass() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let early = integrate_stellar_mass(&ConstantRate, &ctx, 2.0, None);
        let late = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        assert!(early < late, "early {early} should be below late {late}");
    }

    #[test]
    fn zero_elapsed_time_gives_negative_infinity_log_mass() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        // z_end at the track's oldest sample: no time has elapsed.
        let got = integrate_stellar_mass(&ConstantRate, &ctx, 3.0, None);
        assert!(got.is_infinite() && got.is_sign_negative(), "got {got}");
    }

    /// `integrate_stellar_mass` returns mass *formed*, with no stellar
    /// mass-loss return fraction applied. STEEL applies mass loss in
    /// `Functions.py::StellarMassLoss` / its Rust port, so applying it
    /// here too would double-count. This test documents the boundary.
    #[test]
    fn integrator_returns_formed_mass_not_surviving_mass() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let got = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        // Exactly rate x elapsed time: no 0.6-0.8 return-fraction factor.
        assert!((got - 1.035e10f64.log10()).abs() < 1e-6);
    }

    #[test]
    fn stellar_growth_as_smhm_matches_integrate_stellar_mass_directly() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let adapter = StellarGrowthAsSmhm::new(ConstantRate);

        // `log_dm` is deliberately a nonsense value here (unrelated to
        // the track): the adapter must ignore it entirely and defer to
        // `ctx.own_track`, exactly like `integrate_stellar_mass` does.
        let via_adapter = adapter.stellar_mass(-999.0, 0.0, &ctx, None);
        let via_direct_call = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        assert_eq!(via_adapter, via_direct_call);
    }
}
