//! Satellite quenching timescales (Wetzel+2013 fade/delay, Fillingham+2016
//! host-mass dependence, Cowley+2019 redshift scaling), a direct port of
//! the quenching-timescale block in `Functions.py::StarFormation`
//! (lines 337-361).

use steel_core::compat::{Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor};
use steel_core::cosmology::MassDefinition;
use steel_core::quenching::{QuenchTimescales, QuenchingModel};

pub struct Wetzel13;

impl Wetzel13 {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Wetzel13 {
    fn default() -> Self {
        Self::new()
    }
}

impl QuenchingModel for Wetzel13 {
    fn timescales(
        &self,
        log_sm_infall: f64,
        z_infall: f64,
        log_host_mass_infall: f64,
        t_infall: f64,
        pre_quenched: bool,
    ) -> QuenchTimescales {
        let mut tau_fade = -0.5 * log_sm_infall + 5.7;
        if tau_fade <= 0.2 {
            tau_fade = 0.2;
        }

        let mut tau_delay = 3.5 - ((log_sm_infall - 10.8) * 2.0).exp();
        if tau_delay <= 1.0 {
            tau_delay = 1.0;
        }

        // Fillingham+2016 host-mass-dependent floor for low-mass
        // satellites of massive hosts.
        //
        // PORT-FIX A8: this used to be `.clamp(0.0, 1.0)`, which pins
        // the cutoff mass at exactly 9.0 for every host below
        // log_host_mass = 15 -- i.e. every host in any realistic run.
        // Paper 2 eq. (8) has no such floor and gives three distinct
        // cutoffs (8.0, 8.5, 9.0) for the paper's own example host
        // masses (10, 12.5, 15), which is what Figure 6 plots. See
        // docs/PORT_CORRECTIONS.md A8.
        let host_dep = (log_host_mass_infall - 15.0) / 5.0;
        if log_sm_infall < 9.0 + host_dep {
            tau_delay = 2.0;
        }

        // Cowley+2019 redshift scaling.
        let z_scale = (1.0 + z_infall).powf(-1.5);
        tau_delay *= z_scale;
        tau_fade *= z_scale;

        // `t` here is age of the universe (increasing with time) —
        // Timeline's convention (see steel-core::baryonic doc comment)
        // — so quenching *later* means a *larger* t_quench, the
        // opposite sign from `Functions.py`'s lookback-time-based
        // `T_quench = t[0] - Tau_d`.
        let t_quench = if pre_quenched { t_infall } else { t_infall + tau_delay };

        QuenchTimescales { tau_fade, tau_delay, t_quench }
    }
}

impl DescribedPlugin for Wetzel13 {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "wetzel13",
            // A satellite quenching-timescale model, not a stellar-mass
            // calibration; it never touches an IMF.
            imf: Imf::NotApplicable,
            mass_definition: MassDefinition::Vir,
            // `timescales` takes `log_host_mass_infall` in STEEL's
            // internal Msun/h convention -- but that is the *host's*
            // mass, a STEEL-internal invariant populated the same way
            // regardless of which `SmhmModel`/`StellarGrowthModel`
            // supplies *this object's own* stellar mass. It is not the
            // run's stellar/halo-mass axis in the sense the
            // compatibility check cares about, so this is
            // `NotApplicable` rather than `PerH`: see
            // `HConvention::NotApplicable`'s doc for why declaring PerH
            // here would wrongly reject EMERGE (HFree), which
            // `docs/model-assumptions.md` documents as compatible.
            h_convention: HConvention::NotApplicable,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            provides: &[Capability::Quenching],
        }
    }
}

/// A `QuenchingModel` that never quenches anything.
///
/// Exists because `build_quenching` in `steel-cli`'s registry used to be
/// called unconditionally: `Wetzel13` was pushed into every run's
/// composition check regardless of what `[stellar_growth]` selected, so
/// any runfile choosing `UniverseMachineGrowth` (which declares its own
/// `Capability::Quenching`, since its SFR PDF already contains
/// quenching) was rejected by the duplicate-capability check for *every*
/// configuration, not just ones that explicitly stacked a second
/// quenching model on top. `[quenching] model = "none"` selects this
/// model instead, so a UM runfile can omit satellite quenching entirely
/// rather than being unable to run at all.
///
/// `t_quench = f64::INFINITY` is not a large-but-finite stand-in: the
/// only place any field of `QuenchTimescales` is read is
/// `steel_core::baryonic::BaryonicPipeline::evolve`'s
/// `quench.t_quench < timeline.t[i] && i != 0` fade-trigger check. Since
/// `timeline.t[i]` is always finite (it is a cosmic time in Gyr), that
/// comparison is `false` for every step of every possible timeline by
/// construction — not merely for the timelines this crate happens to
/// exercise. `tau_fade` and `tau_delay` are consequently never read
/// (the fade branch they feed is dead), so their values do not matter;
/// they are set to the same `INFINITY` for documentation symmetry, not
/// because anything consumes them.
pub struct NoQuenching;

impl QuenchingModel for NoQuenching {
    fn timescales(
        &self,
        _log_sm_infall: f64,
        _z_infall: f64,
        _log_host_mass_infall: f64,
        _t_infall: f64,
        _pre_quenched: bool,
    ) -> QuenchTimescales {
        QuenchTimescales { tau_fade: f64::INFINITY, tau_delay: f64::INFINITY, t_quench: f64::INFINITY }
    }
}

impl DescribedPlugin for NoQuenching {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "none",
            imf: Imf::NotApplicable,
            mass_definition: MassDefinition::Vir,
            // Applies nothing, so it has no opinion on any axis; see
            // `Wetzel13`'s descriptor doc for why this is `NotApplicable`
            // rather than a specific convention.
            h_convention: HConvention::NotApplicable,
            // Cosmology-agnostic: it applies no calibration at all, so
            // there is nothing to check against the run's cosmology.
            calibrated_cosmology: None,
            // Deliberately empty: this plugin supplies nothing, so it
            // can never trip the duplicate-`Capability::Quenching`
            // check -- including against a model (like UM) that
            // legitimately does claim `Capability::Quenching` itself.
            // Claiming `&[Capability::Quenching]` here would be false
            // (it applies no quenching) and would also make combining
            // it with UM impossible for no physical reason.
            provides: &[],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_massive_satellites_have_shorter_fade_timescales() {
        // tau_fade = -0.5*log_sm + 5.7 (floored at 0.2): once quenching
        // starts, more massive satellites fade *faster* (shorter
        // tau_fade), not slower.
        let model = Wetzel13::new();
        let low_mass = model.timescales(9.0, 0.5, 13.0, 5.0, false);
        let high_mass = model.timescales(11.0, 0.5, 13.0, 5.0, false);
        assert!(high_mass.tau_fade < low_mass.tau_fade, "{} vs {}", high_mass.tau_fade, low_mass.tau_fade);
    }

    #[test]
    fn pre_quenched_forces_immediate_quenching() {
        let model = Wetzel13::new();
        let t_infall = 5.0;
        let q = model.timescales(10.0, 0.5, 13.0, t_infall, true);
        assert_eq!(q.t_quench, t_infall);
    }

    /// PORT-FIX A8: Paper 2 Figure 6 plots three visibly distinct
    /// cutoff masses (log M* ~ 8.0, 8.5, 9.0) for host masses 10, 12.5,
    /// 15 -- eq. (8) unclamped. The clamped version this replaced gave
    /// cutoff 9.0 for all three, since (Mh-15)/5 is negative for every
    /// Mh < 15 and was floored to zero.
    #[test]
    fn fillingham_cutoff_mass_differs_between_host_masses_below_1e15() {
        let model = Wetzel13::new();
        // z=0 removes the Cowley+2019 (1+z)^-1.5 scaling so tau_delay is
        // directly comparable to the paper's static eq. (8)/(9) plot.
        let overridden = |log_sm: f64, log_host: f64| model.timescales(log_sm, 0.0, log_host, 0.0, false).tau_delay == 2.0;

        // Between the host=10 cutoff (8.0) and the host=12.5 cutoff (8.5):
        // only the more massive hosts have reduced this satellite's delay.
        assert!(!overridden(8.2, 10.0), "host=10 should not yet reduce tau_delay at log M*=8.2");
        assert!(overridden(8.2, 12.5), "host=12.5 should reduce tau_delay at log M*=8.2");
        assert!(overridden(8.2, 15.0), "host=15 should reduce tau_delay at log M*=8.2");

        // Between the host=12.5 cutoff (8.5) and the host=15 cutoff (9.0):
        // only the most massive host has reduced this satellite's delay.
        assert!(!overridden(8.6, 10.0), "host=10 should not reduce tau_delay at log M*=8.6");
        assert!(!overridden(8.6, 12.5), "host=12.5 should not yet reduce tau_delay at log M*=8.6");
        assert!(overridden(8.6, 15.0), "host=15 should reduce tau_delay at log M*=8.6");
    }

    #[test]
    fn quench_time_is_after_infall_when_not_pre_quenched() {
        let model = Wetzel13::new();
        let t_infall = 5.0;
        let q = model.timescales(10.0, 0.5, 13.0, t_infall, false);
        assert!(q.t_quench > t_infall, "t_quench={} should exceed t_infall={t_infall}", q.t_quench);
    }

    #[test]
    fn no_quenching_returns_infinite_timescales_regardless_of_input() {
        let model = NoQuenching;
        // Deliberately varied, physically-plausible-looking inputs
        // (including `pre_quenched = true`, which forces immediate
        // quenching for every *other* model in this file): NoQuenching
        // must ignore all of them.
        for &(log_sm, z, log_host, t_infall, pre_quenched) in &[
            (9.0, 0.5, 13.0, 5.0, false),
            (11.0, 2.0, 15.0, 1.0, true),
            (7.5, 0.0, 10.0, 13.5, false),
        ] {
            let q = model.timescales(log_sm, z, log_host, t_infall, pre_quenched);
            assert_eq!(q.t_quench, f64::INFINITY);
            assert_eq!(q.tau_fade, f64::INFINITY);
            assert_eq!(q.tau_delay, f64::INFINITY);
        }
    }

    #[test]
    fn no_quenching_declares_no_capabilities() {
        // Empty `provides`: it must never trip the duplicate-capability
        // check against a model (e.g. UniverseMachine) that legitimately
        // claims `Capability::Quenching` for itself.
        let d = NoQuenching.descriptor();
        assert!(d.provides.is_empty(), "{:?}", d.provides);
    }

    /// PROOF OF INERTNESS.
    ///
    /// This does not merely assert `NoQuenching`'s output looks
    /// harmless in isolation (the test above already does that); it
    /// drives a full `BaryonicPipeline::evolve` satellite trajectory --
    /// the only place a `QuenchingModel`'s output is ever consumed --
    /// and checks the result is bit-identical to an independently
    /// constructed "never quenches" baseline that does *not* rely on
    /// `f64::INFINITY` at all.
    ///
    /// `NeverQuenchesFinite` returns a large but finite `t_quench`
    /// (`1e12`, far beyond any cosmic-time timeline `evolve` will ever
    /// see, but still an ordinary float) with arbitrary finite
    /// `tau_fade`/`tau_delay`. If `NoQuenching`'s infinity were doing
    /// anything unusual -- producing a `NaN` through some arithmetic
    /// path, or behaving differently from "the fade branch never
    /// triggers" for some other reason -- the two trajectories would
    /// diverge. They must not.
    #[test]
    fn no_quenching_satellite_trajectory_is_bit_identical_to_an_unquenched_baseline() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use steel_core::accretion::AccretionContext;
        use steel_core::baryonic::{BaryonicPipeline, SatelliteState, Timeline};
        use steel_core::cosmology::{Cosmology, MassDefinition};
        use steel_core::halo_growth::GrowthTrack;

        use crate::gas::StewartScaling;
        use crate::sfr::TomczakFormSfr;
        use crate::stripping::Cattaneo11;
        use crate::Planck15;

        /// An independently-implemented "unquenched" baseline: a large
        /// but *finite* `t_quench`, not `f64::INFINITY`. Exists purely
        /// so this test does not just check that infinity round-trips
        /// through the same comparison NoQuenching itself performs.
        struct NeverQuenchesFinite;
        impl QuenchingModel for NeverQuenchesFinite {
            fn timescales(&self, _: f64, _: f64, _: f64, _: f64, _: bool) -> QuenchTimescales {
                QuenchTimescales { tau_fade: 1.0, tau_delay: 1.0, t_quench: 1.0e12 }
            }
        }

        fn build_timeline(cosmo: &Planck15) -> Timeline {
            let z: Vec<f64> = (0..11).map(|i| 1.0 - i as f64 * 0.05).collect();
            let t: Vec<f64> = z.iter().map(|&zi| cosmo.age(zi)).collect();
            let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
            dt.push(*dt.last().unwrap());
            Timeline { z, t, dt, log_host_mass: vec![13.0; 11], t_dyn_friction: 3.0 }
        }

        let cosmo = Planck15::new();
        let timeline = build_timeline(&cosmo);
        let galaxy = SatelliteState {
            log_sm_infall: 10.0,
            log_host_mass_infall: 13.0,
            log_sat_mass_infall: 11.5,
            z_infall: timeline.z[0],
            pre_quenched: false,
        };
        let track = GrowthTrack { z: vec![0.0], log_mass: vec![13.0] };
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);

        let no_quenching_pipeline = BaryonicPipeline::new(
            Box::new(TomczakFormSfr::ce()),
            Box::new(NoQuenching),
            Box::new(StewartScaling::from_cosmology(&cosmo)),
            Box::new(Cattaneo11),
        );
        let baseline_pipeline = BaryonicPipeline::new(
            Box::new(TomczakFormSfr::ce()),
            Box::new(NeverQuenchesFinite),
            Box::new(StewartScaling::from_cosmology(&cosmo)),
            Box::new(Cattaneo11),
        );

        // Both stripping on and scatter on, so the comparison exercises
        // the full trajectory (stripping factor, gas cap, main-sequence
        // SFR, scatter draws), not just a bare-bones path.
        let mut rng_a = StdRng::seed_from_u64(7);
        let a = no_quenching_pipeline.evolve(&galaxy, &timeline, true, true, true, &ctx, &mut rng_a);
        let mut rng_b = StdRng::seed_from_u64(7);
        let b = baseline_pipeline.evolve(&galaxy, &timeline, true, true, true, &ctx, &mut rng_b);

        assert_eq!(
            a.log_sm, b.log_sm,
            "NoQuenching's trajectory must be bit-identical to an independently \
             constructed unquenched baseline"
        );
        assert_eq!(a.log_ssfr, b.log_ssfr);

        // And a sanity check that this fixture is not accidentally
        // vacuous: the satellite's mass must actually have moved from
        // its infall value over the timeline (stripping dominates star
        // formation for this fixture, so the net direction is down --
        // the point is that real per-step dynamics ran, not that both
        // pipelines trivially agree on a frozen trajectory).
        assert!(
            (*a.log_sm.last().unwrap() - a.log_sm[0]).abs() > 0.1,
            "fixture should show real evolution: {:?}",
            a.log_sm
        );
    }
}
