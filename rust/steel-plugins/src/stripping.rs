//! Stripping models: stellar tidal mass loss (Cattaneo+2011, live in
//! every real STEEL run) and dark-matter subhalo mass loss (van den
//! Bosch 2005 / Jiang+2016 form, currently dead code in the Python —
//! `Stripping_DM = False #Future use` — kept live here per the trait's
//! own doc comment).

use steel_core::cosmology::Cosmology;
use steel_core::stripping::{HaloStrippingModel, HaloStrippingTrack, StellarStrippingModel};

/// `Functions.py::StellarMassLoss`'s `factor_only=True` path.
///
/// The trailing `* 2.0` is not part of Cattaneo et al. (2011); it is
/// applied on top by `StellarMassLoss` on the `PipGrylls` branch:
///
/// ```python
/// Strip_f = np.log10(Strip + (1-Strip)*(1-Factor))
/// Strip_f = Strip_f*2
/// Strip_f[Strip_f>1] = 1
/// ```
///
/// Since `Strip + (1-Strip)(1-Factor) <= 1` for `Strip, Factor` in
/// `[0, 1]`, `Strip_f` is always `<= 0` and the `> 1` clamp on the next
/// line can never fire — it is dead, and reproducing it would be
/// reproducing nothing, so it is omitted here. The doubling itself is
/// live and is *not* dead: it doubles the stripping in dex, i.e.
/// squares the surviving stellar-mass fraction, and it is what Papers 2
/// and 3 were run with. `master` has neither line, which is one of the
/// reasons a `master`-baselined port could not reproduce the published
/// satellite stellar mass functions.
///
/// The sibling doubling in `Functions.py::StarFormation` (`StripFactor
/// = StripFactor*2`, commented `#For reviwer`) is commented out on
/// every branch and is deliberately not reproduced; applying both would
/// quadruple the stripping.
pub struct Cattaneo11;

impl StellarStrippingModel for Cattaneo11 {
    fn strip_factor(&self, log_host_mass: f64, log_sat_mass: f64, time_fraction: f64) -> f64 {
        let mh_ms = 10f64.powf(log_host_mass - log_sat_mass);
        let strip = 0.6f64.powf((1.428 / (2.0 * std::f64::consts::PI)) * (mh_ms / (1.0 + mh_ms).ln()));
        2.0 * (strip + (1.0 - strip) * (1.0 - time_fraction)).log10()
    }
}

/// `Functions.py::HaloMassLoss_w` + `Functions_c.pyx::HaloMassLoss_c`.
///
/// The Cython hardcodes the dark-energy density to `Ol = 0` in its
/// local `Ez`/overdensity calculation (rather than `Omega_de0`) and
/// separately gets the overdensity fitting formula's linear-term sign
/// backwards relative to `Delta_c`/`Delta_crit`'s own convention
/// elsewhere in the same codebase (`x = 1 - Om(z)` there vs.
/// `x = Om(z) - 1` everywhere else) — two bugs in a path that's never
/// actually exercised (`Stripping_DM` is always `False`). Since this
/// is dead code either way, it's implemented correctly here by simply
/// reusing `Cosmology::e_z`/`Cosmology::delta_vir` directly instead of
/// re-deriving a redundant (and, in the original, wrong) local
/// formula.
pub struct HaloStrippingVdb05;

impl HaloStrippingModel for HaloStrippingVdb05 {
    fn strip(
        &self,
        log_m_infall: f64,
        log_host_mass_track: &[f64],
        z_track: &[f64],
        dt_track: &[f64],
        cosmology: &dyn Cosmology,
    ) -> HaloStrippingTrack {
        const ZETA: f64 = 0.07;
        const ZETA_PWR: f64 = -1.0 / ZETA;
        const A: f64 = 0.81;

        let n = log_host_mass_track.len();
        debug_assert_eq!(n, z_track.len());
        debug_assert_eq!(n, dt_track.len());

        let mut log_mass = vec![0.0_f64; n.max(1)];
        if n == 0 {
            return HaloStrippingTrack { log_mass };
        }
        log_mass[0] = log_m_infall;

        for i in 0..n.saturating_sub(1) {
            let m_m = 10f64.powf(log_mass[i] - log_host_mass_track[i]);
            let e_z = cosmology.e_z(z_track[i]);
            let delta_vir = cosmology.delta_vir(z_track[i]);
            let tau = (1.628 / cosmology.h() * (delta_vir / 178.0).powf(-0.5) * e_z.powi(-1)) / A;

            let part1 = ZETA * m_m.powf(ZETA);
            let part2 = dt_track[i] / tau;
            log_mass[i + 1] = (10f64.powf(log_mass[i]) * (1.0 + part1 * part2).powf(ZETA_PWR)).log10();
        }

        HaloStrippingTrack { log_mass }
    }
}

/// Scales any [`StellarStrippingModel`]'s strip factor, turning
/// stripping strength into a sweepable parameter.
///
/// `strip_factor` is log10 of the fraction still bound, so multiplying
/// it by `strength` raises that fraction to the power `strength`:
/// `f_bound -> f_bound^strength`. `strength = 1` leaves the wrapped
/// model untouched, `> 1` strips harder, `0` disables stripping
/// entirely. Monotonic in `strength`, since `f_bound <= 1`.
///
/// This is a generalisation of something the codebase already does:
/// [`Cattaneo11`] hardcodes a `* 2.0`, the `Strip_f = Strip_f*2` that
/// `Functions.py::StellarMassLoss` applies on the `PipGrylls` branch and
/// that Papers 2 and 3 were run with. **The scale here composes with
/// that**, so `ScaledStripping::new(Cattaneo11, 1.0)` is the published
/// baseline (already doubled), not raw Cattaneo et al. (2011); a
/// `strength` of 2 is four times the paper's stripping in dex.
///
/// The point of making this a parameter is the self-consistency
/// argument: stripping sets a lower bound on how much stellar mass
/// mergers deliver to a central, so "how hard would we have to strip to
/// keep the delivered mass inside the SMHM's budget" is the question
/// that decides whether an SMHM relation is physically attainable. See
/// `steel_postprocess::central_assembly`.
pub struct ScaledStripping<S> {
    pub inner: S,
    pub strength: f64,
}

impl<S> ScaledStripping<S> {
    pub fn new(inner: S, strength: f64) -> Self {
        assert!(strength >= 0.0, "stripping strength must be non-negative, got {strength}");
        Self { inner, strength }
    }
}

impl<S: StellarStrippingModel> StellarStrippingModel for ScaledStripping<S> {
    fn strip_factor(&self, log_host_mass: f64, log_sat_mass: f64, time_fraction: f64) -> f64 {
        self.inner.strip_factor(log_host_mass, log_sat_mass, time_fraction) * self.strength
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;

    #[test]
    fn scaled_stripping_at_unit_strength_is_the_wrapped_model() {
        let base = Cattaneo11;
        let scaled = ScaledStripping::new(Cattaneo11, 1.0);
        for tf in [0.0, 0.25, 0.5, 1.0] {
            let a = base.strip_factor(13.0, 11.0, tf);
            let b = scaled.strip_factor(13.0, 11.0, tf);
            assert!((a - b).abs() < 1e-15, "tf={tf}: {a} vs {b}");
        }
    }

    /// Raising the strength must strip strictly harder (a more negative
    /// log-space factor), and zero strength must strip nothing at all --
    /// the two ends of the sweep the falsification harness relies on.
    #[test]
    fn scaled_stripping_is_monotonic_in_strength() {
        let tf = 0.5;
        let (h, s) = (13.0, 11.0);
        let f1 = ScaledStripping::new(Cattaneo11, 1.0).strip_factor(h, s, tf);
        let f2 = ScaledStripping::new(Cattaneo11, 2.0).strip_factor(h, s, tf);
        let f0 = ScaledStripping::new(Cattaneo11, 0.0).strip_factor(h, s, tf);

        assert!(f1 < 0.0, "baseline should suppress: {f1}");
        assert!(f2 < f1, "more strength must strip harder: {f2} vs {f1}");
        assert_eq!(f0, 0.0, "zero strength must leave the mass untouched, got {f0}");
    }

    /// The documented meaning of the scale: the *bound fraction* is
    /// raised to the power `strength`.
    #[test]
    fn scaling_exponentiates_the_bound_fraction() {
        let (h, s, tf) = (13.5, 11.5, 0.7);
        let base_bound = 10f64.powf(Cattaneo11.strip_factor(h, s, tf));
        let scaled_bound = 10f64.powf(ScaledStripping::new(Cattaneo11, 3.0).strip_factor(h, s, tf));
        assert!(
            (scaled_bound - base_bound.powf(3.0)).abs() < 1e-15,
            "{scaled_bound} vs {}",
            base_bound.powf(3.0)
        );
    }

    #[test]
    fn stellar_stripping_factor_is_zero_or_negative() {
        // strip_factor is log10 of a fraction <= 1, so always <= 0.
        let model = Cattaneo11;
        for &tf in &[0.0, 0.25, 0.5, 0.75, 1.0] {
            let f = model.strip_factor(14.0, 11.0, tf);
            assert!(f <= 1e-9, "strip_factor({tf}) = {f}, expected <= 0");
        }
    }

    #[test]
    fn stellar_stripping_increases_over_time() {
        // More elapsed time (higher time_fraction) should mean more
        // mass stripped (a more negative log-fraction).
        let model = Cattaneo11;
        let early = model.strip_factor(14.0, 11.0, 0.1);
        let late = model.strip_factor(14.0, 11.0, 0.9);
        assert!(late < early, "early={early} late={late}");
    }

    #[test]
    fn stellar_stripping_stays_finite_at_the_time_fraction_limit() {
        // `Cattaneo11` takes log10 of `strip + (1-strip)(1-tf)`, which
        // goes non-positive (NaN) for tf > 1. `BaryonicPipeline` clamps
        // to the documented [0,1] domain; this pins down that the
        // endpoint itself is well-behaved so the clamp is sufficient.
        let model = Cattaneo11;
        let at_limit = model.strip_factor(14.0, 11.0, 1.0);
        assert!(at_limit.is_finite(), "strip_factor at tf=1 should be finite, got {at_limit}");
    }

    #[test]
    fn stellar_stripping_applies_the_pipgrylls_doubling() {
        // `PipGrylls` doubles the log strip factor, i.e. squares the
        // surviving fraction. Pin it against the undoubled Cattaneo+11
        // expression so a revert to the `master` baseline fails loudly
        // rather than quietly changing every satellite SMF.
        let model = Cattaneo11;
        let (log_host, log_sat, tf) = (14.0, 11.0, 0.5);
        let mh_ms = 10f64.powf(log_host - log_sat);
        let strip = 0.6f64.powf((1.428 / (2.0 * std::f64::consts::PI)) * (mh_ms / (1.0 + mh_ms).ln()));
        let undoubled = (strip + (1.0 - strip) * (1.0 - tf)).log10();
        assert!(undoubled < 0.0, "test setup: the undoubled factor should be a suppression");
        assert!((model.strip_factor(log_host, log_sat, tf) - 2.0 * undoubled).abs() < 1e-12);
    }

    #[test]
    fn halo_stripping_mass_is_non_increasing() {
        let cosmo = Planck15::new();
        let model = HaloStrippingVdb05;
        let z: Vec<f64> = (0..10).map(|i| i as f64 * 0.2).collect();
        let dt = vec![0.5; 10];
        let host_mass = vec![13.0; 10];
        let track = model.strip(11.5, &host_mass, &z, &dt, &cosmo);
        for w in track.log_mass.windows(2) {
            assert!(w[1] <= w[0] + 1e-9, "mass should not increase: {:?}", w);
        }
    }
}
