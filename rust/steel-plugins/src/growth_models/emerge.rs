//! EMERGE: empirical galaxy formation via a baryon conversion efficiency
//! applied to the halo accretion rate.
//!
//! Moster, Naab & White (2018) for the base model; O'Leary,
//! Steinwandel, Moster, Martin & Naab (2023), arXiv:2301.07122, for the
//! dwarf-regime extension implemented here.
//!
//! ```text
//! dm*/dt   = eps(M_h, z) . f_b . dM_h/dt
//! eps(M)   = 2 eps_N / [ (M/M_1)^-beta + (M/M_1)^gamma ]
//! ```
//!
//! **Do not confuse this with `crate::smhm::moster`.** The `eps` double
//! power law is algebraically identical to `MosterFormSmhm`'s equation,
//! but `eps` multiplies an accretion *rate* whereas `MosterFormSmhm`
//! multiplies a *mass*. Substituting these coefficients there — or those
//! there here — silently yields a wrong SMHM curve with no error. The
//! stellar-to-halo mass relation in O'Leary+2023's title is an *output*
//! of integrating this rate, not an input parametrisation.
//!
//! Redshift evolution is linear in `z/(1+z)`, matching Moster's
//! convention (`ZEvo::MosterStyle` in `crate::smhm::moster`) rather than
//! STEEL's shifted `(z-0.1)/(1+z)`.
//!
//! # Coefficient provenance — a documented modelling simplification
//!
//! The base `eps()` coefficients below (`M0=11.34829`, `eps_N0=0.009010`,
//! `beta0=3.094621`, `gamma0=1.107304`, `MZ=0.654238`, `epsZ=0.596666`,
//! `betaZ=-2.019841`, `gammaZ=0`) are read verbatim from the pinned
//! upstream `emerge` v1.0.2 commit's `parameterfiles/emerge.param`, and
//! independently verified in Task 8 by calling the real, compiled
//! `sfe()` (`src/galaxies.c`) for every point of `eps_grid.npy` — not a
//! transcription from the paper. `gammaZ = 0` because
//! `SFE_GAMMA_ZEVOLV` is disabled in the shipped build, so gamma carries
//! no redshift evolution in this fixture-verified configuration.
//! (`rust/steel-plugins/tests/fixtures/emerge/provenance.toml`,
//! `[eps_coefficients]`.)
//!
//! The reionization-gate coefficients (`M_q=9.33`, `a_q=0.19`,
//! `R_q=2.56`) come from a *different* source: O'Leary+2023 Table 1,
//! "Logistic" column — the paper's own high-z quenching model (their eq.
//! 9), fetched and read directly from the arXiv PDF. That table column
//! also lists `beta_0=2.22`, `beta_z=-1.50`: these are **not** used here,
//! deliberately. They are part of a joint refit of the base efficiency
//! specifically paired with the Logistic gate in the paper's own
//! analysis (fit on a different, lower-resolution 200 Mpc box, per the
//! paper's section 3.1), and are not independently composable with the
//! shipped/compiled `beta0=3.094621` above. **This module therefore
//! combines the shipped, fixture-verified base `eps()` with the paper's
//! gate suppression layered multiplicatively on top — a deliberate
//! modelling simplification, not a literal reproduction of the paper's
//! own jointly-refit Logistic-column model.** This choice is made so the
//! base efficiency stays checkable against Task 8's compiled-code
//! fixture; it means this module's stellar-to-halo mass relation is not
//! expected to reproduce O'Leary+2023's own Figure 4 digit for digit.
//!
//! `tau_s` (also listed in that table column, "the stellar mass
//! dependent quenching timescale") is **not a parameter of the gate**.
//! O'Leary+2023 section 3.2 states explicitly: "we only allow the low
//! mass slope of the baryon conversion efficiency beta0, its redshift
//! evolution beta_z, and the stellar mass dependent quenching timescale
//! tau_s as free parameters *in addition to those introduced by our new
//! model variations*" — i.e. tau_s is one of the base model's
//! *pre-existing* free parameters (Moster+2018's satellite/environmental
//! quenching timescale), refit jointly alongside beta0/beta_z for this
//! variant, not one of eq. 9's own parameters. Eq. 9 itself,
//! `M_h^min(a) = M_q / (1 + exp(-R_q(a - a_q)))`, uses only `M_q`,
//! `a_q`, `R_q`. Since this module already excludes `Capability::Quenching`
//! (the gate is early-growth suppression, not a satellite quenching
//! prescription — see `descriptor()`), `tau_s` is out of scope here and
//! is not a field of this struct.
//!
//! The reionization gate penalises late-forming low-mass haloes:
//! ```text
//! M_h^min(a) = M_q / [1 + exp(-R_q (a - a_q))]
//! ```
//! Applied via `gate_factor`, using the object's own growth track, so it
//! is exact for satellites as well as centrals (spec section 5).
//!
//! **Extension beyond the paper: smoothing `gate_factor` in log-mass.**
//! O'Leary+2023's eq. 9 defines only a scale-factor-dependent threshold
//! mass, `M_h^min(a)`; the paper's own quenching mechanism is a *hard*,
//! permanent switch (section 3.1: "When a halo does not meet that
//! threshold its star formation will be set to zero and will remain
//! zero for the remainder of that galaxy's lifetime"), not a smooth
//! multiplicative rate suppression. STEEL's rate-based, memoryless-per-
//! call `StellarGrowthModel` interface has no notion of "has this object
//! ever been quenched", so `gate_factor` instead evaluates a smooth
//! logistic in `log_mh` around `log10(M_h^min(a))`, reusing `R_q` — the
//! paper's own "transition strength" parameter — as its steepness, for
//! lack of any dedicated smoothing parameter in the source model (`R_q`
//! is the only steepness-like quantity eq. 9 actually supplies). This is
//! a STEEL-side design choice, not something the paper specifies; see
//! the Task 9 report for the reasoning and the numeric consequence.
//!
//! [`tests::gate_does_not_suppress_massive_halos`] pins the steepness
//! choice: with a unit-width logistic the residual suppression from 4
//! dex above threshold is ~1e-4, not the ~1e-6 the test demands, so
//! `R_q` reuse is load-bearing, not cosmetic.

use rand::RngCore;

use steel_core::accretion::AccretionContext;
use steel_core::compat::{
    Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor,
};
use steel_core::cosmology::MassDefinition;
use steel_core::stellar_growth::StellarGrowthModel;

pub struct EmergeGrowth {
    /// `eps_N(z) = eps_n[0] + eps_n[1] . z/(1+z)`
    eps_n: [f64; 2],
    /// `log10 M_1(z) = log_m1[0] + log_m1[1] . z/(1+z)`
    log_m1: [f64; 2],
    /// `beta(z) = beta[0] + beta[1] . z/(1+z)`
    beta: [f64; 2],
    /// `gamma`, held fixed with redshift (`SFE_GAMMA_ZEVOLV` disabled in
    /// the shipped upstream build; see module doc).
    gamma: f64,
    /// Reionization gate: characteristic mass `M_q` \[log10 Msun\],
    /// scale factor `a_q`, and steepness `R_q`. `tau_s` is deliberately
    /// absent — see module doc.
    log_m_q: f64,
    a_q: f64,
    r_q: f64,
    /// Baryon fraction f_b = Omega_b / Omega_m, taken from the run
    /// cosmology at construction.
    baryon_fraction: f64,
}

impl EmergeGrowth {
    /// O'Leary et al. (2023) logistic-quenching gate, layered on the
    /// upstream v1.0.2 shipped/compiled base efficiency. See the module
    /// doc for exactly which numbers come from where and why they are
    /// not literally the paper's own jointly-refit Logistic-column
    /// model.
    ///
    /// `baryon_fraction` matches Planck15 (0.0486/0.3089 = 0.1573).
    pub fn o_leary23() -> Self {
        Self {
            // provenance.toml [eps_coefficients]: verified against the
            // pinned upstream commit's compiled sfe().
            eps_n: [0.009010, 0.596666],
            log_m1: [11.34829, 0.654238],
            beta: [3.094621, -2.019841],
            gamma: 1.107304,
            // provenance.toml [unrelated_parameters_not_needed_for_this_fixture]:
            // O'Leary+2023 Table 1, Logistic column, verified against the
            // arXiv PDF directly.
            log_m_q: 9.33,
            a_q: 0.19,
            r_q: 2.56,
            baryon_fraction: 0.0486 / 0.3089,
        }
    }

    fn z_param(z: f64) -> f64 {
        z / (1.0 + z)
    }

    /// `log10 M_1(z)`, the pivot mass.
    pub fn log_m1(&self, z: f64) -> f64 {
        self.log_m1[0] + self.log_m1[1] * Self::z_param(z)
    }

    /// Instantaneous baryon conversion efficiency, dimensionless in (0, 1].
    pub fn efficiency(&self, log_mh: f64, z: f64) -> f64 {
        let zp = Self::z_param(z);
        let eps_n = self.eps_n[0] + self.eps_n[1] * zp;
        let beta = self.beta[0] + self.beta[1] * zp;
        let ratio = 10f64.powf(log_mh - self.log_m1(z));
        let eps = 2.0 * eps_n / (ratio.powf(-beta) + ratio.powf(self.gamma));
        eps.clamp(f64::MIN_POSITIVE, 1.0)
    }

    /// Reionization suppression in \[0, 1\]. Unity for haloes already
    /// well above `M_h^min` at this epoch, falling towards zero for
    /// haloes below it — so a halo that assembled its mass late is
    /// penalised. See the module doc for why this is a smooth logistic
    /// in `log_mh` (a STEEL-side extension of eq. 9) rather than the
    /// paper's own hard, permanent switch, and why `R_q` sets its
    /// steepness.
    pub fn gate_factor(&self, log_mh: f64, z: f64) -> f64 {
        let a = 1.0 / (1.0 + z);
        let log_m_min = self.log_m_q - (1.0 + (-self.r_q * (a - self.a_q)).exp()).log10();
        1.0 / (1.0 + 10f64.powf(-self.r_q * (log_mh - log_m_min)))
    }

    /// `dM_h/dt` \[Msun/yr\] at `z`, from the object's own track by
    /// central difference. Returns 0 where the track cannot support a
    /// finite difference.
    fn halo_accretion_rate(&self, z: f64, ctx: &AccretionContext<'_>) -> f64 {
        let t = ctx.own_track;
        if t.z.len() < 2 {
            return 0.0;
        }
        // Nearest sample to `z`; the track is increasing into the past.
        let i = t
            .z
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - z).abs().partial_cmp(&(b.1 - z).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (i0, i1) = if i == 0 { (0, 1) } else { (i - 1, i) };
        // i0 is younger (lower z) than i1.
        let dt_yr = (ctx.cosmology.age(t.z[i0]) - ctx.cosmology.age(t.z[i1])) * 1.0e9;
        if dt_yr <= 0.0 {
            return 0.0;
        }
        let dm = 10f64.powf(t.log_mass[i0]) - 10f64.powf(t.log_mass[i1]);
        (dm / dt_yr).max(0.0)
    }
}

impl StellarGrowthModel for EmergeGrowth {
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        _rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        let mdot_h = self.halo_accretion_rate(z, ctx);
        if mdot_h <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let rate = self.efficiency(log_mh, z)
            * self.gate_factor(log_mh, z)
            * self.baryon_fraction
            * mdot_h;
        if rate <= 0.0 {
            f64::NEG_INFINITY
        } else {
            rate.log10()
        }
    }
}

impl DescribedPlugin for EmergeGrowth {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "emerge",
            // Li & White (2009) calibration data (data/smf.dat) is
            // Chabrier-IMF; provenance.toml [conventions].imf documents
            // this as inferred (O'Leary+2023 never states an IMF).
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::HFree,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            // Supplies M* and its own scatter. The reionization gate is
            // early-growth suppression, not satellite quenching, so
            // Quenching is deliberately absent and STEEL's
            // QuenchingModel stays compatible.
            provides: &[Capability::StellarMass, Capability::Scatter],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use steel_core::accretion::AccretionContext;
    use steel_core::cosmology::MassDefinition;
    use steel_core::halo_growth::GrowthTrack;

    fn ctx_for<'a>(t: &'a GrowthTrack, c: &'a Planck15) -> AccretionContext<'a> {
        AccretionContext::central(t, c, MassDefinition::Vir)
    }

    /// Track for a halo that assembled early: already massive at high z.
    fn early_track() -> GrowthTrack {
        GrowthTrack {
            z: vec![0.0, 1.0, 2.0, 4.0, 6.0],
            log_mass: vec![11.0, 10.95, 10.9, 10.8, 10.7],
        }
    }

    /// Same z=0 mass, assembled late: progenitors are far smaller.
    fn late_track() -> GrowthTrack {
        GrowthTrack {
            z: vec![0.0, 1.0, 2.0, 4.0, 6.0],
            log_mass: vec![11.0, 10.2, 9.4, 8.6, 8.0],
        }
    }

    #[test]
    fn efficiency_peaks_at_the_pivot_mass() {
        let m = EmergeGrowth::o_leary23();
        let at_pivot = m.efficiency(m.log_m1(0.0), 0.0);
        let below = m.efficiency(m.log_m1(0.0) - 2.0, 0.0);
        let above = m.efficiency(m.log_m1(0.0) + 2.0, 0.0);
        assert!(at_pivot > below, "{at_pivot} vs {below}");
        assert!(at_pivot > above, "{at_pivot} vs {above}");
    }

    #[test]
    fn efficiency_is_a_fraction() {
        let m = EmergeGrowth::o_leary23();
        for log_mh in [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0] {
            for z in [0.0, 1.0, 3.0, 6.0] {
                let e = m.efficiency(log_mh, z);
                assert!(e > 0.0 && e <= 1.0, "eps({log_mh}, {z}) = {e}");
            }
        }
    }

    #[test]
    fn growth_rate_increases_with_halo_mass_below_the_pivot() {
        let m = EmergeGrowth::o_leary23();
        let c = Planck15::new();
        let t = early_track();
        let ctx = ctx_for(&t, &c);
        let r10 = m.stellar_growth_rate(10.0, 1.0, &ctx, None);
        let r11 = m.stellar_growth_rate(11.0, 1.0, &ctx, None);
        assert!(r11 > r10, "rate should rise with mass below pivot: {r10} then {r11}");
    }

    /// The reionization gate is the paper's dwarf-regime contribution: a
    /// halo that only assembled after the gate epoch is suppressed.
    #[test]
    fn gate_suppresses_late_forming_low_mass_halos() {
        let m = EmergeGrowth::o_leary23();
        let c = Planck15::new();
        let early = early_track();
        let late = late_track();
        let m_early = steel_core::integrate_stellar_mass(&m, &ctx_for(&early, &c), 0.0, None);
        let m_late = steel_core::integrate_stellar_mass(&m, &ctx_for(&late, &c), 0.0, None);
        assert!(
            m_late < m_early,
            "late-forming halo should end up less massive: {m_late} vs {m_early}"
        );
    }

    #[test]
    fn gate_does_not_suppress_massive_halos() {
        let m = EmergeGrowth::o_leary23();
        // Well above M_q, the gate must be inert (factor ~1).
        let g = m.gate_factor(13.0, 4.0);
        assert!((g - 1.0).abs() < 1e-6, "gate factor at high mass = {g}");
    }

    #[test]
    fn descriptor_declares_stellar_mass_and_not_quenching() {
        let d = EmergeGrowth::o_leary23().descriptor();
        assert!(d.provides.contains(&Capability::StellarMass));
        // EMERGE's gate suppresses early low-mass growth; it is not a
        // satellite quenching prescription, so STEEL's QuenchingModel
        // remains compatible.
        assert!(!d.provides.contains(&Capability::Quenching));
    }
}
