//! UniverseMachine: SFR assigned from a halo's peak circular velocity
//! and assembly history, drawn from a bimodal star-forming / quenched
//! distribution.
//!
//! Behroozi, Wechsler, Hearin & Conroy (2019) for the base (DR1) model;
//! Wang, Nadler, Mao, Wechsler, Behroozi et al. (2024), arXiv:2404.14500,
//! for the UM-SAGA low-mass quenching extension implemented here.
//!
//! All coefficients below are read verbatim from the pinned UM-SAGA
//! commit's real best-fit parameter file
//! (`scripts/bestfit_var15_fin.dat`, commit `6aff8d792e81b...`,
//! independently re-verified against a fresh clone of that exact commit
//! while implementing this module) and cross-checked term-by-term
//! against the real, compiling `calc_sf_model()` (`sf_model.c:73-130`).
//! See `rust/steel-plugins/tests/fixtures/um_saga/provenance.toml` for
//! the full citation of every field.
//!
//! # SFR (star-forming mode)
//!
//! Linear in SFR, **not** a pure double power law in log space — it is a
//! double power law PLUS a Gaussian bump, all inside `epsilon`'s linear
//! scaling (verbatim from four byte-identical copies at the pinned
//! commit: `gen_total_csfr.c:36-40` and three siblings):
//!
//! ```text
//! vd            = log_v - v_1(z)
//! SFR_linear(v) = epsilon(z) * ( 1 / (10^(alpha(z) vd) + 10^(beta(z) vd))
//!                                + gamma(z) * exp(-0.5 (vd/delta)^2) )
//! ```
//!
//! this module returns `log10(SFR_linear)`, matching the trait's
//! convention. Do **not** assume this turns over near the pivot the way
//! a pure double power law would: with the real best-fit coefficients it
//! is *monotonically rising* out to log_v=3.0 at low z (confirmed
//! against the committed `sfr_sf_grid.npy` fixture) and only develops a
//! visible peak-then-fall shape at higher z, once the Gaussian bump and
//! the z-evolving pivot/slopes combine. The apparent "SFR-Vmax relation
//! turns over at high mass" seen in UM's population-level plots is
//! mostly `f_Q` diluting the *mean*, not this star-forming branch
//! falling.
//!
//! # Each coefficient's z-evolution is verified individually
//!
//! `calc_sf_model()` does **not** use one shared expansion template.
//! Every coefficient below cites its own exact source line; several
//! differ in term count or in which basis function (`flow20`, `fhigh20`,
//! `fmid20`) fills which slot:
//!
//! - `epsilon` (`EFF_0` family, sf_model.c:90): 4 terms, capped basis,
//!   `base + flow20*A + fhigh20*A2 + fmid20*A3`.
//! - `v_1` (`V_1` family, sf_model.c:91): 4 terms, **UNcapped** basis
//!   (`flow`/`fhigh`/`fmid`, no z=20 ceiling) — the only coefficient that
//!   uses the uncapped basis besides `r_cen`.
//! - `alpha` (`ALPHA` family, sf_model.c:92): 4 terms, capped basis,
//!   same term order as `epsilon`.
//! - `beta` (`BETA` family, sf_model.c:94): only **3** terms — `base +
//!   flow20*A + fhigh20*A2`. No `fmid20`/"la" term exists upstream.
//! - `gamma` (`GAMMA` family, sf_model.c:96, log10-valued): 3 terms,
//!   same shape as `beta` (`base + flow20*A + fhigh20*A2`).
//! - `delta` (`DELTA`, sf_model.c:95): a single constant, no z evolution
//!   at all.
//! - `q_lvmp`/DR1 high-mass quenching midpoint (sf_model.c:124): 3
//!   terms, `base + flow20*A + fhigh20*Z`.
//! - `q_sig_lvmp`/DR1 high-mass quenching width (sf_model.c:125): 3
//!   terms, but the **third term uses `fmid20`, not `fhigh20`** — a
//!   different basis slot from its own midpoint sibling. Floored at
//!   0.01.
//! - `q_lvmp_low`/UM-SAGA low-mass quenching midpoint (sf_model.c:128):
//!   same 3-term/`fhigh20` shape as `q_lvmp`.
//! - `q_sig_lvmp_low`/UM-SAGA low-mass quenching width (sf_model.c:129):
//!   same 3-term/`fmid20` shape as `q_sig_lvmp`. Floored at 0.01.
//! - `fq_min` (`Q_MIN` family, sf_model.c:121): only **2** terms, `base +
//!   flow20*A`, floored at 0.
//!
//! `flow20`/`fmid20`/`fhigh20` are computed on a z capped at 20
//! (`z_ceiling`); `flow`/`fmid`/`fhigh` are the same functional forms on
//! the true, uncapped z. All grid points this module is validated
//! against (z <= 6) sit far below that ceiling, so it does not affect
//! any committed number, but the distinction is real in the source and
//! preserved here for correctness at higher z.
//!
//! # Quenched fraction
//!
//! ```text
//! fq = fq_min + (1-fq_min)*Phi((log_v-q_lvmp)/q_sig_lvmp)
//!            + (1-fq_min)*(1 - Phi((log_v-q_lvmp_low)/q_sig_lvmp_low))
//! ```
//! clamped to `[0,1]` (`make_sf_catalog.c:687-688`, `if (fq > 1.) fq =
//! 1.;`, applied immediately after this exact expression). `Phi` is the
//! standard normal CDF, `0.5 + 0.5*erf(x/sqrt(2))` — proven exactly equal
//! to upstream's own `cached_rank` lookup table by that table's own
//! generator (`make_sf_catalog.c`: `erf_cache[i] = 0.5+0.5*erf(x/M_SQRT2)`).
//!
//! # `log_vmpeak`: a per-*snapshot* running peak, not one value held
//! # fixed across an object's whole assembly history
//!
//! UM's `vMpeak` is `Vmax` evaluated at a halo's epoch of peak
//! historical mass *so far* (`vmax_at_mpeak`; resolved directly from
//! upstream's own merger-tree builder,
//! `split_halo_trees_phase2.c:462-463,736,738` — **not** the historical
//! maximum of `Vmax` itself, a different field upstream never uses as
//! `vmp`). Upstream computes this per **snapshot**: every node in a
//! merger tree is its own catalog entry with its own Mpeak-so-far, fixed
//! once that snapshot exists, and it is *that* per-snapshot value each
//! snapshot's own SFR/quenching draw uses.
//!
//! An earlier version of this module conflated "fixed once a snapshot
//! exists" with "fixed across every snapshot of one integration call":
//! `stellar_growth_rate` computed `log_v` once from `ctx.own_track`'s
//! root (observed-epoch) sample and reused it for every redshift
//! `integrate_stellar_mass` visits while walking that object's whole
//! progenitor track. That retroactively stamped a halo's *final*
//! quenched fraction onto every earlier progenitor — including
//! progenitors that were small, unremarkable, actively star-forming
//! halos at high z — collapsing the integrated M* for massive halos and
//! making M*(Mh) non-monotonic (confirmed: at z=0.1 a 10^14.2 Msun/h
//! halo's root-fixed vMpeak gave f_Q ~= 0.98 even paired with a z~5.5,
//! 10^11.75 Msun/h progenitor that should be essentially unquenched).
//! `docs/VALIDATION.md` §6.5 records a self-consistency check that hit
//! exactly this discrepancy and, at the time, concluded the check's
//! "per-step" context was the mistake rather than the fixed-root
//! implementation — the wrong direction.
//!
//! The fix: `AccretionContext::own_track` is monotonic (mass
//! non-decreasing forward in time), so a progenitor's peak-so-far *is*
//! its own contemporary mass — exactly the `log_mh` argument
//! `StellarGrowthModel::stellar_growth_rate` already receives at every
//! integration step. `vmpeak_at(log_mh, z, ctx)` uses that per-step
//! pair; [`UniverseMachineGrowth::log_vmpeak`] remains as a convenience
//! query for "this object's vMpeak right now" (its root/observed epoch)
//! but is no longer what `stellar_growth_rate` itself consults.
//!
//! **Velocity-keyed, not mass-keyed.** Converting `log_mh -> vMpeak`
//! goes through `crate::harmonise::mpeak_to_vmax`, which needs a
//! concentration-mass relation — an injected assumption that materially
//! affects results, not an implementation detail (spec section 7):
//! constructor-supplied and recorded in the runfile. It must match the
//! `vmpeak_definition` recorded in the fixture provenance
//! (`vmax_at_mpeak`, as above).
//!
//! # Quenching is internal; scatter is real
//!
//! The bimodal PDF already contains quenching, so this plugin declares
//! `Capability::Quenching` and the composition validator will reject
//! pairing it with STEEL's separate `QuenchingModel`. `Capability::Scatter`
//! is likewise honestly earned, not copied from a template: given an
//! RNG, `stellar_growth_rate` draws a genuine Bernoulli(`f_Q`) mode
//! selection *and*, within the star-forming mode, log-normal scatter
//! with the real, z-evolving width `sig_sf(z)` (`INTR_SCATTER_SF`
//! family, sf_model.c:100-103, capped at the fixed `OBS_SCATTER_SFR_SF =
//! 0.3`, sf_model.h:5).
//!
//! # Assembly-history dependence — a documented simplification
//!
//! Real UM correlates each halo's SFR-distribution rank with a
//! *persistent*, tree-wide rank state (`rank1`/`rarank` in
//! `make_sf_catalog.c`'s `_calc_sf`), at a strength `r1(log_v, z) =
//! r_min + (1-r_min)*Phi((r_cen(z)-log_v)/r_width)` (sf_model.c:110-118,
//! using the real, verified `R_MIN`/`R_WIDTH`/`R_CENTER`/`R_CENTER_A`
//! coefficients). That persistent rank is a Markov chain carried across
//! an entire merger tree — it has no equivalent in
//! `StellarGrowthModel`'s per-call, per-object interface, which supplies
//! only this object's own growth track. This module therefore uses the
//! real `r1(log_v, z)` correlation strength but pairs it with a
//! track-local proxy for "has this halo grown fast recently"
//! (`delta_vmax_proxy`) rather than upstream's own persistent rank —
//! a deliberate modelling simplification, parallel in spirit to
//! `EmergeGrowth::gate_factor`'s smoothing choice (see that module's
//! doc). It nudges the effective quenched fraction down for
//! fast-growing haloes; it does not reproduce upstream's own two-point
//! rank correlation function.

use std::sync::Arc;

use rand::{Rng, RngCore};
use rand_distr::{Distribution, Normal};

use steel_core::accretion::AccretionContext;
use steel_core::compat::{
    Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor,
};
use steel_core::cosmology::MassDefinition;
use steel_core::stellar_growth::StellarGrowthModel;

use crate::harmonise::{mpeak_to_vmax, ConcentrationMassRelation};

/// The z-dependent basis functions `calc_sf_model()` builds its
/// coefficient expansions from (sf_model.c:75-85). `flow`/`fmid`/`fhigh`
/// use the true z; `flow20`/`fmid20`/`fhigh20` use `z` capped at 20
/// first. Different coefficients pick different subsets of these six
/// values — see the module doc.
struct ZBasis {
    flow: f64,
    fmid: f64,
    fhigh: f64,
    flow20: f64,
    fmid20: f64,
    fhigh20: f64,
}

impl ZBasis {
    fn new(z: f64) -> Self {
        let a = 1.0 / (1.0 + z);
        let flow = 1.0 - a;
        let fmid = (1.0 + z).ln() - flow;
        let fhigh = z - flow;

        let z20 = z.min(20.0);
        let a20 = 1.0 / (1.0 + z20);
        let flow20 = 1.0 - a20;
        let fmid20 = (1.0 + z20).ln() - flow20;
        let fhigh20 = z20 - flow20;

        Self { flow, fmid, fhigh, flow20, fmid20, fhigh20 }
    }
}

/// Standard normal CDF, `0.5 + 0.5*erf(x/sqrt(2))` — see the module doc
/// for why this exactly matches upstream's `cached_rank`.
fn std_normal_cdf(x: f64) -> f64 {
    0.5 + 0.5 * erf(x / std::f64::consts::SQRT_2)
}

/// Abramowitz & Stegun 7.1.26 error-function approximation, max absolute
/// error 1.5e-7 — several orders of magnitude below the 0.02 absolute
/// tolerance this feeds into (spec section 6).
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

pub struct UniverseMachineGrowth {
    /// `EFF_0` family: log10 SFR normalisation `c.epsilon` (sf_model.c:90).
    /// `[EFF_0, EFF_0_A, EFF_0_A2, EFF_0_A3]`, capped basis.
    eff_0: [f64; 4],
    /// `V_1` family: log10 pivot velocity `c.v_1` (sf_model.c:91).
    /// `[V_1, V_1_A, V_1_A2, V_1_A3]`, **UNcapped** basis.
    v_1: [f64; 4],
    /// `ALPHA` family: low-mass slope `c.alpha` (sf_model.c:92).
    /// `[ALPHA, ALPHA_A, ALPHA_A2, ALPHA_A3]`, capped basis.
    alpha: [f64; 4],
    /// `BETA` family: high-mass slope `c.beta` (sf_model.c:94). Only 3
    /// terms exist upstream. `[BETA, BETA_A, BETA_A2]`, capped basis.
    beta: [f64; 3],
    /// `DELTA`: Gaussian bump width `c.delta` (sf_model.c:95). No z
    /// evolution.
    delta: f64,
    /// `GAMMA` family: log10 Gaussian bump amplitude `c.gamma`
    /// (sf_model.c:96). `[GAMMA, GAMMA_A, GAMMA_A2]`, capped basis.
    gamma: [f64; 3],
    /// `Q_LVMP` family: DR1 high-mass quenching midpoint `c.q_lvmp`
    /// (sf_model.c:124). `[Q_LVMP, Q_LVMP_A, Q_LVMP_Z]`; 3rd term uses
    /// `fhigh20`.
    q_lvmp: [f64; 3],
    /// `Q_SIG_LVMP` family: DR1 high-mass quenching width `c.q_sig_lvmp`
    /// (sf_model.c:125), floored at 0.01. 3rd term uses `fmid20` (NOT
    /// `fhigh20` — different from `q_lvmp` above).
    q_sig_lvmp: [f64; 3],
    /// `Q_MIN` family: quenched-fraction floor `c.fq_min`
    /// (sf_model.c:121), floored at 0. Only 2 terms.
    q_min: [f64; 2],
    /// `Q_LVMP_low` family: UM-SAGA low-mass quenching midpoint
    /// `c.q_lvmp_low` (sf_model.c:128). Same 3-term/`fhigh20` shape as
    /// `q_lvmp`.
    q_lvmp_low: [f64; 3],
    /// `Q_SIG_low_LVMP` family: UM-SAGA low-mass quenching width
    /// `c.q_sig_lvmp_low` (sf_model.c:129), floored at 0.01. Same
    /// 3-term/`fmid20` shape as `q_sig_lvmp`.
    q_sig_lvmp_low: [f64; 3],
    /// `INTR_SCATTER_SF` family: star-forming-mode scatter `c.sig_sf`
    /// (sf_model.c:100). `[INTR_SCATTER_SF, INTR_SCATTER_SF_A]`,
    /// uncapped `flow` basis, clamped to `[0, obs_scatter_sfr_sf]`.
    intr_scatter_sf: [f64; 2],
    /// `OBS_SCATTER_SFR_SF`: fixed cap on `sig_sf` (sf_model.h:5 = 0.3).
    /// A model constant, not a fitted parameter.
    obs_scatter_sfr_sf: f64,
    /// `R_MIN`, `R_WIDTH`: assembly-history rank-correlation strength
    /// `r1(log_v,z)` (sf_model.c:112, 111).
    r_min: f64,
    r_width: f64,
    /// `R_CENTER`/`R_CENTER_A`: `c.r_cen = R_CENTER + R_CENTER_A*flow`
    /// (sf_model.c:110), **UNcapped** basis.
    v_r: [f64; 2],
    cm: Arc<dyn ConcentrationMassRelation>,
}

impl UniverseMachineGrowth {
    /// UM-SAGA `var15` "final" MCMC best fit (chi2=402.707), read
    /// verbatim from `scripts/bestfit_var15_fin.dat` at the pinned
    /// commit and cross-checked term-by-term against the real,
    /// compiling `calc_sf_model()`. See
    /// `rust/steel-plugins/tests/fixtures/um_saga/provenance.toml` for
    /// the full per-field citation.
    pub fn um_saga(cm: Arc<dyn ConcentrationMassRelation>) -> Self {
        Self {
            eff_0: [0.219183863, 0.601031497, -0.767542291, 5.21295141],
            v_1: [2.13901060, -0.141230884, -0.215809966, 1.56135690],
            alpha: [-6.16624844, -3.97921077, -0.571438200, 6.60432668],
            beta: [-1.81310646, 0.767409694, 1.77222666],
            delta: 0.0658936762,
            gamma: [-1.91445849, 4.13279629, -0.920527434],
            q_lvmp: [2.23213145, 0.138994602, 0.107465062],
            q_sig_lvmp: [0.270386706, -0.203837747, 0.0542323029],
            q_min: [-1.08574167, 0.721040181],
            q_lvmp_low: [1.65023184, -0.264936233, -0.754329105],
            q_sig_lvmp_low: [0.162369262, 0.386523504, 0.0807415281],
            intr_scatter_sf: [-0.302161453, 4.96066877],
            obs_scatter_sfr_sf: 0.3,
            r_min: 0.464005834,
            r_width: 0.240743627,
            v_r: [2.32425410, -6.10531145],
            cm,
        }
    }

    fn log10_epsilon(&self, b: &ZBasis) -> f64 {
        self.eff_0[0] + self.eff_0[1] * b.flow20 + self.eff_0[2] * b.fhigh20 + self.eff_0[3] * b.fmid20
    }

    fn v_1(&self, b: &ZBasis) -> f64 {
        self.v_1[0] + self.v_1[1] * b.flow + self.v_1[2] * b.fhigh + self.v_1[3] * b.fmid
    }

    fn alpha(&self, b: &ZBasis) -> f64 {
        self.alpha[0] + self.alpha[1] * b.flow20 + self.alpha[2] * b.fhigh20 + self.alpha[3] * b.fmid20
    }

    fn beta(&self, b: &ZBasis) -> f64 {
        self.beta[0] + self.beta[1] * b.flow20 + self.beta[2] * b.fhigh20
    }

    fn log10_gamma(&self, b: &ZBasis) -> f64 {
        self.gamma[0] + self.gamma[1] * b.flow20 + self.gamma[2] * b.fhigh20
    }

    fn q_lvmp(&self, b: &ZBasis) -> f64 {
        self.q_lvmp[0] + self.q_lvmp[1] * b.flow20 + self.q_lvmp[2] * b.fhigh20
    }

    fn q_sig_lvmp(&self, b: &ZBasis) -> f64 {
        (self.q_sig_lvmp[0] + self.q_sig_lvmp[1] * b.flow20 + self.q_sig_lvmp[2] * b.fmid20).max(0.01)
    }

    fn q_min(&self, b: &ZBasis) -> f64 {
        (self.q_min[0] + self.q_min[1] * b.flow20).max(0.0)
    }

    fn q_lvmp_low(&self, b: &ZBasis) -> f64 {
        self.q_lvmp_low[0] + self.q_lvmp_low[1] * b.flow20 + self.q_lvmp_low[2] * b.fhigh20
    }

    fn q_sig_lvmp_low(&self, b: &ZBasis) -> f64 {
        (self.q_sig_lvmp_low[0] + self.q_sig_lvmp_low[1] * b.flow20 + self.q_sig_lvmp_low[2] * b.fmid20)
            .max(0.01)
    }

    /// Star-forming-mode log-normal scatter width (sf_model.c:100-103).
    fn sig_sf(&self, b: &ZBasis) -> f64 {
        (self.intr_scatter_sf[0] + self.intr_scatter_sf[1] * b.flow)
            .clamp(0.0, self.obs_scatter_sfr_sf)
    }

    /// log10 vMpeak \[km/s\] for a halo of mass `log_mh` \[log10 Msun,
    /// h-free\] observed at `z`.
    ///
    /// `log_mh` must be the halo's own peak mass *so far*, as of `z` —
    /// not necessarily its final/observed-epoch mass. Because
    /// `GrowthTrack` masses are monotonically non-decreasing forward in
    /// time, the peak-so-far at any epoch on a track *is* that epoch's
    /// own contemporary mass, which is exactly what
    /// `StellarGrowthModel::stellar_growth_rate` receives as `log_mh` at
    /// every integration step (bug fix, was: this method took only
    /// `&ctx` and always read `ctx.own_track`'s root/observed-epoch
    /// sample, applying the object's *final* vMpeak retroactively to
    /// every earlier progenitor. For a halo massive enough to be
    /// strongly quenched today, that stamped today's near-total quenched
    /// fraction onto ancestors that were small, unremarkable,
    /// unquenched halos — collapsing the integrated M* for massive
    /// halos and making M*(Mh) non-monotonic. See
    /// `docs/VALIDATION.md` §6.5 for the self-consistency check that,
    /// pre-fix, mistook this bug's symptom for a bug in the check
    /// itself and papered over it instead of catching it).
    fn vmpeak_at(&self, log_mh: f64, z: f64, ctx: &AccretionContext<'_>) -> f64 {
        // `log_mh` is h-free (physical Msun; see `AccretionContext`'s
        // construction in `context.rs`, which subtracts `log10(h)` off
        // the internal Msun/h sampling mass before storing it on the
        // track). `mpeak_to_vmax`, by contrast, expects `Msun/h` (its
        // NFW conversion runs through `Cosmology::rho_crit`, which is in
        // `Msun h^2 / kpc^3`, and `DuttonMaccio14`'s concentration fit
        // pivots on `1e12 h^-1 Msun` — matching every other call site of
        // `m_to_r` in this codebase). Converting back to `Msun/h`
        // requires `+log10(h)` (`M_true = M_quoted / h` => `log_true =
        // log_quoted - log_h` => `log_quoted = log_true + log_h`).
        // Omitting this silently biased every UM vMpeak high by
        // ~0.052-0.053 dex.
        let log_m_peak = log_mh + ctx.cosmology.h().log10();
        mpeak_to_vmax(log_m_peak, z, ctx.cosmology, self.cm.as_ref(), ctx.mass_definition).log10()
    }

    /// log10 vMpeak \[km/s\] for this object *right now*: `Vmax` at
    /// `ctx.own_track`'s root (observed/current) epoch. A convenience
    /// query for "what is this object's vMpeak today" — `stellar_growth_rate`
    /// does not use this; it calls [`Self::vmpeak_at`] with each
    /// integration step's own `(log_mh, z)` instead (see that method's
    /// doc for why).
    pub fn log_vmpeak(&self, ctx: &AccretionContext<'_>) -> f64 {
        self.vmpeak_at(ctx.own_track.log_mass[0], ctx.own_track.z[0], ctx)
    }

    /// log10 SFR \[Msun/yr\] for the star-forming mode: the double
    /// power law plus Gaussian bump, evaluated in linear SFR then
    /// re-logged. See the module doc for the exact formula and why it
    /// is not a pure double power law in log space.
    pub fn log_sfr_star_forming(&self, log_v: f64, z: f64) -> f64 {
        let b = ZBasis::new(z);
        let vd = log_v - self.v_1(&b);
        let alpha = self.alpha(&b);
        let beta = self.beta(&b);
        let denom = 10f64.powf(alpha * vd) + 10f64.powf(beta * vd);
        let gauss = 10f64.powf(self.log10_gamma(&b)) * (-0.5 * (vd / self.delta).powi(2)).exp();
        let sfr_linear = 10f64.powf(self.log10_epsilon(&b)) * (1.0 / denom + gauss);
        sfr_linear.log10()
    }

    /// Quenched fraction: the DR1 high-mass term plus the UM-SAGA
    /// low-mass term, clamped to `[0,1]` exactly as upstream does
    /// immediately after computing this same expression
    /// (`make_sf_catalog.c:687-688`).
    pub fn quenched_fraction(&self, log_v: f64, z: f64) -> f64 {
        let b = ZBasis::new(z);
        let fq_min = self.q_min(&b);
        let hi = std_normal_cdf((log_v - self.q_lvmp(&b)) / self.q_sig_lvmp(&b));
        let lo = std_normal_cdf((log_v - self.q_lvmp_low(&b)) / self.q_sig_lvmp_low(&b));
        let fq = fq_min + (1.0 - fq_min) * hi + (1.0 - fq_min) * (1.0 - lo);
        fq.clamp(0.0, 1.0)
    }

    /// Assembly-history rank-correlation strength `r1(log_v, z)`
    /// (sf_model.c:110-118), clamped to `[-1,1]` as upstream does at the
    /// point of use. See the module doc for the simplification this
    /// feeds into.
    fn assembly_bias_correlation(&self, log_v: f64, z: f64) -> f64 {
        let b = ZBasis::new(z);
        let r_cen = self.v_r[0] + self.v_r[1] * b.flow;
        let r1 = self.r_min + (1.0 - self.r_min) * std_normal_cdf((r_cen - log_v) / self.r_width);
        r1.clamp(-1.0, 1.0)
    }

    /// Track-local proxy for "how fast has this halo grown recently":
    /// the change in log halo mass over the most recent track interval.
    /// A stand-in for upstream's own persistent tree-wide SFR rank — see
    /// the module doc. Unaffected by the per-step `vmpeak_at` fix: this
    /// deliberately reflects recent growth, not a vMpeak derivation.
    fn delta_vmax_proxy(ctx: &AccretionContext<'_>) -> f64 {
        let t = ctx.own_track;
        if t.log_mass.len() < 2 {
            return 0.0;
        }
        t.log_mass[0] - t.log_mass[1]
    }
}

impl StellarGrowthModel for UniverseMachineGrowth {
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        // UM is keyed on vMpeak: this progenitor's own peak mass-so-far,
        // which — since `GrowthTrack` is monotonic — is exactly its
        // contemporary `(log_mh, z)` at this call (see `vmpeak_at`'s
        // doc: this is NOT the object's final/observed-epoch mass held
        // fixed across the whole integration; that was a bug).
        let log_v = self.vmpeak_at(log_mh, z, ctx);
        let f_q = self.quenched_fraction(log_v, z);
        let log_sfr_sf = self.log_sfr_star_forming(log_v, z);

        match rng {
            Some(r) => {
                let b = ZBasis::new(z);
                let r1 = self.assembly_bias_correlation(log_v, z);
                let growth_rank = Self::delta_vmax_proxy(ctx).clamp(-1.0, 1.0);
                // Fast-growing haloes are nudged towards star-forming;
                // see the module doc on why this is a simplification of
                // upstream's own persistent-rank correlation.
                let boost = (r1 * growth_rank).max(0.0);
                let f_q_eff = (f_q - boost).clamp(0.0, 1.0);

                let u: f64 = r.gen();
                if u < f_q_eff {
                    // Quenched mode. Real UM sets `sfr_q` from the
                    // current stellar mass plus a fixed quenched sSFR
                    // (`Q_SSFR`) once `sm > 1`; a rate-only call has no
                    // running M* to consult, so this always takes
                    // upstream's own fallback branch for `sm <= 1`:
                    // `sfr_q = sfr_sf - 2.0`.
                    log_sfr_sf - 2.0
                } else {
                    let sig_sf = self.sig_sf(&b);
                    if sig_sf > 0.0 {
                        let n = Normal::new(0.0, sig_sf).expect("sig_sf clamped to [0, 0.3]");
                        log_sfr_sf + n.sample(r)
                    } else {
                        log_sfr_sf
                    }
                }
            }
            // Deterministic: the population mean over both modes, so
            // callers without an RNG get a usable value rather than an
            // arbitrary mode. Averaged in linear SFR, then re-logged.
            None => {
                let sf = 10f64.powf(log_sfr_sf);
                let q = 10f64.powf(log_sfr_sf - 2.0);
                let mean = (1.0 - f_q) * sf + f_q * q;
                if mean <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    mean.log10()
                }
            }
        }
    }
}

impl DescribedPlugin for UniverseMachineGrowth {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "universe_machine",
            // provenance.toml [conventions]: README.pdf section 7.1.
            imf: Imf::Chabrier,
            // Correction 6: must match DuttonMaccio14's calibration.
            mass_definition: MassDefinition::Vir,
            // provenance.toml [conventions].h_convention: vmax/vmp carry
            // no h factor anywhere in the upstream source.
            h_convention: HConvention::HFree,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            provides: &[
                Capability::StellarMass,
                // The bimodal SFR PDF already contains quenching;
                // dropping this would let a run pair UM with STEEL's
                // QuenchingModel and quench twice, silently.
                Capability::Quenching,
                // Honestly earned (see module doc): `stellar_growth_rate`
                // draws a genuine mode selection plus log-normal scatter
                // within the star-forming mode when given an RNG.
                Capability::Scatter,
                Capability::StarFormationRate,
            ],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use crate::harmonise::{mpeak_to_vmax, DuttonMaccio14};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Arc;
    use steel_core::accretion::AccretionContext;
    use steel_core::cosmology::{Cosmology, MassDefinition};
    use steel_core::halo_growth::GrowthTrack;

    fn model() -> UniverseMachineGrowth {
        UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14))
    }

    fn track() -> GrowthTrack {
        GrowthTrack {
            z: vec![0.0, 1.0, 2.0, 4.0],
            log_mass: vec![12.0, 11.6, 11.2, 10.6],
        }
    }

    #[test]
    fn quenched_fraction_is_a_fraction() {
        let m = model();
        for log_v in [1.4, 1.8, 2.2, 2.6, 3.0] {
            for z in [0.0, 1.0, 3.0] {
                let f = m.quenched_fraction(log_v, z);
                assert!((0.0..=1.0).contains(&f), "f_Q({log_v}, {z}) = {f}");
            }
        }
    }

    /// UM-SAGA's contribution is enhanced quenching at low mass, so f_Q
    /// must be non-monotonic: high at low velocity (the new term), low in
    /// the middle, high again at cluster scales (the DR1 term). Verified
    /// against the real best-fit coefficients by hand before writing this
    /// test (see task-11-report.md).
    #[test]
    fn quenched_fraction_is_elevated_at_both_extremes() {
        let m = model();
        let low = m.quenched_fraction(1.5, 0.0);
        let mid = m.quenched_fraction(2.1, 0.0);
        let high = m.quenched_fraction(2.9, 0.0);
        assert!(low > mid, "low-mass quenching: f_Q({low}) should exceed mid ({mid})");
        assert!(high > mid, "high-mass quenching: f_Q({high}) should exceed mid ({mid})");
    }

    /// The real UM-SAGA SFR(v) main sequence does NOT turn over at z=0
    /// within this velocity range (confirmed against Task 10's own
    /// fixture: monotonically rising at z=0.1/0.5/1.0). The turnover this
    /// test exercises only appears at higher z, where the pivot moves to
    /// lower v and the fixture grid shows a real peak near log_v=2.6 that
    /// falls off by log_v=3.0 (confirmed by direct evaluation against
    /// sfr_sf_grid.npy before writing this test).
    #[test]
    fn star_forming_sfr_rises_then_falls_with_velocity_at_high_z() {
        let m = model();
        let a = m.log_sfr_star_forming(1.6, 6.0);
        let b = m.log_sfr_star_forming(2.6, 6.0);
        let c = m.log_sfr_star_forming(3.0, 6.0);
        assert!(b > a, "SFR should rise below the peak: {a} then {b}");
        assert!(b > c, "SFR should fall above the peak: {b} then {c}");
    }

    /// The rate is drawn from a bimodal PDF, so it is stochastic — but
    /// reproducible for a fixed seed.
    #[test]
    fn rate_is_stochastic_but_seed_reproducible() {
        let m = model();
        let c = Planck15::new();
        let t = track();
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let mut r1 = StdRng::seed_from_u64(7);
        let mut r2 = StdRng::seed_from_u64(7);
        let a = m.stellar_growth_rate(12.0, 0.5, &ctx, Some(&mut r1));
        let b = m.stellar_growth_rate(12.0, 0.5, &ctx, Some(&mut r2));
        assert_eq!(a.to_bits(), b.to_bits(), "same seed must give the same draw");
    }

    /// Without an RNG the model must still return something usable: the
    /// quenched-fraction-weighted mean rather than panicking or picking a
    /// mode arbitrarily.
    #[test]
    fn no_rng_gives_the_population_mean_rate() {
        let m = model();
        let c = Planck15::new();
        let t = track();
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let mean = m.stellar_growth_rate(12.0, 0.5, &ctx, None);
        assert!(mean.is_finite(), "mean rate should be finite, got {mean}");
        // Must sit at or below the pure star-forming rate, since some of
        // the population is quenched. `stellar_growth_rate` keys vMpeak
        // off this call's own `(log_mh, z) = (12.0, 0.5)` via
        // `vmpeak_at`, not off `ctx`'s root epoch.
        let log_v = m.vmpeak_at(12.0, 0.5, &ctx);
        assert!(mean <= m.log_sfr_star_forming(log_v, 0.5) + 1e-9);
    }

    /// `UniverseMachineGrowth::log_vmpeak` is a convenience query for "this
    /// object's vMpeak right now": `Vmax` at the epoch of peak historical
    /// mass. `GrowthTrack` is monotonic (mass strictly decreasing into
    /// the past), so the peak epoch is always the track's own first
    /// sample, `own_track.z[0]` / `own_track.log_mass[0]`. `log_vmpeak`
    /// must use exactly that. (`stellar_growth_rate` itself no longer
    /// calls this method — see `vmpeak_at`'s doc for why holding vMpeak
    /// fixed across a whole integration was a bug, not this method.)
    #[test]
    fn log_vmpeak_is_fixed_at_the_tracks_peak_epoch_not_recomputed_per_step() {
        let m = model();
        let c = Planck15::new();
        let t = track();
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let log_v = m.log_vmpeak(&ctx);

        // `own_track.log_mass` is h-free; `mpeak_to_vmax` expects
        // `Msun/h` (Correction: log_vmpeak's h-conversion fix). The
        // direct comparison must apply the same `+log10(h)` on the
        // right-hand side, or it is comparing the fixed implementation
        // against an old, wrong invocation.
        let log_h = c.h().log10();
        let expected = mpeak_to_vmax(t.log_mass[0] + log_h, t.z[0], &c, &DuttonMaccio14, MassDefinition::Vir)
            .log10();
        assert!((log_v - expected).abs() < 1e-12, "log_v={log_v} expected={expected}");

        // The wrong alternative pairs the peak MASS with a LATER step's
        // redshift; mpeak_to_vmax's NFW conversion is z-dependent (via
        // rho_crit(z), delta_vir(z), and concentration(z)), so this must
        // give a materially different answer.
        let wrong = mpeak_to_vmax(t.log_mass[0] + log_h, t.z[2], &c, &DuttonMaccio14, MassDefinition::Vir)
            .log10();
        assert!(
            (log_v - wrong).abs() > 1e-6,
            "log_vmpeak must depend on the peak epoch's z, not a later step's z"
        );
    }

    /// Final-review fix: the full composed path
    /// (`own_track -> log_vmpeak -> mpeak_to_vmax`), which no other test
    /// exercised. Task 11's fixture-agreement tests
    /// (`sfr_sf_grid.npy`/`quenched_fraction_grid.npy`) feed explicit
    /// `log_v` grid values straight into `log_sfr_star_forming`/
    /// `quenched_fraction`, bypassing `own_track` and `log_vmpeak`
    /// entirely, so they could not have caught the missing `+log10(h)`
    /// conversion between `own_track.log_mass` (h-free) and
    /// `mpeak_to_vmax` (expects `Msun/h`). Uses a Milky-Way-mass halo
    /// (`log_mh ~ 12.0`, h-free, matching `own_track`'s convention --
    /// same as `track()` above), analogous to Task 6's
    /// `vmax_is_physically_plausible_for_a_milky_way_halo` in
    /// `harmonise.rs`.
    #[test]
    fn log_vmpeak_composed_path_gives_a_physically_plausible_milky_way_vmax() {
        let m = model();
        let c = Planck15::new();
        let t = track(); // log_mass[0] = 12.0 [h-free Msun], z[0] = 0.0
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);

        let log_v = m.log_vmpeak(&ctx);
        let v = 10f64.powf(log_v);

        // A Milky-Way-mass halo should have Vmax of order 100-300 km/s
        // (the same wide sanity bound harmonise.rs's
        // vmax_is_physically_plausible_for_a_milky_way_halo test uses
        // for mpeak_to_vmax directly) -- catches a unit error of either
        // sign, not just this specific one.
        assert!((100.0..300.0).contains(&v), "Vmax = {v} km/s for a MW-mass halo");

        // Cross-check against an independently hand-derived expected
        // value that explicitly encodes the +log10(h) correction, to
        // prove this SPECIFIC fix is present in the composed path -- not
        // just "some plausible-looking number".
        let log_h = c.h().log10();
        let expected_v =
            mpeak_to_vmax(t.log_mass[0] + log_h, t.z[0], &c, &DuttonMaccio14, MassDefinition::Vir);
        assert!((v - expected_v).abs() < 1e-9, "v={v} expected_v={expected_v}");

        // And confirm the pre-fix (buggy) invocation -- feeding the
        // h-free mass straight into mpeak_to_vmax with no h conversion
        // -- would have given a measurably higher answer with
        // approximately the reported ~0.05 dex bias, so this test
        // actually discriminates the fix rather than passing either way.
        let old_wrong_v = mpeak_to_vmax(t.log_mass[0], t.z[0], &c, &DuttonMaccio14, MassDefinition::Vir);
        assert!(old_wrong_v > v, "pre-fix invocation should overestimate Vmax: old={old_wrong_v} fixed={v}");
        let bias_dex = old_wrong_v.log10() - v.log10();
        assert!(
            (0.03..0.08).contains(&bias_dex),
            "expected bias of omitting +log10(h) to be ~0.05 dex, got {bias_dex}"
        );
    }

    /// UM's SFR encodes quenching, so its descriptor must claim the
    /// Quenching capability — that is what makes the validator reject
    /// pairing it with STEEL's QuenchingModel.
    #[test]
    fn descriptor_claims_quenching_to_prevent_double_counting() {
        let d = model().descriptor();
        assert!(d.provides.contains(&Capability::StellarMass));
        assert!(
            d.provides.contains(&Capability::Quenching),
            "UM's bimodal SFR PDF contains quenching; not declaring it would let a run \
             silently quench twice"
        );
    }

    /// The stochastic branch genuinely draws a mode (star-forming vs
    /// quenched) *and* applies log-normal scatter within the
    /// star-forming mode, so `Capability::Scatter` is honestly earned
    /// (Correction 7) — not copied blindly from the brief.
    #[test]
    fn rng_draws_actually_vary_across_seeds() {
        let m = model();
        let c = Planck15::new();
        let t = track();
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let draws: Vec<f64> = (0..20_u64)
            .map(|seed| {
                let mut r = StdRng::seed_from_u64(seed);
                m.stellar_growth_rate(12.0, 0.5, &ctx, Some(&mut r))
            })
            .collect();
        let distinct = draws.iter().any(|&d| (d - draws[0]).abs() > 1e-9);
        assert!(distinct, "draws across 20 seeds were all identical: {draws:?}");
    }

    /// Regression test for the vMpeak-retroactive-quenching bug: holding
    /// vMpeak fixed at an object's root/observed-epoch mass across its
    /// whole integration stamped a massive halo's near-total final
    /// quenched fraction onto every earlier, unquenched progenitor,
    /// collapsing its integrated M* enough to make M*(Mh) non-monotonic.
    /// log_mh0 = 13.0 and 14.2 (h-free, z0 = 0.1) bracket the dip
    /// (peak ~13.2, trough ~14.5) observed via
    /// `steel-plugins/examples/dump_um_pure.rs` before this fix.
    #[test]
    fn integrated_stellar_mass_does_not_collapse_for_a_more_massive_halo() {
        use steel_core::halo_growth::HaloGrowthModel;
        use steel_core::stellar_growth::integrate_stellar_mass;

        let m = model();
        let c = Planck15::new();
        let growth = crate::halo_growth::VandenBosch14::new(&c);
        let z0 = 0.1;
        let track_a = growth.growth_history(13.0, z0);
        let track_b = growth.growth_history(14.2, z0);
        let ctx_a = AccretionContext::central(&track_a, &c, MassDefinition::Vir);
        let ctx_b = AccretionContext::central(&track_b, &c, MassDefinition::Vir);
        let sm_a = integrate_stellar_mass(&m, &ctx_a, z0, None);
        let sm_b = integrate_stellar_mass(&m, &ctx_b, z0, None);
        assert!(
            sm_b >= sm_a,
            "a more massive halo must not end up with less stellar mass: \
             log_mh0=13.0 -> log_sm={sm_a}, log_mh0=14.2 -> log_sm={sm_b}"
        );
    }
}
