//! The composed per-timestep baryonic evolution pipeline.
//!
//! `Functions_c.pyx::Starformation_c` calls star formation, quenching,
//! the gas-mass cap, and stellar stripping together, every timestep, for
//! every satellite: the stripping factor feeds the gas cap and the SFR,
//! and the quench state gates which SFR branch applies. Injecting those
//! four as independent top-level plugins would force the orchestrator to
//! either re-implement that coupling itself (defeating the plugin
//! boundary) or pay four `dyn` dispatches per timestep for no
//! swappability benefit. `BaryonicPipeline` owns the coupling and is
//! itself the single thing the orchestrator injects; each of its four
//! fields remains independently swappable when the pipeline is built.

use rand::RngCore;
use rand_distr::{Distribution, Normal};

use crate::gas::GasMassModel;
use crate::quenching::QuenchingModel;
use crate::sfr::SfrModel;
use crate::stripping::StellarStrippingModel;

/// Mass-loss-rate recycling constants (Moster+2018), shared by every
/// `BaryonicPipeline` instance — `Functions_c.pyx::Starformation_c`'s
/// `C0`/`Lambda`.
const RECYCLING_C0: f64 = 0.05;
/// In years (the Cython's `1.4*10^6`, i.e. 1.4 Myr).
const RECYCLING_LAMBDA_YR: f64 = 1.4e6;
/// dex scatter applied to the star-forming main sequence
/// (`Functions_c.pyx`'s `gsl_ran_gaussian(RNG_set, 0.3)`).
const SFR_SCATTER_DEX: f64 = 0.3;
/// sSFR floor below which a galaxy is treated as fully quenched
/// (`Functions_c.pyx`'s `10.0**-12`).
const SSFR_FLOOR: f64 = 1e-12;

/// A satellite at the moment of infall.
pub struct SatelliteState {
    /// Stellar mass at infall \[log10 Msun\].
    pub log_sm_infall: f64,
    /// Host halo mass at infall \[log10 Msun/h\].
    pub log_host_mass_infall: f64,
    /// Subhalo (satellite) mass at infall \[log10 Msun/h\].
    pub log_sat_mass_infall: f64,
    /// Redshift of infall.
    pub z_infall: f64,
    /// Whether this satellite was pre-processed (partially quenched)
    /// before infall (`Paramaters['PreProcessing']`). Which
    /// realizations of an N-realization ensemble get pre-quenched is an
    /// orchestrator-level decision (it needs the ensemble mean stellar
    /// mass); this field just carries that decision into the pipeline.
    pub pre_quenched: bool,
}

/// The time track a satellite is evolved along, from infall to its
/// return redshift (merger, or z=0 if it never merges).
///
/// `t` is the **age of the universe** (Gyr), increasing with index —
/// index 0 is infall (earliest time / highest z in this track), later
/// indices approach the return redshift. This is the opposite time
/// sense from `Functions.py::StarFormation`'s internal `t` (lookback
/// time, decreasing with index) — deliberately: an increasing "cosmic
/// time" is the more standard, less error-prone convention, and every
/// formula below that depends on time direction (the quench-time
/// comparison, the stripping-factor time fraction, the exponential
/// fade) has been re-derived for it rather than ported with the
/// Python's sign flipped in three different places.
pub struct Timeline {
    /// Redshift at each step, decreasing from `z_infall` to the return
    /// redshift.
    pub z: Vec<f64>,
    /// Age of the universe at each step \[Gyr\], increasing with index.
    pub t: Vec<f64>,
    /// Per-step time interval \[Gyr\] (`t[i+1] - t[i]`, same length as
    /// `t`, last entry repeats the previous interval).
    pub dt: Vec<f64>,
    /// Host halo mass \[log10 Msun/h\] at each step. Not currently
    /// consumed by `BaryonicPipeline::evolve` (the Python's
    /// `Starformation_c` hot loop doesn't take a host-mass track either
    /// — only the fixed infall-time host mass matters, via
    /// `SatelliteState::log_host_mass_infall`) but kept for future
    /// physics that might need it and for orchestrator bookkeeping.
    pub log_host_mass: Vec<f64>,
    /// Dynamical-friction infall/merger timescale for this satellite
    /// \[Gyr\].
    pub t_dyn_friction: f64,
}

/// Per-timestep evolution output for one satellite.
pub struct EvolutionHistory {
    /// Stellar mass \[log10 Msun\] at each step of the timeline.
    pub log_sm: Vec<f64>,
    /// Specific star formation rate \[log10 yr^-1\] at each step.
    pub log_ssfr: Vec<f64>,
}

/// The composed baryonic evolution pipeline: SFR, quenching, gas supply,
/// and stellar stripping, wired together.
pub struct BaryonicPipeline {
    pub sfr: Box<dyn SfrModel>,
    pub quenching: Box<dyn QuenchingModel>,
    pub gas: Box<dyn GasMassModel>,
    pub stripping: Box<dyn StellarStrippingModel>,
}

impl BaryonicPipeline {
    pub fn new(
        sfr: Box<dyn SfrModel>,
        quenching: Box<dyn QuenchingModel>,
        gas: Box<dyn GasMassModel>,
        stripping: Box<dyn StellarStrippingModel>,
    ) -> Self {
        Self {
            sfr,
            quenching,
            gas,
            stripping,
        }
    }

    /// Apply the gas-depletion cap and sSFR floor
    /// (`Functions_c.pyx::Starformation_c`'s shared logic between its
    /// star-forming and quenched branches — factored into one place
    /// here instead of duplicated).
    ///
    /// `ssfr_floor_inclusive`: the Cython uses `sSFR < floor` in the
    /// star-forming branch but `sSFR <= floor` in the quenched branch —
    /// a real (if practically inconsequential) difference, preserved
    /// for fidelity.
    #[allow(clippy::too_many_arguments)]
    fn apply_sfr_caps(
        sfr: f64,
        log_sm_i: f64,
        log_sm_0: f64,
        strip_factor_i: f64,
        apply_stripping: bool,
        max_gas: f64,
        ssfr_floor_inclusive: bool,
    ) -> f64 {
        let sm_new = if apply_stripping {
            10f64.powf(log_sm_i) - 10f64.powf(log_sm_0 + strip_factor_i)
        } else {
            10f64.powf(log_sm_i) - 10f64.powf(log_sm_0)
        };

        let mut sfr = sfr;
        if sm_new > 0.0 && sm_new.log10() > max_gas {
            sfr = 10f64.powf(log_sm_i - 12.0);
        }

        let ssfr = sfr / 10f64.powf(log_sm_i);
        let below_floor = if ssfr_floor_inclusive { ssfr <= SSFR_FLOOR } else { ssfr < SSFR_FLOOR };
        if below_floor {
            sfr = 10f64.powf(log_sm_i - 12.0);
        }
        sfr
    }

    /// Evolve one satellite's stellar mass and sSFR along `timeline`,
    /// starting from `galaxy`. Port of `Functions_c.pyx::Starformation_c`.
    ///
    /// `scatter_on` mirrors the Cython's own `Scatter_On` parameter,
    /// which every real STEEL run leaves at its default `true` (the
    /// Python call sites never override it) — exposed here mainly so a
    /// noiseless, exactly-reproducible trajectory is available for
    /// testing/validation against the Python.
    pub fn evolve(
        &self,
        galaxy: &SatelliteState,
        timeline: &Timeline,
        apply_stripping: bool,
        scatter_on: bool,
        rng: &mut dyn RngCore,
    ) -> EvolutionHistory {
        let n = timeline.t.len();
        assert!(n >= 1, "Timeline must have at least one step");
        // Real asserts, not `debug_assert!`: these compile out in release,
        // which is the profile every actual STEEL run uses. All four
        // vectors are indexed below, so a mismatch must fail here with a
        // clear message rather than as an opaque out-of-bounds panic.
        assert_eq!(timeline.z.len(), n, "Timeline.z length must match Timeline.t");
        assert_eq!(timeline.dt.len(), n, "Timeline.dt length must match Timeline.t");
        assert_eq!(
            timeline.log_host_mass.len(),
            n,
            "Timeline.log_host_mass length must match Timeline.t"
        );

        let quench = self.quenching.timescales(
            galaxy.log_sm_infall,
            galaxy.z_infall,
            galaxy.log_host_mass_infall,
            timeline.t[0],
            galaxy.pre_quenched,
        );

        // Gas ceiling, set once at infall from the noiseless infall SFR
        // and the satellite's own (subhalo) mass at infall.
        let sfr_at_infall = self.sfr.log_sfr(galaxy.log_sm_infall, galaxy.z_infall);
        let max_gas = self.gas.gas_mass(
            sfr_at_infall,
            galaxy.log_sat_mass_infall,
            if scatter_on { Some(&mut *rng) } else { None },
        );

        // Stripping factor track: both masses fixed at their infall
        // values (matching `StellarMassLoss(AvaHaloMass[i,j],
        // SatHaloMass[k], ...)` in STEEL.py, which passes scalars, not
        // a track), only `time_fraction` varies per step.
        let strip_factor: Vec<f64> = (0..n)
            .map(|i| {
                if !apply_stripping {
                    return 0.0;
                }
                // Clamped to [0,1]: that's the contract
                // `StellarStrippingModel::strip_factor` documents, and
                // `Cattaneo11` in particular takes log10 of
                // `strip + (1-strip)(1-time_fraction)`, which goes
                // non-positive (NaN) once `time_fraction > 1`. Reachable
                // whenever a timeline outlives its dynamical-friction
                // timescale.
                let time_fraction = if timeline.t_dyn_friction > 0.0 {
                    ((timeline.t[i] - timeline.t[0]) / timeline.t_dyn_friction).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                self.stripping.strip_factor(
                    galaxy.log_host_mass_infall,
                    galaxy.log_sat_mass_infall,
                    time_fraction,
                )
            })
            .collect();

        let mut log_sm = vec![0.0_f64; n];
        let mut sfh = vec![0.0_f64; n]; // Msun formed during step i
        let mut gmlr = vec![0.0_f64; n]; // Msun/yr, recycled mass-loss rate
        log_sm[0] = galaxy.log_sm_infall;

        let mut sfr_at_quench_onset = 0.0_f64;
        let normal = Normal::new(0.0, SFR_SCATTER_DEX).unwrap();

        for i in 0..n {
            let mut sfr = if quench.t_quench < timeline.t[i] && i != 0 {
                // Quenched: exponential fade from the SFR at the moment
                // quenching began.
                let faded = sfr_at_quench_onset * (-((timeline.t[i] - quench.t_quench) / quench.tau_fade)).exp();
                Self::apply_sfr_caps(faded, log_sm[i], log_sm[0], strip_factor[i], apply_stripping, max_gas, true)
            } else {
                // Star-forming main sequence.
                let sf = 10f64.powf(self.sfr.log_sfr(log_sm[i], timeline.z[i]));
                let sf = Self::apply_sfr_caps(sf, log_sm[i], log_sm[0], strip_factor[i], apply_stripping, max_gas, false);
                sfr_at_quench_onset = sf;
                sf
            };

            // Scatter around the main sequence / fade track.
            if scatter_on {
                sfr = 10f64.powf(sfr.log10() + normal.sample(rng));
            }

            sfh[i] = sfr * timeline.dt[i] * 1.0e9;

            // Mass-loss-rate recycling (Moster+2018): sum, over every
            // earlier star-formation episode j, the mass returned to
            // the ISM between step i and i+1.
            if i > 0 && i < n - 1 {
                for (j, &sfh_j) in sfh.iter().enumerate().take(i) {
                    let t_j = timeline.t[j];
                    let f_mr_1 = 1.0 - RECYCLING_C0 * (((t_j - timeline.t[i]).abs() * 1.0e9 / RECYCLING_LAMBDA_YR) + 1.0).ln();
                    let f_mr_2 = 1.0 - RECYCLING_C0 * (((t_j - timeline.t[i + 1]).abs() * 1.0e9 / RECYCLING_LAMBDA_YR) + 1.0).ln();
                    gmlr[i] += (sfh_j * (f_mr_1 - f_mr_2)).abs() / ((timeline.t[i] - timeline.t[i + 1]).abs() * 1.0e9);
                }
            }

            let m_dot = sfr - gmlr[i]; // Msun/yr
            if i < n - 1 {
                log_sm[i + 1] = if apply_stripping {
                    (10f64.powf(log_sm[i] + (strip_factor[i + 1] - strip_factor[i])) + m_dot * timeline.dt[i] * 1.0e9)
                        .log10()
                } else {
                    (10f64.powf(log_sm[i]) + m_dot * timeline.dt[i] * 1.0e9).log10()
                };
            }
        }

        // sSFR = (mass formed in step i) / (dt_i in yr * mass at start
        // of step i) — `Functions.py::StarFormation`'s post-hoc
        // sSFR = SFH / (d_t * 1e9 * M_out).
        let log_ssfr: Vec<f64> = (0..n)
            .map(|i| (sfh[i] / (timeline.dt[i] * 1.0e9 * 10f64.powf(log_sm[i]))).log10())
            .collect();

        EvolutionHistory { log_sm, log_ssfr }
    }
}
