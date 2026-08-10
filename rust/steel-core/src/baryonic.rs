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

use crate::gas::GasMassModel;
use crate::quenching::QuenchingModel;
use crate::sfr::SfrModel;
use crate::stripping::StellarStrippingModel;

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
}

/// The time/host-mass track a satellite is evolved along, from infall to
/// its return redshift (merger, or z=0 if it never merges).
pub struct Timeline {
    /// Redshift at each step, decreasing from `z_infall` to the return
    /// redshift.
    pub z: Vec<f64>,
    /// Cosmic time at each step \[Gyr\].
    pub t: Vec<f64>,
    /// Per-step time interval \[Gyr\].
    pub dt: Vec<f64>,
    /// Host halo mass \[log10 Msun/h\] at each step.
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

    /// Evolve one satellite's stellar mass and sSFR along `timeline`,
    /// starting from `galaxy`. Port of `Functions_c.pyx::Starformation_c`.
    ///
    /// Implemented in Milestone 4 (see project plan); the signature is
    /// fixed now so the orchestrator (Milestone 5) can be written against
    /// it.
    pub fn evolve(
        &self,
        galaxy: &SatelliteState,
        timeline: &Timeline,
        apply_stripping: bool,
        rng: &mut dyn RngCore,
    ) -> EvolutionHistory {
        let _ = (galaxy, timeline, apply_stripping, rng);
        unimplemented!(
            "BaryonicPipeline::evolve is implemented in Milestone 4 \
             (port of Functions_c.pyx::Starformation_c)"
        )
    }
}
