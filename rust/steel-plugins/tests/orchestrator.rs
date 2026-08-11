//! Smoke test for `Simulation::run` (the `STEEL.py::OneRealization`
//! port) using a deliberately small grid — the full STEEL.py-equivalent
//! resolution (`log_m_bin=0.1` over `[11, 16.6]`) takes ~90s even in
//! release mode, too slow to run on every `cargo test`. This exercises
//! the same code path at a size that finishes in well under a second,
//! checking the output is a sane, physically shaped stellar mass
//! function rather than zeros or NaNs.

use std::sync::Arc;

use steel_core::context::{ModelContext, RunConfig, Simulation};
use steel_core::baryonic::BaryonicPipeline;
use steel_plugins::{
    Cattaneo11, Despali16, DoublePowerLawSfr, Jiang16, McCavanaBK08, MosterFormSmhm, Planck15,
    StewartScaling, VandenBosch14, Wetzel13,
};

fn build_small_simulation() -> Simulation {
    let cosmology = Planck15::new();
    let halo_growth = Arc::new(VandenBosch14::new(&cosmology));
    let hmf = Arc::new(Despali16::new(&cosmology));
    let shmf = Arc::new(Jiang16::default_calibration());
    let merger_time = Arc::new(McCavanaBK08::default());
    let baryonic = BaryonicPipeline::new(
        Box::new(DoublePowerLawSfr),
        Box::new(Wetzel13::new()),
        Box::new(StewartScaling::from_cosmology(&cosmology)),
        Box::new(Cattaneo11),
    );
    let smhm = Arc::new(MosterFormSmhm::g19_se(true));

    Simulation {
        context: ModelContext { cosmology: Arc::new(cosmology), rng_seed: 1 },
        halo_growth,
        hmf,
        shmf,
        merger_time,
        halo_stripping: None,
        smhm,
        baryonic,
    }
}

fn small_config() -> RunConfig {
    RunConfig {
        log_m_min: 12.5,
        log_m_max: 14.5,
        log_m_bin: 0.5,
        sat_min_offset: -1.0,
        z_reference_min: 0.1,
        star_formation: false,
        stellar_stripping: false,
        n_realizations: 3,
        sat_sm_min: 9.0,
        sat_sm_max: 12.0,
        sat_sm_bin: 0.25,
    }
}

#[test]
fn produces_a_well_formed_satellite_smf() {
    let sim = build_small_simulation();
    let output = sim.run(&small_config());

    assert_eq!(output.sat_sm_range.len(), output.surviving_sat_smf.len());
    assert!(!output.z.is_empty());
    assert!(!output.host_halo_mass.is_empty());

    assert!(output.surviving_sat_smf.iter().all(|v| v.is_finite() && *v >= 0.0));
    assert!(output.surviving_sat_smf.iter().any(|&v| v > 0.0), "SMF should have at least some mass");
}

#[test]
fn satellite_smf_declines_toward_high_mass() {
    // A Schechter-like satellite SMF should have (much) more low-mass
    // satellites than high-mass ones.
    let sim = build_small_simulation();
    let output = sim.run(&small_config());

    let low = output.surviving_sat_smf[0];
    let high = *output.surviving_sat_smf.last().unwrap();
    assert!(low > high, "low-mass bin ({low}) should exceed high-mass bin ({high})");
}

#[test]
fn enabling_star_formation_runs_without_panicking() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.star_formation = true;
    let output = sim.run(&config);
    assert!(output.surviving_sat_smf.iter().all(|v| v.is_finite()));
}

#[test]
fn enabling_stripping_runs_without_panicking() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.stellar_stripping = true;
    let output = sim.run(&config);
    assert!(output.surviving_sat_smf.iter().all(|v| v.is_finite()));
}
