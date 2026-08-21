//! `falsification_sweep`, but with satellite stellar masses assigned by a
//! SMHM relation *independent* of the one being tested as the central's
//! fiducial curve.
//!
//! `falsification_sweep` and `dump_central_assembly` both use the same
//! `MosterFormSmhm::moster13` instance to (a) assign infalling satellites
//! their stellar mass and (b) evaluate the fiducial SMHM curve the
//! assembled/accreted mass is compared against. That's a mild circularity
//! risk in the self-consistency argument (`research_notes` 2026-08-21,
//! "same caveats... compound here"): if Moster+13 systematically over- or
//! under-predicts stellar mass at a given halo mass, that bias shows up on
//! *both* sides of the comparison and could hide or manufacture tension
//! that isn't really there.
//!
//! This binary breaks that link: satellites are assigned mass by
//! [`RodriguezPuebla17`] (a Behroozi10-form fit, functionally independent
//! of Moster's double power law), while the fiducial comparison curve
//! stays `MosterFormSmhm::moster13` -- the curve actually being tested.
//! Output columns match `falsification_sweep` exactly, so the same
//! downstream plotting scripts (`plot_icl_stripping_bound.py`,
//! `plot_smhm_steepness_bound.py`) work unchanged by pointing them at this
//! binary's CSV instead.
//!
//! Usage: `cargo run --release -p steel-postprocess --example
//! falsification_sweep_indep_sat -- [log_m_min] [log_m_max] [log_m_bin]`

use std::sync::Arc;

use steel_core::accretion::AccretionContext;
use steel_core::baryonic::BaryonicPipeline;
use steel_core::context::{ModelContext, RunConfig, Simulation};
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::{
    Cattaneo11, Despali16, DoublePowerLawSfr, Jiang16, McCavanaBK08, MosterFormSmhm, Planck15,
    RodriguezPuebla17, ScaledStripping, StewartScaling, VandenBosch14, Wetzel13,
};
use steel_postprocess::merged_mass_per_central;

struct Row {
    log_mh_perh: f64,
    log_sm_smhm: f64,
    accreted: f64,
    icl: f64,
}

fn run_point(
    satellite_star_formation: bool,
    strength: f64,
    log_m_min: f64,
    log_m_max: f64,
    log_m_bin: f64,
) -> Vec<Row> {
    let cosmology = Planck15::new();
    let log_h = cosmology.h().log10();

    // Satellite masses: independent form (Behroozi10-style), NOT the
    // curve being tested.
    let smhm_satellite = Arc::new(RodriguezPuebla17);
    // Fiducial comparison curve: the one the self-consistency argument is
    // actually about.
    let smhm_central = Arc::new(MosterFormSmhm::moster13(true));

    let sim = Simulation {
        context: ModelContext { cosmology: Arc::new(Planck15::new()), rng_seed: 1 },
        halo_growth: Arc::new(VandenBosch14::new(&cosmology)),
        hmf: Arc::new(Despali16::new(&cosmology)),
        shmf: Arc::new(Jiang16::default_calibration()),
        merger_time: Arc::new(McCavanaBK08::default()),
        halo_stripping: None,
        smhm: smhm_satellite,
        baryonic: BaryonicPipeline::new(
            Box::new(DoublePowerLawSfr::satellite()),
            Box::new(Wetzel13::new()),
            Box::new(StewartScaling::from_cosmology(&cosmology)),
            Box::new(ScaledStripping::new(Cattaneo11, strength)),
        ),
    };

    let config = RunConfig {
        log_m_min,
        log_m_max,
        log_m_bin,
        star_formation: satellite_star_formation,
        stellar_stripping: true,
        ..RunConfig::default()
    };

    let out = sim.run(&config);
    let merged =
        merged_mass_per_central(out.accretion_history.view(), &out.sat_sm_range, config.sat_sm_bin);

    let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![12.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmology, MassDefinition::Vir);

    (0..out.host_halo_mass.ncols())
        .map(|j| {
            let log_mh_perh = out.host_halo_mass[[0, j]];
            let log_sm_smhm = smhm_central.stellar_mass(log_mh_perh - log_h, out.z[0], &ctx, None);
            Row {
                log_mh_perh,
                log_sm_smhm,
                accreted: merged.column(j).sum(),
                icl: if out.icl_stripped_mass.is_empty() {
                    0.0
                } else {
                    out.icl_stripped_mass.column(j).sum()
                },
            }
        })
        .collect()
}

fn main() {
    let mut args = std::env::args().skip(1);
    let log_m_min: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(12.0);
    let log_m_max: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(15.0);
    let log_m_bin: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(0.4);

    let strengths = [0.0_f64, 1.0, 2.0, 3.0, 4.0];

    println!("satellite_sf,strength,log_mh_perh,log_sm_smhm,log_accreted,ratio,log_icl,f_icl");

    for &sf in &[true, false] {
        for &strength in &strengths {
            let rows = run_point(sf, strength, log_m_min, log_m_max, log_m_bin);
            for r in &rows {
                let sm_smhm = 10f64.powf(r.log_sm_smhm);
                let ratio = r.accreted / sm_smhm;
                let f_icl = if r.icl + sm_smhm > 0.0 { r.icl / (r.icl + sm_smhm) } else { 0.0 };
                let log10_or_nan = |v: f64| if v > 0.0 { v.log10() } else { f64::NAN };

                println!(
                    "{sf},{strength},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
                    r.log_mh_perh,
                    r.log_sm_smhm,
                    log10_or_nan(r.accreted),
                    ratio,
                    log10_or_nan(r.icl),
                    f_icl,
                );
            }
        }
    }
}
