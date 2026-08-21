//! Closes STEEL's loop end to end: runs the model, feeds each host
//! bin's merging-satellite stellar mass into that host's own central as
//! an ex-situ accretion rate, and reports the assembled central mass
//! against what the SMHM relation says the same halo should host.
//!
//! This is the comparison `research_notes` calls the core of the
//! self-consistency argument. A central grows two ways -- star
//! formation and accretion -- so if accretion alone already delivers
//! more than the empirical SMHM allows, something in the chain is
//! non-physical. Before `steel_postprocess::central_assembly` existed
//! the two halves were computed separately and compared by hand
//! outside the model; here they are one budget.
//!
//! Columns, one row per host halo bin:
//!
//! * `log_mh_perh`      -- host halo mass at the reference epoch [log10 Msun/h]
//! * `log_sm_smhm`      -- what the SMHM says that halo's central weighs
//! * `log_sm_insitu`    -- assembled from star formation alone (accretion zeroed)
//! * `log_sm_assembled` -- assembled from star formation *plus* merged satellites
//! * `log_accreted`     -- total ex-situ stellar mass delivered [log10 Msun]
//! * `log_icl`          -- total stellar mass stripped to the ICL [log10 Msun]
//! * `accreted_over_smhm` -- the falsification ratio: >1 means mergers alone
//!   already overfill the SMHM's budget for this halo
//!
//! Usage: `cargo run --release -p steel-postprocess --example
//! dump_central_assembly [log_m_min] [log_m_max] [log_m_bin]`
//! (defaults to the reduced validation grid, which finishes in seconds;
//! the full 0.1-dex grid takes minutes).

use std::sync::Arc;

use rand::SeedableRng;
use steel_core::accretion::AccretionContext;
use steel_core::baryonic::BaryonicPipeline;
use steel_core::context::{ModelContext, RunConfig, Simulation};
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::{
    Cattaneo11, Despali16, DoublePowerLawSfr, Jiang16, McCavanaBK08, MosterFormSmhm, Planck15,
    StewartScaling, VandenBosch14, Wetzel13,
};
use steel_postprocess::{accretion_rate_msun_per_yr, merged_mass_per_central, CentralEvolution};

fn main() {
    let mut args = std::env::args().skip(1);
    let log_m_min: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(11.0);
    let log_m_max: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(12.6);
    let log_m_bin: f64 = args.next().map(|s| s.parse().unwrap()).unwrap_or(0.5);

    let cosmology = Planck15::new();
    let log_h = cosmology.h().log10();
    let smhm = Arc::new(MosterFormSmhm::moster13(true));

    let sim = Simulation {
        context: ModelContext { cosmology: Arc::new(Planck15::new()), rng_seed: 1 },
        halo_growth: Arc::new(VandenBosch14::new(&cosmology)),
        hmf: Arc::new(Despali16::new(&cosmology)),
        shmf: Arc::new(Jiang16::default_calibration()),
        merger_time: Arc::new(McCavanaBK08::default()),
        halo_stripping: None,
        smhm: smhm.clone(),
        baryonic: BaryonicPipeline::new(
            Box::new(DoublePowerLawSfr::satellite()),
            Box::new(Wetzel13::new()),
            Box::new(StewartScaling::from_cosmology(&cosmology)),
            Box::new(Cattaneo11),
        ),
    };

    // Star formation and stripping both on: satellites must be allowed
    // to evolve for the merged mass to mean anything, and stripping is
    // what the ICL column measures.
    let config = RunConfig {
        log_m_min,
        log_m_max,
        log_m_bin,
        star_formation: true,
        stellar_stripping: true,
        ..RunConfig::default()
    };

    let out = sim.run(&config);
    let n_z = out.z.len();
    let n_host = out.host_halo_mass.ncols();

    let merged = merged_mass_per_central(out.accretion_history.view(), &out.sat_sm_range, config.sat_sm_bin);

    // `out.z` runs low -> high (increasing into the past).
    // `CentralEvolution::evolve` wants time increasing, i.e. z
    // decreasing, so every per-step vector below is reversed.
    let z_rev: Vec<f64> = out.z.iter().rev().copied().collect();
    let t_rev: Vec<f64> = z_rev.iter().map(|&zi| cosmology.age(zi)).collect();
    let mut dt_rev: Vec<f64> = t_rev.windows(2).map(|w| w[1] - w[0]).collect();
    dt_rev.push(*dt_rev.last().unwrap());

    let central = CentralEvolution::new(Box::new(DoublePowerLawSfr::central()));
    let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![12.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmology, MassDefinition::Vir);

    println!(
        "log_mh_perh,log_sm_smhm,log_sm_insitu,log_sm_assembled,log_accreted,log_icl,accreted_over_smhm"
    );
    for j in 0..n_host {
        // Merged mass for this host bin, reversed onto the time-increasing axis.
        let merged_rev: Vec<f64> = (0..n_z).rev().map(|i| merged[[i, j]]).collect();
        let acc_rate = accretion_rate_msun_per_yr(&merged_rev, &dt_rev);
        let zeros = vec![0.0; n_z];

        // The central starts at the SMHM mass of its own halo at the
        // earliest epoch on the track, then grows from there.
        let log_mh_first = out.host_halo_mass[[n_z - 1, j]] - log_h; // h-free, oldest epoch
        let log_sm_start = smhm.stellar_mass(log_mh_first, z_rev[0], &ctx, None);

        // Never quenched (t_quench below the track's first age), so the
        // in-situ leg is the maximal star-formation case -- the
        // conservative choice when asking whether accretion *alone*
        // already overfills the budget.
        let t_quench = t_rev[0] - 1.0;

        let mut rng_a = rand::rngs::StdRng::seed_from_u64(1);
        let assembled = central.evolve(
            log_sm_start, &z_rev, &t_rev, &dt_rev, &acc_rate, t_quench, false, &ctx, &mut rng_a,
        );
        let mut rng_b = rand::rngs::StdRng::seed_from_u64(1);
        let in_situ = central.evolve(
            log_sm_start, &z_rev, &t_rev, &dt_rev, &zeros, t_quench, false, &ctx, &mut rng_b,
        );

        // SMHM's own answer for this halo at the reference epoch.
        let log_mh_ref = out.host_halo_mass[[0, j]] - log_h;
        let log_sm_smhm = smhm.stellar_mass(log_mh_ref, out.z[0], &ctx, None);

        let total_accreted: f64 = merged_rev.iter().sum();
        let total_icl: f64 = if out.icl_stripped_mass.is_empty() {
            0.0
        } else {
            out.icl_stripped_mass.column(j).sum()
        };

        let log10_or_nan = |v: f64| if v > 0.0 { v.log10() } else { f64::NAN };
        let ratio = total_accreted / 10f64.powf(log_sm_smhm);

        println!(
            "{:.4},{:.4},{:.4},{:.4},{:.4},{:.4},{:.4}",
            out.host_halo_mass[[0, j]],
            log_sm_smhm,
            in_situ.log_sm.last().unwrap(),
            assembled.log_sm.last().unwrap(),
            log10_or_nan(total_accreted),
            log10_or_nan(total_icl),
            ratio,
        );
    }
}
