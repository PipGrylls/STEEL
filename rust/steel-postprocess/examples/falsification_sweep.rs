//! The self-consistency falsification test, run to its limits.
//!
//! The argument (`research_notes`): a central galaxy grows two ways,
//! star formation and accretion. Stripping sets a *lower bound* on how
//! much stellar mass mergers actually deliver — strip harder and less
//! arrives. So if, even at implausibly hard stripping and with
//! satellite star formation switched off, the mass delivered by mergers
//! alone still exceeds what an empirical SMHM relation allows the
//! central to weigh, then no amount of tuning inside the model can
//! rescue it: the SMHM relation (or its redshift evolution) is not
//! physically attainable given the halo accretion history.
//!
//! This sweeps the two knobs that minimise delivered mass —
//! `star_formation` off, and [`ScaledStripping`] strength — and reports
//! for every host halo mass:
//!
//! * `ratio` — accreted stellar mass / SMHM stellar mass. `> 1` is the
//!   falsification condition: mergers alone overfill the budget.
//! * `f_icl` — implied ICL fraction, `M_ICL / (M_ICL + M*_SMHM)`. This
//!   is what stops "strip harder" from being a free escape: harder
//!   stripping lowers `ratio` but raises `f_icl`, and observed ICL
//!   fractions bound how far that can go. (See the ICL literature
//!   review TODO in `research_notes` — until those numbers are in, this
//!   column is a prediction, not yet a constraint.)
//!
//! Note `strength = 1` is the *published* baseline, not raw Cattaneo
//! et al. (2011): `Cattaneo11` already carries the `PipGrylls`-branch
//! doubling Papers 2 and 3 were run with. See [`ScaledStripping`].
//!
//! Usage: `cargo run --release -p steel-postprocess --example
//! falsification_sweep [log_m_min] [log_m_max] [log_m_bin]`

use std::sync::Arc;

use steel_core::accretion::AccretionContext;
use steel_core::baryonic::BaryonicPipeline;
use steel_core::context::{ModelContext, RunConfig, Simulation};
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::{
    Cattaneo11, Despali16, DoublePowerLawSfr, Jiang16, McCavanaBK08, MosterFormSmhm, Planck15,
    ScaledStripping, StewartScaling, VandenBosch14, Wetzel13,
};
use steel_postprocess::merged_mass_per_central;

/// One (host mass) row of one sweep point.
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
            Box::new(ScaledStripping::new(Cattaneo11, strength)),
        ),
    };

    // `stellar_stripping` stays on even at strength 0 -- the strength
    // scale, not the switch, is what turns stripping off, so the ICL
    // accumulator stays allocated and reports its (zero) total rather
    // than vanishing.
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
            let log_sm_smhm = smhm.stellar_mass(log_mh_perh - log_h, out.z[0], &ctx, None);
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

    // Track, per host bin, the weakest stripping that already brings the
    // delivered mass inside the SMHM budget -- with satellite star
    // formation off, i.e. the most forgiving case for the SMHM.
    let mut min_strength_ok: Vec<Option<f64>> = Vec::new();
    let mut host_masses: Vec<f64> = Vec::new();

    for &sf in &[true, false] {
        for &strength in &strengths {
            let rows = run_point(sf, strength, log_m_min, log_m_max, log_m_bin);

            if !sf && min_strength_ok.is_empty() {
                min_strength_ok = vec![None; rows.len()];
                host_masses = rows.iter().map(|r| r.log_mh_perh).collect();
            }

            for (j, r) in rows.iter().enumerate() {
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

                if !sf && ratio <= 1.0 && min_strength_ok[j].is_none() {
                    min_strength_ok[j] = Some(strength);
                }
            }
        }
    }

    // Summary to stderr so the CSV on stdout stays machine-readable.
    eprintln!("\n=== weakest stripping that keeps accreted mass within the SMHM budget ===");
    eprintln!("(satellite star formation OFF -- the most forgiving case for the SMHM)");
    eprintln!("{:>12}  {:>8}", "log_mh[Msun/h]", "strength");
    for (j, &mh) in host_masses.iter().enumerate() {
        match min_strength_ok[j] {
            Some(s) => eprintln!("{mh:>12.2}  {s:>8.1}"),
            None => eprintln!(
                "{mh:>12.2}  {:>8}  <-- never satisfied up to strength {}",
                "NONE",
                strengths.last().unwrap()
            ),
        }
    }
}
