//! Smoke test for `Simulation::run` (the `STEEL.py::OneRealization`
//! port) using a deliberately small grid — the full STEEL.py-equivalent
//! resolution (`log_m_bin=0.1` over `[11, 16.6]`) takes ~90s even in
//! release mode, too slow to run on every `cargo test`. This exercises
//! the same code path at a size that finishes in well under a second,
//! checking the output is a sane, physically shaped stellar mass
//! function rather than zeros or NaNs.

use std::sync::Arc;

use steel_core::context::{ModelContext, OutputSelection, RunConfig, Simulation};
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
        Box::new(DoublePowerLawSfr::satellite()),
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
        ..RunConfig::default()
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

/// Stripped stellar mass has to go somewhere. Before the ICL
/// accumulator existed, `BaryonicPipeline::evolve` reduced a satellite's
/// mass and simply dropped the difference, so the model could not
/// report an intracluster-light mass or close the budget
/// `accreted = retained + stripped`.
#[test]
fn stripping_banks_intracluster_light_that_grows_with_host_mass() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.stellar_stripping = true;
    let output = sim.run(&config);

    assert!(!output.icl_stripped_mass.is_empty(), "stripping on must populate the ICL array");
    assert!(
        output.icl_stripped_mass.iter().all(|v| v.is_finite() && *v >= 0.0),
        "ICL mass must be finite and non-negative"
    );

    // Total ICL per central, by host bin. More massive hosts strip more
    // -- they accrete more satellites and strip each one harder -- so
    // this must rise monotonically across the (ascending) host grid.
    let totals: Vec<f64> = output.icl_stripped_mass.columns().into_iter().map(|c| c.sum()).collect();
    assert!(totals.iter().any(|&t| t > 0.0), "no ICL banked at all: {totals:?}");
    for w in totals.windows(2) {
        assert!(w[1] >= w[0], "ICL must not fall with increasing host mass: {totals:?}");
    }
}

/// The counterpart: with stripping off nothing is stripped, so the
/// family stays unallocated rather than silently banking zeros (or,
/// worse, the unstripped baseline).
#[test]
fn no_stripping_leaves_the_icl_array_empty() {
    let sim = build_small_simulation();
    let config = small_config(); // stellar_stripping defaults to false here
    let output = sim.run(&config);
    assert!(output.icl_stripped_mass.is_empty());
}

/// The merger code path did not exist before Phase 1: `Simulation::run`
/// `continue`d past every satellite with `tdyf < ttz0`, so these two
/// arrays were structurally guaranteed to be all-zero. Papers 2 and 3
/// are built entirely on them.
#[test]
fn merging_satellites_populate_the_accretion_history() {
    let sim = build_small_simulation();
    let output = sim.run(&small_config());

    let nonzero = output.accretion_history.iter().filter(|v| **v > 0.0).count();
    assert!(nonzero > 0, "Accretion_History is all zeros -- the merger path is not running");
    assert!(output.accretion_history.iter().all(|v| v.is_finite()));

    let halo_nonzero = output.accretion_history_halo.iter().filter(|v| **v > 0.0).count();
    assert!(halo_nonzero > 0, "Accretion_History_Halo is all zeros");
}

/// PORT-FIX 2: `STEEL.py:436` guards the whole pair-fraction block with
/// `if len(np.shape(SM_Sat)) == 1:` and has no `else`, so with star
/// formation or stripping on -- the configuration Papers 2 and 3 use --
/// the Python writes zeros. Both cases must produce data here.
#[test]
fn pair_fractions_are_populated_with_and_without_baryonic_evolution() {
    let sim = build_small_simulation();

    for (label, sf, stripping) in
        [("unevolved", false, false), ("star formation", true, false), ("stripping", false, true)]
    {
        let mut config = small_config();
        config.star_formation = sf;
        config.stellar_stripping = stripping;
        let output = sim.run(&config);

        let nonzero = output.pair_frac.iter().filter(|v| **v > 0.0).count();
        assert!(nonzero > 0, "Pair_Frac is all zeros with {label}");
        assert!(output.pair_frac.iter().all(|v| v.is_finite()), "Pair_Frac has non-finite entries with {label}");
        assert!(
            output.pair_frac_halo.iter().any(|&v| v > 0.0),
            "Pair_Frac_Halo is all zeros with {label}"
        );
    }
}

#[test]
fn every_enabled_output_family_is_populated_and_finite() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.star_formation = true;
    let output = sim.run(&config);

    let checks: [(&str, bool, bool); 8] = [
        (
            "surviving_sat_smf_highz",
            output.surviving_sat_smf_highz.iter().any(|&v| v > 0.0),
            output.surviving_sat_smf_highz.iter().all(|v| v.is_finite()),
        ),
        (
            "surviving_sat_smf_by_host_highz",
            output.surviving_sat_smf_by_host_highz.iter().any(|&v| v > 0.0),
            output.surviving_sat_smf_by_host_highz.iter().all(|v| v.is_finite()),
        ),
        (
            "sat_smhm",
            output.sat_smhm.iter().any(|&v| v > 0.0),
            output.sat_smhm.iter().all(|v| v.is_finite()),
        ),
        (
            "sat_smhm_host",
            output.sat_smhm_host.iter().any(|&v| v > 0.0),
            output.sat_smhm_host.iter().all(|v| v.is_finite()),
        ),
        (
            "satellite_ssfr",
            output.satellite_ssfr.iter().any(|&v| v > 0.0),
            output.satellite_ssfr.iter().all(|v| v.is_finite()),
        ),
        (
            "z_infall",
            output.z_infall.iter().any(|&v| v > 0.0),
            output.z_infall.iter().all(|v| v.is_finite()),
        ),
        (
            "cuts_nofrac",
            output.cuts_nofrac.iter().any(|&v| v > 0.0),
            output.cuts_nofrac.iter().all(|v| v.is_finite()),
        ),
        (
            "cuts_nofrac_highz",
            output.cuts_nofrac_highz.iter().any(|&v| v > 0.0),
            output.cuts_nofrac_highz.iter().all(|v| v.is_finite()),
        ),
    ];
    for (name, populated, finite) in checks {
        assert!(populated, "{name} is entirely zero");
        assert!(finite, "{name} has non-finite entries");
    }

    // `Total_StarFormation` means are NaN wherever no satellite merged
    // into that (z, host, mass) cell -- that is `np.mean([])`'s value and
    // is deliberate -- so only the populated cells must be finite.
    assert!(
        output.total_star_formation_mean.iter().any(|v| v.is_finite()),
        "Total_StarFormation means are entirely NaN"
    );
    assert!(
        output
            .total_star_formation_std
            .iter()
            .all(|v| v.is_nan() || (v.is_finite() && *v >= 0.0)),
        "Total_StarFormation std must be NaN or non-negative"
    );
}

/// Deterministic mode: with `scatter = false` every stochastic source
/// is off, so the run is a pure function of the grid. This is what the
/// three-way validation compares against the Python's `ScatterOn=False`
/// / `Scatter_On=0` path, and the claim only holds if *all* the sources
/// are switched -- abundance matching, the star-formation main
/// sequence, and the gas-mass relation.
#[test]
fn deterministic_mode_makes_realizations_identical() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.star_formation = true;
    config.stellar_stripping = true;
    config.scatter = false;
    config.n_realizations = 4;

    let a = sim.run(&config);

    // With no scatter the N realizations are the same galaxy, so the
    // result must not depend on how many of them there are.
    config.n_realizations = 1;
    let b = sim.run(&config);
    for (i, (x, y)) in a.surviving_sat_smf.iter().zip(b.surviving_sat_smf.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-9 * x.abs().max(1.0),
            "bin {i}: {x} vs {y} -- realizations still differ, so some scatter source is live"
        );
    }

    // And it must differ from the scattered run, i.e. the flag is doing
    // something rather than being ignored.
    config.n_realizations = 4;
    config.scatter = true;
    let scattered = sim.run(&config);
    assert_ne!(a.surviving_sat_smf, scattered.surviving_sat_smf);
}

/// `Paramaters['PreProcessing']` (the `_PP` run-tuple suffix) quenches
/// a mass-dependent prefix of each satellite's realization ensemble at
/// infall. Paper 2's cmodel and DPL suites both use it, so a run that
/// silently ignored the flag would produce a different result under the
/// same name.
#[test]
fn pre_processing_suppresses_satellite_growth() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.star_formation = true;
    // `PP_Frac` is quantised by `int(PP_Frac * n_realizations)`, exactly
    // as in the Python. `PP_Frac` is 0.3 for an ensemble mean above
    // log M* = 8, so the default 3 realizations of `small_config` give
    // `int(0.9) = 0` -- nothing is pre-quenched and the flag is a no-op.
    // 10 realizations puts at least 3 in the pre-quenched prefix.
    config.n_realizations = 10;

    let normal = sim.run(&config);
    config.pre_processing = true;
    let pre_processed = sim.run(&config);

    // Quenching part of the ensemble at infall means less stellar mass
    // is formed, so the merged-satellite star formation total must drop.
    let total = |o: &steel_core::RunOutput| {
        o.total_star_formation_mean.iter().filter(|v| v.is_finite()).sum::<f64>()
    };
    assert!(
        total(&pre_processed) < total(&normal),
        "pre-processing should reduce total star formation: {} vs {}",
        total(&pre_processed),
        total(&normal)
    );

    // ...and the satellite SMF must actually differ, i.e. the flag is
    // not being silently ignored.
    assert_ne!(normal.surviving_sat_smf, pre_processed.surviving_sat_smf);
}

/// The subhalo mass functions are only accumulated when nothing evolves
/// the subhalo, matching `STEEL.py:298`'s
/// `(Stripping_DM == False) and (Stripping or SF) == False` guard.
#[test]
fn subhalo_mass_functions_track_the_python_guard() {
    let sim = build_small_simulation();

    let unevolved = sim.run(&small_config());
    assert!(
        unevolved.surviving_subhalos.iter().any(|&v| v > 0.0),
        "SurvivingSubhalos should be populated with no stripping or SF"
    );
    assert!(unevolved.surviving_subhalos_by_parent.iter().any(|&v| v > 0.0));
    assert!(unevolved.surviving_subhalos_z_z.iter().any(|&v| v > 0.0));

    let mut config = small_config();
    config.star_formation = true;
    let evolved = sim.run(&config);
    // Not merely zero -- absent. The Python allocates and saves the
    // zero-filled arrays here, which reads downstream as "the unevolved
    // SHMF is flat zero" rather than "this run does not define one".
    assert!(
        evolved.surviving_subhalos.is_empty(),
        "the unevolved SHMF is not defined once star formation is on"
    );
    assert!(evolved.surviving_subhalos_by_parent.is_empty());
    assert!(evolved.surviving_subhalos_z_z.is_empty());
}

#[test]
fn disabling_an_output_family_leaves_its_arrays_empty() {
    let sim = build_small_simulation();
    let mut config = small_config();
    config.outputs = OutputSelection::smf_only();
    let output = sim.run(&config);

    assert!(output.accretion_history.is_empty());
    assert!(output.pair_frac.is_empty());
    assert!(output.surviving_subhalos_z_z.is_empty());
    assert!(output.sat_smhm.is_empty());
    assert!(output.satellite_ssfr.is_empty());
    // ...but the SMFs the grid search needs are still there.
    assert!(!output.surviving_sat_smf_highz.is_empty());
    assert!(output.surviving_sat_smf.iter().any(|&v| v > 0.0));
}

/// The same seed must reproduce the run exactly -- the reproducibility
/// claim the port is built on, and the thing `STEEL.py`'s per-call
/// `np.random.seed(...)` inside `DarkMatterToStellarMass` gives up.
#[test]
fn a_fixed_seed_reproduces_the_whole_run() {
    let mut config = small_config();
    config.star_formation = true;
    let a = build_small_simulation().run(&config);
    let b = build_small_simulation().run(&config);

    assert_eq!(a.surviving_sat_smf, b.surviving_sat_smf);
    assert_eq!(a.accretion_history, b.accretion_history);
    assert_eq!(a.pair_frac, b.pair_frac);
    assert_eq!(a.sat_smhm, b.sat_smhm);
}

/// A shorter dynamical friction time means satellites sink faster, so
/// more of them merge and fewer survive to the reference epoch. This is
/// the `f_tdyn` knob Paper 1 scans, and it only has an effect at all now
/// that both sides of the merge/survive branch do something.
#[test]
fn a_shorter_dynamical_friction_time_moves_satellites_from_survivors_to_mergers() {
    let build = |factor: f64| {
        let cosmology = Planck15::new();
        let mut sim = build_small_simulation();
        sim.merger_time = Arc::new(McCavanaBK08::new(factor, false));
        let _ = cosmology;
        sim
    };

    let fast = build(0.5).run(&small_config());
    let slow = build(2.5).run(&small_config());

    let survivors = |o: &steel_core::RunOutput| o.surviving_sat_smf.iter().sum::<f64>();
    let merged = |o: &steel_core::RunOutput| o.accretion_history.iter().sum::<f64>();

    assert!(
        survivors(&fast) < survivors(&slow),
        "f_tdyn=0.5 should leave fewer survivors than f_tdyn=2.5: {} vs {}",
        survivors(&fast),
        survivors(&slow)
    );
    assert!(
        merged(&fast) > merged(&slow),
        "f_tdyn=0.5 should produce more mergers than f_tdyn=2.5: {} vs {}",
        merged(&fast),
        merged(&slow)
    );
}

/// `[run].star_formation` must actually gate star formation in satellites.
///
/// `STEEL.py:386-392` branches three ways — `SF and Stripping`, `elif SF`,
/// `elif Stripping` — and that third branch calls `StellarMassLoss` alone,
/// with no `StarFormation` call at all. The Rust port originally passed
/// only `config.stellar_stripping` into `BaryonicPipeline::evolve` and
/// never plumbed `config.star_formation` through, so with stripping on the
/// pipeline ran and formed stars regardless of the flag. A 300-row sweep
/// over both settings came back byte-identical in every column.
///
/// Satellites that never form stars must deliver strictly less stellar
/// mass, so the merging-satellite mass function must differ and must be
/// lower overall.
#[test]
fn disabling_star_formation_reduces_delivered_satellite_mass() {
    let sim = build_small_simulation();

    let mut with_sf = small_config();
    with_sf.star_formation = true;
    with_sf.stellar_stripping = true;

    let mut without_sf = small_config();
    without_sf.star_formation = false;
    without_sf.stellar_stripping = true;

    let a = sim.run(&with_sf);
    let b = sim.run(&without_sf);

    let total_a: f64 = a.accretion_history.iter().sum();
    let total_b: f64 = b.accretion_history.iter().sum();

    assert!(total_a > 0.0, "test is vacuous if the SF-on run delivers nothing");
    assert!(
        total_b < total_a,
        "switching satellite star formation off must reduce delivered mass, \
         got {total_b} with SF off vs {total_a} with SF on"
    );
}

/// Guards against the two `bool` switches being transposed at the call
/// site. Both are adjacent arguments of the same type, so a swap compiles
/// silently; these two configurations are each other's mirror image and
/// must not produce the same answer.
#[test]
fn star_formation_and_stripping_switches_are_not_interchangeable() {
    let sim = build_small_simulation();

    let mut sf_only = small_config();
    sf_only.star_formation = true;
    sf_only.stellar_stripping = false;

    let mut strip_only = small_config();
    strip_only.star_formation = false;
    strip_only.stellar_stripping = true;

    let a: f64 = sim.run(&sf_only).accretion_history.iter().sum();
    let b: f64 = sim.run(&strip_only).accretion_history.iter().sum();

    assert!(
        (a - b).abs() > 1e-12 * a.abs().max(b.abs()).max(1.0),
        "SF-only and stripping-only must differ; got {a} and {b}"
    );
}
