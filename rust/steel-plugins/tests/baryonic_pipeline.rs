//! Regression test locking in `BaryonicPipeline::evolve`'s validated
//! output: values here were cross-checked against a faithful Python
//! translation of `Functions_c.pyx::Starformation_c` (both the
//! unstripped and stripped, `Scatter_On=0` paths) and matched to
//! floating-point noise (~1e-11). See the Milestone 4 PR description
//! for the full comparison. This test exists so that match doesn't
//! silently regress.

use rand::rngs::StdRng;
use rand::SeedableRng;

use steel_core::baryonic::{BaryonicPipeline, SatelliteState, Timeline};
use steel_core::cosmology::Cosmology;
use steel_plugins::{Cattaneo11, Planck15, StewartScaling, TomczakFormSfr, Wetzel13};

fn build_timeline(cosmo: &Planck15) -> Timeline {
    let z: Vec<f64> = (0..11).map(|i| 1.0 - i as f64 * 0.05).collect();
    let t: Vec<f64> = z.iter().map(|&zi| cosmo.age(zi)).collect();
    let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    dt.push(*dt.last().unwrap());
    // `t_dyn_friction` must exceed the timeline's own span (2.74 Gyr
    // here), otherwise the fixture describes a satellite still infalling
    // past its own merger timescale — which the real pipeline never
    // produces (for mergers the window *is* the dynamical friction time;
    // for survivors it is shorter) and which the Python reference has no
    // notion of, since `StellarMassLoss`'s `Factor` is <= 1 by
    // construction there.
    Timeline { z, t, dt, log_host_mass: vec![13.0; 11], t_dyn_friction: 3.0 }
}

fn build_pipeline(cosmo: &Planck15) -> BaryonicPipeline {
    BaryonicPipeline::new(
        Box::new(TomczakFormSfr::ce()),
        Box::new(Wetzel13::new()),
        Box::new(StewartScaling::from_cosmology(cosmo)),
        Box::new(Cattaneo11),
    )
}

fn build_galaxy(z0: f64) -> SatelliteState {
    SatelliteState {
        log_sm_infall: 10.0,
        log_host_mass_infall: 13.0,
        log_sat_mass_infall: 11.5,
        z_infall: z0,
        pre_quenched: false,
    }
}

#[test]
fn unstripped_noiseless_trajectory_matches_python_reference() {
    let cosmo = Planck15::new();
    let timeline = build_timeline(&cosmo);
    let galaxy = build_galaxy(timeline.z[0]);
    let pipeline = build_pipeline(&cosmo);

    let mut rng = StdRng::seed_from_u64(1);
    let history = pipeline.evolve(&galaxy, &timeline, false, false, &mut rng);

    let expected = [
        10.0000000000,
        10.0278564290,
        10.0552312774,
        10.0825115496,
        10.1098555745,
        10.1373443982,
        10.1639858886,
        10.1709886546,
        10.1718105974,
        10.1708481418,
        10.1694998222,
    ];
    assert_eq!(
        history.log_sm.len(),
        expected.len(),
        "trajectory length changed -- the regression values below no longer describe the whole output"
    );
    for (i, (&got, &want)) in history.log_sm.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-6, "step {i}: got {got}, want {want}");
    }
}

#[test]
fn stripped_noiseless_trajectory_matches_python_reference() {
    let cosmo = Planck15::new();
    let timeline = build_timeline(&cosmo);
    let galaxy = build_galaxy(timeline.z[0]);
    let pipeline = build_pipeline(&cosmo);

    let mut rng = StdRng::seed_from_u64(1);
    let history = pipeline.evolve(&galaxy, &timeline, true, false, &mut rng);

    // Regenerated from the Python reference after the fixture's
    // `t_dyn_friction` was corrected to 3.0 Gyr (see `build_timeline`).
    // With the span/timescale now self-consistent, `time_fraction` peaks
    // at 0.91, so the [0,1] clamp never engages and Rust and the
    // (unclamped) Python agree exactly, as they should.
    let expected = [
        10.0000000000,
        10.0090146417,
        10.0152894450,
        10.0188344175,
        10.0193207958,
        10.0161948256,
        10.0098522534,
        9.9759422899,
        9.9266938900,
        9.8642899422,
        9.7859900404,
    ];
    assert_eq!(
        history.log_sm.len(),
        expected.len(),
        "trajectory length changed -- the regression values below no longer describe the whole output"
    );
    for (i, (&got, &want)) in history.log_sm.iter().zip(expected.iter()).enumerate() {
        assert!((got - want).abs() < 1e-6, "step {i}: got {got}, want {want}");
    }
}

#[test]
fn stripped_satellite_ends_lower_mass_than_unstripped() {
    let cosmo = Planck15::new();
    let timeline = build_timeline(&cosmo);
    let galaxy = build_galaxy(timeline.z[0]);
    let pipeline = build_pipeline(&cosmo);

    let mut rng_a = StdRng::seed_from_u64(1);
    let unstripped = pipeline.evolve(&galaxy, &timeline, false, false, &mut rng_a);
    let mut rng_b = StdRng::seed_from_u64(1);
    let stripped = pipeline.evolve(&galaxy, &timeline, true, false, &mut rng_b);

    assert!(*stripped.log_sm.last().unwrap() < *unstripped.log_sm.last().unwrap());
}

#[test]
fn scatter_on_gives_reproducible_results_for_a_fixed_seed() {
    let cosmo = Planck15::new();
    let timeline = build_timeline(&cosmo);
    let galaxy = build_galaxy(timeline.z[0]);
    let pipeline = build_pipeline(&cosmo);

    let mut rng1 = StdRng::seed_from_u64(99);
    let h1 = pipeline.evolve(&galaxy, &timeline, false, true, &mut rng1);
    let mut rng2 = StdRng::seed_from_u64(99);
    let h2 = pipeline.evolve(&galaxy, &timeline, false, true, &mut rng2);

    // Exact equality is deliberate here, not an oversight. The claim
    // under test is specifically that an identical seed reproduces
    // bit-identical output -- same process, same code path, same
    // deterministic IEEE operations. Comparing within an epsilon would
    // weaken this into "roughly similar", which is no longer a
    // determinism test at all.
    assert_eq!(h1.log_sm, h2.log_sm);
}
