//! Dumps `BaryonicPipeline::evolve` trajectories (noiseless, both
//! stripped and unstripped) for cross-checking against the real
//! `Functions_c.pyx::Starformation_c` -- see
//! `Scripts/Validation/reference_baryonic.py`, which drives the
//! committed Cython on the same fixture.
//!
//! Fixture matches `steel-plugins/tests/baryonic_pipeline.rs`.

use rand::rngs::StdRng;
use rand::SeedableRng;

use steel_core::accretion::AccretionContext;
use steel_core::baryonic::{BaryonicPipeline, SatelliteState, Timeline};
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::GrowthTrack;
use steel_plugins::{Cattaneo11, Planck15, StewartScaling, TomczakFormSfr, Wetzel13};

fn main() {
    let cosmo = Planck15::new();

    let z: Vec<f64> = (0..11).map(|i| 1.0 - i as f64 * 0.05).collect();
    let t: Vec<f64> = z.iter().map(|&zi| cosmo.age(zi)).collect();
    let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    dt.push(*dt.last().unwrap());

    println!("# z_grid");
    for zi in &z {
        println!("{zi:.10}");
    }
    println!("# t_grid (age, Gyr)");
    for ti in &t {
        println!("{ti:.10}");
    }
    println!("# dt_grid (Gyr)");
    for dti in &dt {
        println!("{dti:.10}");
    }

    let galaxy = SatelliteState {
        log_sm_infall: 10.0,
        log_host_mass_infall: 13.0,
        log_sat_mass_infall: 11.5,
        z_infall: z[0],
        pre_quenched: false,
    };
    let timeline = Timeline {
        z: z.clone(),
        t: t.clone(),
        dt: dt.clone(),
        log_host_mass: vec![13.0; z.len()],
        t_dyn_friction: 3.0,
    };

    let pipeline = BaryonicPipeline::new(
        Box::new(TomczakFormSfr::ce()),
        Box::new(Wetzel13::new()),
        Box::new(StewartScaling::from_cosmology(&cosmo)),
        Box::new(Cattaneo11),
    );

    // `BaryonicPipeline::evolve`/`SfrModel::log_sfr` take an
    // `AccretionContext`, but `TomczakFormSfr` is memoryless and ignores
    // it, so a single flat point satisfies the trait argument.
    let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![13.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmo, MassDefinition::Vir);

    // With scatter off the gas ceiling is the relation's mean, so both
    // sides can compute it independently and still agree -- no need to
    // (impossibly) sync RNG streams across languages.
    let sfr_at_infall = steel_core::sfr::SfrModel::log_sfr(
        &TomczakFormSfr::ce(),
        galaxy.log_sm_infall,
        galaxy.z_infall,
        &ctx,
    );
    let max_gas = steel_core::gas::GasMassModel::gas_mass(
        &StewartScaling::from_cosmology(&cosmo),
        sfr_at_infall,
        galaxy.log_sat_mass_infall,
        None,
    );
    println!("# max_gas (log10 Msun)");
    println!("{max_gas:.10}");

    for (label, stripping) in [("unstripped", false), ("stripped", true)] {
        let mut rng = StdRng::seed_from_u64(1);
        let history = pipeline.evolve(&galaxy, &timeline, stripping, false, &ctx, &mut rng);
        println!("# log_sm ({label}, noiseless)");
        for v in &history.log_sm {
            println!("{v:.10}");
        }
    }
}
