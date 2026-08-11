//! Dumps a `BaryonicPipeline::evolve` trajectory (noiseless,
//! unstripped) plus the age(z) grid it ran on, for cross-checking
//! against a Python translation of `Functions_c.pyx::Starformation_c`.
//! Throwaway validation tool.

use rand::rngs::StdRng;
use rand::SeedableRng;

use steel_core::baryonic::{BaryonicPipeline, SatelliteState, Timeline};
use steel_core::cosmology::Cosmology;
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
        t_dyn_friction: 2.0,
    };

    let pipeline = BaryonicPipeline::new(
        Box::new(TomczakFormSfr::ce()),
        Box::new(Wetzel13::new()),
        Box::new(StewartScaling::from_cosmology(&cosmo)),
        Box::new(Cattaneo11),
    );

    // Print the exact MaxGas draw too, so the Python reference can be
    // given the same value rather than trying to (impossibly) sync
    // RNG streams across languages.
    let sfr_at_infall = steel_core::sfr::SfrModel::log_sfr(&TomczakFormSfr::ce(), galaxy.log_sm_infall, galaxy.z_infall);
    let mut gas_rng = StdRng::seed_from_u64(1);
    let max_gas = steel_core::gas::GasMassModel::gas_mass(
        &StewartScaling::from_cosmology(&cosmo),
        sfr_at_infall,
        galaxy.log_sat_mass_infall,
        &mut gas_rng,
    );
    println!("# max_gas (log10 Msun)");
    println!("{max_gas:.10}");

    let mut rng = StdRng::seed_from_u64(1);
    let history = pipeline.evolve(&galaxy, &timeline, false, false, &mut rng);

    println!("# log_sm (unstripped, noiseless)");
    for v in &history.log_sm {
        println!("{v:.10}");
    }
    println!("# log_ssfr (unstripped, noiseless)");
    for v in &history.log_ssfr {
        println!("{v:.10}");
    }

    let mut rng2 = StdRng::seed_from_u64(1);
    let stripped = pipeline.evolve(&galaxy, &timeline, true, false, &mut rng2);
    println!("# log_sm (stripped, noiseless)");
    for v in &stripped.log_sm {
        println!("{v:.10}");
    }
}
