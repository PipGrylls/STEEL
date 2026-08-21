//! Dumps `McCavanaBK08::infall_time` over a subhalo-mass grid for three
//! host masses at z=1.5, plus the same masses converted to satellite
//! stellar mass via the G19_SE SMHM relation at z=1.5 -- reproducing
//! Paper 2 Figure 7 (dynamical-friction merging time-scale vs.
//! subhalo/satellite mass). Also dumps the age-of-universe difference
//! between z=1.5 and z=0, the figure's "time to z=0" dotted reference
//! line. See `Scripts/Validation/paper2_figures.py` for the
//! py-as-is/py-corrected side (`Functions.py::DynamicalFriction`).

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::GrowthTrack;
use steel_core::merger_time::MergerTimescaleModel;
use steel_core::smhm::SmhmModel;
use steel_plugins::{McCavanaBK08, MosterFormSmhm, Planck15};

fn main() {
    let cosmo = Planck15::new();
    let model = McCavanaBK08::default();
    let smhm = MosterFormSmhm::g19_se(true);
    let z = 1.5;
    let host_masses = [12.0, 13.0, 14.0];

    // `host`/`log_sub` below are in STEEL's internal Msun/h convention
    // (that's what `McCavanaBK08::infall_time` expects -- see
    // `merger_time.rs`, which feeds them straight into `m_to_r` and then
    // divides by `h`). `SmhmModel::stellar_mass`, by contrast, takes an
    // **h-free** halo mass (STEEL.py:380:
    // `DarkMatterToStellarMass(SatHaloMass[k] - np.log10(h), ...)`, and
    // `Simulation::run` mirrors it as `sat_mass[k] - log_h` in
    // context.rs), so it needs the Msun/h value converted first.
    let log_h = cosmo.h().log10();

    // `SmhmModel::stellar_mass` takes an `AccretionContext`, but G19_SE
    // is memoryless and ignores it, so a single flat point suffices.
    let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![12.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmo, MassDefinition::Vir);

    println!("time_to_z0_gyr,{:.10}", cosmo.age(0.0) - cosmo.age(z));

    println!("log_host_mass,log_subhalo_mass,log_sat_stellar_mass,t_merge_gyr");
    for &host in &host_masses {
        let mut log_sub = 9.0;
        while log_sub <= host + 1e-9 {
            let t_merge = model.infall_time(host, log_sub, z, &cosmo);
            let log_sm = smhm.stellar_mass(log_sub - log_h, z, &ctx, None);
            println!("{host:.2},{log_sub:.3},{log_sm:.6},{t_merge:.10}");
            log_sub += 0.05;
        }
    }
}
