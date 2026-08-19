//! Dumps SMHM values for cross-checking against a standalone Python
//! re-implementation of the same equations. Throwaway validation tool.

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::{BehrooziFormSmhm, MosterFormSmhm, Planck15};

fn main() {
    // Every SMHM relation dumped here is memoryless and ignores `ctx`;
    // a single flat point satisfies the trait's context argument.
    let cosmo = Planck15::new();
    let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![12.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmo, MassDefinition::Vir);

    let g19_se = MosterFormSmhm::g19_se(true);
    for log_dm in [11.0, 12.0, 13.0] {
        println!("G19_SE log_dm={log_dm} z=0.1 -> log_sm={:.6}", g19_se.stellar_mass(log_dm, 0.1, &ctx, None));
    }
    println!();
    let b18c = BehrooziFormSmhm::behroozi18c();
    for log_dm in [11.0, 12.0, 13.0] {
        println!("B18c log_dm={log_dm} z=0.1 -> log_sm={:.6}", b18c.stellar_mass(log_dm, 0.1, &ctx, None));
    }
    println!();
    let b13 = BehrooziFormSmhm::behroozi13();
    for log_dm in [11.0, 12.0, 13.0] {
        println!("Behroozi13 log_dm={log_dm} z=0.1 -> log_sm={:.6}", b13.stellar_mass(log_dm, 0.1, &ctx, None));
    }
}
