//! Dumps the G19_SE (PyMorph) and G19_cMod (cmodel) SMHM relations at
//! z=0.1 and z=2.0, reproducing the left panels of Paper 1 Fig. 3,
//! Paper 2 Fig. 4, and Paper 3 Figs. 4-6. See
//! `Scripts/Validation/smhm_curves.py` for the Python side.

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::{MosterFormSmhm, Planck15};

fn main() {
    let g19_se = MosterFormSmhm::g19_se(true);
    let g19_c_mod = MosterFormSmhm::g19_c_mod(true);

    // Both presets are memoryless and ignore `ctx`; a single flat point
    // satisfies the trait's context argument.
    let cosmo = Planck15::new();
    let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![12.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmo, MassDefinition::Vir);

    println!("model,z,log_dm,log_sm");
    let mut log_dm = 10.5;
    while log_dm <= 15.0 + 1e-9 {
        for &z in &[0.1, 2.0] {
            println!("G19_SE,{z},{log_dm:.3},{:.6}", g19_se.stellar_mass(log_dm, z, &ctx, None));
            println!("G19_cMod,{z},{log_dm:.3},{:.6}", g19_c_mod.stellar_mass(log_dm, z, &ctx, None));
        }
        log_dm += 0.05;
    }
}
