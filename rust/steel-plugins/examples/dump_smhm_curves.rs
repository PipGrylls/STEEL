//! Dumps the G19_SE (PyMorph) and G19_cMod (cmodel) SMHM relations at
//! z=0.1 and z=2.0, reproducing the left panels of Paper 1 Fig. 3,
//! Paper 2 Fig. 4, and Paper 3 Figs. 4-6. See
//! `Scripts/Validation/smhm_curves.py` for the Python side.

use steel_core::smhm::SmhmModel;
use steel_plugins::MosterFormSmhm;

fn main() {
    let g19_se = MosterFormSmhm::g19_se(true);
    let g19_c_mod = MosterFormSmhm::g19_c_mod(true);

    println!("model,z,log_dm,log_sm");
    let mut log_dm = 10.5;
    while log_dm <= 15.0 + 1e-9 {
        for &z in &[0.1, 2.0] {
            println!("G19_SE,{z},{log_dm:.3},{:.6}", g19_se.stellar_mass(log_dm, z, None));
            println!("G19_cMod,{z},{log_dm:.3},{:.6}", g19_c_mod.stellar_mass(log_dm, z, None));
        }
        log_dm += 0.05;
    }
}
