//! Dumps the HMevo SMHM preset (Paper 3's high-mass-slope-evolution
//! family) at z=0.1 and z=2.0 for gamma11 in {0.1, 0.2, 0.5},
//! reproducing Paper 3 Fig. 6's left panel. See
//! `Scripts/Validation/hmevo_smhm_curves.py` for the Python side.

use steel_core::smhm::SmhmModel;
use steel_plugins::MosterFormSmhm;

fn main() {
    println!("gamma11,z,log_dm,log_sm");
    for &gamma11 in &[0.1, 0.2, 0.5] {
        let smhm = MosterFormSmhm::hmevo(gamma11, true);
        let mut log_dm = 10.5;
        while log_dm <= 15.0 + 1e-9 {
            for &z in &[0.1, 2.0] {
                println!("{gamma11},{z},{log_dm:.3},{:.6}", smhm.stellar_mass(log_dm, z, None));
            }
            log_dm += 0.05;
        }
    }
}
