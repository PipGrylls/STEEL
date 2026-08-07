//! Dumps SHMF values for cross-checking against Python. Throwaway
//! validation tool.

use steel_core::shmf::SubhaloMassFunctionModel;
use steel_plugins::Jiang16;

fn main() {
    let shmf = Jiang16::default_calibration();
    for x in [0.001, 0.01, 0.1, 0.5, 0.9] {
        println!("x={x} -> dn/dlog10x={:.8}", shmf.dn_dlog10x(x));
    }
}
