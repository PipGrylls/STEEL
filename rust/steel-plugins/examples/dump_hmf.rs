//! Dumps Despali16 dn/dlog10M at a few masses for a sanity check against
//! well-known literature HMF normalizations. Throwaway validation tool.

use steel_core::hmf::HaloMassFunctionModel;
use steel_plugins::{Despali16, Planck15};

fn main() {
    let cosmo = Planck15::new();
    let hmf = Despali16::new(&cosmo);
    for log_m in [11.0, 12.0, 13.0, 14.0, 15.0] {
        let n = hmf.dn_dlog10m(log_m, 0.0);
        println!("log10(M)={log_m:.1}  dn/dlog10M(z=0) = {n:.4e}  [h^3 Mpc^-3 dex^-1]");
    }
}
