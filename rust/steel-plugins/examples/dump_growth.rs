//! Dumps a VandenBosch14 growth history as "z log_M/M0" for comparison
//! against a live `getPWGH` run. Not part of the crate's public API —
//! a throwaway validation tool for Milestone 2's plan-mandated
//! cross-check against the real Fortran binary.

use steel_core::halo_growth::HaloGrowthModel;
use steel_plugins::{Planck15, VandenBosch14};

fn main() {
    let log_m0: f64 = std::env::args()
        .nth(1)
        .expect("usage: dump_growth <log10_M0>")
        .parse()
        .unwrap();
    let cosmo = Planck15::new();
    let model = VandenBosch14::new(&cosmo);
    let track = model.growth_history(log_m0, 0.0);
    for (z, log_m) in track.z.iter().zip(track.log_mass.iter()) {
        println!("{:.6} {:.6}", z, log_m - log_m0);
    }
}
