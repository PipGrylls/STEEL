//! Dumps STEEL's own central SMHM stellar mass (Moster13, the
//! `steel-default.toml` preset) for an arbitrary list of host halo
//! masses [log10 Msun/h] at z=0.1, read one per line from stdin.
//!
//! Answers "how much stellar mass comes in from satellites (assigned
//! via EMERGE's/UniverseMachine's own model) relative to the central's
//! own STEEL-SMHM stellar mass": pipe in the exact per-bin z=0.1 host
//! masses read from an EMERGE/UM run's own `Mergers_AvaHaloMass.npy`
//! (row 0) so the two line up bin-for-bin exactly, rather than
//! reconstructing that grid a second, independent way here.
//!
//! Usage: `echo 12.3 | cargo run --release -p steel-plugins --example
//! dump_central_smhm_grid`, or pipe a whole column of values.

use std::io::{BufRead, Write};

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::{MosterFormSmhm, Planck15};

fn main() {
    let cosmo = Planck15::new();
    let smhm = MosterFormSmhm::moster13(true); // steel-default.toml's preset
    let z_ref = 0.1_f64;

    let flat_track = GrowthTrack { z: vec![z_ref], log_mass: vec![0.0] };
    let ctx = AccretionContext::central(&flat_track, &cosmo, MassDefinition::Vir);

    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    writeln!(out, "log_mh_perh,log_sm_central").unwrap();
    for line in std::io::stdin().lock().lines() {
        let line = line.unwrap();
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let log_mh_perh: f64 = line.parse().expect("expected one float per line");
        let log_sm = smhm.stellar_mass(log_mh_perh, z_ref, &ctx, None);
        writeln!(out, "{log_mh_perh:.6},{log_sm:.6}").unwrap();
    }
}
