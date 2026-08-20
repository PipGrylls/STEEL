//! Dumps the numeric evidence behind `docs/VALIDATION.md` §6.5's two
//! quantitative self-consistency criteria for EMERGE and UniverseMachine,
//! as CSVs for plotting:
//!
//! 1. Baryon budget (`baryon_budget.rs`'s exact grid/formula): does
//!    integrated M* ever exceed `f_b . M_h`?
//! 2. SFR-vs-accretion-history consistency (`ssfr_self_consistency.rs`'s
//!    exact grid/formula): does the discrete trapezoidal integrator agree
//!    with a direct rate evaluation at the same step?
//!
//! `Scripts/Validation/plot_self_consistency.py` renders these.

use std::io::Write;
use std::sync::Arc;

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::{integrate_stellar_mass, StellarGrowthModel};
use steel_plugins::harmonise::DuttonMaccio14;
use steel_plugins::{EmergeGrowth, Planck15, UniverseMachineGrowth, VandenBosch14};

fn dump_baryon_budget(model: &dyn StellarGrowthModel, label: &str, w: &mut impl std::io::Write) {
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);
    let f_b = cosmo.omega_b0() / cosmo.omega_m0();

    for i in 0..=50 {
        let log_mh = 10.0 + i as f64 * 0.1;
        for &z_end in &[0.1, 0.5, 1.0, 2.0, 4.0] {
            let track = growth.growth_history(log_mh, z_end);
            let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
            let log_sm = integrate_stellar_mass(model, &ctx, z_end, None);
            if !log_sm.is_finite() {
                continue;
            }
            let ceiling = log_mh + f_b.log10();
            writeln!(w, "{label},{log_mh:.2},{z_end},{log_sm:.6},{ceiling:.6}").unwrap();
        }
    }
}

fn dump_ssfr_consistency(model: &dyn StellarGrowthModel, label: &str, w: &mut impl std::io::Write) {
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);

    for &log_mh0 in &[11.0, 12.0, 13.0, 14.0] {
        let track = growth.growth_history(log_mh0, 0.0);
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);

        for &z_query in &[0.2, 1.0, 2.0, 4.0] {
            let idx = track.z.partition_point(|&zt| zt < z_query);
            if idx == 0 || idx >= track.z.len() {
                continue;
            }
            let (z_lo, z_hi) = (track.z[idx - 1], track.z[idx]);
            if z_lo >= z_query || z_hi < z_query {
                continue;
            }

            let dt_yr = (cosmo.age(z_lo) - cosmo.age(z_hi)) * 1.0e9;
            if dt_yr <= 0.0 {
                continue;
            }
            let r_lo = 10f64.powf(model.stellar_growth_rate(track.log_mass[idx - 1], z_lo, &ctx, None));
            let r_hi = 10f64.powf(model.stellar_growth_rate(track.log_mass[idx], z_hi, &ctx, None));
            let local_implied_rate = 0.5 * (r_lo + r_hi);
            if local_implied_rate <= 0.0 {
                continue;
            }
            let local_implied_dex = local_implied_rate.log10();

            let log_mh_query = track.log_mass[idx - 1]
                + (track.log_mass[idx] - track.log_mass[idx - 1]) * (z_query - z_lo) / (z_hi - z_lo);
            let direct_dex = model.stellar_growth_rate(log_mh_query, z_query, &ctx, None);
            if !direct_dex.is_finite() {
                continue;
            }

            writeln!(
                w,
                "{label},{log_mh0:.2},{z_query:.2},{direct_dex:.6},{local_implied_dex:.6}"
            )
            .unwrap();
        }
    }
}

fn main() {
    let emerge = EmergeGrowth::o_leary23();
    let um = UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14));

    let mut budget = std::fs::File::create("baryon_budget.csv").unwrap();
    writeln!(budget, "model,log_mh,z,log_sm,ceiling").unwrap();
    dump_baryon_budget(&emerge, "EMERGE", &mut budget);
    dump_baryon_budget(&um, "UniverseMachine", &mut budget);

    let mut ssfr = std::fs::File::create("ssfr_consistency.csv").unwrap();
    writeln!(ssfr, "model,log_mh0,z,direct_dex,local_implied_dex").unwrap();
    dump_ssfr_consistency(&emerge, "EMERGE", &mut ssfr);
    dump_ssfr_consistency(&um, "UniverseMachine", &mut ssfr);

    println!("wrote baryon_budget.csv and ssfr_consistency.csv");
}
