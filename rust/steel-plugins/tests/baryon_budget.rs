//! No model may convert more than the available baryons into stars.
//! Applies to both rate-based models over the full mass and redshift
//! range STEEL runs on. Spec section 10.

use std::sync::Arc;

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::{integrate_stellar_mass, StellarGrowthModel};
use steel_plugins::harmonise::DuttonMaccio14;
use steel_plugins::{EmergeGrowth, Planck15, UniverseMachineGrowth, VandenBosch14};

fn assert_within_baryon_budget(model: &dyn StellarGrowthModel, label: &str) {
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
                continue; // no elapsed time on this track segment
            }
            let ceiling = log_mh + f_b.log10();
            assert!(
                log_sm <= ceiling,
                "{label}: M*={log_sm:.3} exceeds baryon budget {ceiling:.3} \
                 at log_mh={log_mh} z={z_end}"
            );
        }
    }
}

#[test]
fn emerge_respects_the_baryon_budget() {
    assert_within_baryon_budget(&EmergeGrowth::o_leary23(), "emerge");
}

#[test]
fn universe_machine_respects_the_baryon_budget() {
    assert_within_baryon_budget(
        &UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14)),
        "universe_machine",
    );
}
