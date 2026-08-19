//! Task 13, self-consistency criterion 3 ("SFR consistent with the
//! driving accretion history", `Grylls2020`, `paper/main.tex:177-183`)
//! for the two rate-based `StellarGrowthModel` plugins.
//!
//! For STEEL's own `[smhm]`+`[sfr]` pipeline this is a genuine,
//! nontrivial check: `SmhmModel` (M* as a function of Mh, z) and
//! `SfrModel` (SFR as a function of M*, z) are two *independently*
//! calibrated relations, and `Scripts/Validation/ssfr_sfr_sweep.py`
//! checks that taking d(SMHM)/dt along a halo's accretion history
//! reproduces the SFR the independent SfrModel predicts at that mass.
//!
//! EMERGE and UniverseMachine have no such second relation: a single
//! `StellarGrowthModel::stellar_growth_rate(log_mh, z, ctx, rng)` call
//! *is* simultaneously the model's SFR and the rate
//! `integrate_stellar_mass` integrates to build the accretion history.
//! There is nothing for it to disagree with -- self-consistency holds
//! by construction, not by coincidence of two independent fits
//! agreeing.
//!
//! What *is* worth checking numerically is that the discrete integrator
//! (trapezoidal in cosmic time, on the halo's actual ~200-point growth
//! track -- the same resolution the real orchestrator uses) doesn't
//! itself introduce an inconsistency: does the *local* implied rate for
//! one native track step -- `d(mass formed in that step)/dt`, i.e.
//! exactly what `integrate_stellar_mass`'s own trapezoidal formula
//! computes internally -- agree with `stellar_growth_rate` evaluated
//! directly at that step's redshift, using the *same*, real,
//! full-history `AccretionContext` both times?
//!
//! Both rate calls below always share one full, real `GrowthTrack`
//! (never an artificial local stub) as `ctx.own_track`, because
//! `UniverseMachineGrowth::stellar_growth_rate` keys on vMpeak -- the
//! *track's own peak mass*, not the per-step mass -- so any call built
//! from a truncated or synthetic track would silently evaluate a
//! different vMpeak than the real orchestrator ever would, which is a
//! bug in the test, not evidence about the model.

use std::sync::Arc;

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::StellarGrowthModel;
use steel_plugins::harmonise::DuttonMaccio14;
use steel_plugins::{EmergeGrowth, Planck15, UniverseMachineGrowth, VandenBosch14};

fn report(label: &str, model: &dyn StellarGrowthModel) {
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);

    println!("=== {label} ===");
    println!("log_mh0  z       direct_dex   local_implied_dex   |diff| dex");
    let mut max_diff = 0.0_f64;
    let mut n_compared = 0;
    for &log_mh0 in &[11.0, 12.0, 13.0, 14.0] {
        // z0 = 0.0: the widest, most demanding track (every self-
        // consistency query below is a redshift this track actually
        // passes through).
        let track = growth.growth_history(log_mh0, 0.0);
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);

        for &z_query in &[0.2, 1.0, 2.0, 4.0] {
            // The track's own bracketing grid points around z_query --
            // exactly the interval `integrate_stellar_mass` would use.
            let idx = track.z.partition_point(|&zt| zt < z_query);
            if idx == 0 || idx >= track.z.len() {
                continue;
            }
            let (z_lo, z_hi) = (track.z[idx - 1], track.z[idx]); // z_lo < z_hi (older = z_hi)
            if z_lo >= z_query || z_hi < z_query {
                continue;
            }

            let dt_yr = (cosmo.age(z_lo) - cosmo.age(z_hi)) * 1.0e9;
            if dt_yr <= 0.0 {
                continue;
            }
            let r_lo = 10f64.powf(model.stellar_growth_rate(track.log_mass[idx - 1], z_lo, &ctx, None));
            let r_hi = 10f64.powf(model.stellar_growth_rate(track.log_mass[idx], z_hi, &ctx, None));
            let local_implied_rate = 0.5 * (r_lo + r_hi); // integrate_stellar_mass's own trapezoidal average
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

            let diff = (direct_dex - local_implied_dex).abs();
            max_diff = max_diff.max(diff);
            n_compared += 1;
            println!("{log_mh0:6.2}  {z_query:.2}   {direct_dex:10.4}   {local_implied_dex:16.4}   {diff:9.4}");
        }
    }
    println!("max |direct - local_implied| over {n_compared} points = {max_diff:.4} dex\n");
}

fn main() {
    report("EMERGE (o_leary23)", &EmergeGrowth::o_leary23());
    report(
        "UniverseMachine (um_saga, deterministic mode: rng=None)",
        &UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14)),
    );
}
