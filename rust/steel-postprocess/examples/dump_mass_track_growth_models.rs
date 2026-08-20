//! Extends `dump_mass_tracks.rs`'s Paper 2 Fig. 6/8 "mass track"
//! reproduction (M*_AM(z) = SMHM(Mh(z), z), the in-situ-only SFH track,
//! and their difference = accreted/merger mass) with EMERGE's and
//! UniverseMachine's own predictions along the *same* halo growth
//! track, anchored the same way (STEEL's own SMHM inverted at the
//! target z=0 central mass).
//!
//! EMERGE/UM have no separate accretion-vs-SFH decomposition of their
//! own: a single `StellarGrowthModel::stellar_growth_rate` call is
//! simultaneously the model's SFR and the rate that builds its whole
//! mass history (docs/VALIDATION.md §6.5.3), so there is nothing here
//! to split into "accreted" vs "in-situ" the way STEEL's independent
//! SmhmModel/SfrModel pair can be split. What this *does* produce is a
//! fair, like-for-like comparison of each model's own total M*(z)
//! against STEEL's `log_sm_am` (total, via abundance matching) and
//! `log_sm_insitu` (pure star formation, zero accretion) tracks, all
//! driven by the identical halo mass history.
//!
//! See `Scripts/Validation/mass_tracks.py` and
//! `Scripts/Validation/mass_track_decomposition.py` for the STEEL-only
//! Python-side figure this extends.

use std::sync::Arc;

use rand::SeedableRng;
use steel_core::accretion::AccretionContext;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::{GrowthTrack, HaloGrowthModel};
use steel_core::smhm::SmhmModel;
use steel_core::stellar_growth::integrate_stellar_mass;
use steel_plugins::harmonise::DuttonMaccio14;
use steel_plugins::{DoublePowerLawSfr, EmergeGrowth, MosterFormSmhm, Planck15, UniverseMachineGrowth, VandenBosch14};
use steel_postprocess::central_evolution::CentralEvolution;

fn invert_smhm(smhm: &dyn SmhmModel, target_log_sm: f64, z: f64, ctx: &AccretionContext<'_>) -> f64 {
    let f = |log_dm: f64| smhm.stellar_mass(log_dm, z, ctx, None) - target_log_sm;
    let (mut lo, mut hi) = (9.0_f64, 17.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) < 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    0.5 * (lo + hi)
}

fn main() {
    let cosmo = Planck15::new();
    let smhm = MosterFormSmhm::g19_se(true); // Paper 2 Fig. 6's preset
    let growth = VandenBosch14::new(&cosmo);
    let log_h = cosmo.h().log10();
    let emerge = EmergeGrowth::o_leary23();
    let um = UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14));

    println!("target_log_sm,z,log_mh,log_sm_am,log_sm_insitu,log_sm_emerge,log_sm_um");

    for &target_log_sm in &[11.0, 11.5, 12.0] {
        let flat_track = GrowthTrack { z: vec![0.0], log_mass: vec![target_log_sm] };
        let flat_ctx = AccretionContext::central(&flat_track, &cosmo, MassDefinition::Vir);

        let log_dm_z0 = invert_smhm(&smhm, target_log_sm, 0.0, &flat_ctx);
        let track = growth.growth_history(log_dm_z0 + log_h, 0.0);

        // growth_history returns z increasing from z0=0 -- GrowthTrack's
        // own convention (index 0 = observed epoch, increasing into the
        // past), exactly what `integrate_stellar_mass` requires of
        // `own_track`. Keep only z<=3 (matching the paper's tracks) in
        // THIS order for EMERGE/UM; a separate, reversed (time-
        // increasing) copy feeds STEEL's AM lookup and
        // `CentralEvolution::evolve` below.
        // Both STEEL's `smhm.stellar_mass` call (dump_mass_tracks.rs's
        // existing convention) and EMERGE/UM's `own_track` want this
        // same h-free mass -- one vector serves both.
        let mut z_asc: Vec<f64> = Vec::new(); // increasing into the past, z_asc[0]=0
        let mut log_mh_hfree_asc: Vec<f64> = Vec::new();
        for (&zi, &log_m) in track.z.iter().zip(track.log_mass.iter()) {
            if zi <= 3.0 {
                z_asc.push(zi);
                log_mh_hfree_asc.push(log_m - log_h);
            }
        }
        let own_track = GrowthTrack { z: z_asc.clone(), log_mass: log_mh_hfree_asc.clone() };
        let growth_ctx = AccretionContext::central(&own_track, &cosmo, MassDefinition::Vir);

        let mut z = z_asc;
        let mut log_mh = log_mh_hfree_asc;
        z.reverse();
        log_mh.reverse();

        let log_sm_am: Vec<f64> =
            z.iter().zip(&log_mh).map(|(&zi, &lm)| smhm.stellar_mass(lm, zi, &flat_ctx, None)).collect();

        let t: Vec<f64> = z.iter().map(|&zi| cosmo.age(zi)).collect();
        let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        dt.push(*dt.last().unwrap());
        let accretion_rate = vec![0.0_f64; z.len()];

        let central = CentralEvolution::new(Box::new(DoublePowerLawSfr::central()));
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let history = central.evolve(
            log_sm_am[0], &z, &t, &dt, &accretion_rate,
            t[0] - 1.0, // never quenches -- see dump_mass_tracks.rs's doc
            false, &flat_ctx, &mut rng,
        );

        // EMERGE/UM: integrate each model's own rate along the *same*
        // halo track (h-free own_track, in its natural ascending-into-
        // the-past order), from formation down to each z on the track
        // -- their own "total M* so far" curve. `z` here is the
        // reversed (time-increasing) copy above; z_end order doesn't
        // matter to `integrate_stellar_mass`, only its value.
        let log_sm_emerge: Vec<f64> =
            z.iter().map(|&z_end| integrate_stellar_mass(&emerge, &growth_ctx, z_end, None)).collect();
        let log_sm_um: Vec<f64> =
            z.iter().map(|&z_end| integrate_stellar_mass(&um, &growth_ctx, z_end, None)).collect();

        for i in 0..z.len() {
            println!(
                "{target_log_sm},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6}",
                z[i], log_mh[i], log_sm_am[i], history.log_sm[i], log_sm_emerge[i], log_sm_um[i]
            );
        }
    }
}
