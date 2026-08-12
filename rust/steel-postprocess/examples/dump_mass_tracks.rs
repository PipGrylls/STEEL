//! Dumps a central-galaxy "mass track" -- the abundance-matching mass
//! M*_AM(z) = SMHM(Mh(z), z) along one halo's van den Bosch (2014)
//! growth history, alongside the in-situ-only stellar mass from
//! evolving `CentralEvolution` with zero external accretion -- for one
//! target z=0 central stellar mass. Reproduces the top-panel /
//! "star formation only" strand of Paper 1 Figs. 6 & 8 and Paper 3
//! Fig. 7 (which anchor at z=0.1, not z=0 -- see below), not the full
//! accretion-decomposed 3x3 grid (that needs the per-track merger
//! accumulation, not yet ported to a standalone tool). See
//! `Scripts/Validation/mass_tracks.py` for the Python side.
//!
//! Anchored at z=0, not the papers' z=0.1: `Functions.py::Halogrowth`
//! has no z0 parameter at all -- it is hardcoded to z0=0 -- while this
//! trait's `growth_history(log_m0, z0)` takes z0 explicitly. Matching
//! the two legs to each other (the point of this comparison) means
//! matching Halogrowth's fixed anchor, not the paper's.
//!
//! Units note: `HaloGrowthModel` works in log10(Msun/h)
//! (`Functions.py`'s "Units are Mvir h-1"), matching
//! `steel_core::context::Simulation::run`'s own convention; `SmhmModel`
//! works in plain log10(Msun). Converted at the log_h boundary here the
//! same way `Simulation::run` does.

use rand::SeedableRng;
use steel_core::cosmology::Cosmology;
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::smhm::SmhmModel;
use steel_plugins::{DoublePowerLawSfr, MosterFormSmhm, Planck15, VandenBosch14};
use steel_postprocess::central_evolution::CentralEvolution;

fn invert_smhm(smhm: &dyn SmhmModel, target_log_sm: f64, z: f64) -> f64 {
    let f = |log_dm: f64| smhm.stellar_mass(log_dm, z, None) - target_log_sm;
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
    let mut args = std::env::args().skip(1);
    let target_log_sm: f64 = args
        .next()
        .expect("usage: dump_mass_tracks <target_log10_Mstar_at_z0> [gamma11|cmod]")
        .parse()
        .unwrap();
    // Optional second arg: either the HMevo preset's high-mass-slope-
    // evolution parameter (Paper 3's family), or the literal "cmod"
    // for the G19_cMod (de Vaucouleurs) preset used by Paper 2 Fig. 8.
    // Omit both for G19_SE (PyMorph), Paper 2 Fig. 6's preset.
    let second_arg = args.next();
    let is_cmod = second_arg.as_deref() == Some("cmod");
    let gamma11: Option<f64> = if is_cmod { None } else { second_arg.map(|s| s.parse().unwrap()) };

    let cosmo = Planck15::new();
    let smhm = match gamma11 {
        Some(g) => MosterFormSmhm::hmevo(g, true),
        None if is_cmod => MosterFormSmhm::g19_c_mod(true),
        None => MosterFormSmhm::g19_se(true),
    };
    let growth = VandenBosch14::new(&cosmo);
    let log_h = cosmo.h().log10();

    let log_dm_z0 = invert_smhm(&smhm, target_log_sm, 0.0);
    let track = growth.growth_history(log_dm_z0 + log_h, 0.0);

    // growth_history returns z increasing from z0; reverse to get time
    // increasing (z decreasing), and cut at z=3 to match the paper's
    // tracks.
    let mut z: Vec<f64> = Vec::new();
    let mut log_mh: Vec<f64> = Vec::new();
    for (&zi, &log_m) in track.z.iter().zip(track.log_mass.iter()) {
        if zi <= 3.0 {
            z.push(zi);
            log_mh.push(log_m - log_h);
        }
    }
    z.reverse();
    log_mh.reverse();

    let log_sm_am: Vec<f64> = z.iter().zip(&log_mh).map(|(&zi, &lm)| smhm.stellar_mass(lm, zi, None)).collect();

    let t: Vec<f64> = z.iter().map(|&zi| cosmo.age(zi)).collect();
    let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
    dt.push(*dt.last().unwrap());
    let accretion_rate = vec![0.0_f64; z.len()];

    let central = CentralEvolution::new(Box::new(DoublePowerLawSfr::central()));
    let mut rng = rand::rngs::StdRng::seed_from_u64(1);
    let history = central.evolve(
        log_sm_am[0], &z, &t, &dt, &accretion_rate,
        // `evolve` recomputes the main-sequence SFR while `t_quench <
        // t[i]`; t increases with age here, so a t_quench *below* the
        // track's first age keeps that true for every step -- never
        // quenches. (t_quench beyond the *last* age does the opposite:
        // it freezes SFR at its i==0 value for the whole track.)
        t[0] - 1.0,
        false, &mut rng,
    );

    println!("z,log_mh,log_sm_am,log_sm_insitu");
    for i in 0..z.len() {
        println!("{:.6},{:.6},{:.6},{:.6}", z[i], log_mh[i], log_sm_am[i], history.log_sm[i]);
    }
}
