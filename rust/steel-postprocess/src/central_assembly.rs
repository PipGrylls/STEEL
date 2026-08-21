//! Closing the loop: feeding merged satellite stellar mass into the
//! central galaxy's own growth.
//!
//! [`crate::CentralEvolution::evolve`] has always taken an
//! `accretion_rate` argument, and `Simulation::run` has always produced
//! `accretion_history` (the merging-satellite stellar mass function per
//! central). Nothing joined them: every caller in the repo passed
//! `vec![0.0; n]`, so a central's assembled mass reflected in-situ star
//! formation only, and "how much mass arrives from satellites" versus
//! "what the SMHM says the central weighs" were two quantities computed
//! separately and compared by hand outside the model.
//!
//! This module is that join. The comparison it enables is the core of
//! STEEL's self-consistency argument: a central grows by star formation
//! *plus* accretion, so if the accreted mass alone already exceeds what
//! an empirical SMHM relation allows, something in the chain is
//! non-physical — and with stripping at its maximum (minimising the
//! delivered mass) that conclusion is forced rather than tunable.
//!
//! # No ad-hoc retention factor
//!
//! `Scripts/CentralPostprocessing.py` multiplies its accreted mass by
//! `0.612` — a Moster+2018 estimate that ~40% of a merging satellite's
//! stars end up in the intracluster light rather than the central. That
//! factor is **deliberately absent here**: `accretion_history` bins each
//! satellite's *post-stripping* mass at merger (`final_sm`, the end of
//! its `BaryonicPipeline` trajectory), and the mass stripped along the
//! way is banked separately in `RunOutput::icl_stripped_mass`. Applying
//! `0.612` on top would count the same loss twice. The retention
//! fraction here is an emergent property of the stripping model, not a
//! constant.

use ndarray::{Array2, ArrayView3};

/// Ex-situ stellar mass delivered to each host bin's central, per
/// redshift step: `[z, host]`, Msun per central.
///
/// `accretion_history` is `[z, host, sm]` and holds dN/dlog(M*) per
/// central (`Simulation::run` divides by `sat_sm_bin` when filling it),
/// so recovering a mass means re-multiplying by that bin width and
/// weighting each bin by its own stellar mass:
///
/// ```text
/// M_merged[i, j] = sum_k  AH[i, j, k] * 10^(sm_centre[k]) * sm_bin
/// ```
///
/// `sat_sm_range` is the vector of bin **left edges**, so bin centres
/// sit half a bin higher — the same convention
/// `Scripts/CentralPostprocessing.py` uses when it builds
/// `SatelliteMasses`.
pub fn merged_mass_per_central(
    accretion_history: ArrayView3<'_, f64>,
    sat_sm_range: &[f64],
    sat_sm_bin: f64,
) -> Array2<f64> {
    let (n_z, n_host, n_sm) = accretion_history.dim();
    let mut out = Array2::<f64>::zeros((n_z, n_host));
    if n_sm == 0 {
        return out;
    }
    assert!(
        sat_sm_range.len() >= n_sm,
        "sat_sm_range has {} entries but accretion_history has {n_sm} stellar-mass bins",
        sat_sm_range.len()
    );

    for i in 0..n_z {
        for j in 0..n_host {
            let mut total = 0.0;
            for k in 0..n_sm {
                let n_per_central = accretion_history[[i, j, k]];
                if n_per_central <= 0.0 {
                    continue;
                }
                let sm_centre = sat_sm_range[k] + 0.5 * sat_sm_bin;
                total += n_per_central * 10f64.powf(sm_centre) * sat_sm_bin;
            }
            out[[i, j]] = total;
        }
    }
    out
}

/// Convert a per-step merged mass \[Msun\] into the Msun/yr accretion
/// rate [`crate::CentralEvolution::evolve`] expects.
///
/// `dt_gyr` is the step duration in Gyr, matching the `dt` handed to
/// `evolve` itself; steps of non-positive duration yield a zero rate
/// rather than an infinity.
pub fn accretion_rate_msun_per_yr(merged_mass: &[f64], dt_gyr: &[f64]) -> Vec<f64> {
    assert_eq!(
        merged_mass.len(),
        dt_gyr.len(),
        "merged_mass and dt_gyr must describe the same steps"
    );
    merged_mass
        .iter()
        .zip(dt_gyr)
        .map(|(&m, &dt)| if dt > 0.0 { m / (dt * 1.0e9) } else { 0.0 })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::Array3;

    /// One populated bin, hand-computable: N galaxies per dex, times the
    /// bin's own mass, times the bin width.
    #[test]
    fn merged_mass_recovers_a_single_populated_bin() {
        let sat_sm_bin = 0.1;
        let sat_sm_range: Vec<f64> = (0..4).map(|k| 9.0 + k as f64 * sat_sm_bin).collect();
        let mut ah = Array3::<f64>::zeros((1, 1, 4));
        ah[[0, 0, 2]] = 3.0; // dN/dlogM* = 3 per central in the 9.2-9.3 bin

        let got = merged_mass_per_central(ah.view(), &sat_sm_range, sat_sm_bin);

        // centre of bin 2 is 9.2 + 0.05 = 9.25
        let want = 3.0 * 10f64.powf(9.25) * sat_sm_bin;
        assert!((got[[0, 0]] - want).abs() / want < 1e-12, "got {}, want {want}", got[[0, 0]]);
    }

    /// Bins add, and each host column is independent of the others.
    #[test]
    fn merged_mass_sums_bins_and_separates_hosts() {
        let sat_sm_bin = 0.5;
        let sat_sm_range = vec![10.0, 10.5];
        let mut ah = Array3::<f64>::zeros((1, 2, 2));
        ah[[0, 0, 0]] = 1.0;
        ah[[0, 0, 1]] = 2.0;
        ah[[0, 1, 0]] = 4.0;

        let got = merged_mass_per_central(ah.view(), &sat_sm_range, sat_sm_bin);

        let host0 = (10f64.powf(10.25) + 2.0 * 10f64.powf(10.75)) * sat_sm_bin;
        let host1 = 4.0 * 10f64.powf(10.25) * sat_sm_bin;
        assert!((got[[0, 0]] - host0).abs() / host0 < 1e-12);
        assert!((got[[0, 1]] - host1).abs() / host1 < 1e-12);
    }

    #[test]
    fn empty_accretion_history_gives_no_mass() {
        let ah = Array3::<f64>::zeros((2, 2, 3));
        let got = merged_mass_per_central(ah.view(), &[9.0, 9.1, 9.2], 0.1);
        assert!(got.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn rate_divides_mass_by_the_step_length_in_years() {
        let rates = accretion_rate_msun_per_yr(&[1.0e10, 0.0], &[1.0, 2.0]);
        assert!((rates[0] - 10.0).abs() < 1e-12, "{rates:?}"); // 1e10 Msun / 1e9 yr
        assert_eq!(rates[1], 0.0);
    }

    /// A zero-length step must not produce an infinite rate — the
    /// redshift grid's final step has no successor and can degenerate.
    #[test]
    fn zero_length_steps_give_zero_rate_not_infinity() {
        let rates = accretion_rate_msun_per_yr(&[1.0e10], &[0.0]);
        assert_eq!(rates[0], 0.0);
    }

    /// An SFR that ignores both mass and redshift, so two `evolve` runs
    /// differing *only* in their accretion rate stay exactly comparable:
    /// `sfr` and the recycling term are then identical step for step,
    /// and the whole difference between the runs is the accreted mass.
    /// A mass-dependent SFR (every real model) would couple the two and
    /// turn the identity below into an approximation.
    struct ConstantSfr(f64);
    impl steel_core::sfr::SfrModel for ConstantSfr {
        fn log_sfr(&self, _log_sm: f64, _z: f64, _ctx: &steel_core::accretion::AccretionContext<'_>) -> f64 {
            self.0
        }
    }

    /// The loop actually closes: mass fed in as `accretion_rate` arrives
    /// in the central, in full.
    ///
    /// `evolve` advances `M(i+1) = M(i) + (acc[i] + sfr - gmlr[i]) * dt[i]`
    /// and only for `i < n - 1`, so against an otherwise-identical
    /// zero-accretion run the difference telescopes to exactly
    /// `sum_{i < n-1} acc[i] * dt[i] * 1e9`. This is the assertion that
    /// would have failed for the entire history of the codebase before
    /// this module existed, because nothing ever passed a non-zero rate.
    #[test]
    fn accreted_mass_arrives_in_the_central_in_full() {
        use rand::rngs::StdRng;
        use rand::SeedableRng;
        use steel_core::accretion::AccretionContext;
        use steel_core::cosmology::MassDefinition;
        use steel_core::halo_growth::GrowthTrack;
        use steel_plugins::Planck15;

        let n = 8;
        let z: Vec<f64> = (0..n).map(|i| 2.0 - i as f64 * 0.2).collect();
        let t: Vec<f64> = (0..n).map(|i| 3.0 + i as f64 * 0.5).collect();
        let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        dt.push(*dt.last().unwrap());

        let merged_mass: Vec<f64> = (0..n).map(|i| 1.0e9 * (i as f64 + 1.0)).collect();
        let acc = accretion_rate_msun_per_yr(&merged_mass, &dt);
        let zeros = vec![0.0; n];

        let cosmo = Planck15::new();
        let track = GrowthTrack { z: vec![0.0], log_mass: vec![12.0] };
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let central = crate::CentralEvolution::new(Box::new(ConstantSfr(0.5)));

        // Scatter off, so the two runs cannot diverge through the RNG.
        let mut rng_a = StdRng::seed_from_u64(1);
        let with_accretion =
            central.evolve(10.5, &z, &t, &dt, &acc, f64::NEG_INFINITY, false, &ctx, &mut rng_a);
        let mut rng_b = StdRng::seed_from_u64(1);
        let in_situ_only =
            central.evolve(10.5, &z, &t, &dt, &zeros, f64::NEG_INFINITY, false, &ctx, &mut rng_b);

        let gained = 10f64.powf(*with_accretion.log_sm.last().unwrap())
            - 10f64.powf(*in_situ_only.log_sm.last().unwrap());
        let expected: f64 =
            (0..n - 1).map(|i| acc[i] * dt[i] * 1.0e9).sum();

        assert!(expected > 0.0, "test would be vacuous with no accretion");
        assert!(
            (gained - expected).abs() / expected < 1e-9,
            "central gained {gained} Msun from accretion, expected {expected}"
        );
    }
}
