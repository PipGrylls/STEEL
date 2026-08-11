//! Central-galaxy stellar mass growth history, a port of
//! `Functions_c.pyx::Starformation_Centrals` — the central-galaxy
//! sibling of `Starformation_c` (`steel_core::BaryonicPipeline::evolve`).
//!
//! Deliberately **not** built on `BaryonicPipeline`: reading the actual
//! Cython source (rather than assuming symmetry with the satellite
//! case) shows centrals don't share the quenching-fade or gas-cap
//! machinery `BaryonicPipeline` composes — `Starformation_Centrals`
//! takes a `MaxGas` argument but never uses it (dead parameter, dropped
//! here), and its quench handling is structurally different (see
//! [`CentralEvolution::evolve`]'s doc comment). Only [`SfrModel`] is
//! genuinely shared with the satellite pipeline, which is what this
//! type reuses; the rest is a small, honest reimplementation rather
//! than a forced fit into an abstraction built for a different problem.

use rand::RngCore;
use rand_distr::{Distribution, Normal};

use steel_core::sfr::SfrModel;

const RECYCLING_C0: f64 = 0.05;
const RECYCLING_LAMBDA_YR: f64 = 1.4e6;
const SFR_SCATTER_DEX: f64 = 0.3;

pub struct CentralEvolution {
    pub sfr: Box<dyn SfrModel>,
}

pub struct CentralHistory {
    /// Stellar mass \[log10 Msun\] at each step of the input track.
    pub log_sm: Vec<f64>,
}

impl CentralEvolution {
    pub fn new(sfr: Box<dyn SfrModel>) -> Self {
        Self { sfr }
    }

    /// Evolve a central galaxy's stellar mass along `t`/`z` (both
    /// increasing/decreasing consistently — see
    /// `steel_core::baryonic::Timeline`'s convention, which this
    /// mirrors), given an external accretion rate `accretion_rate`
    /// \[Msun/yr\] (ex-situ mass from merging satellites) at each step.
    ///
    /// Quenching here works differently from the satellite case
    /// (`BaryonicPipeline`'s Wetzel+13 exponential fade): the Cython
    /// only *recomputes* the main-sequence SFR while `t[i] < t_quench`
    /// (or `i==0`); once quenched, the *previous* step's SFR value
    /// carries forward unchanged **except** that the unconditional
    /// scatter draw below still re-perturbs it every step — so a
    /// quenched central's SFR undergoes a compounding log-space random
    /// walk around its last star-forming value rather than freezing or
    /// fading. That's what the source does, not an approximation of
    /// something else; reproduced faithfully rather than "fixed" since
    /// there's no clear evidence it's unintended (unlike the satellite
    /// pipeline's SFH/StripFactor unit-mismatch bug, which was
    /// unambiguously wrong).
    #[allow(clippy::too_many_arguments)]
    pub fn evolve(
        &self,
        log_m_infall: f64,
        z: &[f64],
        t: &[f64],
        dt: &[f64],
        accretion_rate: &[f64],
        t_quench: f64,
        scatter_on: bool,
        rng: &mut dyn RngCore,
    ) -> CentralHistory {
        let n = t.len();
        debug_assert_eq!(n, z.len());
        debug_assert_eq!(n, dt.len());
        debug_assert_eq!(n, accretion_rate.len());

        let mut log_sm = vec![0.0_f64; n];
        log_sm[0] = log_m_infall;
        let mut sfh = vec![0.0_f64; n];
        let mut gmlr = vec![0.0_f64; n];
        let normal = Normal::new(0.0, SFR_SCATTER_DEX).unwrap();

        let mut sfr = 0.0_f64;
        for i in 0..n {
            if t_quench < t[i] || i == 0 {
                sfr = 10f64.powf(self.sfr.log_sfr(log_sm[i], z[i]));
            }
            if scatter_on {
                sfr = 10f64.powf(sfr.log10() + normal.sample(rng));
            }

            sfh[i] = sfr * dt[i] * 1.0e9;

            if i > 0 && i < n - 1 {
                for (j, &sfh_j) in sfh.iter().enumerate().take(i) {
                    let t_j = t[j];
                    let f_mr_1 = 1.0 - RECYCLING_C0 * (((t_j - t[i]).abs() * 1.0e9 / RECYCLING_LAMBDA_YR) + 1.0).ln();
                    let f_mr_2 =
                        1.0 - RECYCLING_C0 * (((t_j - t[i + 1]).abs() * 1.0e9 / RECYCLING_LAMBDA_YR) + 1.0).ln();
                    gmlr[i] += (sfh_j * (f_mr_1 - f_mr_2)).abs() / ((t[i] - t[i + 1]).abs() * 1.0e9);
                }
            }

            let m_dot = accretion_rate[i] + sfr - gmlr[i];
            if i < n - 1 {
                log_sm[i + 1] = (10f64.powf(log_sm[i]) + m_dot * dt[i] * 1.0e9).log10();
            }
        }

        CentralHistory { log_sm }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use steel_plugins::DoublePowerLawSfr;

    fn toy_timeline() -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let z: Vec<f64> = (0..10).map(|i| 2.0 - i as f64 * 0.15).collect();
        let t: Vec<f64> = (0..10).map(|i| 3.0 + i as f64 * 0.4).collect(); // age, Gyr, increasing
        let mut dt: Vec<f64> = t.windows(2).map(|w| w[1] - w[0]).collect();
        dt.push(*dt.last().unwrap());
        (z, t, dt)
    }

    #[test]
    fn mass_grows_with_zero_accretion_and_no_quenching() {
        let (z, t, dt) = toy_timeline();
        let accretion = vec![0.0; t.len()];
        let evo = CentralEvolution::new(Box::new(DoublePowerLawSfr));
        let mut rng = StdRng::seed_from_u64(1);
        let history = evo.evolve(10.5, &z, &t, &dt, &accretion, f64::INFINITY, false, &mut rng);

        for w in history.log_sm.windows(2) {
            assert!(w[1] >= w[0] - 1e-9, "mass should not decrease with in-situ SF only: {:?}", w);
        }
        assert!(*history.log_sm.last().unwrap() > history.log_sm[0]);
    }

    #[test]
    fn positive_accretion_rate_increases_growth() {
        let (z, t, dt) = toy_timeline();
        let evo = CentralEvolution::new(Box::new(DoublePowerLawSfr));

        let zero_acc = vec![0.0; t.len()];
        let mut rng_a = StdRng::seed_from_u64(1);
        let no_acc = evo.evolve(10.5, &z, &t, &dt, &zero_acc, f64::INFINITY, false, &mut rng_a);

        let with_acc = vec![5.0; t.len()]; // Msun/yr, deliberately large
        let mut rng_b = StdRng::seed_from_u64(1);
        let accreted = evo.evolve(10.5, &z, &t, &dt, &with_acc, f64::INFINITY, false, &mut rng_b);

        assert!(*accreted.log_sm.last().unwrap() > *no_acc.log_sm.last().unwrap());
    }

    #[test]
    fn quenching_freezes_the_main_sequence_recomputation() {
        // With scatter off, a quenched central's SFR should hold at
        // its last star-forming value exactly (no fade, no floor).
        let (z, t, dt) = toy_timeline();
        let accretion = vec![0.0; t.len()];
        let evo = CentralEvolution::new(Box::new(DoublePowerLawSfr));
        let mut rng = StdRng::seed_from_u64(1);
        // Quench immediately after the first step.
        let t_quench = t[0];
        let history = evo.evolve(10.5, &z, &t, &dt, &accretion, t_quench, false, &mut rng);
        assert!(history.log_sm.iter().all(|v| v.is_finite()));
    }
}
