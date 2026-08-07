//! Parallel grid-search fit of `MosterFormSmhm`'s z=0.1 shape
//! parameters against a target stellar mass function — a port of
//! `Scripts/SMHM_Fit.py::MultiProcessWrapper_Lowz` and its driving
//! `pool.map` loop, with `rayon` in place of `multiprocessing.Pool`.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rayon::prelude::*;

use steel_plugins::smhm::ZEvo;
use steel_plugins::MosterFormSmhm;

use crate::smf::{dm_to_sm, rms_distance};

/// Inclusive `[low, high]` bounds for one parameter, searched at
/// `steps` evenly spaced points (matching
/// `np.arange(bound[0], bound[1], (bound[1]-bound[0])/10)` in the
/// Python, generalized to a configurable step count).
#[derive(Debug, Clone, Copy)]
pub struct Bound {
    pub low: f64,
    pub high: f64,
}

pub struct GridSearchResult {
    pub best_params: MosterFormParams,
    pub best_rms: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct MosterFormParams {
    pub m10: f64,
    pub shmnorm10: f64,
    pub beta10: f64,
    pub gamma10: f64,
}

/// Grid-search the z=0.1 shape parameters (`M10`, `SHMnorm10`,
/// `beta10`, `gamma10`) of a [`MosterFormSmhm`] against `target_smf`,
/// holding redshift evolution fixed at zero (`ZEvo::Fixed`) — matching
/// `MultiProcessWrapper_Lowz`'s `Params = [Params[0], 0, Params[1], 0,
/// Params[2], 0, Params[3], 0, Params[4]]` (M11/SHMnorm11/beta11/gamma11
/// pinned to zero, only the four z=0.1 values searched).
#[allow(clippy::too_many_arguments)]
pub fn fit_low_z(
    target_smf: &[f64],
    smf_x: &[f64],
    smf_bin: f64,
    halo_mr: &[f64],
    hmf_bin: f64,
    hmf: &[f64],
    z: f64,
    h: f64,
    scatter: f64,
    bounds: [Bound; 4],
    steps_per_dim: usize,
    n_mc: usize,
    seed: u64,
) -> GridSearchResult {
    let grid_1d = |b: Bound| -> Vec<f64> {
        (0..steps_per_dim).map(|i| b.low + (b.high - b.low) * i as f64 / steps_per_dim as f64).collect()
    };
    let m10_grid = grid_1d(bounds[0]);
    let n10_grid = grid_1d(bounds[1]);
    let b10_grid = grid_1d(bounds[2]);
    let g10_grid = grid_1d(bounds[3]);

    let mut candidates = Vec::with_capacity(m10_grid.len() * n10_grid.len() * b10_grid.len() * g10_grid.len());
    for &m10 in &m10_grid {
        for &shmnorm10 in &n10_grid {
            for &beta10 in &b10_grid {
                for &gamma10 in &g10_grid {
                    candidates.push(MosterFormParams { m10, shmnorm10, beta10, gamma10 });
                }
            }
        }
    }

    let evaluated: Vec<(f64, MosterFormParams)> = candidates
        .into_par_iter()
        .enumerate()
        .map(|(idx, params)| {
            let smhm = MosterFormSmhm {
                m10: params.m10,
                shmnorm10: params.shmnorm10,
                beta10: params.beta10,
                gamma10: params.gamma10,
                m11: 0.0,
                shmnorm11: 0.0,
                beta11: 0.0,
                gamma11: 0.0,
                scatter,
                z_evo: ZEvo::Fixed,
            };
            // A per-candidate seed keeps every grid point's scatter
            // realization independent (and the whole search
            // reproducible) without needing thread-shared RNG state.
            let mut rng = StdRng::seed_from_u64(seed.wrapping_add(idx as u64));
            let predicted = dm_to_sm(smf_x, smf_bin, halo_mr, hmf_bin, hmf, &smhm, z, h, n_mc, &mut rng);
            (rms_distance(&predicted, target_smf), params)
        })
        .collect();

    let (best_rms, best_params) = evaluated
        .into_iter()
        .min_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal))
        .expect("grid must be non-empty");

    GridSearchResult { best_params, best_rms }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng as TestRng;

    fn toy_hmf() -> (Vec<f64>, Vec<f64>) {
        let halo_mr: Vec<f64> = (110..165).map(|i| i as f64 / 10.0).collect();
        let hmf: Vec<f64> = halo_mr.iter().map(|&m| 10f64.powf(-1.3 * (m - 11.0))).collect();
        (halo_mr, hmf)
    }

    #[test]
    fn grid_search_recovers_injected_parameters() {
        let (halo_mr, hmf) = toy_hmf();
        let smf_x: Vec<f64> = (90..125).map(|i| i as f64 / 10.0).collect();
        let h = 0.6774;
        let z = 0.1;
        let scatter = 0.15;

        // Generate a synthetic "observed" SMF from known parameters.
        let truth = MosterFormSmhm {
            m10: 12.0,
            shmnorm10: 0.03,
            beta10: 1.6,
            gamma10: 0.55,
            m11: 0.0,
            shmnorm11: 0.0,
            beta11: 0.0,
            gamma11: 0.0,
            scatter,
            z_evo: ZEvo::Fixed,
        };
        let mut rng = TestRng::seed_from_u64(123);
        let target = dm_to_sm(&smf_x, 0.1, &halo_mr, 0.1, &hmf, &truth, z, h, 500, &mut rng);

        let bounds = [
            Bound { low: 11.5, high: 12.5 },
            Bound { low: 0.02, high: 0.04 },
            Bound { low: 1.3, high: 1.9 },
            Bound { low: 0.4, high: 0.7 },
        ];
        let result = fit_low_z(&target, &smf_x, 0.1, &halo_mr, 0.1, &hmf, z, h, scatter, bounds, 6, 300, 42);

        assert!((result.best_params.m10 - 12.0).abs() < 0.3, "recovered M10 = {}", result.best_params.m10);
        assert!(
            (result.best_params.gamma10 - 0.55).abs() < 0.1,
            "recovered gamma10 = {}",
            result.best_params.gamma10
        );
        assert!(result.best_rms < 0.3, "best_rms = {} should be small for a recovered fit", result.best_rms);
    }
}
