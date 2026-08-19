//! Convolves a halo mass function with an SMHM relation (with scatter)
//! to produce a predicted stellar mass function — a direct port of
//! `Scripts/SMHM_Fit.py::DM_to_SM`.

use rand::RngCore;

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::GrowthTrack;
use steel_core::smhm::SmhmModel;
use steel_plugins::Planck15;

fn histogram_bin_index(x: f64, min: f64, bin_width: f64, n_bins: usize) -> Option<usize> {
    if x < min {
        return None;
    }
    let idx = ((x - min) / bin_width) as usize;
    if idx < n_bins {
        Some(idx)
    } else {
        None
    }
}

/// Predicted stellar mass function `dn/dlog10(M*)` \[Mpc^-3 dex^-1\],
/// in log10, one entry per `smf_x` point.
///
/// `smf_x` holds bin **centres**, matching the Python: `SMHM_Fit.py:339`
/// bins with `np.append(SMF_X, ...) - 0.05`, i.e. it shifts the edges
/// down by half a 0.1 dex bin so each `SMF_X` value sits at the middle
/// of its bin. (`Functions.py:776` writes the same operation as
/// `-(X_Bin/2)`, confirming `-0.05` means "half a bin" rather than a
/// magic constant — so this uses `smf_bin/2`, which is also correct for
/// bin widths other than 0.1, where the Python's hardcoded `0.05` would
/// silently be wrong.)
///
/// `halo_mr`/`hmf`: halo mass grid \[log10 Msun/h\] and its number
/// density \[h^3 Mpc^-3 dex^-1\] at each point (`Halo_MR`/`HMF` in the
/// Python). `n_mc`: Monte Carlo draws per halo mass bin used to sample
/// the SMHM relation's scatter (`N` in the Python, which used `1000`
/// by default).
#[allow(clippy::too_many_arguments)]
pub fn dm_to_sm(
    smf_x: &[f64],
    smf_bin: f64,
    halo_mr: &[f64],
    hmf_bin: f64,
    hmf: &[f64],
    smhm: &dyn SmhmModel,
    z: f64,
    h: f64,
    n_mc: usize,
    rng: &mut dyn RngCore,
) -> Vec<f64> {
    let h3 = h * h * h;
    let mut smf_y = vec![0.0_f64; smf_x.len()];
    // `smf_x` are bin centres, so the first bin's left edge sits half a
    // bin below `smf_x[0]` (the Python's `- 0.05`).
    let hist_min = smf_x[0] - smf_bin / 2.0;

    // The SMF fit evaluates the mean relation (with scatter) at each
    // halo mass independently; no accretion history is involved. Every
    // `SmhmModel` this function is called with today is memoryless, so
    // the concrete cosmology and the single-point track below are never
    // actually dereferenced -- they exist only to satisfy the trait.
    let cosmology = Planck15::new();

    for (&log_m_h, &n_h) in halo_mr.iter().zip(hmf) {
        let dm = log_m_h - h.log10();
        let weight_per_draw = n_h * h3 * hmf_bin / n_mc as f64;
        let track = GrowthTrack { z: vec![z], log_mass: vec![dm] };
        let ctx = AccretionContext::central(&track, &cosmology, MassDefinition::Vir);
        for _ in 0..n_mc {
            let sm = smhm.stellar_mass(dm, z, &ctx, Some(rng));
            if let Some(bin) = histogram_bin_index(sm, hist_min, smf_bin, smf_x.len()) {
                smf_y[bin] += weight_per_draw;
            }
        }
    }

    smf_y.iter().map(|&y| (y / smf_bin).log10()).collect()
}

/// Masked RMS distance between a predicted and target log-SMF,
/// `inf` if fewer than 90% of bins overlap in finite values — a direct
/// port of `MultiProcessWrapper_Lowz`'s fit-quality gate.
pub fn rms_distance(predicted: &[f64], target: &[f64]) -> f64 {
    debug_assert_eq!(predicted.len(), target.len());
    let mut sum = 0.0;
    let mut n = 0usize;
    for (&p, &t) in predicted.iter().zip(target) {
        if p.is_finite() && t.is_finite() {
            sum += (p - t).powi(2);
            n += 1;
        }
    }
    let coverage = n as f64 / predicted.len() as f64;
    if coverage > 0.9 {
        (sum / n as f64).sqrt()
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use steel_plugins::MosterFormSmhm;

    #[test]
    fn dm_to_sm_produces_a_declining_smf() {
        let smhm = MosterFormSmhm::g19_se(false);
        let halo_mr: Vec<f64> = (110..160).map(|i| i as f64 / 10.0).collect();
        let hmf: Vec<f64> = halo_mr.iter().map(|&m| 10f64.powf(-(m - 11.0))).collect(); // toy declining HMF
        let smf_x: Vec<f64> = (90..125).map(|i| i as f64 / 10.0).collect();
        let mut rng = StdRng::seed_from_u64(1);

        let smf = dm_to_sm(&smf_x, 0.1, &halo_mr, 0.1, &hmf, &smhm, 0.1, 0.6774, 200, &mut rng);

        let low = smf[2];
        let high = smf[smf.len() - 3];
        assert!(low > high, "low={low} high={high}");
    }

    #[test]
    fn smf_x_values_are_treated_as_bin_centres() {
        // Matches SMHM_Fit.py's `- 0.05` edge shift: a stellar mass
        // landing exactly on an `smf_x` value belongs to *that* bin, and
        // one just under the half-bin boundary below it belongs to the
        // bin beneath. With the old left-edge convention the first case
        // would still land in bin 1 but the second would wrongly land
        // there too.
        let smf_x = [10.0, 10.1, 10.2];
        let bin = 0.1;
        let hist_min = smf_x[0] - bin / 2.0; // 9.95
        let idx = |x: f64| super::histogram_bin_index(x, hist_min, bin, smf_x.len());
        assert_eq!(idx(10.1), Some(1), "a value on a centre belongs to that centre's bin");
        assert_eq!(idx(10.04), Some(0), "just below the 10.05 boundary belongs to the lower bin");
        assert_eq!(idx(10.06), Some(1), "just above the 10.05 boundary belongs to the upper bin");
        assert_eq!(idx(9.9), None, "below the first bin's left edge falls outside");
    }

    #[test]
    fn rms_distance_is_zero_for_identical_arrays() {
        let a = [1.0, 2.0, 3.0];
        assert_eq!(rms_distance(&a, &a), 0.0);
    }

    #[test]
    fn rms_distance_is_infinite_below_coverage_threshold() {
        let predicted = [1.0, f64::NAN, f64::NAN, f64::NAN, f64::NAN];
        let target = [1.0, 2.0, 3.0, 4.0, 5.0];
        assert!(rms_distance(&predicted, &target).is_infinite());
    }
}
