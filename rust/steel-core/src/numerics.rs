//! Small numerical utilities used by the orchestrator (not physics —
//! see `steel-plugins::numerics` for that).

/// Mirrors `numpy.digitize(x, bins, right=False)`: for monotonically
/// increasing `bins`, returns `i` such that `bins[i-1] <= x < bins[i]`;
/// for monotonically decreasing `bins`, returns `i` such that
/// `bins[i-1] > x >= bins[i]`. Returns an index in `[0, bins.len()]`.
pub fn digitize(x: f64, bins: &[f64]) -> usize {
    let increasing = bins.len() < 2 || bins[0] <= bins[bins.len() - 1];
    if increasing {
        bins.partition_point(|&b| b <= x)
    } else {
        bins.partition_point(|&b| b > x)
    }
}

/// Number of points `numpy.arange(start, stop, step)` produces, i.e.
/// `ceil((stop - start) / step)`.
///
/// Deliberately `ceil`, not `round`: they differ whenever
/// `(stop-start)/step` isn't an exact integer *including* when floating
/// point pushes an "exact" case just over. STEEL's own default halo
/// grid is exactly such a case — `(16.6 - 11.0) / 0.1` evaluates to
/// `56.000000000000014`, so `round` gives 56 bins where numpy gives 57,
/// silently dropping the most massive halo bin. Pass the same `start`
/// and `stop` numpy is given (h-offsets already applied to both), since
/// the offset only cancels in exact arithmetic.
pub fn arange_len(start: f64, stop: f64, step: f64) -> usize {
    assert!(step > 0.0, "arange_len: step must be positive");
    let n = ((stop - start) / step).ceil();
    if n <= 0.0 {
        0
    } else {
        n as usize
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arange_len_matches_numpy_for_steel_default_grids() {
        // Verified against numpy: np.arange(11.0+log10(h), 16.6+log10(h), 0.1)
        // has 57 elements, and the satellite grid has 65. `round()` would
        // give 56 for the first -- the off-by-one this helper exists to
        // prevent.
        let log_h = 0.6774_f64.log10();
        assert_eq!(arange_len(11.0 + log_h, 16.6 + log_h, 0.1), 57);
        assert_eq!(arange_len(10.0 + log_h, 16.5 + log_h, 0.1), 65);
    }

    #[test]
    fn arange_len_handles_exact_and_empty_ranges() {
        assert_eq!(arange_len(0.0, 1.0, 0.25), 4);
        assert_eq!(arange_len(9.0, 13.0, 0.1), 40);
        assert_eq!(arange_len(5.0, 5.0, 0.1), 0);
        assert_eq!(arange_len(5.0, 4.0, 0.1), 0);
    }

    #[test]
    fn matches_numpy_digitize_for_increasing_bins() {
        let bins = [0.0, 1.0, 2.0, 3.0];
        assert_eq!(digitize(-0.5, &bins), 0);
        assert_eq!(digitize(0.0, &bins), 1);
        assert_eq!(digitize(0.5, &bins), 1);
        assert_eq!(digitize(2.5, &bins), 3);
        assert_eq!(digitize(3.5, &bins), 4);
    }

    #[test]
    fn matches_numpy_digitize_for_decreasing_bins() {
        let bins = [3.0, 2.0, 1.0, 0.0];
        assert_eq!(digitize(3.5, &bins), 0);
        assert_eq!(digitize(2.5, &bins), 1);
        assert_eq!(digitize(0.5, &bins), 3);
        assert_eq!(digitize(-0.5, &bins), 4);
    }
}
