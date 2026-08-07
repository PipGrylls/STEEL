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

#[cfg(test)]
mod tests {
    use super::*;

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
