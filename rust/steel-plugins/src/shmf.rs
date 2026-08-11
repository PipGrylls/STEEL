//! Unevolved subhalo mass function (Jiang & van den Bosch 2016), a
//! direct port of `Functions.py::dn_dlnX`.

use steel_core::shmf::SubhaloMassFunctionModel;

/// `dn/dlog10(X) = ln(10) * gamma * (a X)^alpha * exp(-beta (a X)^omega)`,
/// `X = M_sub/M_host` at infall.
///
/// The Python hardcodes the `dn/dlnX -> dn/dlog10X` conversion factor as
/// the literal `2.30` rather than `ln(10) = 2.302585...` — a minor
/// rounding, not a documented behavioral choice, so per the "clean
/// reimplementation" decision this uses the exact constant.
pub struct Jiang16 {
    gamma: f64,
    alpha: f64,
    beta: f64,
    omega: f64,
    a: f64,
}

impl Jiang16 {
    pub fn new(gamma: f64, alpha: f64, beta: f64, omega: f64, a: f64) -> Self {
        Self { gamma, alpha, beta, omega, a }
    }

    /// The default calibration STEEL always runs with
    /// (`STEEL.py::Unevolved`, lines 102-108).
    pub fn default_calibration() -> Self {
        Self::new(0.22, -0.91, 6.0, 3.0, 1.0)
    }
}

impl Default for Jiang16 {
    fn default() -> Self {
        Self::default_calibration()
    }
}

impl SubhaloMassFunctionModel for Jiang16 {
    fn dn_dlog10x(&self, x: f64) -> f64 {
        let ax = self.a * x;
        let part1 = self.gamma * ax.powf(self.alpha);
        let part2 = (-self.beta * ax.powf(self.omega)).exp();
        std::f64::consts::LN_10 * part1 * part2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dn_dlog10x_is_positive_for_typical_mass_ratios() {
        let shmf = Jiang16::default_calibration();
        for &x in &[0.001, 0.01, 0.1, 0.5, 0.9] {
            let n = shmf.dn_dlog10x(x);
            assert!(n > 0.0 && n.is_finite(), "dn/dlog10x({x}) = {n}");
        }
    }

    #[test]
    fn dn_dlog10x_declines_toward_the_high_mass_ratio_cutoff() {
        // The exp(-beta (aX)^omega) term should dominate near X~1,
        // suppressing the number density of subhaloes close in mass to
        // their host.
        let shmf = Jiang16::default_calibration();
        let n_small = shmf.dn_dlog10x(0.01);
        let n_large = shmf.dn_dlog10x(0.9);
        assert!(n_large < n_small, "n_small={n_small} n_large={n_large}");
    }
}
