//! Star-formation-rate main-sequence models, direct ports of
//! `Functions_c.pyx::Starformation_c`'s SFR branches (the actual
//! per-timestep hot loop STEEL runs — not `Functions.py::StarFormationRate`,
//! which is a non-accelerated sibling used only for other bookkeeping
//! and disagrees with the Cython on the Schreiber-form clamp direction;
//! see `SchreiberFormSfr`'s doc comment).

use steel_core::sfr::SfrModel;

/// `s0 - log10(1 + (10^(SM - logM0))^Gamma)`, with `s0`, `logM0`, and
/// `Gamma` each a redshift polynomial `c[0] + c[1] z + c[2] z^2`. Covers
/// three named presets (`T16`, `CE`, `Illustris`) that share this
/// equation and differ only in coefficients — `T16`'s constant `Gamma`
/// and `CE`'s linear `Gamma` both fit the same 3-term pattern trivially
/// (`gamma[2] = 0`, or `gamma[1] = gamma[2] = 0`).
pub struct TomczakFormSfr {
    s0: [f64; 3],
    log_m0: [f64; 3],
    /// `Gamma(z) = -(gamma[0] + gamma[1] z + gamma[2] z^2)` — stored
    /// without the sign flip the Python source applies inline, applied
    /// in `log_sfr`.
    gamma: [f64; 3],
}

impl TomczakFormSfr {
    fn poly(c: [f64; 3], z: f64) -> f64 {
        c[0] + c[1] * z + c[2] * z * z
    }

    /// Tomczak+2016 ("T16").
    pub fn t16() -> Self {
        Self { s0: [0.195, 1.157, -0.143], log_m0: [9.244, 0.753, -0.09], gamma: [1.118, 0.0, 0.0] }
    }

    /// Grylls+2019 continuity fit ("CE") — STEEL's default SFR model.
    pub fn ce() -> Self {
        Self { s0: [0.6, 1.22, -0.2], log_m0: [10.3, 0.753, -0.15], gamma: [1.3, -0.1, 0.0] }
    }

    /// Illustris-calibrated fit.
    pub fn illustris() -> Self {
        Self { s0: [0.6, 1.22, -0.2], log_m0: [10.7, 0.5, -0.09], gamma: [1.6, -0.25, 0.01] }
    }
}

impl SfrModel for TomczakFormSfr {
    fn log_sfr(&self, log_sm: f64, z: f64) -> f64 {
        let s0 = Self::poly(self.s0, z);
        let log_m0 = Self::poly(self.log_m0, z);
        let gamma = -Self::poly(self.gamma, z);
        s0 - (1.0 + 10f64.powf((log_sm - log_m0) * gamma)).log10()
    }
}

/// Schreiber+2015-style: `m - m0 + a0 r - a1 max(m - m1 - a2 r, 0)^2`
/// (with the sign of the clamp as detailed below), `m = SM - 9`,
/// `r = log10(1+z)`. Covers `S15`/`S16CE`.
///
/// `Functions.py::StarFormationRate` (`Max[Max<0] = 0`, clip the
/// *negative* branch) and `Functions_c.pyx::Starformation_c`
/// (`if Max>0: Max=0`, clip the *positive* branch) disagree on the
/// clamp direction — a real inconsistency in the original source, not
/// a deliberate choice. Per the port's "clean reimplementation, not
/// bug-for-bug" decision, this uses the Cython's direction (clip
/// positive), since the Cython is what every actual STEEL run's hot
/// loop executes.
pub struct SchreiberFormSfr {
    m0: f64,
    a0: f64,
    a1: f64,
    m1: f64,
    a2: f64,
}

impl SchreiberFormSfr {
    /// Schreiber+2015 ("S15" in `Functions.py`, `SFR_Model_int==3`
    /// "S16" in the Cython — same coefficients despite the differing
    /// names).
    pub fn s15() -> Self {
        Self { m0: 0.5, a0: 1.5, a1: 0.3, m1: 0.36, a2: 2.5 }
    }

    /// The "S16CE" variant (`SFR_Model_int==4`).
    pub fn s16ce() -> Self {
        Self { m0: 0.75, a0: 1.75, a1: 0.3, m1: 0.36, a2: 1.75 }
    }
}

impl SfrModel for SchreiberFormSfr {
    fn log_sfr(&self, log_sm: f64, z: f64) -> f64 {
        let m = log_sm - 9.0;
        let r = (1.0 + z).log10();
        let mut max_term = m - self.m1 - self.a2 * r;
        if max_term > 0.0 {
            max_term = 0.0;
        }
        m - self.m0 + self.a0 * r - self.a1 * max_term * max_term
    }
}

/// Double-power-law SFR-mass relation ("G19_DPL").
pub struct DoublePowerLawSfr;

impl SfrModel for DoublePowerLawSfr {
    fn log_sfr(&self, log_sm: f64, z: f64) -> f64 {
        let log_m_n = 10.7 + 0.34 * z - 0.079 * z * z;
        let norm = 10f64.powf(0.74 + 0.71 * z - 0.087 * z * z);
        let alpha = 1.035 - 0.022 * z + 0.0077 * z * z;
        let beta = 1.55 - 0.35 * z - 0.02 * z * z;
        let x = log_sm - log_m_n;
        let m_per_y = 2.0 * norm / (10f64.powf(-alpha * x) + 10f64.powf(beta * x));
        m_per_y.log10()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ce_is_monotonically_increasing_below_the_turnover() {
        let sfr = TomczakFormSfr::ce();
        let s1 = sfr.log_sfr(9.0, 0.1);
        let s2 = sfr.log_sfr(10.0, 0.1);
        assert!(s2 > s1, "{s1} {s2}");
    }

    #[test]
    fn schreiber_form_clamp_direction_matches_the_cython() {
        // At high enough mass/low z, max_term = m - m1 - a2*r > 0, and
        // the Cython's fixed direction clips it to 0 -- verify the
        // clamp actually engages (log_sfr should equal the unclamped
        // linear part, i.e. the quadratic term vanishes) rather than
        // silently taking the Functions.py direction (which would clip
        // negative values instead and leave this term unclamped,
        // giving a different, larger-magnitude quadratic penalty).
        let sfr = SchreiberFormSfr::s15();
        let log_sm = 14.0; // m = 5.0, comfortably above m1=0.36 at z~0
        let z: f64 = 0.01;
        let m = log_sm - 9.0;
        let r = (1.0 + z).log10();
        let max_term = m - 0.36 - 2.5 * r;
        assert!(max_term > 0.0, "test setup should produce a positive max_term");
        let expected = m - 0.5 + 1.5 * r; // quadratic term clipped to 0
        assert!((sfr.log_sfr(log_sm, z) - expected).abs() < 1e-9);
    }

    #[test]
    fn double_power_law_peaks_near_the_knee_mass() {
        let sfr = DoublePowerLawSfr;
        let s_low = sfr.log_sfr(9.0, 1.0);
        let s_knee = sfr.log_sfr(10.7, 1.0);
        let s_high = sfr.log_sfr(12.5, 1.0);
        assert!(s_knee > s_low, "{s_knee} vs {s_low}");
        assert!(s_knee > s_high, "{s_knee} vs {s_high}");
    }
}
