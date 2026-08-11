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

/// Schreiber+2015 main sequence:
/// `log SFR = m - m0 + a0 r - a1 [max(0, m - m1 - a2 r)]^2`,
/// `m = log10(M*/1e9 Msun)`, `r = log10(1+z)`. Covers `S15`/`S16CE`.
///
/// **The two Python implementations disagree on the clamp, and the one
/// that actually runs is wrong.** `Functions.py::StarFormationRate`
/// writes `Max[Max<0] = 0` — clamp *below* at zero, i.e. `max(0, ·)`,
/// which is Schreiber et al. (2015, A&A 575, A74) Eq. 9 exactly.
/// `Functions_c.pyx::Starformation_c` writes `if Max > 0: Max = 0` —
/// clamp *above* at zero, the opposite — and the Cython is the version
/// every real STEEL run executes in its hot loop.
///
/// The published relation bends the main sequence *down at high mass*
/// and leaves it linear at low mass. Inverting the clamp does the
/// reverse: it removes the high-mass bend entirely and applies the
/// quadratic penalty to low-mass galaxies instead. The effect is large
/// where STEEL's satellites live — at `M* = 1e11`, `z = 0` the
/// published form suppresses SFR by 0.81 dex and the Cython's by
/// nothing at all.
///
/// This implements the published relation (equivalently,
/// `Functions.py`'s direction). That is a deliberate behavioural
/// departure from the code that produced the papers, not a
/// transcription of it — `S16CE` runs will not reproduce their
/// published figures without also correcting the Python. See
/// `docs/PORT_CORRECTIONS.md`.
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
        let max_term = (m - self.m1 - self.a2 * r).max(0.0);
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
    fn schreiber_form_bends_down_at_high_mass_and_stays_linear_at_low_mass() {
        // Schreiber+2015 Eq. 9 is `max(0, m - m1 - a2 r)`: the
        // quadratic term switches ON above the mass threshold and OFF
        // below it. `Functions_c.pyx` has this backwards; this pins the
        // published direction so the correction cannot silently revert.
        let sfr = SchreiberFormSfr::s15();
        let z: f64 = 0.01;
        let r = (1.0 + z).log10();
        let linear = |log_sm: f64| (log_sm - 9.0) - 0.5 + 1.5 * r;

        // Low mass: m - m1 - a2 r < 0, so no quadratic penalty at all.
        let low = 9.2;
        assert!(low - 9.0 - 0.36 - 2.5 * r < 0.0, "test setup: low mass should be below the knee");
        assert!(
            (sfr.log_sfr(low, z) - linear(low)).abs() < 1e-9,
            "below the knee the main sequence must be linear"
        );

        // High mass: the bend engages and suppresses the SFR.
        let high = 11.0;
        let max_term = high - 9.0 - 0.36 - 2.5 * r;
        assert!(max_term > 0.0, "test setup: high mass should be above the knee");
        let expected = linear(high) - 0.3 * max_term * max_term;
        assert!((sfr.log_sfr(high, z) - expected).abs() < 1e-9);
        assert!(
            sfr.log_sfr(high, z) < linear(high) - 0.5,
            "the high-mass bend should be a large suppression, got {} vs linear {}",
            sfr.log_sfr(high, z),
            linear(high)
        );
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
