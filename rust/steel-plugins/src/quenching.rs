//! Satellite quenching timescales (Wetzel+2013 fade/delay, Fillingham+2016
//! host-mass dependence, Cowley+2019 redshift scaling), a direct port of
//! the quenching-timescale block in `Functions.py::StarFormation`
//! (lines 337-361).

use steel_core::quenching::{QuenchTimescales, QuenchingModel};

pub struct Wetzel13;

impl Wetzel13 {
    pub fn new() -> Self {
        Self
    }
}

impl Default for Wetzel13 {
    fn default() -> Self {
        Self::new()
    }
}

impl QuenchingModel for Wetzel13 {
    fn timescales(
        &self,
        log_sm_infall: f64,
        z_infall: f64,
        log_host_mass_infall: f64,
        t_infall: f64,
        pre_quenched: bool,
    ) -> QuenchTimescales {
        let mut tau_fade = -0.5 * log_sm_infall + 5.7;
        if tau_fade <= 0.2 {
            tau_fade = 0.2;
        }

        let mut tau_delay = 3.5 - ((log_sm_infall - 10.8) * 2.0).exp();
        if tau_delay <= 1.0 {
            tau_delay = 1.0;
        }

        // Fillingham+2016 host-mass-dependent floor for low-mass
        // satellites of massive hosts.
        //
        // PORT-FIX A8: this used to be `.clamp(0.0, 1.0)`, which pins
        // the cutoff mass at exactly 9.0 for every host below
        // log_host_mass = 15 -- i.e. every host in any realistic run.
        // Paper 2 eq. (8) has no such floor and gives three distinct
        // cutoffs (8.0, 8.5, 9.0) for the paper's own example host
        // masses (10, 12.5, 15), which is what Figure 6 plots. See
        // docs/PORT_CORRECTIONS.md A8.
        let host_dep = (log_host_mass_infall - 15.0) / 5.0;
        if log_sm_infall < 9.0 + host_dep {
            tau_delay = 2.0;
        }

        // Cowley+2019 redshift scaling.
        let z_scale = (1.0 + z_infall).powf(-1.5);
        tau_delay *= z_scale;
        tau_fade *= z_scale;

        // `t` here is age of the universe (increasing with time) —
        // Timeline's convention (see steel-core::baryonic doc comment)
        // — so quenching *later* means a *larger* t_quench, the
        // opposite sign from `Functions.py`'s lookback-time-based
        // `T_quench = t[0] - Tau_d`.
        let t_quench = if pre_quenched { t_infall } else { t_infall + tau_delay };

        QuenchTimescales { tau_fade, tau_delay, t_quench }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn more_massive_satellites_have_shorter_fade_timescales() {
        // tau_fade = -0.5*log_sm + 5.7 (floored at 0.2): once quenching
        // starts, more massive satellites fade *faster* (shorter
        // tau_fade), not slower.
        let model = Wetzel13::new();
        let low_mass = model.timescales(9.0, 0.5, 13.0, 5.0, false);
        let high_mass = model.timescales(11.0, 0.5, 13.0, 5.0, false);
        assert!(high_mass.tau_fade < low_mass.tau_fade, "{} vs {}", high_mass.tau_fade, low_mass.tau_fade);
    }

    #[test]
    fn pre_quenched_forces_immediate_quenching() {
        let model = Wetzel13::new();
        let t_infall = 5.0;
        let q = model.timescales(10.0, 0.5, 13.0, t_infall, true);
        assert_eq!(q.t_quench, t_infall);
    }

    /// PORT-FIX A8: Paper 2 Figure 6 plots three visibly distinct
    /// cutoff masses (log M* ~ 8.0, 8.5, 9.0) for host masses 10, 12.5,
    /// 15 -- eq. (8) unclamped. The clamped version this replaced gave
    /// cutoff 9.0 for all three, since (Mh-15)/5 is negative for every
    /// Mh < 15 and was floored to zero.
    #[test]
    fn fillingham_cutoff_mass_differs_between_host_masses_below_1e15() {
        let model = Wetzel13::new();
        // z=0 removes the Cowley+2019 (1+z)^-1.5 scaling so tau_delay is
        // directly comparable to the paper's static eq. (8)/(9) plot.
        let overridden = |log_sm: f64, log_host: f64| model.timescales(log_sm, 0.0, log_host, 0.0, false).tau_delay == 2.0;

        // Between the host=10 cutoff (8.0) and the host=12.5 cutoff (8.5):
        // only the more massive hosts have reduced this satellite's delay.
        assert!(!overridden(8.2, 10.0), "host=10 should not yet reduce tau_delay at log M*=8.2");
        assert!(overridden(8.2, 12.5), "host=12.5 should reduce tau_delay at log M*=8.2");
        assert!(overridden(8.2, 15.0), "host=15 should reduce tau_delay at log M*=8.2");

        // Between the host=12.5 cutoff (8.5) and the host=15 cutoff (9.0):
        // only the most massive host has reduced this satellite's delay.
        assert!(!overridden(8.6, 10.0), "host=10 should not reduce tau_delay at log M*=8.6");
        assert!(!overridden(8.6, 12.5), "host=12.5 should not yet reduce tau_delay at log M*=8.6");
        assert!(overridden(8.6, 15.0), "host=15 should reduce tau_delay at log M*=8.6");
    }

    #[test]
    fn quench_time_is_after_infall_when_not_pre_quenched() {
        let model = Wetzel13::new();
        let t_infall = 5.0;
        let q = model.timescales(10.0, 0.5, 13.0, t_infall, false);
        assert!(q.t_quench > t_infall, "t_quench={} should exceed t_infall={t_infall}", q.t_quench);
    }
}
