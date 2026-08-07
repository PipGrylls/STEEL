//! Satellite quenching plugin: fade/delay timescales after infall.

/// Quenching timescales for a satellite, computed once at infall.
#[derive(Debug, Clone, Copy)]
pub struct QuenchTimescales {
    /// Exponential SFR fade timescale after quenching starts \[Gyr\].
    pub tau_fade: f64,
    /// Delay between infall and the onset of quenching \[Gyr\].
    pub tau_delay: f64,
    /// Cosmic time at which quenching begins \[Gyr\].
    pub t_quench: f64,
}

pub trait QuenchingModel: Send + Sync {
    /// Compute quenching timescales for a satellite that fell in with
    /// stellar mass `log_sm_infall` \[log10 Msun\] onto a host of
    /// `log_host_mass_infall` \[log10 Msun/h\] at redshift `z_infall`,
    /// cosmic time `t_infall` \[Gyr\].
    fn timescales(
        &self,
        log_sm_infall: f64,
        z_infall: f64,
        log_host_mass_infall: f64,
        t_infall: f64,
    ) -> QuenchTimescales;
}
