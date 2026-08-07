//! Stripping plugins: dark-matter subhalo mass loss and stellar mass
//! (tidal) stripping. Two independent traits since they act on different
//! quantities and are driven by different physics.

use crate::cosmology::Cosmology;

/// A track of subhalo dark-matter mass, stripped over time as it orbits
/// its host.
pub struct HaloStrippingTrack {
    /// log10(subhalo mass) \[Msun/h\] at each step of the input track.
    pub log_mass: Vec<f64>,
}

pub trait HaloStrippingModel: Send + Sync {
    /// Strip a subhalo that fell in with `log_m_infall` \[log10 Msun/h\]
    /// as it orbits within a host whose mass history is
    /// `log_host_mass_track` \[log10 Msun/h\] at each `z_track` step,
    /// with per-step time intervals `dt_track` \[Gyr\]. Needs
    /// `cosmology` for the background expansion/collapse-overdensity
    /// terms the VDB05/Jiang16 mass-loss rate depends on.
    fn strip(
        &self,
        log_m_infall: f64,
        log_host_mass_track: &[f64],
        z_track: &[f64],
        dt_track: &[f64],
        cosmology: &dyn Cosmology,
    ) -> HaloStrippingTrack;
}

pub trait StellarStrippingModel: Send + Sync {
    /// log10 of the fraction of stellar mass still bound, given the host
    /// and satellite (subhalo) masses \[log10 Msun/h\] and the fraction of
    /// the total infall-to-merger time elapsed, `time_fraction` in
    /// `[0, 1]` (0 = infall, 1 = merger).
    fn strip_factor(&self, log_host_mass: f64, log_sat_mass: f64, time_fraction: f64) -> f64;
}
