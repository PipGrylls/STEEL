//! Dynamical-friction merger/infall timescale plugin.

use crate::cosmology::Cosmology;

pub trait MergerTimescaleModel: Send + Sync {
    /// Dynamical-friction infall timescale \[Gyr\] for a subhalo of
    /// `log_sat_mass` \[log10 Mvir/h\] orbiting a host of `log_host_mass`
    /// \[log10 Mvir/h\] at redshift `z`.
    fn infall_time(
        &self,
        log_host_mass: f64,
        log_sat_mass: f64,
        z: f64,
        cosmology: &dyn Cosmology,
    ) -> f64;
}
