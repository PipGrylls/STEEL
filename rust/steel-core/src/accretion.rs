//! Read-only accretion history and environment, passed to every
//! mass-assigning plugin.
//!
//! STEEL's `SmhmModel` is a memoryless M*(Mh,z) map, but rate-based
//! models (EMERGE, UniverseMachine) need the halo's assembly history.
//! Rather than a second parallel trait, every mass-assigning plugin
//! receives this context and ignores what it does not need — so any
//! future model has equal access to history without another interface.
//!
//! All fields are shared references or `Copy` scalars: constructing one
//! allocates nothing.

use crate::cosmology::{Cosmology, MassDefinition};
use crate::halo_growth::GrowthTrack;

pub struct AccretionContext<'a> {
    /// Main-progenitor track of *this* object treated as a central: to
    /// z0 for a central, to `z_infall` for a satellite. Always present.
    pub own_track: &'a GrowthTrack,
    /// Main-progenitor track of the host halo. `None` for centrals.
    pub host_track: Option<&'a GrowthTrack>,
    /// Infall redshift. `None` for centrals.
    pub z_infall: Option<f64>,
    /// Peak halo mass \[log10 Msun\] where it differs from the current
    /// mass. `None` when the caller cannot distinguish them.
    pub log_m_peak: Option<f64>,
    pub cosmology: &'a dyn Cosmology,
    /// Mass definition the `log_dm` / `log_mh` arguments are in.
    pub mass_definition: MassDefinition,
}

impl<'a> AccretionContext<'a> {
    pub fn central(
        own_track: &'a GrowthTrack,
        cosmology: &'a dyn Cosmology,
        mass_definition: MassDefinition,
    ) -> Self {
        Self { own_track, host_track: None, z_infall: None, log_m_peak: None, cosmology, mass_definition }
    }

    pub fn satellite(
        own_track: &'a GrowthTrack,
        host_track: &'a GrowthTrack,
        z_infall: f64,
        cosmology: &'a dyn Cosmology,
        mass_definition: MassDefinition,
    ) -> Self {
        Self {
            own_track,
            host_track: Some(host_track),
            z_infall: Some(z_infall),
            log_m_peak: None,
            cosmology,
            mass_definition,
        }
    }

    /// Redshift at which the main progenitor first exceeded `log_m`
    /// \[log10 Msun\], interpolated linearly in `log_mass` between the
    /// bracketing samples. `None` if the track never crosses it.
    ///
    /// `own_track.z` is increasing into the past and `log_mass` is
    /// decreasing, so the crossing is the first index whose mass falls
    /// below `log_m`.
    pub fn formation_redshift(&self, log_m: f64) -> Option<f64> {
        let t = self.own_track;
        let i = t.log_mass.iter().position(|&m| m < log_m)?;
        if i == 0 {
            return Some(t.z[0]);
        }
        let (m_hi, m_lo) = (t.log_mass[i - 1], t.log_mass[i]);
        let (z_lo, z_hi) = (t.z[i - 1], t.z[i]);
        let span = m_hi - m_lo;
        if span.abs() < f64::EPSILON {
            return Some(z_hi);
        }
        Some(z_lo + (m_hi - log_m) / span * (z_hi - z_lo))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::halo_growth::GrowthTrack;

    struct StubCosmo;
    impl crate::cosmology::Cosmology for StubCosmo {
        fn h0(&self) -> f64 { 67.74 }
        fn omega_m0(&self) -> f64 { 0.3089 }
        fn omega_b0(&self) -> f64 { 0.0486 }
        fn omega_de0(&self) -> f64 { 0.6911 }
        fn omega_r0(&self) -> f64 { 0.0 }
        fn sigma8(&self) -> f64 { 0.8159 }
        fn n_spec(&self) -> f64 { 0.9667 }
        fn e_z(&self, z: f64) -> f64 {
            (self.omega_m0() * (1.0 + z).powi(3) + self.omega_de0()).sqrt()
        }
        fn age(&self, _z: f64) -> f64 { 13.8 }
    }

    fn track() -> GrowthTrack {
        GrowthTrack { z: vec![0.0, 1.0, 2.0], log_mass: vec![12.0, 11.5, 11.0] }
    }

    #[test]
    fn central_context_has_no_host_or_infall() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        assert!(ctx.host_track.is_none());
        assert!(ctx.z_infall.is_none());
        assert_eq!(ctx.own_track.log_mass[0], 12.0);
    }

    #[test]
    fn satellite_context_carries_host_and_infall() {
        let own = track();
        let host = GrowthTrack { z: vec![0.0, 1.0], log_mass: vec![14.0, 13.5] };
        let c = StubCosmo;
        let ctx = AccretionContext::satellite(&own, &host, 1.5, &c, MassDefinition::Vir);
        assert_eq!(ctx.z_infall, Some(1.5));
        assert_eq!(ctx.host_track.expect("host").log_mass[0], 14.0);
    }

    #[test]
    fn formation_redshift_interpolates_between_samples() {
        // log_mass 12.0 -> 11.5 -> 11.0 at z 0, 1, 2. Crossing 11.75
        // sits halfway through the first interval, i.e. z = 0.5.
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let z = ctx.formation_redshift(11.75).expect("should cross");
        assert!((z - 0.5).abs() < 1e-12, "z = {z}");
    }

    #[test]
    fn formation_redshift_is_none_when_never_crossed() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        assert!(ctx.formation_redshift(9.0).is_none());
    }
}
