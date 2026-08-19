//! Unit and definition conversions between STEEL and external models.
//!
//! No physics here. These are the mismatches that silently invalidate an
//! SMHM overlay: an IMF offset comparable in size to the signal being
//! compared, an `Msun/h` vs `Msun` slip, or a halo mass quoted at a
//! different overdensity. Spec section 7.

use steel_core::cosmology::{Cosmology, MassDefinition};

/// Stellar initial mass function a stellar-mass calibration assumes.
///
/// Offsets are in dex, to be *added* to log10 M* when converting. Values
/// are the conventional ones; each must be re-verified against the
/// source paper before results are published (spec section 6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imf {
    Chabrier,
    Kroupa,
    Salpeter,
    /// The plugin's output does not carry an IMF (e.g. a halo-only model).
    /// Compatibility checks skip it.
    NotApplicable,
}

impl Imf {
    /// log10 M* offset relative to Chabrier. Chabrier is the zero point
    /// because it is STEEL's own calibration basis.
    fn dex_from_chabrier(self) -> f64 {
        match self {
            Imf::Chabrier => 0.0,
            // Kroupa masses are ~0.05 dex above Chabrier.
            Imf::Kroupa => 0.05,
            // Salpeter masses are ~0.24 dex above Chabrier.
            Imf::Salpeter => 0.24,
            Imf::NotApplicable => 0.0,
        }
    }

    /// Offset in dex to add to a log10 M* on `self` to express it on
    /// `other`. Zero if either side is `NotApplicable`.
    pub fn log_offset_to(self, other: Imf) -> f64 {
        if self == Imf::NotApplicable || other == Imf::NotApplicable {
            return 0.0;
        }
        other.dex_from_chabrier() - self.dex_from_chabrier()
    }
}

/// Whether masses carry a factor of `h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HConvention {
    /// Masses in Msun.
    HFree,
    /// Masses in Msun/h — STEEL's internal convention.
    PerH,
    /// The plugin's h-sensitive argument, where it has one, is not the
    /// run's own stellar/halo-mass axis (e.g. `QuenchingModel`'s
    /// `log_host_mass_infall`, always populated from STEEL's own native
    /// grid regardless of which stellar-mass model the run selects).
    /// Compatibility checks skip it, mirroring [`Imf::NotApplicable`].
    /// No numeric convention to convert, so `to_h_free`/`from_h_free`
    /// are the identity for it.
    NotApplicable,
}

impl HConvention {
    /// Convert a log10 mass in `self`'s convention to h-free log10 Msun.
    pub fn to_h_free(self, log_m: f64, h: f64) -> f64 {
        match self {
            HConvention::HFree | HConvention::NotApplicable => log_m,
            HConvention::PerH => log_m - h.log10(),
        }
    }

    /// Inverse of [`to_h_free`](Self::to_h_free).
    pub fn from_h_free(self, log_m: f64, h: f64) -> f64 {
        match self {
            HConvention::HFree | HConvention::NotApplicable => log_m,
            HConvention::PerH => log_m + h.log10(),
        }
    }
}

/// Halo concentration as a function of mass and redshift.
///
/// A trait rather than a constant because UniverseMachine is keyed on
/// peak circular velocity while STEEL is mass-keyed throughout, so this
/// relation sits on the conversion path and materially affects UM's
/// results. It is a selectable modelling assumption, not an
/// implementation detail (spec section 7).
pub trait ConcentrationMassRelation: Send + Sync {
    /// NFW concentration c = R_delta / r_s for `log_mh` \[log10 Msun\].
    fn concentration(&self, log_mh: f64, z: f64) -> f64;
}

/// Dutton & Maccio (2014) NFW concentration fit for a Planck cosmology,
/// virial mass definition:
///
/// ```text
/// log10 c = a + b (log10 M_vir/[1e12 h^-1 Msun])
/// a = 0.537 + (1.025 - 0.537) exp(-0.718 z^1.08)
/// b = -0.097 + 0.024 z
/// ```
///
/// Chosen as the default because it is a simple closed form calibrated on
/// Planck parameters, matching STEEL's `Planck15`. Swappable: implement
/// `ConcentrationMassRelation` and select it in the runfile.
pub struct DuttonMaccio14;

impl ConcentrationMassRelation for DuttonMaccio14 {
    fn concentration(&self, log_mh: f64, z: f64) -> f64 {
        let a = 0.537 + (1.025 - 0.537) * (-0.718 * z.powf(1.08)).exp();
        let b = -0.097 + 0.024 * z;
        // The fit is in units of 1e12 h^-1 Msun.
        10f64.powf(a + b * (log_mh - 12.0))
    }
}

/// Peak circular velocity \[km/s\] for a halo of `log_mh` \[log10
/// Msun/h\] at `z`, under an NFW profile.
///
/// For NFW, `Vmax^2 / V_delta^2 = 0.216 c / [ln(1+c) - c/(1+c)]`, with
/// `V_delta = sqrt(G M / R_delta)`. Masses in Msun/h and radii in kpc/h
/// leave the h factors cancelling in `G M / R`.
pub fn mpeak_to_vmax(
    log_mh: f64,
    z: f64,
    cosmo: &dyn Cosmology,
    cm: &dyn ConcentrationMassRelation,
    mdef: MassDefinition,
) -> f64 {
    /// kpc (km/s)^2 Msun^-1 — matches `Cosmology::rho_crit`.
    const G: f64 = 4.30091e-6;

    let m = 10f64.powf(log_mh); // Msun/h
    let r = cosmo.m_to_r(m, z, mdef); // kpc/h
    let v_delta_sq = G * m / r; // (km/s)^2

    let c = cm.concentration(log_mh, z);
    let denom = (1.0 + c).ln() - c / (1.0 + c);
    debug_assert!(denom > 0.0, "NFW mass factor must be positive, c = {c}");

    (v_delta_sq * 0.216 * c / denom).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use steel_core::cosmology::MassDefinition;

    #[test]
    fn imf_offset_is_zero_to_itself() {
        for imf in [Imf::Chabrier, Imf::Kroupa, Imf::Salpeter] {
            assert_eq!(imf.log_offset_to(imf), 0.0);
        }
    }

    #[test]
    fn imf_offset_is_antisymmetric() {
        let a = Imf::Chabrier.log_offset_to(Imf::Salpeter);
        let b = Imf::Salpeter.log_offset_to(Imf::Chabrier);
        assert!((a + b).abs() < 1e-12, "{a} and {b} should sum to zero");
    }

    #[test]
    fn salpeter_masses_exceed_chabrier() {
        // A Salpeter IMF infers more stellar mass for the same light.
        assert!(Imf::Chabrier.log_offset_to(Imf::Salpeter) > 0.0);
    }

    #[test]
    fn not_applicable_offset_is_zero_and_does_not_panic() {
        assert_eq!(Imf::NotApplicable.log_offset_to(Imf::Chabrier), 0.0);
        assert_eq!(Imf::Chabrier.log_offset_to(Imf::NotApplicable), 0.0);
    }

    #[test]
    fn not_applicable_h_conversion_is_the_identity() {
        assert_eq!(HConvention::NotApplicable.to_h_free(12.0, 0.6774), 12.0);
        assert_eq!(HConvention::NotApplicable.from_h_free(12.0, 0.6774), 12.0);
    }

    #[test]
    fn h_conversion_round_trips() {
        let h = 0.6774;
        let log_m_per_h = 12.0;
        let free = HConvention::PerH.to_h_free(log_m_per_h, h);
        let back = HConvention::PerH.from_h_free(free, h);
        assert!((back - log_m_per_h).abs() < 1e-12);
        // Msun/h -> Msun divides by h, so the h-free value is larger.
        assert!(free > log_m_per_h);
    }

    #[test]
    fn concentration_decreases_with_mass_and_redshift() {
        let cm = DuttonMaccio14;
        assert!(cm.concentration(11.0, 0.0) > cm.concentration(14.0, 0.0));
        assert!(cm.concentration(12.0, 0.0) > cm.concentration(12.0, 2.0));
    }

    #[test]
    fn vmax_increases_with_halo_mass() {
        let cosmo = Planck15::new();
        let cm = DuttonMaccio14;
        let v11 = mpeak_to_vmax(11.0, 0.0, &cosmo, &cm, MassDefinition::Vir);
        let v13 = mpeak_to_vmax(13.0, 0.0, &cosmo, &cm, MassDefinition::Vir);
        assert!(v13 > v11, "v(13)={v13} should exceed v(11)={v11}");
    }

    /// A Milky-Way-mass halo should have Vmax of order 150-250 km/s.
    /// Wide bounds: this catches unit errors, not fit quality.
    #[test]
    fn vmax_is_physically_plausible_for_a_milky_way_halo() {
        let cosmo = Planck15::new();
        let cm = DuttonMaccio14;
        let v = mpeak_to_vmax(12.1, 0.0, &cosmo, &cm, MassDefinition::Vir);
        assert!((120.0..300.0).contains(&v), "Vmax = {v} km/s");
    }
}
