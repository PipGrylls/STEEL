//! Unit and definition conversions between STEEL and external models.
//!
//! No physics here. These are the mismatches that silently invalidate an
//! SMHM overlay: an IMF offset comparable in size to the signal being
//! compared, an `Msun/h` vs `Msun` slip, or a halo mass quoted at a
//! different overdensity. Spec section 7.

use std::f64::consts::PI;

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
    /// NFW concentration c = R_delta / r_s for `log_mh`
    /// \[log10 **Msun/h**\], virial mass definition.
    ///
    /// The h-convention is load-bearing: `DuttonMaccio14`'s fit is
    /// pivoted at 1e12 h^-1 Msun, so passing an h-free mass shifts the
    /// concentration by `b * log10(h)`. Pinned by
    /// `dutton_maccio_pivot_is_defined_at_1e12_msun_per_h`.
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

/// NFW characteristic mass function, `mu(x) = ln(1+x) - x/(1+x)`.
///
/// The enclosed mass of an NFW halo is `M(<r) = 4 pi rho_s r_s^3 mu(r/r_s)`,
/// so mass ratios between radii reduce to ratios of `mu`.
fn nfw_mu(x: f64) -> f64 {
    (1.0 + x).ln() - x / (1.0 + x)
}

/// Overdensity threshold for `mdef`, relative to the critical density —
/// the same convention `Cosmology::m_to_r` uses.
fn delta_wrt_critical(mdef: MassDefinition, z: f64, cosmology: &dyn Cosmology) -> f64 {
    match mdef {
        MassDefinition::Vir => cosmology.delta_vir(z),
        MassDefinition::Critical(d) => d,
        MassDefinition::Mean(d) => d * cosmology.omega_m(z),
    }
}

/// Mass \[Msun/h\] enclosed by radius `r` \[kpc/h\] under definition
/// `mdef` — the exact inverse of [`Cosmology::m_to_r`].
fn mass_from_radius(r: f64, z: f64, mdef: MassDefinition, cosmology: &dyn Cosmology) -> f64 {
    let delta = delta_wrt_critical(mdef, z, cosmology);
    (4.0 / 3.0) * PI * r.powi(3) * cosmology.rho_crit(z) * delta
}

/// Mass at definition `mdef` implied by the NFW halo whose virial mass is
/// `m_vir` \[Msun/h\].
///
/// Solves for the radius where the profile's enclosed mass equals the
/// mass that `mdef` itself assigns to that radius. Both sides are
/// continuous and cross exactly once for physical concentrations, so
/// bisection is safe.
fn implied_mass_at(
    m_vir: f64,
    mdef: MassDefinition,
    z: f64,
    cosmology: &dyn Cosmology,
    concentration: &dyn ConcentrationMassRelation,
) -> f64 {
    let c_vir = concentration.concentration(m_vir.log10(), z);
    let r_vir = cosmology.m_to_r(m_vir, z, MassDefinition::Vir);
    let r_s = r_vir / c_vir;
    let mu_c = nfw_mu(c_vir);

    // f(r) > 0 while the profile encloses more than the definition
    // demands. At tiny r the profile term dominates; at large r the r^3
    // term does. Bracket generously around r_vir.
    let f = |r: f64| m_vir * nfw_mu(r / r_s) / mu_c - mass_from_radius(r, z, mdef, cosmology);

    let (mut lo, mut hi) = (1.0e-4 * r_vir, 1.0e2 * r_vir);
    // Promoted from debug_assert!: this is exactly the check that matters
    // in release builds, where the science runs. Two extra evaluations of
    // `f` are negligible against the 200-iteration bisection below, so
    // there is no performance case for leaving it debug-only. An
    // unbracketed root would otherwise return a silently wrong-but-
    // plausible-looking mass.
    assert!(f(lo) > 0.0 && f(hi) < 0.0, "root not bracketed for {mdef:?}");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r = 0.5 * (lo + hi);
    mass_from_radius(r, z, mdef, cosmology)
}

/// Convert a spherical-overdensity halo mass between definitions under an
/// NFW profile, in log10 Msun/h.
///
/// Conversions anchor on the virial definition because
/// [`ConcentrationMassRelation`] is virial-calibrated: `from` is first
/// inverted to a virial mass, then the profile is re-evaluated at `to`.
/// The inversion is a bisection on virial mass, since `implied_mass_at`
/// increases monotonically with it.
pub fn convert_mass_definition(
    log_m_from: f64,
    from: MassDefinition,
    to: MassDefinition,
    z: f64,
    cosmology: &dyn Cosmology,
    concentration: &dyn ConcentrationMassRelation,
) -> f64 {
    if from == to {
        return log_m_from;
    }
    let m_from = 10f64.powf(log_m_from);

    // Invert `from` to a virial mass, unless it already is one.
    let m_vir = if from == MassDefinition::Vir {
        m_from
    } else {
        let (mut lo, mut hi) = (log_m_from - 3.0, log_m_from + 3.0);
        // `implied_mass_at` increases monotonically with virial mass, so a
        // valid bracket has the implied mass below `m_from` at `lo` and
        // above it at `hi`. Without this check an out-of-range target
        // (e.g. an absurd overdensity) silently returns a bracket
        // endpoint rather than failing loudly.
        let implied_lo = implied_mass_at(10f64.powf(lo), from, z, cosmology, concentration);
        let implied_hi = implied_mass_at(10f64.powf(hi), from, z, cosmology, concentration);
        assert!(
            implied_lo < m_from && implied_hi > m_from,
            "convert_mass_definition: outer bisection root not bracketed for \
             from={from:?} at z={z}, log_m_from={log_m_from} \
             (tried bracket [{lo}, {hi}] in log10 Msun/h, implied masses \
             [{implied_lo:e}, {implied_hi:e}] vs target {m_from:e})"
        );
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let implied =
                implied_mass_at(10f64.powf(mid), from, z, cosmology, concentration);
            if implied < m_from {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        10f64.powf(0.5 * (lo + hi))
    };

    if to == MassDefinition::Vir {
        return m_vir.log10();
    }
    implied_mass_at(m_vir, to, z, cosmology, concentration).log10()
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

    #[test]
    fn dutton_maccio_pivot_is_defined_at_1e12_msun_per_h() {
        // D&M14 eq. 7 pivot: at z=0 and M_vir = 1e12 h^-1 Msun the mass term
        // vanishes, leaving log10 c = a(0) = 0.537 + 0.488 = 1.025. This test
        // pins the argument convention as log10 Msun/h -- passing an h-free
        // mass here would silently shift c.
        let c = DuttonMaccio14.concentration(12.0, 0.0);
        assert!((c - 10f64.powf(1.025)).abs() < 1e-9, "c = {c}");
    }

    #[test]
    fn concentration_falls_with_mass_and_redshift() {
        let c_low = DuttonMaccio14.concentration(11.0, 0.0);
        let c_high = DuttonMaccio14.concentration(14.0, 0.0);
        let c_z1 = DuttonMaccio14.concentration(12.0, 1.0);
        assert!(c_low > c_high, "concentration must fall with mass");
        assert!(c_z1 < DuttonMaccio14.concentration(12.0, 0.0));
    }

    #[test]
    fn converting_to_the_same_definition_is_the_identity() {
        let c = crate::cosmology::Planck15::new();
        let got = convert_mass_definition(
            14.0, MassDefinition::Vir, MassDefinition::Vir, 0.1, &c, &DuttonMaccio14);
        assert!((got - 14.0).abs() < 1e-6, "got {got}");
    }

    #[test]
    fn virial_mass_exceeds_m500c() {
        // Delta_vir(z~0) ~ 100x critical, well below 500x, so the virial
        // radius encloses more mass than r500c.
        let c = crate::cosmology::Planck15::new();
        let log_mvir = convert_mass_definition(
            14.0, MassDefinition::Critical(500.0), MassDefinition::Vir, 0.1, &c, &DuttonMaccio14);
        assert!(log_mvir > 14.0, "Mvir {log_mvir} should exceed M500c 14.0");
        assert!(log_mvir < 14.5, "conversion should be a modest shift, got {log_mvir}");
    }

    #[test]
    fn mass_definition_conversion_round_trips() {
        let c = crate::cosmology::Planck15::new();
        let fwd = convert_mass_definition(
            14.0, MassDefinition::Critical(500.0), MassDefinition::Vir, 0.3, &c, &DuttonMaccio14);
        let back = convert_mass_definition(
            fwd, MassDefinition::Vir, MassDefinition::Critical(500.0), 0.3, &c, &DuttonMaccio14);
        assert!((back - 14.0).abs() < 1e-4, "round trip gave {back}");
    }

    #[test]
    fn m200m_exceeds_m200c() {
        // Mean-density overdensities enclose more mass than critical ones at
        // the same Delta, since rho_mean < rho_crit.
        let c = crate::cosmology::Planck15::new();
        let m200m = convert_mass_definition(
            14.0, MassDefinition::Vir, MassDefinition::Mean(200.0), 0.0, &c, &DuttonMaccio14);
        let m200c = convert_mass_definition(
            14.0, MassDefinition::Vir, MassDefinition::Critical(200.0), 0.0, &c, &DuttonMaccio14);
        assert!(m200m > m200c, "M200m {m200m} should exceed M200c {m200c}");
    }

    #[test]
    fn m500c_to_mvir_pinned_value_at_z_0_1() {
        // Pins the exact numeric output so a future silent regression is
        // loud. Three of four deliberately-wrong variants tried during
        // review (mass_from_radius 3x too large, nfw_mu missing its
        // -x/(1+x) term, concentration hard-wired to 5.0) still passed
        // every other test in this file -- none of them constrain the
        // actual numerics. This value, 14.22438425687037, is this
        // implementation's own output for M500c=10^14 Msun/h at z=0.1; it
        // was independently cross-checked during code review, where a
        // from-scratch reimplementation reproduced it to all 16 digits
        // and a separate analytic root-find agreed to 15 digits
        // (c_vir ~= 6.11, Mvir/M500c ~= 1.676).
        let c = crate::cosmology::Planck15::new();
        let log_mvir = convert_mass_definition(
            14.0, MassDefinition::Critical(500.0), MassDefinition::Vir, 0.1, &c, &DuttonMaccio14);
        assert!(
            (log_mvir - 14.22438425687037).abs() < 1e-3,
            "expected log10(Mvir) ~= 14.2244, got {log_mvir}"
        );
    }

    #[test]
    fn mass_from_radius_inverts_m_to_r() {
        // mass_from_radius must be the exact inverse of Cosmology::m_to_r
        // -- implied_mass_at's bisection relies on that algebraic fact.
        // A mutant that scales either side by a stray constant (e.g. the
        // 3x mass_from_radius bug found in review) still passes every
        // other test here, since those only check ordering/identity/
        // round-trip properties that survive a shared constant-factor
        // error. This test would have caught it directly.
        let c = crate::cosmology::Planck15::new();
        let z = 0.2;
        let log_m = 13.5;
        let m = 10f64.powf(log_m);
        for mdef in [
            MassDefinition::Vir,
            MassDefinition::Critical(500.0),
            MassDefinition::Mean(200.0),
        ] {
            let r = c.m_to_r(m, z, mdef);
            let back = mass_from_radius(r, z, mdef, &c);
            let rel_err = (back - m).abs() / m;
            assert!(
                rel_err < 1e-9,
                "{mdef:?}: m_to_r/mass_from_radius round trip gave relative error {rel_err}"
            );
        }
    }
}
