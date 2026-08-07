//! Cosmology plugin trait.
//!
//! One implementation (`steel_plugins::cosmology::Planck15`) ships as the
//! default, but the trait exists so alternate cosmologies are a drop-in —
//! the STEEL thesis notes the statistical accretion history is
//! "theoretically independent" of this choice (Ch. 2, Method).

/// Spherical-overdensity mass definition used by [`Cosmology::m_to_r`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MassDefinition {
    /// Overdensity relative to the critical density using the
    /// Bryan & Norman (1998) virial threshold, Delta_vir(z).
    Vir,
    /// Overdensity `delta` x critical density, e.g. `Critical(200.0)` = "200c".
    Critical(f64),
    /// Overdensity `delta` x mean matter density, e.g. `Mean(200.0)` = "200m".
    Mean(f64),
}

/// A flat background cosmology, providing the handful of quantities the
/// STEEL physics plugins need: expansion history, cosmic time, and
/// spherical-overdensity radii.
pub trait Cosmology: Send + Sync {
    /// Hubble constant, H0 \[km/s/Mpc\].
    fn h0(&self) -> f64;

    /// Dimensionless Hubble parameter, h = H0/100.
    fn h(&self) -> f64 {
        self.h0() / 100.0
    }

    /// Present-day matter density parameter (baryons + CDM), dimensionless.
    fn omega_m0(&self) -> f64;

    /// Present-day baryon density parameter, dimensionless.
    fn omega_b0(&self) -> f64;

    /// Present-day dark-energy density parameter, dimensionless.
    fn omega_de0(&self) -> f64;

    /// Present-day radiation density parameter, dimensionless.
    fn omega_r0(&self) -> f64;

    /// sigma_8: rms matter fluctuation in 8 Mpc/h spheres at z=0.
    fn sigma8(&self) -> f64;

    /// Scalar spectral index of the primordial power spectrum.
    fn n_spec(&self) -> f64;

    /// Dimensionless Hubble parameter E(z) = H(z)/H0.
    fn e_z(&self, z: f64) -> f64;

    /// Hubble parameter at redshift z \[km/s/Mpc\].
    fn h_z(&self, z: f64) -> f64 {
        self.h0() * self.e_z(z)
    }

    /// Matter density parameter at redshift z, Omega_m(z), dimensionless.
    fn omega_m(&self, z: f64) -> f64 {
        self.omega_m0() * (1.0 + z).powi(3) / self.e_z(z).powi(2)
    }

    /// Age of the universe at redshift z \[Gyr\].
    fn age(&self, z: f64) -> f64;

    /// Lookback time to redshift z \[Gyr\].
    fn lookback_time(&self, z: f64) -> f64 {
        self.age(0.0) - self.age(z)
    }

    /// Virial overdensity relative to the critical density,
    /// Bryan & Norman (1998).
    fn delta_vir(&self, z: f64) -> f64 {
        let x = self.omega_m(z) - 1.0;
        18.0 * std::f64::consts::PI.powi(2) + 82.0 * x - 39.0 * x.powi(2)
    }

    /// Critical density at redshift z \[Msun h^2 / kpc^3\].
    fn rho_crit(&self, z: f64) -> f64 {
        // rho_crit(z) = 3 H(z)^2 / (8 pi G). Masses in Msun/h and lengths
        // in kpc/h mean H(z)/h [km/s/kpc] is the natural unit to plug in,
        // which leaves the density in Msun h^2 kpc^-3.
        const G_KPC_MSUN_KM2_S2: f64 = 4.30091e-6; // kpc (km/s)^2 Msun^-1
        let hz_over_h_km_s_kpc = 100.0 * self.e_z(z) / 1000.0; // (H(z)/h) in km/s/kpc
        3.0 * hz_over_h_km_s_kpc.powi(2) / (8.0 * std::f64::consts::PI * G_KPC_MSUN_KM2_S2)
    }

    /// Radius \[kpc/h\] enclosing overdensity `mdef` for a halo of mass
    /// `m` \[Msun/h\] at redshift `z`.
    fn m_to_r(&self, m: f64, z: f64, mdef: MassDefinition) -> f64 {
        let delta = match mdef {
            MassDefinition::Vir => self.delta_vir(z),
            MassDefinition::Critical(d) => d,
            // Mean-density overdensities are expressed relative to
            // rho_crit here too: rho_mean(z) = Omega_m(z) rho_crit(z).
            MassDefinition::Mean(d) => d * self.omega_m(z),
        };
        let rho_ref = self.rho_crit(z); // Msun h^2 kpc^-3
        (3.0 * m / (4.0 * std::f64::consts::PI * rho_ref * delta)).cbrt()
    }
}
