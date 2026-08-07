//! Galaxy gas-mass plugin: caps star formation by available gas supply.

use rand::RngCore;

pub trait GasMassModel: Send + Sync {
    /// Maximum gas mass \[log10 Msun\] a halo of `log_halo_mass`
    /// \[log10 Msun\] can supply (a cosmic baryon-fraction ceiling).
    fn max_gas_mass(&self, log_halo_mass: f64) -> f64;

    /// Instantaneous gas mass \[log10 Msun\] estimated from an SFR
    /// scaling relation, with scatter drawn from `rng`, capped at
    /// `max_gas_mass`.
    fn gas_mass(&self, log_sfr: f64, log_halo_mass: f64, rng: &mut dyn RngCore) -> f64;
}
