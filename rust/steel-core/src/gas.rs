//! Galaxy gas-mass plugin: caps star formation by available gas supply.

use rand::RngCore;

pub trait GasMassModel: Send + Sync {
    /// Maximum gas mass \[log10 Msun\] a halo of `log_halo_mass`
    /// \[log10 Msun\] can supply (a cosmic baryon-fraction ceiling).
    fn max_gas_mass(&self, log_halo_mass: f64) -> f64;

    /// Instantaneous gas mass \[log10 Msun\] estimated from an SFR
    /// scaling relation, capped at `max_gas_mass`.
    ///
    /// `rng` is `None` for a noiseless draw (the relation's mean),
    /// matching [`crate::smhm::SmhmModel::stellar_mass`]'s convention.
    /// The validation harness needs every stochastic source in the
    /// model switchable from one place: with scatter off on both sides,
    /// the Rust and Python run the same arithmetic and should agree to
    /// floating point, which is the strong fidelity claim. Leaving one
    /// un-switchable source would silently weaken that to "roughly
    /// similar".
    fn gas_mass(&self, log_sfr: f64, log_halo_mass: f64, rng: Option<&mut dyn RngCore>) -> f64;
}
