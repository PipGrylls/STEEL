//! Stellar-mass-halo-mass (abundance matching) plugin.

use rand::RngCore;

pub trait SmhmModel: Send + Sync {
    /// Stellar mass \[log10 Msun\] given halo mass `log_dm` \[log10
    /// Msun, h-free\] and redshift `z`. When `rng` is `Some`, draws and
    /// adds the model's intrinsic scatter; when `None`, returns the
    /// noiseless relation.
    fn stellar_mass(&self, log_dm: f64, z: f64, rng: Option<&mut dyn RngCore>) -> f64;
}
