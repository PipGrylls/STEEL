//! Stellar-mass-halo-mass (abundance matching) plugin.

use rand::RngCore;

use crate::accretion::AccretionContext;

pub trait SmhmModel: Send + Sync {
    /// Stellar mass \[log10 Msun\] given halo mass `log_dm` \[log10
    /// Msun, h-free\] and redshift `z`. `ctx` supplies the object's
    /// accretion history; memoryless relations ignore it. When `rng` is
    /// `Some`, draws and adds the model's intrinsic scatter.
    fn stellar_mass(
        &self,
        log_dm: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64;
}
