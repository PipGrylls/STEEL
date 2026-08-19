//! Star-formation-rate main-sequence plugin.

use crate::accretion::AccretionContext;

pub trait SfrModel: Send + Sync {
    /// log10 star formation rate \[Msun/yr\] on the star-forming main
    /// sequence, before quenching/scatter is applied by the caller.
    /// `ctx` supplies accretion history; M*-keyed relations ignore it.
    fn log_sfr(&self, log_sm: f64, z: f64, ctx: &AccretionContext<'_>) -> f64;
}
