//! Star-formation-rate main-sequence plugin.

pub trait SfrModel: Send + Sync {
    /// log10 star formation rate \[Msun/yr\] given stellar mass `log_sm`
    /// \[log10 Msun\] and redshift `z`, on the star-forming main sequence
    /// (before any quenching/scatter is applied by the caller).
    fn log_sfr(&self, log_sm: f64, z: f64) -> f64;
}
