//! Average halo mass growth history plugin (van den Bosch 2014 by default).

/// The average mass accretion history of the main progenitor of a halo of
/// mass `log_m0` at `z0 = 0`, sampled on the model's own redshift grid
/// (van den Bosch 2014 uses 200 points log-spaced in `1+z`). All masses in
/// a single STEEL run share the same z0, so they share the same grid.
pub struct GrowthTrack {
    /// Redshift steps, increasing from `z0` to the model's z_max.
    pub z: Vec<f64>,
    /// log10( M(z) ) \[Msun/h\], the mass of the main progenitor at each
    /// redshift in `z`.
    pub log_mass: Vec<f64>,
}

pub trait HaloGrowthModel: Send + Sync {
    /// The redshift grid this model evaluates growth histories on, for a
    /// halo observed at `z0`. Shared across all masses for a fixed `z0`.
    fn redshift_grid(&self, z0: f64) -> Vec<f64>;

    /// Average mass growth history of the main progenitor of a halo with
    /// `log10(M0)` \[Msun/h\] observed at `z0`.
    fn growth_history(&self, log_m0: f64, z0: f64) -> GrowthTrack;
}
