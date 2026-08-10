//! Subhalo mass function plugin (unevolved SHMF, Jiang & van den Bosch
//! 2016 by default).

pub trait SubhaloMassFunctionModel: Send + Sync {
    /// Unevolved subhalo mass function, dn/dlog10(X) \[dex^-1\], where
    /// `x = M_sub / M_host` is the subhalo-to-host mass ratio at infall.
    fn dn_dlog10x(&self, x: f64) -> f64;

    /// Vectorized convenience wrapper over [`Self::dn_dlog10x`].
    fn dn_dlog10x_arr(&self, x: &[f64]) -> Vec<f64> {
        x.iter().map(|&xi| self.dn_dlog10x(xi)).collect()
    }
}
