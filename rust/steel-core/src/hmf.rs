//! Halo mass function plugin (Despali+2016 by default).

pub trait HaloMassFunctionModel: Send + Sync {
    /// Differential halo number density, dn/dlog10(M) \[h^3 Mpc^-3 dex^-1\].
    ///
    /// `log_m`: log10(Mvir) \[Msun/h\], `z`: redshift.
    fn dn_dlog10m(&self, log_m: f64, z: f64) -> f64;

    /// Vectorized convenience wrapper over [`Self::dn_dlog10m`]. Override
    /// if a batched/interpolated implementation is cheaper than N scalar
    /// calls.
    fn dn_dlog10m_arr(&self, log_m: &[f64], z: f64) -> Vec<f64> {
        log_m.iter().map(|&m| self.dn_dlog10m(m, z)).collect()
    }
}
