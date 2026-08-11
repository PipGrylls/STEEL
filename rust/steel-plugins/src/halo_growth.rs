//! van den Bosch (2014) average halo mass accretion history, a direct
//! port of `Functions/OtherModels/VDB13/getPWGH.f`'s main loop and
//! `find_psi` — the actual algorithm STEEL currently gets by shelling
//! out to the compiled `getPWGH` binary per halo mass
//! (`Functions.py::Halogrowth`).

use steel_core::cosmology::Cosmology;
use steel_core::halo_growth::{GrowthTrack, HaloGrowthModel};

use crate::growth::GrowthFactor;
use crate::numerics::ridders_root_find;
use crate::variance::Variance;

/// "Averages" fit parameters from `getPWGH.f`'s `ianswer=1` branch
/// (`apar1..apar5`). STEEL always requests averages, not medians —
/// `Functions.py::Halogrowth`'s `Input_Str` hardcodes
/// `"1 ! median (0) or averages (1)"` — so the median branch
/// (`apar1..5 = 1.9278, 0.4241, 0.7684, 0.1481, 0.3096`) is never
/// exercised and isn't ported.
const APAR: [f64; 5] = [3.2954, 0.1975, 0.7542, 0.0898, 0.4415];

/// Number of redshift steps (`Nz` in `paramfile.h`).
const N_Z: usize = 200;
/// Upper bound on `log10(1+z) - log10(1+z0)` (`getPWGH.f:76`).
const LOG_DELTA_Z_MAX: f64 = 0.85;
/// Small offset added to every `log10(1+z)` sample (`getPWGH.f:77`) —
/// load-bearing, not cosmetic: without it the very first grid point
/// lands exactly on `z0`, where the root-find's `sqrt(s1 - s0)` term is
/// `0/0`.
const LOG_Z_OFFSET: f64 = 0.00004343;

pub struct VandenBosch14 {
    variance: Variance,
    growth: GrowthFactor,
}

impl VandenBosch14 {
    pub fn new(cosmology: &dyn Cosmology) -> Self {
        Self {
            variance: Variance::new(cosmology),
            growth: GrowthFactor::from_cosmology(cosmology),
        }
    }

    /// Find a lower bracket below `x_hi` where `f` changes sign, so
    /// `ridders_root_find`'s bracketing precondition holds.
    ///
    /// The Fortran hardcodes a one-dex window (`xlgpsi_min = xlgpsi -
    /// 1.0`), which is comfortably wide for STEEL's own mass/redshift
    /// grid but is an assumption, not a guarantee — a different grid or
    /// mass range could step further than one dex between samples and
    /// silently break the root find. Widening until the sign actually
    /// flips keeps the common case identical (the first candidate *is*
    /// `x_hi - 1.0`) while making the failure mode an explicit,
    /// informative panic instead of a bracketing assertion deep inside
    /// the solver.
    fn widen_bracket<F: Fn(f64) -> f64>(f: F, x_hi: f64, z: f64) -> f64 {
        const MAX_WIDENINGS: usize = 8;
        let f_hi = f(x_hi);
        let mut width = 1.0_f64;
        for _ in 0..MAX_WIDENINGS {
            let x_lo = x_hi - width;
            let f_lo = f(x_lo);
            if f_lo.is_finite() && f_lo * f_hi < 0.0 {
                return x_lo;
            }
            width *= 2.0;
        }
        panic!(
            "VandenBosch14: could not bracket the find_psi root below xlgpsi={x_hi} at z={z} \
             after widening to {width} dex"
        );
    }

    fn find_psi(&self, xlgpsi: f64, m0: f64, s0: f64, dc0: f64, delta_dc: f64) -> f64 {
        let psi = 10f64.powf(xlgpsi);
        let s1 = self.variance.sigma(psi * m0).powi(2);

        let omega_fid = APAR[0]
            * (1.0 - APAR[1] * xlgpsi).powf(APAR[2])
            * (1.0 - psi.powf(APAR[3])).powf(APAR[4]);
        let omega = delta_dc / (s1 - s0).sqrt();
        let g_corr = (0.57 * (s1 / s0).powf(0.19) * (dc0 / s0.sqrt()).powf(-0.01)).powf(0.4);

        omega * g_corr - omega_fid
    }
}

impl HaloGrowthModel for VandenBosch14 {
    fn redshift_grid(&self, z0: f64) -> Vec<f64> {
        let log1pz0 = (1.0 + z0).log10();
        (0..N_Z)
            .map(|j| {
                let frac = j as f64 / (N_Z - 1) as f64;
                let xlgz = log1pz0 + frac * LOG_DELTA_Z_MAX + LOG_Z_OFFSET;
                10f64.powf(xlgz) - 1.0
            })
            .collect()
    }

    fn growth_history(&self, log_m0: f64, z0: f64) -> GrowthTrack {
        let m0 = 10f64.powf(log_m0);
        let z = self.redshift_grid(z0);

        let s0 = self.variance.sigma(m0).powi(2);
        let dc0 = self.growth.delta_collapse(z0);

        let mut xlgpsi = 0.0_f64;
        let mut log_mass = Vec::with_capacity(N_Z);
        for &zi in &z {
            let delta_dc = self.growth.delta_collapse(zi) - dc0;

            // At delta_dc == 0 the halo is at its own observation epoch,
            // so psi == 1 by definition (xlgpsi == 0). `find_psi` would
            // otherwise evaluate 0/sqrt(s1-s0) with s1 == s0 -- a 0/0.
            // `LOG_Z_OFFSET` keeps the grid off this point today, but
            // handling it explicitly means the function is total rather
            // than relying on that offset staying in place.
            if delta_dc == 0.0 {
                xlgpsi = 0.0;
                log_mass.push(log_m0);
                continue;
            }

            let x_hi = xlgpsi;
            let x_lo = Self::widen_bracket(
                |x| self.find_psi(x, m0, s0, dc0, delta_dc),
                x_hi,
                zi,
            );
            xlgpsi = ridders_root_find(
                |x| self.find_psi(x, m0, s0, dc0, delta_dc),
                x_lo,
                x_hi,
                1.0e-5,
            );
            log_mass.push(xlgpsi + log_m0);
        }

        GrowthTrack { z, log_mass }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;

    #[test]
    fn growth_history_starts_at_m0() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        let track = model.growth_history(12.0, 0.0);
        assert!((track.log_mass[0] - 12.0).abs() < 1e-3, "log_mass[0] = {}", track.log_mass[0]);
    }

    #[test]
    fn growth_history_is_monotonically_decreasing_into_the_past() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        let track = model.growth_history(13.0, 0.0);
        for w in track.log_mass.windows(2) {
            assert!(w[1] <= w[0] + 1e-6, "mass should not increase going to higher z: {:?}", w);
        }
    }

    #[test]
    fn growth_history_covers_the_expected_redshift_range() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        let track = model.growth_history(12.0, 0.0);
        assert_eq!(track.z.len(), N_Z);
        assert!((track.z[0] - 0.0).abs() < 1e-3);
        // 10^0.85 - 1 =~ 6.08
        assert!((track.z[N_Z - 1] - 6.08).abs() < 0.05, "z_max = {}", track.z[N_Z - 1]);
    }

    #[test]
    fn more_massive_halos_have_steeper_mass_accretion_histories() {
        // Hierarchical "downsizing": massive halos are rare peaks that
        // assemble *later*, so their progenitors are a *smaller*
        // fraction of the z=0 mass at fixed z than less massive halos'
        // progenitors are (van den Bosch 2002; McBride+2009) — a 10^14
        // halo should have lost a larger fraction of its z=0 mass by
        // z=2 than a 10^11 halo has.
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        let small = model.growth_history(11.0, 0.0);
        let large = model.growth_history(14.0, 0.0);
        let z_bin = small.z.iter().position(|&z| z > 2.0).unwrap();
        let frac_small = small.log_mass[z_bin] - 11.0;
        let frac_large = large.log_mass[z_bin] - 14.0;
        assert!(frac_large < frac_small, "frac_large={frac_large} frac_small={frac_small}");
    }
}
