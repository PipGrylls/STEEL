//! Eisenstein & Hu (1998) CDM+baryon transfer function and the linear
//! matter mass variance sigma(M) built from it.
//!
//! Formulas transcribed directly from `Functions/OtherModels/VDB13/cosmo_sub.f`
//! (`init_cosmology` for the fixed EH98 coefficients, `power_spec` for
//! the per-k transfer function, `var_numerical`/`toint1` for the
//! sigma(M) integral) — read from source, not recalled from memory.

use steel_core::cosmology::Cosmology;

use crate::numerics::{simpson, CubicSpline};

/// Fixed CMB-temperature parameter EH98 hardcodes (`theta = 1.0093` in
/// `cosmo_sub.f::init_cosmology`).
const THETA: f64 = 1.0093;

/// Cosmology-dependent coefficients of the Eisenstein & Hu (1998)
/// transfer function, computed once from the background cosmology.
struct Eh98Params {
    f_baryon: f64,
    k_eq: f64,
    k_silk: f64,
    bnode: f64,
    sound_horizon: f64,
    alpha_c: f64,
    beta_c: f64,
    alpha_b: f64,
    beta_b: f64,
}

impl Eh98Params {
    fn new(cosmology: &dyn Cosmology) -> Self {
        let h = cosmology.h();
        let omega_m0 = cosmology.omega_m0();
        let omega_b0 = cosmology.omega_b0();
        let f = omega_m0 * h * h; // Omega_m0 h^2
        let omega_b_h2 = omega_b0 * h * h;
        let f_baryon = omega_b0 / omega_m0;

        let b1 = 0.313 * f.powf(-0.419) * (1.0 + 0.607 * f.powf(0.674));
        let b2 = 0.238 * f.powf(0.223);
        let bnode = 8.41 * f.powf(0.435);
        let k_eq = 7.46e-2 * f / THETA.powi(2);
        let k_silk = 1.6 * omega_b_h2.powf(0.52) * f.powf(0.73) * (1.0 + (10.4 * f).powf(-0.95));

        let z_eq = 2.5e4 * f / THETA.powi(4);
        let z_d = 1291.0 * (f.powf(0.251) / (1.0 + 0.659 * f.powf(0.828)))
            * (1.0 + b1 * omega_b_h2.powf(b2));

        let y = (1.0 + z_eq) / (1.0 + z_d);
        let g_y = y
            * (-6.0 * (1.0 + y).sqrt()
                + (2.0 + 3.0 * y) * (((1.0 + y).sqrt() + 1.0) / ((1.0 + y).sqrt() - 1.0)).ln());

        let r_eq = 31.5 * omega_b_h2 * (1000.0 / z_eq) / THETA.powi(4);
        let r_d = 31.5 * omega_b_h2 * (1000.0 / z_d) / THETA.powi(4);

        let sound_horizon = (2.0 / (3.0 * k_eq)) * (6.0 / r_eq).sqrt()
            * (((1.0 + r_d).sqrt() + (r_d + r_eq).sqrt()) / (1.0 + r_eq.sqrt())).ln();

        let a1 = (46.9 * f).powf(0.670) * (1.0 + (32.1 * f).powf(-0.532));
        let a2 = (12.0 * f).powf(0.424) * (1.0 + (45.0 * f).powf(-0.582));
        let b1c = 0.944 / (1.0 + (458.0 * f).powf(-0.708));
        let b2c = (0.395 * f).powf(-0.0266);

        let alpha_c = a1.powf(-f_baryon) * a2.powf(-f_baryon.powi(3));
        let beta_c = 1.0 / (1.0 + b1c * ((1.0 - f_baryon).powf(b2c) - 1.0));

        let alpha_b = 2.07 * k_eq * sound_horizon * (1.0 + r_d).powf(-0.75) * g_y;
        let beta_b = 0.5 + f_baryon + (3.0 - 2.0 * f_baryon) * ((17.2 * f).powi(2) + 1.0).sqrt();

        Self {
            f_baryon,
            k_eq,
            k_silk,
            bnode,
            sound_horizon,
            alpha_c,
            beta_c,
            alpha_b,
            beta_b,
        }
    }

    /// T(k) at physical wavenumber `k_mpc` \[Mpc^-1\] (already converted
    /// from `h`/Mpc — see [`Eh98Params::transfer_function`]).
    fn transfer_at_physical_k(&self, k: f64) -> f64 {
        let s = self.sound_horizon;
        let q = k / (13.41 * self.k_eq);
        let s_tilde = s / (1.0 + (self.bnode / (k * s)).powi(3)).cbrt();

        let c1 = 14.2 + 386.0 / (1.0 + 69.9 * q.powf(1.08));
        let c2 = 14.2 / self.alpha_c + 386.0 / (1.0 + 69.9 * q.powf(1.08));
        let t11 = (std::f64::consts::E + 1.8 * self.beta_c * q).ln();
        let t12 = (std::f64::consts::E + 1.8 * q).ln();

        let f_win = 1.0 / (1.0 + (k * s / 5.4).powi(4));
        let t1 = t11 / (t11 + c1 * q * q);
        let t2 = t11 / (t11 + c2 * q * q);
        let t3 = t12 / (t12 + c1 * q * q);

        let t_c = f_win * t1 + (1.0 - f_win) * t2;

        let tb1 = t3 / (1.0 + (k * s / 5.2).powi(2));
        let tb2 = (self.alpha_b / (1.0 + (self.beta_b / (k * s)).powi(3)))
            * (-((k / self.k_silk).powf(1.4))).exp();

        let ks_tilde = k * s_tilde;
        let sinc = if ks_tilde.abs() < 1e-8 { 1.0 } else { ks_tilde.sin() / ks_tilde };
        let t_b = (tb1 + tb2) * sinc;

        self.f_baryon * t_b + (1.0 - self.f_baryon) * t_c
    }

    /// T(k) at `k_h_mpc` \[h/Mpc\] — the convention every other STEEL
    /// quantity (masses, radii) uses.
    fn transfer_function(&self, k_h_mpc: f64, h: f64) -> f64 {
        self.transfer_at_physical_k(k_h_mpc * h)
    }
}

/// Top-hat window function in Fourier space, `W(x) = 3(sin x - x cos x)/x^3`.
fn top_hat_window(x: f64) -> f64 {
    if x.abs() < 1e-6 {
        // W(x) -> 1 - x^2/10 as x -> 0; avoid the 0/0 cancellation.
        return 1.0 - x * x / 10.0;
    }
    3.0 * (x.sin() - x * x.cos()) / x.powi(3)
}

/// The linear matter mass variance sigma(M) at z=0, and its logarithmic
/// mass derivative, both needed by [`crate::halo_growth::VandenBosch14`]
/// and [`crate::hmf::Despali16`].
pub struct Variance {
    eh98: Eh98Params,
    h: f64,
    n_spec: f64,
    /// rho_bar_m in Msun h^2 Mpc^-3 (comoving mean matter density).
    rho_bar_m: f64,
    /// Calibration factor: sigma8 / sigma_raw(M8), so that
    /// `sigma(M) = calibration * sigma_raw(M)`.
    calibration: f64,
    /// sigma(M), tabulated over log10(M) in [4, 15.5] \[Msun/h\]
    /// (matching `paramfile.h`'s `Mminvar`/`Mmaxvar`/`Nsigma=1000`) and
    /// cubic-spline interpolated; queries outside that range fall back
    /// to the direct numerical integral, mirroring `variance()`'s
    /// branch in `cosmo_sub.f`.
    table: CubicSpline,
    log10_m_min: f64,
    log10_m_max: f64,
}

impl Variance {
    const N_TABLE: usize = 1000;
    const LOG10_M_MIN: f64 = 4.0;
    const LOG10_M_MAX: f64 = 15.5;
    /// log10(k) grid [h/Mpc] for the sigma(M) integral: wide enough to
    /// cover every R of interest (k ~ 1/R) across the table's mass
    /// range, dense enough that composite Simpson's rule converges.
    const N_K: usize = 4000;
    const LOG10_K_MIN: f64 = -6.0;
    const LOG10_K_MAX: f64 = 4.0;

    pub fn new(cosmology: &dyn Cosmology) -> Self {
        let eh98 = Eh98Params::new(cosmology);
        let h = cosmology.h();
        let n_spec = cosmology.n_spec();
        // rho_crit(z=0) [Msun h^2 kpc^-3] -> Msun h^2 Mpc^-3 (1 Mpc = 1000 kpc).
        let rho_bar_m = cosmology.omega_m0() * cosmology.rho_crit(0.0) * 1.0e9;

        let mut v = Self {
            eh98,
            h,
            n_spec,
            rho_bar_m,
            calibration: 1.0,
            table: CubicSpline::fit(vec![0.0, 1.0, 2.0], vec![0.0, 0.0, 0.0]), // placeholder
            log10_m_min: Self::LOG10_M_MIN,
            log10_m_max: Self::LOG10_M_MAX,
        };

        // Calibrate against sigma8 at M8 = 5.9543e14 * Omega_m0 [Msun/h]
        // (cosmo_sub.f:566, init_variance:508), then build the table
        // with that calibration applied.
        let m8 = 5.9543e14 * cosmology.omega_m0();
        let raw_at_m8 = v.sigma_raw(m8);
        v.calibration = cosmology.sigma8() / raw_at_m8;

        let log10_m: Vec<f64> = (0..Self::N_TABLE)
            .map(|i| Self::LOG10_M_MIN + (Self::LOG10_M_MAX - Self::LOG10_M_MIN) * i as f64 / (Self::N_TABLE - 1) as f64)
            .collect();
        let sigma_vals: Vec<f64> = log10_m
            .iter()
            .map(|&lm| v.calibration * v.sigma_raw(10f64.powf(lm)))
            .collect();
        v.table = CubicSpline::fit(log10_m, sigma_vals);

        v
    }

    fn radius_mpc_h(&self, m_msun_h: f64) -> f64 {
        (3.0 * m_msun_h / (4.0 * std::f64::consts::PI * self.rho_bar_m)).cbrt()
    }

    /// Un-calibrated (arbitrary normalization) sigma(M), from direct
    /// numerical integration in ln(k).
    fn sigma_raw(&self, m_msun_h: f64) -> f64 {
        let r = self.radius_mpc_h(m_msun_h);
        let integrand = |ln_k: f64| {
            let k = ln_k.exp();
            let t = self.eh98.transfer_function(k, self.h);
            let p_k = t * t * k.powf(self.n_spec);
            let w = top_hat_window(k * r);
            p_k * w * w * k.powi(3) // extra k from d(ln k) = dk/k, so k^2 dk = k^3 d(ln k)
        };
        let ln_k_min = Self::LOG10_K_MIN * std::f64::consts::LN_10;
        let ln_k_max = Self::LOG10_K_MAX * std::f64::consts::LN_10;
        let integral = simpson(integrand, ln_k_min, ln_k_max, Self::N_K);
        (integral / (2.0 * std::f64::consts::PI.powi(2))).sqrt()
    }

    /// sigma(M), the rms linear-theory density fluctuation in spheres of
    /// mass `m_msun_h` \[Msun/h\] at z=0.
    pub fn sigma(&self, m_msun_h: f64) -> f64 {
        let log10_m = m_msun_h.log10();
        if log10_m < self.log10_m_min || log10_m > self.log10_m_max {
            self.calibration * self.sigma_raw(m_msun_h)
        } else {
            self.table.eval(log10_m)
        }
    }

    /// `d ln(sigma^2) / d ln(M)`, central difference with `eps=0.15`
    /// (matching `cosmo_sub.f::dlnSdlnM`).
    pub fn dln_sigma2_dln_m(&self, m_msun_h: f64) -> f64 {
        const EPS: f64 = 0.15;
        let m1 = (1.0 - EPS) * m_msun_h;
        let m2 = (1.0 + EPS) * m_msun_h;
        let s1 = self.sigma(m1).powi(2);
        let s2 = self.sigma(m2).powi(2);
        (s2.ln() - s1.ln()) / (m2.ln() - m1.ln())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;

    #[test]
    fn sigma_decreases_with_mass() {
        let cosmo = Planck15::new();
        let v = Variance::new(&cosmo);
        let s_small = v.sigma(1e8);
        let s_mid = v.sigma(1e12);
        let s_large = v.sigma(1e15);
        assert!(s_small > s_mid && s_mid > s_large, "{s_small} {s_mid} {s_large}");
    }

    #[test]
    fn sigma_at_8_mpc_h_matches_sigma8_calibration() {
        // sigma(M8) should equal sigma8 by construction.
        let cosmo = Planck15::new();
        let v = Variance::new(&cosmo);
        let m8 = 5.9543e14 * cosmo.omega_m0();
        let s8 = v.sigma(m8);
        assert!((s8 - cosmo.sigma8()).abs() < 1e-6, "sigma(M8) = {s8}, expected {}", cosmo.sigma8());
    }

    #[test]
    fn dln_sigma2_dln_m_is_negative() {
        // sigma decreases with mass, so its log derivative is negative.
        let cosmo = Planck15::new();
        let v = Variance::new(&cosmo);
        let d = v.dln_sigma2_dln_m(1e12);
        assert!(d < 0.0, "dlnS/dlnM = {d}, expected negative");
    }
}
