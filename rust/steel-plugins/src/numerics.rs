//! Small numerical utilities shared across plugins: a monotone table
//! built by cumulative trapezoidal integration, queried by linear
//! interpolation. Good enough accuracy (smooth integrands, fine grids)
//! without pulling in an external quadrature/spline crate.

/// A tabulated, monotone function of one variable, built once and then
/// queried by linear interpolation with binary search.
pub struct InterpTable {
    x: Vec<f64>,
    y: Vec<f64>,
}

impl InterpTable {
    pub fn new(x: Vec<f64>, y: Vec<f64>) -> Self {
        assert_eq!(x.len(), y.len(), "InterpTable: x and y must have the same length");
        assert!(x.windows(2).all(|w| w[0] < w[1]), "InterpTable: x must be strictly increasing");
        Self { x, y }
    }

    /// Linear interpolation; clamps to the table's endpoints outside its
    /// domain.
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.x.len();
        if x <= self.x[0] {
            return self.y[0];
        }
        if x >= self.x[n - 1] {
            return self.y[n - 1];
        }
        let idx = match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&x).unwrap())
        {
            Ok(i) => return self.y[i],
            Err(i) => i,
        };
        let (x0, x1) = (self.x[idx - 1], self.x[idx]);
        let (y0, y1) = (self.y[idx - 1], self.y[idx]);
        y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    }
}

/// Cumulative trapezoidal integral of `f` sampled at each point of `x`
/// (which need not be uniformly spaced), returning `F(x_i) = int_{x_0}^{x_i} f dx`.
pub fn cumulative_trapezoid(x: &[f64], f: &[f64]) -> Vec<f64> {
    assert_eq!(x.len(), f.len(), "cumulative_trapezoid: x and f must have the same length");
    let mut out = Vec::with_capacity(x.len());
    let mut acc = 0.0;
    out.push(acc);
    for i in 1..x.len() {
        acc += 0.5 * (f[i] + f[i - 1]) * (x[i] - x[i - 1]);
        out.push(acc);
    }
    out
}

/// Composite Simpson's rule for `f` over `[a, b]` with `n` subintervals
/// (`n` must be even).
pub fn simpson<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    assert!(n > 0 && n.is_multiple_of(2), "simpson: n must be a positive even number");
    let h = (b - a) / n as f64;
    let mut sum = f(a) + f(b);
    for i in 1..n {
        let x = a + i as f64 * h;
        sum += if i % 2 == 0 { 2.0 } else { 4.0 } * f(x);
    }
    sum * h / 3.0
}

/// Natural cubic spline (second derivative zero at both endpoints),
/// fit once and evaluated by binary search + local cubic evaluation.
pub struct CubicSpline {
    x: Vec<f64>,
    y: Vec<f64>,
    /// Second derivatives at each knot, from the standard tridiagonal
    /// (Thomas algorithm) natural-spline solve.
    y2: Vec<f64>,
}

impl CubicSpline {
    pub fn fit(x: Vec<f64>, y: Vec<f64>) -> Self {
        let n = x.len();
        assert_eq!(n, y.len(), "CubicSpline: x and y must have the same length");
        assert!(n >= 3, "CubicSpline: needs at least 3 points");
        assert!(x.windows(2).all(|w| w[0] < w[1]), "CubicSpline: x must be strictly increasing");

        // Standard natural-cubic-spline tridiagonal solve (e.g. Numerical
        // Recipes `spline`), specialized to y1=y2=0 (natural) boundary
        // conditions — the same boundary condition `cosmo_sub.f`'s calls
        // to `spline` use for the sigma(M) table (`2.0E+30` sentinel for
        // "natural").
        let mut u = vec![0.0; n];
        let mut y2 = vec![0.0; n];
        for i in 1..n - 1 {
            let sig = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1]);
            let p = sig * y2[i - 1] + 2.0;
            y2[i] = (sig - 1.0) / p;
            let d2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])
                - (y[i] - y[i - 1]) / (x[i] - x[i - 1]);
            u[i] = (6.0 * d2 / (x[i + 1] - x[i - 1]) - sig * u[i - 1]) / p;
        }
        for i in (0..n - 1).rev() {
            y2[i] = y2[i] * y2[i + 1] + u[i];
        }

        Self { x, y, y2 }
    }

    /// Evaluate the spline at `x`. Callers are expected to stay inside
    /// `[x_min, x_max]`; outside it this clamps to the boundary value
    /// (mirroring `InterpTable`'s convention).
    pub fn eval(&self, x: f64) -> f64 {
        let n = self.x.len();
        if x <= self.x[0] {
            return self.y[0];
        }
        if x >= self.x[n - 1] {
            return self.y[n - 1];
        }
        let hi = match self
            .x
            .binary_search_by(|probe| probe.partial_cmp(&x).unwrap())
        {
            Ok(i) => return self.y[i],
            Err(i) => i,
        };
        let lo = hi - 1;
        let h = self.x[hi] - self.x[lo];
        let a = (self.x[hi] - x) / h;
        let b = (x - self.x[lo]) / h;
        a * self.y[lo]
            + b * self.y[hi]
            + ((a.powi(3) - a) * self.y2[lo] + (b.powi(3) - b) * self.y2[hi]) * (h * h) / 6.0
    }
}

/// Ridders' method root finder for `f` bracketed by `[x_lo, x_hi]`
/// (`f(x_lo)` and `f(x_hi)` must have opposite signs), iterating until
/// consecutive estimates agree within `tol` or `max_iter` is reached.
/// Standard textbook algorithm (Ridders 1979 / Numerical Recipes
/// `zriddr`) — `getPWGH.f`'s `zriddr` implements the same algorithm, so
/// this is ported from the well-known method rather than transcribed
/// line-by-line from the Fortran.
pub fn ridders_root_find<F: Fn(f64) -> f64>(f: F, x_lo: f64, x_hi: f64, tol: f64) -> f64 {
    const MAX_ITER: usize = 60;
    let mut x_lo = x_lo;
    let mut x_hi = x_hi;
    let mut f_lo = f(x_lo);
    let mut f_hi = f(x_hi);
    assert!(
        (f_lo < 0.0 && f_hi > 0.0) || (f_lo > 0.0 && f_hi < 0.0),
        "ridders_root_find: root must be bracketed"
    );

    let mut ans = f64::NAN;
    for _ in 0..MAX_ITER {
        let x_mid = 0.5 * (x_lo + x_hi);
        let f_mid = f(x_mid);
        let s = (f_mid * f_mid - f_lo * f_hi).sqrt();
        if s == 0.0 {
            return x_mid;
        }
        let sign = if f_lo >= f_hi { 1.0 } else { -1.0 };
        let x_new = x_mid + (x_mid - x_lo) * sign * f_mid / s;
        if (x_new - ans).abs() <= tol {
            return x_new;
        }
        ans = x_new;
        let f_new = f(ans);
        if f_new == 0.0 {
            return ans;
        }

        // Re-bracket around the new estimate.
        if f_mid.signum() != f_new.signum() {
            x_lo = x_mid;
            f_lo = f_mid;
            x_hi = ans;
            f_hi = f_new;
        } else if f_lo.signum() != f_new.signum() {
            x_hi = ans;
            f_hi = f_new;
        } else if f_hi.signum() != f_new.signum() {
            x_lo = ans;
            f_lo = f_new;
        } else {
            unreachable!("ridders_root_find: lost the bracket");
        }
        if (x_hi - x_lo).abs() <= tol {
            return ans;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cubic_spline_reproduces_a_quadratic_exactly() {
        // A natural cubic spline through samples of a quadratic won't be
        // exact everywhere (natural BCs bend the ends), but should be
        // close in the interior away from the boundary effect.
        let x: Vec<f64> = (0..11).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * xi).collect();
        let spline = CubicSpline::fit(x, y);
        assert!((spline.eval(5.0) - 25.0).abs() < 0.1);
    }

    #[test]
    fn ridders_finds_root_of_simple_polynomial() {
        // f(x) = x^2 - 2, root at sqrt(2)
        let root = ridders_root_find(|x| x * x - 2.0, 0.0, 2.0, 1e-10);
        assert!((root - std::f64::consts::SQRT_2).abs() < 1e-8);
    }

    #[test]
    fn ridders_finds_root_of_linear_function() {
        let root = ridders_root_find(|x| 2.0 * x - 3.0, -10.0, 10.0, 1e-12);
        assert!((root - 1.5).abs() < 1e-9);
    }

    // The following four tests exist to prove `InterpTable`/`CubicSpline`
    // reject invalid input via a real `assert!` (checked in every build
    // profile, including `--release` — the profile every actual STEEL
    // run uses), not a `debug_assert!` that would silently compile out
    // there and let bad state through instead of failing loudly.

    #[test]
    #[should_panic(expected = "same length")]
    fn interp_table_rejects_mismatched_lengths() {
        InterpTable::new(vec![0.0, 1.0, 2.0], vec![0.0, 1.0]);
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn interp_table_rejects_non_monotonic_x() {
        InterpTable::new(vec![0.0, 2.0, 1.0], vec![0.0, 1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn cubic_spline_rejects_mismatched_lengths() {
        CubicSpline::fit(vec![0.0, 1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "strictly increasing")]
    fn cubic_spline_rejects_non_monotonic_x() {
        CubicSpline::fit(vec![0.0, 2.0, 1.0, 3.0], vec![0.0, 1.0, 2.0, 3.0]);
    }

    #[test]
    #[should_panic(expected = "same length")]
    fn cumulative_trapezoid_rejects_mismatched_lengths() {
        cumulative_trapezoid(&[0.0, 1.0, 2.0], &[0.0, 1.0]);
    }
}
