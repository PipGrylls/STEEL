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
        debug_assert_eq!(x.len(), y.len());
        debug_assert!(x.windows(2).all(|w| w[0] < w[1]), "x must be strictly increasing");
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
    debug_assert_eq!(x.len(), f.len());
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
/// (`n` must be even). Not yet called anywhere — kept for Milestone 2's
/// sigma(M) mass-variance integral, which needs a quadrature routine
/// this generic.
#[allow(dead_code)]
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
