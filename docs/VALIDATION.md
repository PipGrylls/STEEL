# Three-way validation results

py-as-is / py-corrected / rs-steel, run on one configuration and
compared. Reproduce with `Scripts/Validation/three_way.py`; see
`Scripts/Validation/README.md` for the mechanics and
`docs/PORT_CORRECTIONS.md` for what "corrected" means.

**Configuration.** `('1.0', True, True, True, 'G19_DPL', 'G19_SE')` —
stripping and star formation on, the family Papers 2 and 3 are built on.
Reduced grid (`log M = 11.0 … 12.6`, 0.5 dex; 190 redshift steps, 4 host
bins, 5 subhalo bins) so a Python run finishes in ~12 s rather than
~45 min. Magnitudes below are grid-specific; signs and mechanisms are
not.

---

## 1. Deterministic mode — is the port numerically faithful?

Scatter off on both sides (`STEEL_SCATTER=0` / `[run] scatter = false`),
so both implementations evaluate the same arithmetic on the same grid
and any disagreement is real rather than Monte-Carlo noise.

py-as-is cannot take part: it has no scatter switch even in principle
(`GetGasMass` scatters unconditionally — correction A7), which is itself
a finding.

Reverse-cumulative agreement along the stellar-mass axis, py-corrected
vs rs-steel:

| output | median | p90 | integral ratio |
|---|---|---|---|
| `Figure3_AnalyticalModel_SMF` | 0.27% | 24% | 1.0015 |
| `Figure10_AnalyticalModel_SMF` | 0.27% | 24% | 1.0015 |
| `SMFhz_AnalyticalModel_SMF_Highz` | 0.35% | 1.7% | 0.9981 |
| `Raw_Richness_..._highz` | 0.40% | 2.5% | 0.9981 |
| `Sat_SMHM_Sat_SMHM` | 0.34% | 1.6% | 0.9981 |
| `Mergers_Accretion_History` | 0.25% | 1.4% | 0.9995 |
| `Pair_Frac_Pair_Frac` | 0.22% | 3.2% | 0.9988 |
| `z_infall` | 0.23% | 0.45% | 1.0015 |

**Why cumulative and not per-bin.** With scatter off, every realization
of a given (redshift, host, subhalo) bin lands on the *same* stellar
mass — the distribution is a delta function. The residual ~0.01 dex
halo-mass difference between the two cosmology implementations can
therefore move an entire bin's weight to its neighbour, which reads as a
100% per-bin deviation while the underlying physics agrees. Deterministic
mode is *more* sensitive to that than stochastic mode, not less. The
reverse-cumulative distribution is what the papers integrate anyway and
is insensitive to a value crossing a bin edge.

The residual itself is the cosmology and halo-growth port, already
measured independently at **0.009 dex** against a freshly compiled
`getPWGH` (Milestone 2), and visible here as `AvaHaloMass` differing by
0.1% and `z` by 0.04%.

**Isolated from the cosmology**, the baryonic pipeline agrees far more
tightly. Driving the committed Cython on the Rust's own `age(z)` grid
(`Scripts/Validation/reference_baryonic.py`):

* stripped satellite, gas cap live: **1e-9 over all 11 steps**, unadjusted
* unstripped, gas cap neutralised: **1e-9 over all 11 steps**
* unstripped, gas cap live: diverges from step 7 — that is correction A6

---

## 2. Stochastic mode — what do the corrections change?

Scatter on, ensemble mean over 5 seeds. The three legs draw from
unrelated generators (NumPy Mersenne Twister, GSL taus, `rand` ChaCha)
so they can never agree element-wise; only ensemble statistics are
comparable.

Integrated over each output:

| output | py-as-is | py-corrected | rs-steel | corr/as-is | rs/corr |
|---|---|---|---|---|---|
| `Figure3_AnalyticalModel_SMF` | 1.3146e-02 | 1.3141e-02 | 1.3166e-02 | 1.0000 | 1.0019 |
| `SMFhz_AnalyticalModel_SMF_Highz` | 1.2843e+00 | 1.2858e+00 | 1.2823e+00 | 1.0012 | 0.9973 |
| `Mergers_Accretion_History` | 1.7477e+01 | 1.7384e+01 | 1.7383e+01 | 0.9947 | 0.9999 |
| **`Pair_Frac_Pair_Frac`** | **0.0000e+00** | 1.1591e+02 | 1.1562e+02 | — | 0.9974 |
| **`Pair_Frac_Halo_Pair_Frac_Halo`** | **0.0000e+00** | 1.8619e+03 | 1.8609e+03 | — | 0.9995 |
| `Satellite_sSFR` | 1.3146e-03 | 1.3141e-03 | 1.2942e-03 | 1.0000 | 0.9848 |
| `Figure4_6_AnalyticalModelNoFrac_` | 2.3092e-03 | 2.4182e-03 | 2.4430e-03 | 1.0472 | 1.0103 |
| `Sat_Env_Highz_AnalyticalModelNoFracHighz` | 1.8214e-01 | 2.0485e-01 | 2.0479e-01 | 1.1247 | 0.9997 |
| `z_infall` | 1.3146e-02 | 1.3141e-02 | 1.3166e-02 | 1.0000 | 1.0019 |

**The two corrected implementations agree to 0.03%–1.5% on every
output** — two independent implementations, in different languages, with
independently written corrections, converging on the same numbers.
That is the validation.

**What the corrections move:**

* **Pair fractions were identically zero.** Not approximately —
  `np.count_nonzero` is 0. The pair-fraction block sits inside
  `if len(np.shape(SM_Sat)) == 1:` with no `else`, and `SM_Sat` is 2-D
  exactly when stripping or star formation is on. So the configurations
  Papers 2 and 3 interpret wrote nothing (correction B1).
* **Richness integrals move by +4.7% and +12.5%** — `np.digitize` used
  as a `fast_histogram` bin index, so "satellites above log M* = X"
  started at X + 0.1 (C1).
* **Everything else moves by under 0.5%**, which is the reassuring half
  of the result: the satellite stellar mass functions the papers'
  headline figures show are not materially changed by any correction.

`Satellite_sSFR`'s 1.5% rs/corr difference is the largest remaining gap
and is expected: it is the one output whose *binning* differs between the
legs (correction C2 gives it 60 bins of 0.1 dex against py-as-is's 59 of
0.1017), so the two are not binning the same way even after correction.

---

## 3. Performance

Same reduced grid, same machine, single-threaded:

| | wall clock |
|---|---|
| py-as-is | 12.0 s |
| py-corrected | 11.2 s |
| rs-steel | 1.4 s |

At full published resolution (`log M = 11.0 … 16.6`, 0.1 dex — 190 × 57
× 65 bins) rs-steel takes **3.9 s** frozen and **61 s** with stripping
and star formation on. The Python at that resolution is ~45 min,
extrapolated.

One caveat on reading that as a language comparison: a large part of the
Rust's margin here is an algorithmic change, not code generation. The
halo mass function is tabulated once on the (redshift, host bin) grid
rather than evaluated per window step — the same move
`Functions.py::Make_HMF_Interp` makes with an interpolation table. Before
that hoist the full-resolution Rust run took over 10 minutes.

---

## 4. Caveats

* The reduced grid stops at `log M = 12.6`, so the massive-satellite tail
  is sparsely populated and ratios in it are noisy. Full-resolution
  numbers will differ.
* No observational data exists in the repository, so nothing here is
  compared against the data overlays in the published figures. This is a
  model-vs-model comparison throughout.
* py-as-is is byte-for-byte the committed `STEEL.py`, run on a
  period-correct Python 3.10 / NumPy 1.22 stack (`env/py-asis`). It does
  not run on any NumPy ≥ 1.23 — see `env/README.md`.
