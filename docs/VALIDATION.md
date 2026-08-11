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

* **`master` writes identically zero pair fractions.** Not
  approximately — `np.count_nonzero` is 0. The block sits inside
  `if len(np.shape(SM_Sat)) == 1:` with no `else`, and `SM_Sat` is 2-D
  exactly when stripping or star formation is on (correction B1).
  **This is a `master`-only defect**: the `PipGrylls`, `Paper2`,
  `Refactor`, `haofu` and `saiduc` branches all carry the missing
  `else:` branch, and `Paper2`'s tip (2020-01-17) is contemporaneous
  with Paper 3's submission. So the published pair fractions are not
  affected; the repository's *public default branch* simply cannot
  reproduce them. See `docs/PORT_CORRECTIONS.md` B1.
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

## 2b. Full published resolution — does the reduced grid mislead?

Everything above is the reduced grid, whose host halos stop at
`log M = 12.6`. STEEL is a model of satellites in groups and clusters,
and the corrections that bite hardest are integrals of satellite counts,
which scale with richness — so the reduced grid tests them in the regime
where they should be *weakest*. Repeated at the published resolution
(`log M = 11.0 … 16.6`, 0.1 dex; hosts spanning `log M = 10.81–16.28`,
i.e. including clusters):

| output | py-as-is | py-corrected | rs-steel | corr/as-is | rs/corr |
|---|---|---|---|---|---|
| **frozen** — `('1.0', False, False, True, 'CE', 'G18')` ||||||
| `Figure3_AnalyticalModel_SMF` | 5.3352e-02 | 5.3139e-02 | 5.2904e-02 | 0.9960 | 0.9956 |
| `Figure4_6_AnalyticalModelNoFrac_` | 9.7037e-03 | 1.0964e-02 | 1.0916e-02 | **1.1299** | 0.9956 |
| `Sat_Env_Highz_AnalyticalModelNoFracHighz` | 6.0875e-01 | 7.0036e-01 | 6.9535e-01 | **1.1505** | 0.9928 |
| `Pair_Frac_Pair_Frac` | 4.3937e+04 | 4.3908e+04 | 4.3649e+04 | 0.9994 | 0.9941 |
| **with baryons** — `('1.0', True, True, True, 'G19_DPL', 'G19_SE')` ||||||
| `Figure3_AnalyticalModel_SMF` | 6.2069e-02 | 6.1327e-02 | 6.1274e-02 | 0.9880 | 0.9991 |
| `Figure4_6_AnalyticalModelNoFrac_` | 1.1495e-02 | 1.2486e-02 | 1.2472e-02 | **1.0862** | 0.9988 |
| `Sat_Env_Highz_AnalyticalModelNoFracHighz` | 7.9529e-01 | 9.0370e-01 | 8.9858e-01 | **1.1363** | 0.9943 |
| `Pair_Frac_Pair_Frac` | **0.0000e+00** | 5.1103e+04 | 5.0855e+04 | — | 0.9951 |
| `Mergers_Accretion_History` | 7.5654e+03 | 7.5313e+03 | 7.4868e+03 | 0.9955 | 0.9941 |

The reduced grid did understate it, though not dramatically: the
richness correction grows from +4.7% to **+8.6%** on `Figure4_6` and
from +12.5% to **+13.6%** on `Sat_Env_Highz`. The headline satellite SMF
moves from "unchanged" to **−1.2%**, still small.

**The two corrected implementations agree to 0.1%–0.7%** at full
resolution — tighter than on the reduced grid, as expected, since 57
host bins and 65 subhalo bins average away far more Monte-Carlo noise
than 4 × 5 did.

### The richness correction is strongly cut-dependent

Per stellar-mass cut, `Figure4_6_AnalyticalModelNoFrac_`, corrected /
as-is:

| SM cut | frozen | with baryons |
|---|---|---|
| 9.0 | 1.1161 | 1.1034 |
| 9.5 | 1.1222 | 1.0542 |
| 10.0 | 1.1270 | 1.0437 |
| 10.5 | 1.1987 | 1.1017 |
| **11.0** | **1.5202** | **1.7056** |
| 11.45 | 0.9425 | 1.0994 |

**+52% and +71% at the `log M* = 11.0` cut.** That is the steep end of
the satellite mass function, where the one bin C1 restores carries a
large share of the integral.

`11.45` is the control, and it works: it is the only cut of the six that
does *not* sit on a bin edge, so `np.digitize` and the histogram
convention agree there and C1 has no effect by construction (pinned by
`cut_bin_index(11.45, …) == 25` in the Rust tests). Its residual 0.94 /
1.10 is everything *except* C1 — other corrections plus Monte-Carlo
noise — and it is the odd one out in both columns, exactly as it should
be.

### Which figure you are looking at decides whether this matters

`Figure4_6` is saved twice: `NoFrac_` (absolute satellite counts above
the cut, per host bin) and `Frac_` (the same, normalised across host
bins). Paper 1's Fig. 4 and 6 plot the **normalised** distribution, and
it is far more robust — the correction adds roughly one bin at every
host mass, so it largely divides out:

| SM cut | max shift in normalised fraction, as % of the distribution's peak |
|---|---|
| 9.0 | 2.8% / 2.9% |
| 9.5 | 3.7% / 2.3% |
| 10.0 | 6.0% / 2.4% |
| 10.5 | 8.7% / 6.2% |
| 11.0 | **17.7% / 16.9%** |
| 11.45 | 8.1% / 10.1% |

(frozen / with baryons). So the satellite *distributions* shift by a few
percent of peak at the low cuts and ~17% at `log M* = 11.0`; the
absolute *richnesses* shift by 10–71%.

### Pair fractions

At full resolution py-as-is's `Pair_Frac` is **0 nonzero cells out of
433 200**, `sum = 0.0`, `min = max = 0.0`. Any pair fraction derived
from it is identically zero — it is a ratio with an exactly-zero
numerator, which is arithmetic rather than an approximation.

The corrected output gives physically sensible values. Running the real
`CentralPostprocessing.PairFractionData.Return_PF_Plot` on it:

| central mass cut | pair fraction at z ≈ 0.1 | range over 0.11 < z < 6 |
|---|---|---|
| `log M* > 10` | 0.0017 | 0 – 0.083 |
| `log M* > 11` | 0.0141 | 0.013 – 0.099 |

A per-cent-level pair fraction at low redshift rising to ~10% by high
redshift is the expected behaviour for the quantity Paper 3 compares
against Mundy+2017. That is an eyeball judgement, not a measurement:
**no published figure has been digitised and compared here**, so this
does not establish that either implementation reproduces Paper 3.

### The 2-D pair-fraction branch — now validated against a reference

Correction B1 required *writing* the branch that handles an evolved
satellite, because `master` has none. That made it the least-validated
part of this work: py-corrected and rs-steel agreeing to 0.5% was
self-consistency (same author, same reading), not corroboration.

A reference does exist — on the branches `master` does not merge.
`Paper2` (tip 2020-01-17) implements it as:

```python
else:
    Counterpart = np.multiply(np.ones_like(SM_Sat), np.arange(z_bin,i,1)).T
    Wt_Corr = np.flipud(np.divide(histogram2d(
        Counterpart.flatten(), SM_Sat.T.flatten(),
        (i-z_bin, SatM_len), ((z_bin, i), (SatM_min, SatM_max))), N))[PF_bin_l:PF_bin_u]
    Corr = np.divide(np.multiply(WeightList_SubOnly[PF_bin_l:PF_bin_u], Wt_Corr.T).T, SatBin)
```

Run against the reconstruction on identical input, the two are
**bit-identical** — `np.array_equal` → `True`, max |diff| exactly 0.
They differ only in whether the row slice is taken before or after the
histogram, which is equivalent. The reconstruction is correct.

The frozen-config check stands as before: 0.06% on the integral against
the original 1-D path, which shows nothing was broken there.

---

## 3. Performance

Same reduced grid, same machine, single-threaded:

Reduced grid:

| | wall clock |
|---|---|
| py-as-is | 12.0 s |
| py-corrected | 11.2 s |
| rs-steel | 1.4 s |

Full published resolution (190 × 57 × 65), measured rather than
extrapolated:

| config | py-as-is | py-corrected | rs-steel | speedup |
|---|---|---|---|---|
| frozen | 276.3 s | 245.6 s | 5.7 s | **48x** |
| stripping + star formation | 514.0 s | 498.9 s | 61.0 s | **8.4x** |

An earlier extrapolation from the reduced grid put the Python at ~45
min; measured, it is 8.5 min. The extrapolation was wrong because the
reduced grid's cost is dominated by fixed start-up, not by the loop.

Caveat on reading the speedups: the two Python configs were run
concurrently on a 4-core machine and so contended for CPU, while the
Rust runs were sequential and had the machine to themselves. The Rust is
also single-threaded, so this is not a parallelism advantage.

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
