# Three-way validation results

py-as-is / py-corrected / rs-steel, run on one configuration and
compared. Reproduce with `Scripts/Validation/three_way.py`; see
`Scripts/Validation/README.md` for the mechanics and
`docs/PORT_CORRECTIONS.md` for what "corrected" means.

**Baseline: `PipGrylls`, not `master`.** `master`'s tip predates all
three papers; `PipGrylls` carries the commit labelled *"the version of
the code used for the 1st submission of Paper2"*. Every number on this
page was re-measured after that rebaseline — the earlier figures, taken
against `master`, were measuring code the papers were not run from. See
`docs/PORT_CORRECTIONS.md` § "Which baseline this is measured against".

The three legs:

| Leg | What | MAH cosmology |
|---|---|---|
| py-as-is | detached worktree at `origin/PipGrylls`, byte-for-byte | `nspec = 1`, as its own `Halogrowth` asks for |
| py-corrected | `py-steel-corrected-pipgrylls` | `n_s = 0.9667` (correction 16) |
| rs-steel | the Rust port | native, `n_s = 0.9667` |

**Configuration.** `('1.0', True, True, True, 'G19_DPL', 'G19_SE')` —
stripping and star formation on, the family Papers 2 and 3 are built on.
Two grids: a *reduced* one (`log M = 11.0 … 12.6`, 0.5 dex; 4 host bins)
for day-to-day work, and the *published* one (`log M = 11.0 … 16.6`,
0.1 dex; 56 host bins), both over 190 redshift steps and 5 subhalo bins.

---

## 1. Deterministic mode — is the port numerically faithful?

Scatter off on both sides (`STEEL_SCATTER=0` / `[run] scatter = false`),
so both implementations evaluate the same arithmetic on the same grid.
py-as-is cannot participate: it has no such switch, because `GetGasMass`
scatters unconditionally (A7).

Compared as **reverse cumulatives along the stellar-mass axis**. A
per-bin comparison of two differently-binned histograms is dominated by
mass sliding across a bin edge, which reports as a large fractional
change in two adjacent bins while the function itself has barely moved;
the cumulative is insensitive to that and is what the papers actually
plot against.

### Published grid (`log M = 11.0 … 16.6`, 0.1 dex)

| Output | median | p90 | integral ratio |
|---|---|---|---|
| `Figure3_AnalyticalModel_SMF` (satellite SMF) | 0.93% | 23.9% | 0.9975 |
| `Figure10_AnalyticalModel_SMF` | 1.27% | 5.5% | 0.9975 |
| `SMFhz_AnalyticalModel_SMF_Highz` | 1.09% | 2.4% | 0.9959 |
| `Sat_SMHM_Sat_SMHM` | 1.48% | 4.4% | 0.9959 |
| `Sat_SMHM_Sat_SMHM_Host` | 2.29% | 20.7% | 0.9959 |
| `Raw_Richness_..._highz` | 2.29% | 20.7% | 0.9959 |
| `Mergers_Accretion_History` | 0.70% | 27.1% | 1.0018 |
| `Pair_Frac_Pair_Frac` | 0.58% | 9.9% | 1.0001 |
| `z_infall` | 0.94% | 11.2% | 0.9975 |

**Every integrated quantity agrees to better than 0.5%.** The p90 column
is carried by the sparse high-mass tail, where a bin holds at most one
satellite in either run. See
`Figures/PortValidation/Paper1_SatelliteSMF.png` for this row as
a picture instead of a table (both differential and reverse-cumulative
views), reproduced with `Scripts/Validation/results_figure3.py`. Not
tied to a specific published figure number -- see that script's
docstring.

### Reduced grid

Tighter, as expected — fewer host bins, less tail:

| Output | median | integral ratio |
|---|---|---|
| `Figure3_AnalyticalModel_SMF` | 0.45% | 1.0040 |
| `SMFhz` / `Sat_SMHM` / `Raw_Richness` | 0.39% | 0.9991 |
| `Mergers_Accretion_History` | 0.26% | 0.9979 |
| `Pair_Frac_Pair_Frac` | 0.27% | 0.9979 |

### A caution the numbers themselves produced

The first run after the rebaseline put this disagreement at **3.7%**,
not 0.4%. The cause was not the port: `SHMFs_Entering_*.npy` is derived
from the mass-accretion histories but keyed without the cosmology
(`STEEL.py:143`), so swapping the MAH table left one cosmology's
accretion histories paired with another's subhalo mass function. Three
and a bit percentage points of "port error" were a stale cache. The
harness now drops the derived caches whenever it rebuilds the MAH table
(`three_way.py::ensure_mah_table`); anyone reproducing this by hand has
to do the same. Recorded as G2 in `docs/PORT_CORRECTIONS.md`.

### What still differs by construction

* `Satellite_sSFR` / `sSFR_Range`: 60 bins in rs-steel and
  py-corrected, 59 in py-as-is (C2).
* `Sat_SMHM_*`: py-as-is carries an unwritten trailing padding slot
  (D2).
* rs-steel writes `Figure3_z.npy` and the `Surviving_Subhalos*` pair
  that py-steel writes as `.dat`/`.png` elsewhere, and omits the latter
  entirely when stripping or star formation is on, where py-steel saves
  arrays of zeros.

---

## 2. Stochastic mode — what do the corrections change?

Scatter on. The three draw from unrelated generators (NumPy's Mersenne
Twister, GSL's `taus`, `rand`'s ChaCha), so only ensemble statistics are
comparable.

### Published grid, py-as-is vs py-corrected

Reverse cumulatives, for the reason given in §1:

| Output | median | p90 | integral ratio |
|---|---|---|---|
| `Figure3_AnalyticalModel_SMF` (satellite SMF) | 5.7% | 13.7% | **0.9908** |
| `Figure10_AnalyticalModel_SMF` | 6.4% | 99.2% | 0.9908 |
| `SMFhz_AnalyticalModel_SMF_Highz` | 3.0% | 72.3% | 1.0080 |
| `Raw_Richness_..._highz` | 5.9% | 53.6% | 1.0080 |
| `Mergers_Accretion_History` | **27.1%** | 78.1% | 0.9942 |
| `Pair_Frac_Pair_Frac` | 11.0% | 75.7% | 0.9905 |
| `z_infall` | 11.3% | 91.8% | 0.9908 |

This is the number that matters for the papers. The **integrals** move
by under 1% — the satellite stellar mass function's normalisation is
robust to every correction found. What moves is the *shape*, and it
moves most in the merger and pair-fraction outputs: the accretion
history's median bin shifts by 27%, and the pair fractions by 11%.
Those are Papers 2 and 3's headline observables, and they are exactly
the outputs that depend on the satellite evolution window (A2), the
histogram convention (C1), and the accretion histories the spectral
index changes (G1).

Contrast §1: py-corrected and rs-steel agree to 0.5% on the same
quantities. The disagreement here is the corrections, not the port.

### Reduced grid, ensemble means over 5 seeds

**py-as-is vs py-corrected**, per bin:

| Output | max abs | max frac | median frac |
|---|---|---|---|
| `Total_StarFormation_Means` | 6.09e+09 | 194% | **32.6%** |
| `Figure3_AnalyticalModel_SMF` | 7.54e-04 | 73.4% | 0.19% |
| `Figure4_6_AnalyticalModelNoFrac_` (cut integrals) | 6.04e-05 | 35.5% | — |
| `Figure4_6_AnalyticalModelFrac_` | 3.23e-02 | 14.4% | — |
| `Sat_Env_Highz_AnalyticalModelFracHighz` | 1.15e-01 | 100% | — |
| `Pair_Frac_Halo_Pair_Frac_Halo` | 4.32e-01 | 100% | 0.32% |
| `AvaHaloMass` (all files) | 6.89e-02 dex | 0.72% | 0.17% |

The `AvaHaloMass` row is correction 16 in isolation: the halo mass
accretion histories themselves move by up to 0.069 dex because py-as-is
hands `getPWGH` a Harrison-Zel'dovich spectral index. Everything below
it in the table inherits that shift on top of whatever else changed.

The star-formation totals are the loudest single effect. They are also
the output with the fewest populated cells on this grid (39 of 30 400),
so the median is over a small sample; the summed total moves by −8.3%
from correction 16 alone.

---

## 3. Performance

Published grid, one process:

| | deterministic | stochastic |
|---|---|---|
| py-as-is | — (no scatter switch) | 308.9 s |
| py-corrected | 296.4 s | 311.3 s |
| rs-steel | 65.6 s | 68.7 s |
| speedup | **4.5×** | **4.5×** |

Scatter costs both implementations about 5%, and the corrections cost
the Python nothing measurable (296.4 s vs 308.9 s across the
scatter-on/off boundary, py-corrected vs py-as-is within 1% at equal
settings). Reduced grid: 3.4 s vs 1.3 s. The Rust advantage is larger
on the
frozen (`Stripping=False, SF=False`) configurations, where the Python's
per-timestep Cython call still dominates and the Rust's does not.

The single largest Rust-side optimisation was tabulating the halo mass
function and virial radius on the `(i, j)` grid once, rather than
re-evaluating `dn/dlog10M` inside the window loop — about 1e8 σ(M)
quadratures removed, and verified bit-identical across all 53 output
files.

---

## 4. Caveats

* One configuration. `S16CE` runs would exercise A1, which is inert
  here; `Stripping=False, SF=False` runs would make A2, A3 and A6 inert.
* Magnitudes are grid-dependent. Signs and mechanisms are not.
* py-as-is cannot be run in deterministic mode at all, so the
  numerical-fidelity claim is strictly about py-corrected vs rs-steel.
  The as-is leg's role is to say what the corrections changed, not to
  validate the port.
* The reduced grid excludes groups and clusters entirely
  (`log M ≤ 12.6`), which is where richness and pair fractions do most
  of their work — hence the published-grid numbers above.
* The Rust's van den Bosch (2014) MAH agrees with a freshly compiled
  `getPWGH` to 0.0021 dex max over `log M0 = 11…15` when the Fortran is
  fed the same cosmology (G1). That is the floor on any py/rs
  comparison and is well below every disagreement reported here.
* **These tables predate PORT_CORRECTIONS.md A8** (the Fillingham+2016
  host-mass-dependence clamp, found while reproducing Paper 2 Figure 6,
  fixed in both py-corrected and rs-steel). A8 touches every satellite
  below its cutoff mass in every host below `1e15 Msun` — most
  satellites in most runs — through the quenching timescale, so the
  py-corrected/rs-steel agreement above is unaffected (both sides of
  §1 already carry the fix, applied identically) but the *magnitudes*
  in both tables should be re-measured against a run built after A8
  before being treated as final. **Partially done:** a fresh
  deterministic published-grid run built after both A8 and H1 gives
  `Figure3_AnalyticalModel_SMF` median 0.93%, p90 23.9%, integral ratio
  0.9977 — indistinguishable from the row above, so A8 does not appear
  to move this particular output's agreement. The other eight rows in
  the table have not been individually re-checked.
