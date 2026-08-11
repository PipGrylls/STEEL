# Defects found by porting STEEL to Rust

Every entry below was found by reading `STEEL.py`, `Functions/Functions.py`
and `Functions/Functions_c.pyx` closely enough to re-implement them, and
each is **fixed in `rust/` rather than reproduced** (the port's stated
"clean reimplementation" mandate). This file is the canonical list.

Every entry is also applied to the Python, one commit per concern, on
the `py-steel-corrected` branch — see `docs/PY_CORRECTED.md` for the
commit series and the measured effect of each. Where a measurement
exists it is quoted below.

"Bites" says which run configurations the defect actually changes. Several
are inert in the `Stripping=False, SF=False` configuration, so a
corrected-vs-as-is comparison that only runs that config will show no
difference and prove nothing.

---

## A. Wrong physics

### A1. The Schreiber+2015 main sequence is clamped the wrong way in the compiled hot loop

* **Where:** `Functions/Functions_c.pyx:107-112` and `:115-120`
  (`SFR_Model_int == 3` and `== 4`).
* **What:** Schreiber et al. (2015, A&A 575, A74) Eq. 9 is

  ```
  log SFR = m - m0 + a0 r - a1 [max(0, m - m1 - a2 r)]^2
  ```

  with `m = log10(M*/1e9)`, `r = log10(1+z)`. The `max(0, ·)` clamps
  **below** at zero, so the quadratic term switches *on* above the knee
  mass and bends the main sequence down at high mass.

  `Functions.py::StarFormationRate:46` gets this right
  (`Max[Max<0] = 0`). `Functions_c.pyx` gets it backwards
  (`if Max > 0: Max = 0`), clamping **above** — which deletes the
  high-mass bend entirely and instead applies the quadratic penalty to
  low-mass galaxies. The two Python implementations therefore disagree
  with each other, and the wrong one is the one every real run executes.
* **Size:** at `M* = 1e11`, `z = 0` the published relation suppresses
  SFR by 0.81 dex; the Cython suppresses it by 0.
* **Bites:** every run with `SFR_Model` in `{S15, S16, S16CE}` — i.e.
  the `('1.0', True, True, True, 'S16CE', 'G19_SE')` family. Runs using
  `CE`, `G19_DPL`, `T16` or `Illustris` are unaffected.
* **Fixed in:** `rust/steel-plugins/src/sfr.rs::SchreiberFormSfr`.
* **Caveat:** this is a deliberate behavioural departure from the code
  that produced the papers. `S16CE` figures will not reproduce without
  correcting the Python too.

### A2. The satellite evolution window is one timestep short

* **Where:** `Functions/Functions.py::StarFormation:311-320`, and the
  compensating `np.flipud` at `STEEL.py:375`, `:398`.
* **What:** `StarFormation` slices `z_all[z_bin_i:z_bin_r]`, giving
  `i - z_bin` grid points running `z[i] … z[z_bin+1]`. `Starformation_c`
  writes `M_out[k,i+1]` only for `i < N-1`, so it applies `N-1` steps
  and the track **never reaches the merge/reference epoch `z[z_bin]`**.
  The accumulators then relabel those columns via `np.flipud` as if they
  spanned `z[i-1] … z[z_bin]` — a second, compensating shift, so the
  reported redshift of every evolved satellite is off by one step and
  the last step of evolution is missing.
* **Bites:** every run with `Stripping` or `SF` on. Inert otherwise
  (with neither on, `SM_Sat` stays 1-D and no window is built).
* **Fixed in:** `rust/steel-core/src/context.rs` (PORT-FIX 1) — the
  timeline spans `z_bin..=i` inclusive, `i - z_bin` steps ending exactly
  at `z[z_bin]`.

### A3. `SFH` accumulates a log-ratio into a linear-Msun total

* **Where:** `Functions/Functions_c.pyx:197`,
  `SFH[k,i] = SFH[k,i]+(StripFactor[i+1]-StripFactor[i])`.
* **What:** `SFH[k,i]` is a mass in Msun (`SFR*delta_t*1e9`).
  `StripFactor` is a base-10 *logarithm* of a surviving fraction. The
  difference of two such logs is a dimensionless log-ratio, of order
  -0.01 to -1, and adding it to a quantity of order 1e8 Msun is a unit
  error. (It is also inside the `for j in range(i)` recycling loop, so
  it is applied `i` times per timestep rather than once.) `SFH` then
  feeds both the recycled mass-loss rate and the reported sSFR.
* **Bites:** runs with `Stripping=True` **and** `SF=True`.
* **Fixed in:** `rust/steel-core/src/baryonic.rs` — the stripping term
  is applied to the stellar mass (`log_sm[i+1]`), where it belongs, and
  not to `sfh`.

### A4. `dn_dlnX` converts per-ln to per-dex with a truncated constant

* **Where:** `Functions/Functions.py:248`, `dn_dlogX_arr = dn_dlnX_arr*2.30`.
* **What:** the conversion factor is `ln(10) = 2.302585…`. Using `2.30`
  scales the whole unevolved subhalo mass function low by 0.11%.
* **Bites:** every run; it is a uniform 0.11% normalisation error on
  every satellite number density.
* **Fixed in:** `rust/steel-plugins/src/shmf.rs`.

### A5. The abundance-matching scatter is not independent between bins

* **Where:** `Functions/Functions.py:524` (`DarkMatterToStellarMass`)
  and `:630` (`DarkMatterToStellarMass_Alt`).
* **What:** the abundance-matching routine reseeds the *global* NumPy
  generator on every call, from the wall clock:

  ```python
  np.random.seed(int(time() + os.getpid()*1000))
  ```

  `os.getpid()*1000` is constant within a process, so the seed only
  changes once per wall-clock second. `DarkMatterToStellarMass` is
  called once per `(i, j, k)` bin — ~700 000 times in a full run — so
  every bin evaluated within the same second is handed the **identical**
  `N`-element scatter vector. Verified directly: four consecutive calls
  return bit-identical draws.

  The `N = 5` realizations exist specifically "to capture upscatter
  effects" (`STEEL.py:38`). Reseeding this way means a ~10-second run
  samples on the order of ten distinct scatter realizations in total
  rather than 700 000 independent ones, so the upscatter is
  systematically under-sampled and correlated across neighbouring mass
  and redshift bins in a way that does not average out.

  It also makes a run unreproducible even in principle: there is no seed
  to set, and re-running the same configuration gives different answers.
* **Bites:** every run.
* **Fixed in:** the Rust threads one explicitly-seeded `StdRng` through
  the whole run (`ModelContext::rng_seed`), drawn from continuously
  rather than reset. A run is bit-reproducible — see the
  `a_fixed_seed_reproduces_the_whole_run` test.

### A6. The gas-supply cap never engages

* **Where:** `Functions/Functions.py::StarFormation:378` (which passes
  the ceiling) and `Functions/Functions_c.pyx:150-152`, `:172-174`
  (which tests it).
* **What:** `StarFormation` computes the ceiling as

  ```python
  MaxGas = np.power(10, GetGasMass(SM_Sat, z, HM_infall, Paramaters))
  ```

  — a **linear** mass in Msun, of order `4e9`. `Starformation_c` then
  tests it against a logarithm:

  ```
  if SM_new > 0:
      if c_log10(SM_new) > MaxGas[k]:
          SFR = c_pow(10, M_out[k,i]-12.0)
  ```

  `log10(SM_new)` cannot exceed ~12 for any physical stellar mass and
  `MaxGas[k]` is ~4e9, so the branch is never taken. **The gas supply
  never limits star formation.** The entire `GetGasMass` /
  `GetMaxGasMass` machinery — a whole physical ingredient, with its own
  scaling relation, scatter and baryon-fraction ceiling — has no effect
  on any result.

  `Starformation_c` also uses `MaxGas` *linearly* four lines earlier
  (filling its `GasMass` array, which is returned and never read), so it
  contradicts itself inside one function.
* **Size:** on the M4 fixture the cap costs 0.013 dex of final stellar
  mass for an unstripped `M* = 1e10` satellite over `z = 1 -> 0.5`. It
  does not engage at all for a stripped satellite, which is losing mass
  rather than accumulating it.
* **Bites:** runs with `SF = True`. Larger for gas-poor, rapidly
  star-forming satellites, i.e. exactly the regime the ceiling exists to
  regulate.
* **Fixed in:** `rust/steel-core/src/baryonic.rs` keeps the ceiling in
  log throughout. Verified against the committed Cython: with the cap
  neutralised the two agree to 1e-9 step for step over the whole
  trajectory, and the stripped case (where the cap never binds) agrees
  to 1e-9 with no adjustment at all. See
  `rust/steel-plugins/tests/baryonic_pipeline.rs` and
  `Scripts/Validation/reference_baryonic.py`.

### A7. `Scatter_On = 0` does not give a noiseless run

* **Where:** `Functions/Functions.py::GetGasMass:110`.
* **What:** `GetGasMass` applies `np.random.normal(GasMass, 0.2)`
  unconditionally — it has no `ScatterOn` parameter, unlike its sibling
  `StarFormationRate`. So even with `Starformation_c`'s `Scatter_On = 0`
  and `DarkMatterToStellarMass(..., ScatterOn=False)`, one stochastic
  source stays live.
* **Bites:** any attempt to run the model deterministically. It is why
  the three-way comparison could not have a trustworthy deterministic
  mode without this fix.
* **Fixed in:** the Rust routes every stochastic source through one
  `RunConfig::scatter` switch, and `GasMassModel::gas_mass` takes
  `Option<&mut dyn RngCore>` (the convention `SmhmModel::stellar_mass`
  already used). The `noiseless_evolution_does_not_consume_randomness`
  test pins it.

---

## B. Dead or unreachable code paths

### B1. `Pair_Frac` and `Pair_Frac_Halo` are silently zero whenever star formation or stripping is on

* **Where:** `STEEL.py:436-442`.
* **What:** the entire pair-fraction block is nested inside
  `if len(np.shape(SM_Sat)) == 1:` with **no `else`**. `SM_Sat` is 2-D
  exactly when the satellite was evolved, i.e. when `Stripping` or `SF`
  is on. So the configurations Papers 2 and 3 are built on write
  all-zero pair fractions. Note `Pair_Frac_Halo` does not even depend on
  `SM_Sat` — it is skipped only because it sits inside the same block.
* **Bites:** every run with `Stripping` or `SF` on. This is the most
  consequential entry in the list.
* **Fixed in:** `rust/steel-core/src/context.rs` (PORT-FIX 2).

### B2. `Make_HMF_Interp`'s cache check can never be true

* **Where:** `Functions/Functions.py:207`,
  `if AbsFP+"/../Data/Model/Input/hmf_fun.pkl" in os.listdir():`
* **What:** `os.listdir()` with no argument lists the *current working
  directory* and returns bare file names; the test compares them against
  an absolute path, so it never matches. The halo mass function
  interpolation table (700 redshift steps × 800 masses of COLOSSUS
  `massFunction` calls) is therefore rebuilt and re-pickled on **every**
  import, and the committed `hmf_fun.pkl` is never read.
* **Bites:** every run — as start-up cost, not as a wrong number.
* **Fixed in:** not applicable to the Rust (the HMF is evaluated
  natively and tabulated on the grid it is queried at); listed because
  the Python fix is worth making.

### B3. `WeightList_SubOnly` is read from the previous loop iteration

* **Where:** `STEEL.py:285-294` (assigned only in the `i != 0 and
  z_bin != i` branch) versus `:416` and `:440` (read).
* **What:** when `z_bin == i` the `else` branch runs and never assigns
  `WeightList_SubOnly`, so the merger accumulator reads whatever the
  previous `k` iteration left in the name — a different subhalo mass
  bin's weight.
* **Bites:** runs where any satellite has `z_bin == i` (a
  dynamical-friction time shorter than one redshift step).
* **Fixed in:** `rust/steel-core/src/context.rs` (PORT-FIX 3).

---

## C. Binning and grid off-by-ones

### C1. `np.digitize` used as a histogram bin index

* **Where:** `STEEL.py:333` (`Total_StarFormation`'s `bin_`) and
  `:450`/`:463` (`AnalyticalModel_Cuts_*`'s `SM_Bin`).
* **What:** every other binning in `OneRealization` goes through
  `fast_histogram`, whose bin index for a value on a bin's left edge is
  that bin. `np.digitize` returns the index *past* it. All of STEEL's
  `SM_Cuts` except 11.45 sit exactly on a bin edge, so e.g. the
  "satellites above `log M* = 9.0`" integral actually started at 9.1 and
  dropped a whole bin.
* **Size:** on the reduced validation grid the integrated
  `Figure4_6_AnalyticalModelNoFrac_` is 10.4% higher in rs-steel than in
  py-as-is, and `Sat_Env_Highz_AnalyticalModelNoFracHighz` 7.9% higher,
  against ≲1.5% for every other output family. Those two arrays are the
  only consumers of `SM_Bin`, so this correction is what separates them
  from the rest — but the figures are aggregate py-vs-rs ratios, not an
  isolated measurement of C1. Isolating it is Phase 2/4 work: apply the
  fix alone on `py-steel-corrected` and difference against py-as-is.
* **Bites:** every run.
* **Fixed in:** `rust/steel-core/src/context.rs::cut_bin_index`
  (PORT-FIX 4).

### C2. The sSFR histogram bins do not match the axis saved beside them

* **Where:** `STEEL.py:232-234`.
* **What:** `sSFR_Range = np.arange(-14, -8, 0.1)` has 60 entries;
  `sSFR_len = np.size(sSFR_Range)-1 = 59` bins are then spread over the
  full `(-14, -8)` range, giving 0.1017-dex bins. The axis saved
  alongside (`sSFR_Range[:-1]`, 59 values) is 0.1-spaced and runs to
  -8.2. So every sSFR distribution is plotted against the wrong
  abscissa, with the error growing to ~0.1 dex at the top of the range.
* **Bites:** every run with `SF=True` (the only ones that fill it).
* **Fixed in:** `rust/steel-core/src/context.rs` (PORT-FIX 5) — 60 bins
  of exactly 0.1 dex, and a 60-entry axis.

### C3. `np.arange` grid sizing (found in the Rust, not the Python)

* **What:** the Rust originally sized the halo grid with `round()`.
  `(16.6 - 11.0)/0.1` evaluates to `56.000000000000014`, so `round`
  gives 56 bins where `np.arange` gives 57 — silently dropping the most
  massive halo bin (`log M ≈ 16.43`).
* **Fixed in:** `rust/steel-core/src/numerics.rs::arange_len` (`ceil`
  semantics, with a test against the real numpy values).

---

## D. Reproducibility and hygiene

### D1. `PrepareToSave`'s `rm` is missing a space

* **Where:** `Functions/Functions.py:785`,
  `os.system("rm -r" + OutputFolder + ...)`.
* **What:** produces `rm -r/path/to/RunParam_...`, i.e. `rm` with the
  unknown option `-r/path/...`. It always fails, so stale output
  directories are never cleared and a re-run silently mixes new output
  with old. (The subsequent `mkdir`, also without `-p`, then fails too.)
* **Fixed in:** not applicable to the Rust (`write_run` uses
  `create_dir_all` and overwrites each file); listed for the Python.

### D2. Unused shape padding in `Sat_SMHM`

* **Where:** `STEEL.py:228-229`, `np.zeros((a, c+1, ...))` and
  `np.zeros((a, b+1, ...))`.
* **What:** the trailing slot on the halo-mass axis is never written and
  is saved as zeros, next to a `SatHaloMass`/`AvaHaloMass` axis array
  with `c`/`b` entries — so the data and its own axis disagree in
  length. Verified all-zero in a real run.
* **Fixed in:** the Rust emits `(a, c, n_sm)` and `(a, b, n_sm)`,
  matching the saved axes.

---

## F. Post-processing (Phase 5)

Found while making `Scripts/CentralPostprocessing.py` run on a current
stack. None of these change `STEEL.py`'s numerics — verified: after all
of them, a reduced-grid run differs from before by 2e-15 relative.

### F1. The module could not be imported without observational data

* **Where:** `Scripts/CentralPostprocessing.py:30-34` (module scope).
* **What:** the SDSS comparison catalogue was constructed at import
  time, so `import CentralPostprocessing` read or rebuilt the whole
  Bernardi catalogue before any function in the file could be called.
  With no observational data present — the state of this repository —
  the module could not be imported at all, and none of the
  post-processing could be inspected, tested, or run against model
  output that needs no data.
* **Fixed:** loaded lazily by `Add_SDSS()` on first use.

### F2. `scipy.interpolate.interp2d`'s call semantics were worked around, not used

* **Where:** `Functions.py::Make_HMF_Interp`,
  `CentralPostprocessing.py::Generate_SMF_interp`, and the consumer at
  `STEEL.py:341`.
* **What:** `interp2d.__call__(x, y)` sorts both inputs and returns the
  full `len(y) × len(x)` outer grid. Every caller in STEEL wants
  *paired* evaluation, and `STEEL.py` recovered it with

  ```python
  Arr2D = HMF_fun(AvaHaloMass[z_bin:i, j], z[z_bin:i])
  WeightList = np.diag(np.fliplr(Arr2D)) * ...
  ```

  The anti-diagonal is only the right pairing because the halo-mass
  slice happens to decrease with index while the redshift slice
  increases — an unstated precondition. Reverse either ordering and it
  silently returns the wrong weights: demonstrated, with both axes
  ascending it is wrong by 0.74 against a directly-evaluated truth,
  where the replacement is exact to 6e-17.
* **Also:** `interp2d` was removed in SciPy 1.14.
* **Fixed:** `Functions.GridInterp2D`, a broadcasting adapter over
  `RegularGridInterpolator`. Verified to reproduce the anti-diagonal to
  2.8e-17 under STEEL's actual usage.

### F3. The HMF cache pickled a live SciPy object

* **Where:** `Functions.py::Make_HMF_Interp`.
* **What:** `hmf_fun.pkl` held an `interp2d` *instance*. That is
  unusable twice over: the class is gone in SciPy ≥ 1.14, and even
  between SciPy 1.8 and 1.13 the interpolator classes moved modules, so
  loading raises `AttributeError` across any version change.
* **Fixed:** the cache (`hmf_table.npz`) stores the three arrays and the
  interpolator is rebuilt.

### F4. A `@jit` that could never compile

* **Where:** `Functions.py::DarkMatterToStellarMass`.
* **What:** a bare `@jit` on a function whose second argument is a
  Python dict, which numba cannot type. It therefore always fell back to
  object mode: no acceleration, only compilation overhead and a
  deprecation warning. numba 0.59 made bare `@jit` mean
  `nopython=True`, at which point the function stops working entirely
  (`TypingError: non-precise type pyobject`).
* **Fixed:** decorator removed (a no-op on the numerics). The two
  genuinely-compilable `@jit`s in `CentralPostprocessing.py` are now
  explicitly `@jit(nopython=True)`.

### F5. A ragged list that NumPy ≥ 1.24 refuses to build

* **Where:** `CentralPostprocessing.py::Return_PF_Plot`.
* **What:** one branch appended a length-1 array (the halo mass function
  call returns one), the other a bare `np.nan`. The resulting ragged
  list built an object array with a `VisibleDeprecationWarning` on
  NumPy < 1.24 and raises from 1.24 on:
  `ValueError: setting an array element with a sequence`.
* **Fixed:** both branches append a plain float.

### F6. Removed pandas and SciPy APIs

`get_values()` → `to_numpy()` (removed pandas 1.0);
`delim_whitespace=True` → `sep=r"\s+"` (removed pandas 2.2, 14 sites);
`cumtrapz` → `cumulative_trapezoid` (removed SciPy 1.14).

**Result:** `CentralPostprocessing` now imports and runs on
`env/py-legacy` (NumPy 1.26 / SciPy 1.13 / pandas 1.5) with no
observational data present, and all six analysis methods —
`Return_PF_Plot`, `Return_Merger_Plot`, `Return_Morph_Plot`,
`Return_satSMF`, `Return_SSFR`, `Return_Cent_SMF` — produce finite,
correctly-shaped output against a real run. The first three could not
have been tested at all before Phase 1, since they consume the merger
outputs the Rust did not produce and the Python wrote as zeros.

---

## E. Environment (not defects, but reproducibility findings)

`STEEL.py` cannot run on any current scientific Python stack. See
`env/README.md` for the full account; in summary:

| API | Used at | Removed in |
|---|---|---|
| multidimensional indexing with a non-tuple sequence | `STEEL.py:300-303` | numpy 1.23 |
| `scipy.interpolate.interp2d` | `Functions.py:229`, `CentralPostprocessing.py:236` | scipy 1.14 |
| `scipy.integrate.cumtrapz` | `CentralPostprocessing.py:20` | scipy 1.14 |
| `pd.Series.get_values()` | `CentralPostprocessing.py:233,235` | pandas 1.0 |
| `delim_whitespace=` | `SDSS_Plots.py:45,52`, `SMHM_Fit.py:156+` | pandas 2.2 |

The committed `Functions/OtherModels/VDB13/getPWGH` binary links against
`libgfortran.so.3` and does not run on any current system; it has been
rebuilt in place from the repository's own `.f` sources.
