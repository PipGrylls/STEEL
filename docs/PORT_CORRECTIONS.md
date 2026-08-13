# Defects found by porting STEEL to Rust

Every entry below was found by reading `STEEL.py`, `Functions/Functions.py`
and `Functions/Functions_c.pyx` closely enough to re-implement them, and
each is **fixed in `rust/` rather than reproduced** (the port's stated
"clean reimplementation" mandate). This file is the canonical list: it is
the input to the `py-steel-corrected` branch, which applies the same
fixes to the Python one commit per entry, and the basis of the
three-way comparison.

"Bites" says which run configurations the defect actually changes. Several
are inert in the `Stripping=False, SF=False` configuration, so a
corrected-vs-as-is comparison that only runs that config will show no
difference and prove nothing.

## Which baseline this is measured against

**`PipGrylls`, not `master`.** `master`'s tip is 2019-03-04, before all
three papers. `PipGrylls` (tip 2019-11-15) is 26 commits ahead of it on
the model core and carries `bfdb4d8`, *"This is the version of the code
used for the 1st submission of Paper2 on 02/05/19"*. Papers 1–3 were run
from that line of development, not from `master`.

The first version of this document, and the first version of the port,
were both written against `master`. That was wrong, and the difference is
not cosmetic — the two branches disagree about the SFR main sequence, two
of the three abundance-matching presets, and the strength of tidal
stripping:

| | `master` | `PipGrylls` |
|---|---|---|
| `G19_DPL` satellites (`Starformation_c`) | `N = 10^(0.74 + 0.71z − 0.087z²)`, `α = 1.035 − 0.022z + 0.0077z²`, `β = 1.55 − 0.35z − 0.02z²` | `N = 10^(0.69 + 0.71z − 0.085z²)`, `α = 1.0 − 0.022z + 0.007z²`, `β = 1.8 − 0.7z − 0.035z²` |
| `G19_DPL` centrals (`Starformation_Centrals`) | *identical to the satellite block* | `M_n = 10.65 + 0.33z − 0.08z²`, `N = 10^(0.69 + 0.71z − 0.088z²)`, `α = 1.0 − 0.022z + 0.009z²`, `β = 1.8 − 1.0z + 0.1z²` |
| `G19_SE` | 12.0, 0.032, 1.5, 0.56 / 0.6, −0.014, −0.7, 0.08 | 11.925, 0.032, 1.639, 0.532 / 0.576, −0.014, −0.693, 0.03 |
| `G19_cMod` | 12.0, 0.032, 1.74, 0.66 / 0.4, −0.024, −0.74, −0.12 | 11.91, 0.029, 2.09, 0.64 / 0.644, −0.019, −1.422, −0.043 |
| `StellarMassLoss` | Cattaneo+11 as published | Cattaneo+11 **doubled in dex** (`Strip_f = Strip_f*2`) |
| `RP17`, `HMevo` presets | absent | present |
| MAH cosmology into `getPWGH` | hardcoded (0.307, 0.678, 0.823, 0.96, 0.02298) | from COLOSSUS, but with `nspec = 1` |

The satellite main sequence alone moves by 0.37 dex at `z = 0`,
`log M* = 12` and 0.76 dex at `z = 3`. These are *not* defects — they are
the published model — and the Rust port now implements them. What follows
below is the list of things that are genuinely wrong on `PipGrylls`
itself.

Three entries changed status in the rebaseline, and are kept with their
history rather than deleted, because "`master` has a bug that the paper
branch does not" is itself a finding about the repository:

* **B1** (zero pair fractions) is a `master`-only defect. `PipGrylls`,
  `Paper2`, `Refactor` and `saiduc` all have the two-dimensional branch.
* **D1** (`rm -r` missing a space) is fixed on `PipGrylls`; the `mkdir`
  without `-p` beside it is not.
* **F4** (`@jit` that could never compile) is already removed on
  `PipGrylls`.

**A limitation this audit cannot fix:** Paper 1 (MNRAS 483, submitted
2018-07-27) predates this repository, whose earliest commit is
2019-02-19 — nearly seven months later. Every "confirmed live at
commit X" claim in this file for Paper 1 specifically rests on the
falsifiability check (does the defect stop the code reproducing the
published figure?) rather than a direct `git show` of Paper 1's actual
submission-time state, because that state was never committed anywhere
in this history. Papers 2 (`bfdb4d8`, 2019-05-02, author-tagged as the
1st-submission commit) and 3 (commits through 2019-11-07 on
`PipGrylls`, corroborated by `haofu`/`Paper2`-branch commits from
2020-01-17) do not have this gap.

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

### A8. The Fillingham+2016 host-mass dependence is clamped to zero for every host below a cluster

* **Where:** `Functions/Functions.py::StarFormation:432-437` (both
  `master` and `PipGrylls` — confirmed present verbatim at
  `origin/PipGrylls:Functions/Functions.py:329-334`, and byte-for-byte
  identical at `bfdb4d8:Functions/Functions.py:343-348`, the commit
  the author tagged *"This is the version of the code used for the 1st
  submission of Paper2 on 02/05/19"*); mirrored in
  `rust/steel-plugins/src/quenching.rs::Wetzel13::timescales`.
* **What:** found while reproducing **Paper 1** Figure 6, not by code
  review (corrected from an earlier "Paper 2" mislabel in this entry —
  checked directly against `MNRAS 483, 2506` p.2517: the host-dependent
  Fillingham cutoff is eq. (8), `log(Mcutoff) = 9 - (15 - log
  Mh,host)/5`, and Fig. 6's caption reads "for three example host
  masses log10 Mh,cent = 10, 12.5, 15"). The code computed

  ```python
  Host_Dep = (AvaHaloMass[0] - 15)/5
  if Host_Dep < 0: Host_Dep = 0
  elif Host_Dep > 1: Host_Dep = 1
  Tau_d[SM_Sat < 9+Host_Dep] = 2.0
  ```

  i.e. `Mcutoff = 9 + Host_Dep` with `Host_Dep = (Mh,host-15)/5` and
  *no* floor in the paper's own equation. For any host below `log Mh =
  15` (every host in any realistic run — the published grids top out
  at 16.6, but typical group/cluster hosts sit at 11-14), `Host_Dep` is
  negative and the floor silently pins it to 0, so the cutoff mass is
  the same 9.0 regardless of host mass. The paper's own Figure 6 plots
  three *visibly distinct* cutoffs (log M* ≈ 8.0, 8.5, 9.0) for its
  three example host masses — confirmed against the published PDF page
  image, not just the equation. As committed, the code cannot reproduce
  its own figure: the host-mass dependence the whole section is about
  is inert for two of the three example hosts, and for every host mass
  typically explored in this model.
* **A likely regression, not live at Paper 1's own submission.** Paper
  1 was submitted 2018 July 27 (received 2018 Nov 27, accepted 2018 Nov
  28) — this repository's earliest commit is 2019-02-19, nearly seven
  months later, so Paper 1's actual submission-time code cannot be
  checked by `git show` the way every other defect in this file can.
  What *can* be checked is the falsifiability test: does the clamped
  code reproduce Paper 1's own published figure? It does not — see
  above. Independently, the same clamp is confirmed present, unchanged,
  in the exact commit tagged as Paper 2's 1st-submission code
  (`bfdb4d8`, 2019-05-02) and is still present at `origin/PipGrylls`'s
  last commit before Paper 3 (2019-11-07). Put together: the defect is
  solidly live for **Papers 2 and 3**, and was most likely introduced
  sometime between Paper 1's July 2018 submission and Paper 2's May
  2019 one — a regression in a period this repository's history does
  not cover, not a defect present when Paper 1's Fig. 6 was drawn.
* **Size:** at `log Mh,cent = 10`, the buggy code keeps a satellite
  quenching-delayed at 2.0 Gyr up to `log M* = 9.0`; the fixed code
  hands it off to the smooth W13 curve (~3.49 Gyr) a full 1.0 dex
  earlier, at `log M* = 8.0`. See
  `Figures/PortValidation/Paper1_Fig6_Quenching.png`.
* **Fixed by:** removing the clamp (`Host_Dep = (AvaHaloMass[0]-15)/5`,
  used directly), matching eq. (8) exactly — and the same in the Rust.
  `rust/steel-plugins/src/quenching.rs`'s
  `fillingham_cutoff_mass_differs_between_host_masses_below_1e15` pins
  three distinct cutoffs for host masses 10, 12.5, 15.
* **Not yet measured** on a full published-grid run (this was found and
  fixed while building the Figure 6 reproduction, not while re-running
  the three-way comparison in `docs/VALIDATION.md`, which predates this
  fix). It touches every satellite below the cutoff mass in every host
  less massive than `1e15 Msun`, i.e. most satellites in most runs, so
  the deterministic/stochastic tables in `docs/VALIDATION.md` should be
  re-measured before being treated as final. This affects Papers 2 and
  3's simulated outputs (both post-date the likely regression); it does
  not affect Paper 1's own published Fig. 6, which the buggy code
  cannot even reproduce.

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
`delim_whitespace=True` → `sep=r"\s+"` (removed pandas 2.2, 31 sites —
14 on `master`, 17 more in the code `PipGrylls` added);
`cumtrapz` → `cumulative_trapezoid` (removed SciPy 1.14); an unused
`interp2d` import in the new `Scripts/SMHM_Fit_MCMC.py`; four bare
`@jit` in the new `Scripts/stew_paper.py`, on functions taking and
returning Python lists and string flags that numba could never type
(object mode before 0.59, a hard error after).

### F7. `DarkMatterToStellarMass` reads `AbnMtch` keys its own callers do not set

* **Where:** `Functions/Functions.py:651`, `:716` on `PipGrylls`.
* **What:** the `RP17` and `HMevo` presets were added with unguarded
  `Paramaters['RP17']` / `Paramaters['HMevo']` lookups, but only
  `STEEL.py`'s `AbnMtch` dict was updated to carry them.
  `Scripts/CentralPostprocessing.py`, `Scripts/SMHM_Fit.py` and the
  notebooks each build their own dict, and all of them therefore raise
  `KeyError: 'RP17'` on the first call.
* **Size:** total. **None of the post-processing can run against the
  model code on its own branch.** This is not a subtle numerical
  defect; it is a hard crash on import-and-use, and it means the
  `PipGrylls` analysis path was only ever exercised from notebooks that
  predated the feature.
* **Bites:** every caller that is not `STEEL.py`.
* **Fixed by:** `.get(key, False)` at both sites — which is exactly what
  a caller predating the feature means — plus the missing key in
  `CentralPostprocessing.py`'s dict.

**Result:** `CentralPostprocessing` now imports and runs on
`env/py-legacy` (NumPy 1.26 / SciPy 1.13 / pandas 1.5) with no
observational data present, and **twelve of its thirteen** `Return_*`
methods produce finite, correctly-shaped output against a real run. The
thirteenth, `Return_NoMerger_Plot`, needs the Bernardi SDSS catalogue,
which this repository does not ship. The merger-consuming methods could
not have been tested at all before Phase 1, since they read outputs the
Rust did not produce and the Python wrote as zeros.

---

## G. Cosmology and the mass-accretion-history cache

### G1. `Halogrowth` passes a Harrison-Zel'dovich spectral index

* **Where:** `Functions/Functions.py::Halogrowth:321` on `PipGrylls`.
* **What:** `PipGrylls` replaced `master`'s five hardcoded `getPWGH`
  inputs with the run's own COLOSSUS cosmology — `Cosmo.Om(0)`, `h`,
  `Cosmo.sigma(8,0)`, `Cosmo.Ob(0)*h²` — except `nspec`, which is the
  literal `1`. Planck15 has `n_s = 0.9667`.
* **Size:** measured on a freshly compiled `getPWGH` over
  `log M0 = 11…15`:

  | change | max ΔlogM(z) |
  |---|---|
  | `master` → `PipGrylls` as written (`nspec = 1`) | **0.080 dex** |
  | `master` → `PipGrylls` with `n_s = 0.9667` | 0.011 dex |

  Seven eighths of the shift is the typo. Downstream on the reduced
  grid: `AvaHaloMass` max 0.069 dex; `Figure4_6` cut integrals median
  1.0%, max 11.9%; total star formation −8.3% summed.
* **Bites:** every run — but only when the MAH cache is cold (see G2).
* **Fixed by:** `Cosmo.ns`, and
  `Scripts/Validation/make_mah_table.py`, which builds the table with
  an explicit, recorded cosmology.

This also sharpens the Milestone 2 validation of the Rust's native van
den Bosch (2014) MAH: against `getPWGH` fed the *same* cosmology it
agrees to **0.0021 dex** max over `log M0 = 11…15`. The 0.009 dex
originally reported was measuring the cosmology mismatch, not the port.

### G2. The MAH cache key does not include the cosmology

* **Where:** `Functions/Functions.py::Get_HM_History:147`.
* **What:** the cached table is named
  `<halo_min><halo_max><halo_bin><h>.dat` and regenerated only when
  absent. Nothing in the key records which cosmology built it, so
  checking out a branch that changes the cosmology and re-running
  silently reuses the other branch's grid. G1 is invisible on a warm
  cache.
* **Fixed by:** a `.cosmology` stamp written beside each table by the
  validation harness, and one table per leg of the three-way
  comparison. The cache key itself is left alone, because changing it
  would invalidate every table a user already has on disk.

### G3. `Halogrowth` cannot run off one machine

* **Where:** `Functions/OtherModels/VDB13/getPWGH.f:119` and
  `Functions/Functions.py::Halogrowth:338,341` on `PipGrylls`.
* **What:** two independent breakages. `getPWGH.f` opens its output as
  `fileplace//outfile` with
  `fileplace = "/data/pg1g15/STEEL/Functions/OtherModels/VDB13/"`, a
  path that exists on one machine (and, being `CHARACTER*47`
  concatenated without `TRIM`, produces a filename with trailing blanks
  even there). The Python then reads the result back through a path
  containing a literal `*`, which neither `np.loadtxt` nor `os.remove`
  globs, so the call raises `OSError` regardless.
* **Size:** total, but latent — `Get_HM_History` only calls
  `Halogrowth` on a cold cache, so this shows up as "the MAH table has
  to be shipped" rather than as a crash.
* **Fixed by:** restoring `master`'s working-directory output on both
  sides. With that, `F.Halogrowth(12.0)` completes and returns a
  200-point history.

---

## H. `Scripts/CentralPostprocessing.py` — a correctness pass on the analysis methods

Phase 5 (§F) made this file *importable and runnable*: removed APIs,
missing dict keys, a crashing SDSS load. It did not check whether the
3,110 lines of analysis logic inside `PairFractionData` compute the
right thing — a materially shallower pass than the one applied to
`Functions.py`/`STEEL.py`/`Functions_c.pyx`, which is where A1–A7,
B1–B3 and C1–C3 came from. This section is that pass, run after the
fact and only on `PairFractionData`'s reusable methods (lines 178–1145,
about a third of the file) plus two exhaustive whole-file AST scans;
the ~1,300-line `if __name__ == "__main__"` figure script (1146–2469)
and the block after it were checked only by tracing which `Return_*`
calls inside it are reachable (`if True:`/`if False:` guards), not read
line by line. **Not a complete audit.**

Each call site below was checked against the live script, because a
defect in a method nobody calls with `if True:` around it did not
change a published figure and a defect in one that is did.

### H1. `Return_PF_Plot`'s upper mass-ratio cut digitized against the wrong axis — **live, feeds a Paper 3 figure — fixed**

* **Where:** `Scripts/CentralPostprocessing.py::Return_PF_Plot:313-314`.
* **What:** with `UpperLimit=True` (the default, and the value every
  call site in the script passes explicitly), the satellite-side upper
  cut was

  ```python
  Sat_Mass_Cut_bin_upper = np.digitize(CND_Mass_Upper, SM_Arr)
  ```

  which is byte-for-byte the same expression already computed three
  lines above as `M_Cut_bin_upper` (`Return_PF_Plot:298`) — digitizing
  a mass against `SM_Arr`, the *central*-halo-bin stellar-mass track
  (57 elements in a published-grid run). The result is then used to
  slice `self.Pair_Frac[i, M_Cut_bin+j, Sat_Mass_Cut_bin:Sat_Mass_Cut_bin_upper]`
  along the *satellite* stellar-mass axis
  (`self.Surviving_Sat_SMF_MassRange`, 40 elements) — the same axis
  `Sat_Mass_Cut_bin` (the lower bound, three lines above) is correctly
  digitized against. Measured directly against a real run: at
  `i=0, Parent_Cut=11`, the lower bound comes out as bin 15-17 of 40
  (via the correct axis); the upper bound as written came out as bin
  29-57, fixed at 29 across every `j` in the first host-mass window and
  pinned to the axis length (57) at higher `i`, because it was
  digitized against a 57-element track covering a similar-looking but
  different mass range and does not vary with the satellite it is
  supposedly bounding. Neither array is out of bounds, so nothing
  raises — the sum was just taken over the wrong slice of the satellite
  mass function, silently.
* **Bites:** `Return_PF_Plot(..., UpperLimit=True)`. Traced against
  `Scripts/CentralPostprocessing.py`'s `__main__` block: 5 of 7 call
  sites sit under `if False:`; the other two (lines 1503, 1515) sit
  under the `if True:` at line 1476, whose block ends by saving
  `Figures/Paper3/PairFractionData.{png,pdf}` — the pair-fraction vs.
  observational-data comparison. **Not a hypothetical defect in
  unreachable code; it was live.**
* **Confirmed directly against the branch's own history**, independent
  of the reachability trace above: `bc1b0ead` (2019-05-23, *"changed
  some plotting routines to make a pairfraction data and satellite
  accretion plot for paper 3"*) is the commit that introduced this
  exact code (`M_Cut_bin_upper = np.digitize(CND_Mass_Upper, SM_Arr)`),
  and it is still present, unchanged, at `ef8229ee` (2019-11-07,
  *"thesis plots"*) — the last commit to touch this file on
  `origin/PipGrylls` before Paper 3's Jan 2020 submission. Also present
  at the same lines in the intervening `haofu`/`Paper2`-branch commits
  from 2020-01-17, three days before the arXiv submission. There is no
  commit anywhere in this file's history, on any branch, that fixes it
  before Paper 3 went out.
* **Fixed by:** every other major-merger mass-ratio cut in this file
  (`Maj_Merge_Bin`, 13 call sites from line 397 on, and
  `Sat_Mass_Cut_bin` three lines above the bug itself) uses the same
  idiom: digitize a *central* stellar mass plus a log mass-ratio offset
  against `self.Surviving_Sat_SMF_MassRange`, the satellite-mass axis.
  `Sat_Mass_Cut_bin`'s lower bound is exactly that, with offset
  `Mass_Ratio = log10(1/4)`. A 1:4–1:1 pair-fraction window's upper
  bound is the same idiom with offset zero — mass ratio 1, the
  central's own mass, no second parameter needed:

  ```python
  Sat_Mass_Cut_bin_upper = np.digitize(SM_Arr[M_Cut_bin + j], self.Surviving_Sat_SMF_MassRange)
  ```

  This resolves the ambiguity the two readings below couldn't: it
  needs no new parameter, matches the codebase's own established
  pattern for this exact kind of cut, and — checked directly — the
  resulting bin always sits `Mass_Ratio`'s width (log10(4) ≈ 0.6 dex,
  6 bins at 0.1 dex spacing) above the lower bound, for every `(i,j)`
  checked, which the buggy version never does.
* **Size:** measured on the published-grid `('1.0', True, True, True,
  'G19_DPL', 'G19_SE')` run, `Parent_Cut=11`, comparing
  `Return_PF_Plot`'s output before and after
  (`Scripts/Validation/verify_h1_fix.py`). Every one of the 2,688
  `(i,j)` cells checked gets a different bin index (buggy range
  [29,57] vs. fixed range [21,26] against the same 40-bin axis).
  In the integrated `PairFracTot` output itself the effect is
  mass-function-weighted and much smaller at low redshift — median
  0.09%, because most of the pair-fraction signal already sits just
  above the lower cut and the buggy slice's extra bins hold little
  mass — but grows with redshift as the satellite mass function
  narrows relative to the cut: p90 5.7%, max 33% (`z≈4.9`,
  0.0280 → 0.0187).

### H2. `Return_Gas_Hard_Threshold_Plot` drops the lenticular fraction to zero the first time the gas threshold isn't met

* **Where:** `Scripts/CentralPostprocessing.py::Return_Gas_Hard_Threshold_Plot:887-892`.
* **What:** the recursive `P_lentic[i,j] = P_lentic[i+1,j] + ...` update
  that carries the lenticular fraction forward through the redshift
  loop is present in the `if CurrentGasFrac >= GasFracThresh:` branch
  but **absent from the `else:`** — only `P_ellip[i,j]` is set there.
  `P_lentic` was zero-initialized, so once a bin fails the threshold at
  any redshift step, every later step in the loop reads
  `P_lentic[i+1,j] == 0` and the lenticular fraction never recovers.
  Confirmed structurally: `Return_Gas_Soft_Threshold_Plot` and
  `Return_New_Gas_Inflow_Plot` are near-identical siblings and both
  update `P_lentic` in their equivalent `else` branch; only this one
  drops the statement (`P_ellip`/`P_lentic` assignment counts 3/2
  here vs. 4/4 and 8/10 in the two siblings).
* **Bites:** `Return_Gas_Hard_Threshold_Plot`. Its one call site
  (`Scripts/CentralPostprocessing.py:2010`) is under `if False:`
  (the "Gas Fraction Restricted Lenticular Plots" block) — **not
  currently live** in the script.
* **Not fixed here.** What the `else` branch should compute is a
  physics choice (carry `P_lentic` forward unchanged? decay it? reset
  it, as `Return_Sai_Idea_Plot`'s deliberate `else: P_lentic[i,j] = 0`
  does for a similar but distinct model?) that isn't recoverable from
  the code alone, and the call site is dead, so there is no live output
  to regression-test a fix against.

### H3. Two parameters are silently overwritten with a literal, discarding the caller's value

* **Where:** `Return_Sai_Idea_Plot:403` (`GasFracThresh = 0.06`,
  overwriting the `GasFracThresh` parameter) and
  `Return_NoMerger_Plot:475` (`z_cut = 1.5`, overwriting `z_cut`).
  Found by an AST scan for `param = <literal>` assignments inside a
  function that also declares that name as a defaulted parameter —
  applied to every function in the file, not just these two.
* **What:** whatever the caller passes for these arguments has no
  effect; the function always uses the hardcoded value.
* **Bites:** both call sites are traceable and both are dead —
  `Return_Sai_Idea_Plot(MassRatio, 2, GasFracThresh)` at line 2298
  sits under `if False:` at line 2223 ("Sai's Model Idea"), and
  `Return_NoMerger_Plot(MassRatio, 1, 0.1)` at line 2202 sits under
  `if False:` at line 2200, whose preceding comment already says *"This
  is currently broken as we need to use a time interval instead of
  redshift interval"* — the author had already found this path broken
  and disabled it independently of this defect. At the live
  `GasFracThresh = 0.0` the call site passes, the overwrite changes the
  value used from 0.0 to 0.06, which would matter if the block were
  ever re-enabled.
* **Not fixed here.** Deleting the overwrite is the obvious fix and
  would cost nothing today (both call sites are dead), but changes
  what re-enabling either block would compute; left as a documented
  trap for whoever re-enables them rather than silently resolved.

### H4. Two defects fixed directly (no live output changed)

* **`CreateAverageSM:229`** compared `AvaStellarMass2[i, -1]` against
  itself (`x < x`, always `False`), so the extrapolation meant to catch
  a still-non-monotonic trailing bin could never run. Fixed to compare
  against `[i, -2]`, matching the extrapolation direction already used
  three lines above. The surrounding `except` block is reached in real
  runs (14/190 rows on the reduced grid), but the specific tail case
  this line exists for did not occur in any row tested (0/190) — the
  main smoothing loop already fixes the trailing bin in every case
  observed. Confirmed the fix is a no-op on current output before and
  after.
* **`JitLoop` (used by `Return_Merger_Plot`)** looped `i in range(m)`
  with `m = SHMF_Entering.shape[0]` (190 on the published grid) while
  indexing `z_step[i]`/`t_step[i]`, both length 189 — an out-of-bounds
  read on the last iteration, inside a numba `nopython` function, so
  it does not raise a Python exception; it is undefined behaviour, not
  a defined error. Harmless for the only current caller
  (`Return_Merger_Plot` only reads rows `0:m-1` of the results, so the
  corrupted final row is discarded), but "currently discarded" is not
  the same as "correct" or "safe under other numpy layouts." Fixed by
  guarding the division with `if i < len(z_step):`. Confirmed the fix
  is a no-op on current output.

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
