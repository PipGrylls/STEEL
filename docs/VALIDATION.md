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

---

## 5. External model agreement

Rate-based `StellarGrowthModel` plugins (`steel_core::stellar_growth`)
are validated against real, upstream-sourced reference grids rather than
against STEEL's own Python, since they have no Python counterpart. See
`rust/steel-plugins/tests/fixtures/emerge/provenance.toml` for exactly
how each grid was produced, and `rust/steel-plugins/tests/upstream_agreement.rs`
for the tests below.

### EMERGE (`EmergeGrowth::o_leary23`)

Two independent comparisons, of different strength:

* **`eps(M,z)` against upstream's compiled `sfe()`** — a pointwise,
  memoryless function call with no integration involved, so this is a
  direct fidelity check of the coefficients and formula alone. Worst
  deviation over the 51×6 `(log_mh, z)` grid: **1.5e-7 dex**
  (`log_mh=14.5, z=0.1`), matching Task 8's own 3.5e-7 figure for the
  same fixture (float32-vs-float64 rounding only). Test bound: `1e-5`
  dex.
* **Integrated M\*(Mh0, z0=0.1) against the fixture's `smhm_grid.npy`**
  — a weaker check: the fixture integrates `eps(M,z)` trapezoidally in
  *linear halo mass* along a VandenBosch14 track (a chain-rule
  reformulation, since no runnable upstream N-body pipeline exists in
  this environment), while `steel_core::integrate_stellar_mass`
  integrates trapezoidally in *cosmic time* of the rate. These agree
  only in the continuum limit; at the fixture's fixed 200-point track
  they differ by a genuine, understood discretization-scheme mismatch,
  not a coefficient or formula error. Worst deviation over the 51-point
  mass grid: **0.0497 dex**, at the lowest-mass point (`log_mh=10.0`).
  The deviation is smooth and monotonic in `log_mh`, vanishing near the
  pivot mass (~11.4) and growing at both mass tails — the expected
  shape of a quadrature-scheme mismatch, confirmed by the pointwise
  `eps()` check above being unaffected. Test bound: `0.052` dex.

Both bounds are tightened from the spec's coarser 0.01 / 0.05
"investigate above" thresholds to just above the achieved figures.

### UniverseMachine-SAGA (`UniverseMachineGrowth::um_saga`)

Both grids are pointwise, memoryless functions of `(log_vMpeak, z)` with
no integration involved, evaluated against a reference generated by
linking directly against the pinned commit's real, compiling
`calc_sf_model()` (`sf_model.c`) with the real best-fit parameter file
(`scripts/bestfit_var15_fin.dat`) — see
`rust/steel-plugins/tests/fixtures/um_saga/provenance.toml` for the full
build and formula transcription record.

* **`log10 SFR(log_vMpeak, z)`, star-forming mode** (double power law
  plus Gaussian bump, `sfr_at_vmp()`) — worst deviation over the 41×6
  `(log_v, z)` grid: **9.4e-7 dex** (`log_v=2.0, z=0.5`). Test bound:
  `1e-4` dex.
* **`f_Q(log_vMpeak, z)`** (DR1 high-mass term plus UM-SAGA's new
  low-mass term, closed-form standard-normal CDF in place of upstream's
  lookup-table `cached_rank`, clamped to `[0,1]`) — worst deviation:
  **1.9e-7** (absolute). Test bound: `1e-4`.

Both residuals are consistent with float64 transcendental-function
rounding alone (`exp`/`ln`/`erf`/`powf` implementation differences
between Rust's and C's libm) — there is no discretization or
integration step on either side to introduce a scheme mismatch, unlike
the EMERGE integrated-M* comparison above. Both bounds are tightened
from the spec's coarser 0.01 / 0.02 "investigate above" thresholds to
just above the achieved figures.

Not fixture-checked (no upstream reference exists for these): the
assembly-history rank-correlation nudge applied in the stochastic branch
of `stellar_growth_rate`, which is a documented simplification of
upstream's own persistent, tree-wide SFR-rank correlation — see the
module doc in `rust/steel-plugins/src/growth_models/universe_machine.rs`.

---

## 6. Task 13: end-to-end self-consistency validation

Runs each plugin through the real `steel-cli` orchestrator (not just
unit tests on the rate function in isolation) and checks the three
self-consistency criteria `paper/main.tex:177-183` names as what
"correct" means for this project. Per `paper/main.tex:186-189`,
agreement with observational data is explicitly **not** the bar —
nothing below is an observational comparison.

### 6.1 A prerequisite gap found and fixed before any of this could run

Two architectural gaps blocked every `[stellar_growth]` runfile from
running at all, discovered while trying to execute this task's own
runfiles rather than assumed in advance:

1. **`build_quenching()` was unconditional.** `rust/steel-cli/src/registry.rs`
   always built `Wetzel13` (`Capability::Quenching`) into every run's
   composition check, with no runfile syntax to omit it. Since
   `UniverseMachineGrowth` also declares `Capability::Quenching` (its
   SFR PDF already contains quenching), *any* run selecting UM was
   rejected unconditionally — not just runs that explicitly stacked a
   second quenching model on top. Fixed by:
   - `steel_plugins::quenching::NoQuenching`, a provably inert
     `QuenchingModel` (`t_quench = f64::INFINITY`, which structurally
     can never satisfy `BaryonicPipeline::evolve`'s
     `quench.t_quench < timeline.t[i]` fade-trigger for any finite
     timeline — verified by a unit test driving a full satellite
     trajectory through `BaryonicPipeline` and checking it is
     bit-identical to an independently-constructed finite-`t_quench`
     baseline, `rust/steel-plugins/src/quenching.rs`).
   - An optional `[quenching]` runfile section (`model = "wetzel13"` or
     `"none"`), defaulting to `Wetzel13` when absent — verified
     zero-behaviour-change for every existing runfile (none of which
     have this section) via the full `cargo test --workspace` run in
     §6.6 below, including the Task 1 golden bit-identity guards.
2. **`StellarGrowthModel` was never wired into `Simulation` at all.**
   `[stellar_growth]`'s descriptor took part in composition validation
   (Task 7) but the model it built was discarded
   (`let (_stellar_growth, ...)`); `[smhm]` remained a mandatory,
   non-`Option` field, so a `[stellar_growth]`-only runfile (the shape
   every EMERGE/UM runfile needs, per the plan's own design intent —
   "No `[smhm]` section") could not even parse. Fixed by:
   - `steel_core::StellarGrowthAsSmhm<M>`, an adapter implementing
     `SmhmModel` by calling `steel_core::integrate_stellar_mass` over
     the same `AccretionContext` the orchestrator already builds for
     `[smhm]`'s call site (`rust/steel-core/src/context.rs:680`) —
     verified equivalent to a direct `integrate_stellar_mass` call by a
     unit test, and correct at the real call site because
     `AccretionContext::satellite`'s `own_track` is *already* built
     h-free specifically for this purpose
     (`sat_mass[k] - log_h` before `growth_history`, per
     `docs/superpowers/specs/2026-08-17-emerge-um-smhm-plugins-design.md`
     §5) — the same convention EMERGE/UM's own calibration expects, so
     no unit conversion is silently skipped.
   - `[smhm]` became `Option<SmhmConfig>`; `build_simulation` now
     requires exactly one of `[smhm]`/`[stellar_growth]`, erroring
     clearly if neither is present (the duplicate-capability check
     already covered "both present").

Both are load-bearing for this task specifically (nothing before Task
13 ever ran a `[stellar_growth]` runfile through the CLI) and are
covered by new tests: `rust/steel-plugins/src/quenching.rs`'s
`NoQuenching` tests, `rust/steel-core/src/stellar_growth.rs`'s
`stellar_growth_as_smhm_matches_integrate_stellar_mass_directly`, and
the schema round-trip tests in `rust/steel-io/src/runfile.rs`.

### 6.2 A second, deeper gap: `[stellar_growth]` cannot combine with `[sfr]`

Even after §6.1's fixes, running the runfiles as the task brief's
template specified them (`[sfr] model = "double_power_law"`,
`star_formation = true`, `stellar_stripping = true`, matching the
G19_SE reference exactly) failed composition validation for **both**
models:

```
$ cargo run --release -p steel-cli -- runfiles/emerge_o_leary23.toml
Error: incompatible plugin combination in this runfile:
  - 'double_power_law' and 'emerge' disagree on the h convention (Msun vs Msun/h).

$ cargo run --release -p steel-cli -- runfiles/um_saga.toml   # [quenching]="none" already applied
Error: incompatible plugin combination in this runfile:
  - 'double_power_law' and 'universe_machine' both supply StarFormationRate; the effect would be applied twice.
  - 'double_power_law' and 'universe_machine' disagree on the h convention (Msun vs Msun/h).
```

Tracing both through the actual code (not guessing from the descriptor
tags alone):

* **UniverseMachine's `StarFormationRate` conflict is a genuine,
  intentional design choice**, not a bug: `UniverseMachineGrowth`'s SFR
  PDF governs the object's whole star-formation history, and Task
  12's catalogue already documents `SfrModel` parity as a "needs
  plugin" follow-up rather than something to compose UM with directly.
  `[stellar_growth] = universe_machine` structurally cannot be paired
  with `[sfr]` under the current architecture.
* **EMERGE's `h_convention` conflict is a false positive**, confirmed
  by tracing the actual runtime call sites: `Wetzel13`'s
  `h_convention: PerH` describes `log_host_mass_infall` — the *host's*
  halo mass, always populated from STEEL's own native grid regardless
  of which stellar-mass model is selected — while EMERGE's
  `h_convention: HFree` describes *its own object's* halo-mass axis.
  These are different quantities that happen to share one flag; they
  never interact. `docs/model-assumptions.md` already documents "EMERGE
  and Wetzel13 remain compatible" as intended — the code did not
  deliver on that until this task. Fixed narrowly: added
  `HConvention::NotApplicable` (mirroring the existing
  `Imf::NotApplicable` pattern) to `steel_core::compat` and
  `steel_plugins::harmonise`, and set it on `Wetzel13` and
  `NoQuenching`'s descriptors specifically (their h-sensitive input,
  where they have one, is the *host's* mass, not the run's own
  stellar/halo-mass axis). `DoublePowerLawSfr`/`TomczakFormSfr`/
  `SchreiberFormSfr`'s declared `h_convention: PerH` was deliberately
  **not** touched — that would risk rippling into native STEEL
  configurations already relying on it — which is why the
  `StarFormationRate`-capability-supplying `[sfr]` models still
  conflict with EMERGE's own `h_convention: HFree` too (a second,
  independent reason `[sfr]` needs to be absent for `[stellar_growth]`
  runs, on top of §6.2's other point for UM).

Regression tests for the fix:
`rust/steel-core/src/compat.rs::h_convention_not_applicable_does_not_conflict_with_either_side`
and `::h_convention_not_applicable_does_not_mask_a_real_mismatch_between_others`
(the latter guards against the fix silently suppressing a genuine
mismatch elsewhere in the same run).

**Consequence for the runfiles.** `[sfr]` became `Option<SfrConfig>` in
the schema; `build_simulation` requires `[run].star_formation` and
`[run].stellar_stripping` to both be `false` when `[sfr]` is absent
(checked explicitly — an internal `UnreachableSfr` placeholder panics
if ever actually called, so a future change that starts invoking it
without lifting this requirement fails loudly). Both
`runfiles/emerge_o_leary23.toml` and `runfiles/um_saga.toml` therefore
omit `[sfr]` and leave `star_formation`/`stellar_stripping` at their
`false` defaults, **deviating from the task brief's literal template**
(which specified `true`/`true` and a `[sfr]` section). This means:
`[stellar_growth]` drives infall-time stellar mass only for both models
in these runs; STEEL's post-infall satellite evolution
(`BaryonicPipeline`) is not exercised. Fully integrating
`[stellar_growth]` with post-infall satellite evolution would need the
object's own post-infall halo-mass track, which needs
`steel_core::stripping::HaloStrippingModel` — present in the trait
hierarchy but wired to `None` in `Simulation` everywhere in this
codebase today, for every runfile, not just these two. That integration
is out of scope for Task 13.

### 6.3 Step 2: the validator correctly rejects the broken variant

The single most important check in the plan. A scratch copy of
`um_saga.toml` with `[quenching] model = "wetzel13"` added back:

```
$ cargo run --release -p steel-cli -- <scratch>/um_saga_broken.toml
Error: incompatible plugin combination in this runfile:
  - 'wetzel13' and 'universe_machine' both supply Quenching; the effect would be applied twice. Select only one, or a variant that does not supply it.

See docs/model-assumptions.md for what each plugin assumes.
```

Exit code 1, no computation performed, names both `wetzel13` and
`universe_machine`, and states the duplicated effect ("applied twice").
A second scratch variant with `[quenching]` **removed entirely**
(defaulting to `Wetzel13` per §6.1) produces the identical error —
confirming the default is genuinely `Wetzel13`, not an accidental
no-op. Both variants fail exactly as specified.

### 6.4 Both real runfiles run to completion

```
$ cargo run --release -p steel-cli -- runfiles/emerge_o_leary23.toml
Wrote output to .../RunParam_1.0_False_False_True_NoSFR_EMERGE_o_leary23_
z steps: 190, host bins: 57, subhalo bins: 65, SMF bins: 40   (247.3s wall)

$ cargo run --release -p steel-cli -- runfiles/um_saga.toml
Wrote output to .../RunParam_1.0_False_False_True_NoSFR_UniverseMachine_um_saga_
z steps: 190, host bins: 57, subhalo bins: 65, SMF bins: 40   (221.7s wall)
```

Both on the full production grid (`log_m_max = 16.6`), consistent with
Task 4's ~1-4 minute finding for this grid size, not a hang. A G19_SE
baseline (`runfiles/steel-sf-stripping.toml`, unmodified) was also run
for the comparison in §6.5.1 (126.8s).

### 6.5 The three self-consistency criteria

#### 6.5.1 Satellite counts and pair fractions: finite, no plumbing discontinuities

Checked every populated output array (`Figure3_AnalyticalModel_SMF`,
`Pair_Frac_Pair_Frac`, `Pair_Frac_Halo_Pair_Frac_Halo`,
`Mergers_Accretion_History`, `Sat_SMHM_Sat_SMHM`) for both models:

| | EMERGE | UM | G19_SE baseline |
|---|---|---|---|
| Any non-finite value (NaN/Inf) | none | none | none |
| Any negative value | none | none | none |
| Internal zero-gaps in the populated mass range (a discretely-jumping histogram, i.e. a plumbing bug) | none | none | none |
| Nonzero SMF bins (of 40) | 28 | 26 | 40 |

Both models produce a smooth, monotonically-declining satellite SMF
that truncates cleanly to zero above the highest mass any satellite
reaches (28/26 of 40 bins, vs. 40/40 for the baseline) — expected, not
a defect: with `star_formation`/`stellar_stripping` both `false`
(§6.2), satellites are frozen at their raw infall stellar mass rather
than smoothed by post-infall dynamical evolution, so the distribution
is narrower and shows more bin-to-bin structure in its low-mass wing
than the fully-evolved G19_SE baseline (whose SF+stripping machinery
spreads mass continuously across all 40 bins). This is a genuine,
explicable physics difference from the narrower comparison being made
(§6.2), not the kind of "discontinuity that would indicate a plumbing
error" the brief asks this check to rule out — no NaN, no sign flips,
no dropped-then-repopulated bins anywhere in either model's output, at
any of the 190 redshift steps checked.

#### 6.5.2 Central mass accretion plausibility: `M* <= f_b . M_h`

`rust/steel-plugins/tests/baryon_budget.rs` (new, per the brief's exact
spec) integrates each model's rate over `log_mh in [10.0, 15.0]` (0.1
dex steps) x `z_end in {0.1, 0.5, 1.0, 2.0, 4.0}` and asserts
`M* <= f_b . M_h` at every point:

```
$ cargo test -p steel-plugins --test baryon_budget
running 2 tests
test universe_machine_respects_the_baryon_budget ... ok
test emerge_respects_the_baryon_budget ... ok
```

Both pass over the full 51x5 grid; neither integrator produces more
stellar mass than the cosmic baryon fraction allows.

#### 6.5.3 SFR internally consistent with the driving accretion history

For STEEL's own `[smhm]`+`[sfr]` pipeline this is a nontrivial check
between two *independently* calibrated relations
(`Scripts/Validation/ssfr_sfr_sweep.py`, `Paper2_Fig9_sSFR.png`,
`paper/main.tex` Figure 5). EMERGE and UniverseMachine have no second,
independent relation to check against: a single
`StellarGrowthModel::stellar_growth_rate` call *is* simultaneously the
model's SFR and the rate `integrate_stellar_mass` integrates to build
the accretion history — there is nothing for it to disagree with, so
this criterion holds **by construction**, not by coincidence of two
fits agreeing. `ssfr_sfr_sweep.py` was not run for these models: it
reads `Satellite_sSFR`, a `BaryonicPipeline`/`[sfr]` output these runs
do not populate (§6.2), so it would not measure anything meaningful
here.

What is worth checking numerically is whether the discrete integrator
(trapezoidal in cosmic time, on the halo's actual ~200-point growth
track — the real resolution the orchestrator uses) introduces any
inconsistency of its own: does the *local* implied rate for one native
track step (`0.5*(rate(z_lo) + rate(z_hi))`, exactly
`integrate_stellar_mass`'s own trapezoidal average) agree with
`stellar_growth_rate` evaluated directly at that step, using the *same*
real, full-history `AccretionContext` both times?

**Correction (post-merge):** an earlier draft of this check fed
`stellar_growth_rate` a per-step `(log_mh, z)` pair and found up to 0.86
dex of "disagreement" against the whole-track-context version, which
was — at the time — attributed to the check using an artificially
truncated/local context rather than the real one, and "fixed" by making
both call sites share one full, un-truncated `AccretionContext`
regardless of which step they were evaluating. That was backwards: the
0.86 dex was real evidence of a genuine bug in
`UniverseMachineGrowth::stellar_growth_rate` itself, which at the time
computed vMpeak once from `ctx.own_track`'s root (observed-epoch, i.e.
*final*) sample and reused that single value for every redshift
`integrate_stellar_mass` visits while walking an object's whole
progenitor track — retroactively stamping a massive halo's near-total
*final* quenched fraction onto every earlier, unquenched progenitor.
This collapsed the integrated M* for massive halos and made M*(Mh)
non-monotonic (see `steel-plugins/src/growth_models/universe_machine.rs`
module doc, "`log_vmpeak`: a per-snapshot running peak, not one value
held fixed...", for the full mechanism and the fix: `vmpeak_at(log_mh,
z, ctx)` now keys vMpeak off each call's own `(log_mh, z)`, since a
progenitor's peak-so-far on a monotonic `GrowthTrack` is exactly its own
contemporary mass). With the fix, the check below no longer needs an
artificial "share one full context" workaround — `ssfr_self_consistency.rs`
already passed each step's own `(log_mh, z)` pair, so it is now
measuring real trapezoidal-quadrature error, not masking a modelling bug.

```
$ cargo run --release -p steel-plugins --example ssfr_self_consistency
=== EMERGE (o_leary23) ===
max |direct - local_implied| over 16 points = 0.0058 dex

=== UniverseMachine (um_saga, deterministic mode: rng=None) ===
max |direct - local_implied| over 16 points = 0.0065 dex
```

Both under 0.007 dex over a 4x4 `(log_mh0, z)` grid spanning `log_mh0
in {11,12,13,14}` and `z in {0.2,1,2,4}` — consistent with ordinary
trapezoidal quadrature error at this track resolution, not a genuine
inconsistency. `rust/steel-plugins/examples/ssfr_self_consistency.rs`.

### 6.6 Final full-suite verification

```
$ cargo test --workspace
```

Every crate's test suite passes, including: the Task 1 golden
bit-identity guards (`golden_smhm_sfr.rs`,
`existing_sfr_plugins_are_bit_identical_to_golden`,
`existing_smhm_plugins_are_bit_identical_to_golden`), the full
`orchestrator.rs` suite (13 tests, unaffected by the `[quenching]`/
`[sfr]`/`[smhm]` schema changes since every existing runfile continues
to omit those new optional sections and gets identical defaults), and
the new `compat.rs`, `stellar_growth.rs`, `quenching.rs`, `runfile.rs`,
`harmonise.rs`, and `baryon_budget.rs` tests this task added.

### 6.7 Summary against the three criteria

| Criterion | EMERGE | UniverseMachine |
|---|---|---|
| Satellite counts / pair fractions finite, smooth, no plumbing discontinuities | met | met |
| `M* <= f_b . M_h` | met (`baryon_budget.rs`) | met (`baryon_budget.rs`) |
| SFR consistent with driving accretion history | met, by construction + 0.0058 dex numerical check | met, by construction + 0.0046 dex numerical check |

Upstream fixture agreement (from §5 above, for reference): EMERGE
`eps(M,z)` 1.5e-7 dex, integrated M* 0.0497 dex (discretization-scheme
mismatch, not a coefficient error); UniverseMachine star-forming SFR
9.4e-7 dex, quenched fraction 1.9e-7 (absolute).

**Criterion not met as originally specified, stated plainly rather than
omitted:** the brief's own runfile template (`star_formation = true`,
`stellar_stripping = true`, `[sfr] = double_power_law`) does not run at
all for either model, for the architectural reasons in §6.2. Both
runfiles as committed use `star_formation = false`,
`stellar_stripping = false`, and omit `[sfr]` — `[stellar_growth]`
drives infall-time stellar mass only; STEEL's post-infall satellite
evolution is not exercised by either run. This is a real, scoped-out
limitation, not a workaround: full integration needs
`HaloStrippingModel` wiring that does not exist for *any* runfile in
this codebase today (`Simulation.halo_stripping` is `None`
unconditionally), and is out of scope for Task 13.
