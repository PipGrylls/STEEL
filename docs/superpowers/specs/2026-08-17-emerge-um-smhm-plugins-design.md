# EMERGE and UniverseMachine as STEEL plugins

**Date:** 2026-08-17
**Status:** design approved, pending implementation plan

## 1. Purpose

Add EMERGE (O'Leary, Steinwandel, Moster, Martin & Naab 2023,
arXiv:2301.07122, building on Moster, Naab & White 2018) and
UniverseMachine / UM-SAGA (Wang, Nadler, Mao, Wechsler, Behroozi et al.
2024, arXiv:2404.14500, building on Behroozi et al. 2019) to `rs-steel`
as first-class plugins, so that each can be run through STEEL's own
statistical accretion-history machinery and compared against STEEL's
native SMHM family on equal terms.

This is a direct test of the claim in `paper/main.tex:157-175`: that the
port's dependency-injection structure reduces the cost of adding a new
prescription to "pointing it at the paper describing the physics and the
trait it must implement." How cleanly this lands is itself a result, and
the friction encountered should be recorded, not smoothed over.

Two secondary deliverables follow from the comparison being *fair*
rather than merely *possible*: a harmonisation layer for unit and
definition mismatches (§7), and a composition-validation mechanism that
rejects physically incoherent plugin combinations (§8).

## 2. Decisions taken

| Decision | Choice | Rationale |
|---|---|---|
| Fidelity | Native mechanism via proper traits | An effective-M*(Mh,z)-map shortcut bakes each model's own accretion history into the input, making STEEL's sSFR self-consistency check circular. |
| Upstream code | Clone, run, pin outputs as committed fixtures | Keeps the build pure `cargo` with no vendored C or MPI, while empirically pinning our Rust to upstream numerics. Mirrors the `getPWGH` precedent (`paper/main.tex:220`, agreement to 0.0021 dex). |
| Assumption parity | Catalogue + harmonisation layer only | Unit/definition conversions are mandatory for any valid overlay; trait-level parity plugins are optional depth and get their own specs. |
| Satellite history | Object's own pre-infall central track | Superseded an earlier decision to use a proxy. See §5. |
| Composition validation | Rigid declared-descriptor rule set now; derived contracts later | §8. |

## 3. Key structural finding: these are rate models, not mass maps

STEEL's `SmhmModel` is a **memoryless mass map**: it answers
"what M* corresponds to this Mh at this z?" (`Functions.py:625-778`,
ported in `rust/steel-plugins/src/smhm/`).

Neither new model is that shape.

**EMERGE** parametrises an instantaneous baryon conversion efficiency
applied to the halo's *accretion rate*:

```
dm*/dt = eps(M_h, z) . Mdot_b,      Mdot_b = f_b . dM_h/dt
```

M* is the **time integral of that rate along the growth track**. The
`eps` double power law,

```
eps(M) = 2 eps_N / [ (M/M_1)^-beta + (M/M_1)^gamma ]
```

is algebraically identical to the Moster form already implemented in
`rust/steel-plugins/src/smhm/moster.rs`, but its *meaning* is different:
`eps` multiplies a rate, whereas `MosterFormSmhm` multiplies a mass.
Substituting EMERGE's published coefficients into `MosterFormSmhm` would
silently reinterpret a rate efficiency as a mass ratio and yield a
confidently wrong SMHM curve. This trap must be called out in the
`emerge.rs` module documentation.

Correspondingly, the stellar-to-halo mass relation in O'Leary+2023's
title is an **output** of integration, not an input parametrisation.

**UniverseMachine** parametrises SFR directly from halo properties: a
double power law in peak circular velocity `vMpeak`, drawn from a
bimodal (star-forming / quenched) PDF, rank-correlated with the
assembly-history proxy `Delta vmax`. It is not an SMHM relation at all.

Both models are therefore **rate-based**, and one new trait serves both.
`SmhmModel` remains a mass map for STEEL's native abundance-matching
family and is not repurposed.

## 4. Trait design

### 4.1 `AccretionContext`

A read-only view of history and environment, passed to every
mass-assigning plugin. All fields are shared references or `Copy`
scalars: no allocation, no cloning, no interior mutability.

```rust
/// Read-only accretion history and environment available at the point a
/// galaxy's stellar mass or growth rate is assigned.
pub struct AccretionContext<'a> {
    /// Main-progenitor track of *this* object treated as a central:
    /// down to z0 for a central, down to z_infall for a satellite.
    /// Always present.
    pub own_track: &'a GrowthTrack,
    /// Main-progenitor track of the host halo. `None` for centrals.
    pub host_track: Option<&'a GrowthTrack>,
    /// Infall redshift. `None` for centrals.
    pub z_infall: Option<f64>,
    /// Peak halo mass [log10 Msun], where it differs from the current
    /// mass. `None` when the caller cannot distinguish them.
    pub log_m_peak: Option<f64>,
    pub cosmology: &'a dyn Cosmology,
    /// Mass definition the `log_dm` / `log_mh` arguments are expressed in.
    pub mass_definition: MassDefinition,
}
```

### 4.2 Widened existing traits

```rust
pub trait SmhmModel: Send + Sync {
    fn stellar_mass(
        &self,
        log_dm: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64;
}

pub trait SfrModel: Send + Sync {
    fn log_sfr(&self, log_sm: f64, z: f64, ctx: &AccretionContext<'_>) -> f64;
}
```

The three existing `SmhmModel` implementations (`MosterFormSmhm`,
`BehrooziFormSmhm`, `RodriguezPuebla17`) gain one ignored parameter and
are otherwise unchanged. Production call sites to update:

- `rust/steel-core/src/context.rs:642` and `:644` — `stellar_mass`
- `rust/steel-fit/src/smf.rs:61` — `stellar_mass`
- `rust/steel-core/src/baryonic.rs:197` and `:249` — `log_sfr`
- `rust/steel-postprocess/src/central_evolution.rs:84` — `log_sfr`

Six sites total, plus tests and examples.

### 4.3 New `StellarGrowthModel`

```rust
/// Rate-based stellar mass assembly. M* is obtained by integrating this
/// rate along the object's growth track, in contrast to `SmhmModel`,
/// which returns M* directly.
pub trait StellarGrowthModel: Send + Sync {
    /// log10 dM*/dt [Msun/yr] for a halo of mass `log_mh` [log10 Msun]
    /// at redshift `z`.
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64;
}
```

`rng` is present because UM's SFR is drawn from a bimodal PDF, so the
rate is intrinsically stochastic rather than a mean relation with
scatter bolted on afterwards.

A model supplies M* through *either* `SmhmModel` or
`StellarGrowthModel`, never both; §8 enforces this.

## 5. Infall and satellite treatment

At the moment of infall the infalling object *was* a central, and so has
a legitimate central assembly history. STEEL can obtain it from the
existing growth model:

- **Central**: `growth_history(log_m0, 0.0)` — unchanged from today.
- **Satellite, pre-infall**: `growth_history(sat_mass[k] - log_h, z[i])`
  — its genuine main-progenitor history as the central it was until
  infall.
- **Satellite, post-infall**: already supplied by `HaloStrippingModel`
  via `HaloStrippingTrack`.

No proxy is required. This is the same average-MAH-per-mass-bin
approximation STEEL already makes for host halos, applied consistently
to subhalos; it is not a new approximation.

Consequences:

- EMERGE's reionization gate is **exact** for satellites, not proxied.
  This matters because the gate governs the dwarf regime, which is
  predominantly satellites.
- UM's `Delta vmax` is derivable from the same track.
- The residual limitation is that subhalo histories are *average* MAHs
  per mass bin rather than individual merger trees — a limitation STEEL
  already carries for hosts and already documents.

### 5.1 Prerequisite: `z0 != 0` coverage

`VandenBosch14` is `z0`-general by construction (`redshift_grid` uses
`log10(1+z0)`; `growth_history` takes `delta_collapse(z0)`), but **every
existing test passes `z0 = 0.0`**
(`rust/steel-plugins/src/halo_growth.rs:150-176`). The code path this
design depends on is therefore unexercised.

Before anything relies on it, add:

1. Tests that `growth_history(m, z0)` for `z0 in {0.5, 1, 2, 4}` starts
   at `m`, is monotonically decreasing into the past, and returns a grid
   of `N_Z` points beginning at `z0`.
2. A continuity check: a track requested at `z0 = z_infall` joins the
   host's track at that epoch without offset or sign error.

### 5.2 Cost

Growth tracks are precomputed once per (mass bin, z0) pair and cached,
following the existing pattern at `rust/steel-core/src/context.rs:425-433`.
Satellite tracks add roughly 190 redshift steps x 5 subhalo bins = 950
root-finds, against 56 for hosts today. Each is a 200-point root-find
computed once at startup. Negligible against total run cost.

## 6. Upstream fixture harness

Upstream repositories, both verified reachable and publicly cloneable on
2026-08-17:

| Model | URL | Ref | HEAD SHA |
|---|---|---|---|
| EMERGE | `https://github.com/bmoster/emerge.git` | `v1.0.2` | `2781b54c21a80acf237daf7f2e71ff6254da8c3b` |
| UM-SAGA | `https://bitbucket.org/RW-Stanford/universemachine-saga.git` | `saga` | `6aff8d792e81bf6049058e3e1bc6f2cfa616b525` |

STEEL is AGPL-3.0, which is compatible with incorporating GPL'd upstream
code. Nonetheless upstream is cloned **out of tree** (into a scratchpad,
never committed): both are MPI whole-pipeline programs with global state,
not callable libraries, so vendoring would add a C toolchain and
cross-platform build burden for no fidelity gain over pinned fixtures.

Procedure:

1. Clone at the pinned ref into a scratch directory.
2. Compile and run each at a cosmology matched to STEEL's run cosmology.
3. Dump reference grids: for EMERGE, `eps(M_h, z)` and integrated
   `M*(M_h, z)`; for UM, `SFR(vMpeak, z)` percentiles and the
   quenched fraction `f_Q(vMpeak, z)`.
4. Commit **only** the resulting `.npy` fixtures plus a
   `provenance.toml` recording upstream URL, commit SHA, cosmology
   parameters, build flags, and exact command line.
5. Validate the Rust implementations against those fixtures and report
   the agreement in dex, in the same style as the existing three-way
   validation (`docs/VALIDATION.md`).

The acceptance tolerance is deliberately not fixed in advance: it cannot
be known before the fixtures exist. The procedure is to measure the
achieved agreement, record it as the reference figure for that plugin
(as `getPWGH`'s 0.0021 dex is recorded), and treat any later regression
beyond it as a failure. Where achieved agreement is worse than roughly
0.01 dex, the cause must be identified and documented rather than
absorbed by widening the tolerance.

### 6.1 Published parameter values must be verified from source

Parameter values gathered during design came from an automated summary
of the arXiv HTML renderings and are **provisional**. Equation *shapes*
were cross-checked and are trusted; individual digits are not. Every
numeric coefficient must be re-read from the published PDF tables or the
upstream parameter files before being committed, and the source of each
recorded inline. This applies in particular to:

- EMERGE: `beta_0`, `beta_z`, `tau_s`, `M_q`, `a_q`, `R_q`, and the
  reference-model `eps` coefficients.
- UM-SAGA: the 15 fitted parameters, of which 9 are inherited from UM DR1
  and 6 are the new low-mass quenching terms.

## 7. Harmonisation layer

Pure unit and definition conversions, introducing no new physics. These
are the failure modes that produce a plausible-looking wrong overlay.

| Dimension | Issue | Treatment |
|---|---|---|
| Halo mass definition | Upstreams use overdensities differing from STEEL's `Vir` | Convert via existing `MassDefinition` and `Cosmology::m_to_r`; a genuine conversion, not a relabelling |
| `h` convention | STEEL carries `Msun/h` internally (see `sat_mass[k] - log_h`, `context.rs:639`); upstreams differ | Explicit conversion at the plugin boundary |
| IMF / SPS | Chabrier vs Kroupa is roughly 0.05-0.25 dex, comparable to the entire signal under comparison | Declared per plugin; conversion applied on a single documented offset |
| `Mpeak -> vMpeak` | UM is velocity-keyed; STEEL is mass-keyed throughout | Requires an injected concentration-mass relation, which is itself a catalogued assumption and a named dependency, not an implementation detail |

The concentration-mass relation needed for `Mpeak -> vMpeak` is a
modelling choice that affects UM's results. It must be selectable and
recorded in the run configuration, not hardcoded.

## 8. Composition validation

### 8.1 Rigid mechanism (this spec)

The dangerous failures are not type errors; they are **silent
double-counting** of a physical effect, which yields plausible output.

A literal N-by-N "model A x model B" matrix is rejected: it requires a
new row *and* column per plugin, and is no stricter than the equivalent
rule set over declared metadata. Instead each plugin declares its
assumptions, and a fixed rule set is checked once at wiring time.

```rust
pub struct PluginDescriptor {
    pub id: &'static str,
    pub imf: Imf,                        // Chabrier | Kroupa | Salpeter | NotApplicable
    pub mass_definition: MassDefinition,
    pub h_convention: HConvention,
    pub calibrated_cosmology: Option<CosmologyTag>,
    /// Exclusive capabilities this plugin supplies.
    pub provides: &'static [Capability], // StellarMass | Quenching | Scatter | ...
}

pub trait DescribedPlugin {
    fn descriptor(&self) -> PluginDescriptor;
}

/// Runs once when the orchestrator wires a run. Hard error, never a warning.
pub fn validate_composition(
    descriptors: &[PluginDescriptor],
) -> Result<(), Vec<Incompatibility>>;
```

Rules:

1. No two plugins may `provide` the same `Capability`.
2. All `imf` values must agree, or a declared conversion must exist.
3. All `mass_definition` values must agree, or a declared conversion
   must exist.
4. All `h_convention` values must agree, or a declared conversion must
   exist.
5. `calibrated_cosmology`, where `Some`, must match the run cosmology.

Incompatibilities this catches, all live in this plan:

| Conflict | Why it is silent | Rule |
|---|---|---|
| UM + STEEL `QuenchingModel` | UM's bimodal SFR PDF already contains quenching | duplicate `Capability::Quenching` |
| `StellarGrowthModel` + `SmhmModel` both driving M* | two sources of truth for M* | duplicate `Capability::StellarMass` |
| SMHM scatter + growth-model scatter | scatter applied twice; sigma inflated | duplicate `Capability::Scatter` |
| EMERGE (Chabrier) + STEEL sSFR (Kroupa) | offset comparable to the measured signal | `imf` mismatch |
| Plugin fitted under WMAP7, run under Planck15 | wrong normalisation, no error raised | `calibrated_cosmology` mismatch |

Validation must fail the run at startup with an actionable message. It
must never downgrade to a warning: a warning in a batch run is
indistinguishable from silence.

**Stated limitation.** This mechanism only detects conflicts along
dimensions enumerated above. It cannot detect a novel incompatibility
nobody thought to declare. That is the motivation for §8.2.

### 8.2 Planned extension: derived contracts for agent-built plugins

Recorded as the next logical step, to be its own spec.

1. **Derived rather than declared compatibility.** Plugins publish
   machine-readable `requires` / `provides` contracts; the system infers
   validity by unification over assumption dimensions instead of
   consulting a hand-maintained rule set, and automatically inserts
   conversion adapters where a registered conversion exists.
2. **`steel validate` CLI plus a published descriptor schema.** An agent
   authoring a plugin self-checks before submitting. Failures are
   actionable diagnostics, not panics.
3. **Property-based cross-validation.** Automatically enumerate trait
   combinations and assert physical invariants: M* monotonic in Mh,
   sSFR non-negative, integrated M* <= f_b . Mh, realised scatter
   variance consistent with the declared sigma. This is what catches
   incompatibilities nobody declared, and is the substantive reason the
   extension exists rather than being cosmetic.

## 9. Assumptions catalogue

Deliverable: `docs/model-assumptions.md`.

A per-trait gap table across {STEEL, EMERGE, UM} covering all eleven
traits (`Cosmology`, `HaloGrowthModel`, `HaloMassFunctionModel`,
`SubhaloMassFunctionModel`, `MergerTimescaleModel`,
`HaloStrippingModel`, `SmhmModel`, `SfrModel`, `QuenchingModel`,
`GasMassModel`, `StellarStrippingModel`), plus the new
`StellarGrowthModel`.

Each row classified as one of:

- **identical** — no action
- **convertible** — handled by §7
- **needs plugin** — candidate for a follow-up spec
- **structurally absent** — the model has no analogue; a documented
  asymmetry in any comparison

The catalogue must record, explicitly:

- The average-MAH-per-mass-bin limitation for subhalos (§5).
- The concentration-mass relation dependency introduced by
  `Mpeak -> vMpeak` (§7).
- Which parity plugins are recommended as follow-up specs, ranked by how
  much each affects the comparison.

## 10. Testing

Test-driven throughout; tests precede implementation.

**Refactor guard (first, before any new model).** A regression test
asserting the three existing `SmhmModel` implementations produce
bit-identical output before and after the signature widening, over a
grid of `(log_dm, z)` and a fixed RNG seed. The widening must be proven
inert before new physics is added.

**Prerequisite.** The `z0 != 0` suite of §5.1.

**Per model.**

1. Unit tests on the rate function against hand-computed values from the
   published equations.
2. An integration test that integrating the rate along a `GrowthTrack`
   reproduces the upstream fixture within the stated dex tolerance.
3. Property tests: M* monotonic in Mh; integrated M* never exceeding
   `f_b . Mh`; the EMERGE gate suppressing late-forming low-mass halos
   in the correct direction.

**Composition.** Tests that each conflict in §8.1's table is rejected by
`validate_composition`, and that valid compositions pass.

**End to end.** A full `rs-steel` run with each new model, checked
against the three self-consistency criteria of `paper/main.tex:177-183`:
satellite counts and pair fractions, plausible central mass accretion,
and SFR internally consistent with the driving accretion history.

## 11. Out of scope

- Trait-level parity plugins for non-SMHM assumptions (§9 identifies
  them; each gets its own spec).
- Per-object subhalo merger trees. STEEL is statistical by design;
  changing that is a structural change with its own justification.
- The derived-contract mechanism of §8.2.
- Any DECODE (Fu, Shankar et al. 2022) comparison. Its abundance-matching
  extension of STEEL is disputed on separate grounds and is not part of
  this work.
- Fitting or re-calibrating either model. Published parameters are used
  as given.

## 12. Risks

| Risk | Mitigation |
|---|---|
| Upstream codes fail to build or need unavailable inputs | Discover early: build both before any Rust is written. If one cannot run, fall back to published tabulated data and record the weakened validation explicitly. |
| Provisional parameter values propagate into results | §6.1: every coefficient re-read from the PDF or upstream parameter file, with its source recorded inline. |
| `z0 != 0` growth path proves subtly wrong | §5.1 tests it directly, before dependence. |
| Signature widening perturbs existing results | §10's refactor guard demands bit-identical output. |
| The comparison remains unfair along an unenumerated dimension | Acknowledged, not solved. §8.1 states the limitation; §8.2 addresses it. |
