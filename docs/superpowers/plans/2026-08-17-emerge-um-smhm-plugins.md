# EMERGE and UniverseMachine STEEL Plugins Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add EMERGE and UniverseMachine as first-class `rs-steel` plugins so each runs through STEEL's own accretion-history machinery and can be compared against STEEL's native SMHM family on equal terms.

**Architecture:** Both new models are *rate-based* (M* is the time integral of a growth rate along a halo's accretion history), unlike STEEL's `SmhmModel`, which is a memoryless M*(Mh,z) map. So we add a `StellarGrowthModel` trait for them, widen `SmhmModel`/`SfrModel` to carry a read-only `AccretionContext` (giving every model equal access to history), and validate plugin compositions to catch silent double-counting.

**Tech Stack:** Rust 2021, cargo workspace at `rust/`. `ndarray` 0.16, `ndarray-npy` 0.9, `rand` 0.8, `rand_distr` 0.4, `anyhow` 1, `thiserror` 1, `serde` 1, `toml` 0.8. Upstream reference codes are C + MPI, built out-of-tree.

**Spec:** `docs/superpowers/specs/2026-08-17-emerge-um-smhm-plugins-design.md`

## Global Constraints

- All work happens in the `rust/` cargo workspace. Build: `cargo build --workspace`. Test: `cargo test --workspace`.
- Edition 2021. Workspace dependency versions are pinned in `rust/Cargo.toml` `[workspace.dependencies]`; use `{ workspace = true }`, never a fresh version string.
- Every new trait must be `Send + Sync` — the orchestrator runs under `rayon`.
- `AccretionContext` carries only shared references and `Copy` scalars. No allocation, no cloning, no interior mutability.
- A model supplies M* through *either* `SmhmModel` or `StellarGrowthModel`, never both.
- Composition validation is a **hard error at startup**, never a warning.
- Published parameter values are **provisional** until re-read from the paper PDF tables or upstream parameter files. Record the source of every coefficient inline. Never commit a coefficient sourced only from an HTML summary.
- Upstream code is cloned **out of tree** into the scratchpad and never committed. Only `.npy` fixtures + `provenance.toml` are committed.
- House comment style: every plugin module documents its provenance (paper citation, and the `Functions.py` line range where a port exists). Match the existing tone in `rust/steel-plugins/src/smhm/moster.rs`.
- Existing `.npy` output must stay byte-compatible with the Python's numpy serialisation (`steel-io`).

---

## File Structure

**New files:**

| Path | Responsibility |
|---|---|
| `rust/steel-core/src/accretion.rs` | `AccretionContext<'a>` — read-only history/environment view |
| `rust/steel-core/src/stellar_growth.rs` | `StellarGrowthModel` trait + `integrate_stellar_mass` |
| `rust/steel-core/src/compat.rs` | `PluginDescriptor`, `Imf`, `HConvention`, `CosmologyTag`, `Capability`, `Incompatibility`, `DescribedPlugin`, `validate_composition` |
| `rust/steel-plugins/src/harmonise.rs` | Unit/definition conversions; `ConcentrationMassRelation`, `DuttonMaccio14`, `mpeak_to_vmax` |
| `rust/steel-plugins/src/growth_models/mod.rs` | Re-exports for rate-based models |
| `rust/steel-plugins/src/growth_models/emerge.rs` | EMERGE ε double power law + reionization gate |
| `rust/steel-plugins/src/growth_models/universe_machine.rs` | UM SFR(vMpeak, Δvmax, z), bimodal PDF, quenched fraction |
| `rust/steel-plugins/tests/golden_smhm_sfr.rs` | Refactor guard: bit-identical existing plugins |
| `rust/steel-plugins/tests/fixtures/emerge/` | Committed upstream reference grids + `provenance.toml` |
| `rust/steel-plugins/tests/fixtures/um_saga/` | Committed upstream reference grids + `provenance.toml` |
| `rust/steel-plugins/tests/upstream_agreement.rs` | Validates Rust against committed fixtures |
| `docs/model-assumptions.md` | Per-trait gap table across {STEEL, EMERGE, UM} |
| `scripts/fixtures/build_emerge_fixture.sh` | Reproducible upstream clone/build/run for EMERGE |
| `scripts/fixtures/build_um_fixture.sh` | Reproducible upstream clone/build/run for UM-SAGA |

**Modified files:**

| Path | Change |
|---|---|
| `rust/steel-core/src/lib.rs` | Register + re-export new modules |
| `rust/steel-core/src/smhm.rs` | Widen `stellar_mass` with `ctx` |
| `rust/steel-core/src/sfr.rs` | Widen `log_sfr` with `ctx` |
| `rust/steel-plugins/src/smhm/{moster,behroozi,rodriguez_puebla}.rs` | Accept + ignore `ctx`; add `DescribedPlugin` |
| `rust/steel-plugins/src/sfr.rs` | Accept + ignore `ctx`; add `DescribedPlugin` |
| `rust/steel-plugins/src/halo_growth.rs` | Add `z0 != 0` tests |
| `rust/steel-plugins/src/lib.rs` | Export `harmonise`, `growth_models` |
| `rust/steel-core/src/context.rs:425-433` | Cache satellite own-tracks |
| `rust/steel-core/src/context.rs:638-647` | Build + pass `AccretionContext` |
| `rust/steel-core/src/baryonic.rs:197,249` | Pass `ctx` to `log_sfr` |
| `rust/steel-postprocess/src/central_evolution.rs:84` | Pass `ctx` to `log_sfr` |
| `rust/steel-fit/src/smf.rs:61` | Pass `ctx` to `stellar_mass` |
| `rust/steel-io/src/runfile.rs` | `[stellar_growth]` section; `[compat]` overrides |
| `rust/steel-cli/src/registry.rs` | Build rate-based models; run `validate_composition` |

**Pre-flight note for the executor:** `rust/Cargo.toml` declares `license = "MIT"` while the repo `LICENSE` is AGPL-3.0. Raise this with the maintainer before Task 8 — the spec's §6 licensing analysis assumes AGPL-3.0. Do not resolve it yourself.

---

### Task 1: Golden-value regression fixture for existing plugins

Locks current behaviour **before** any signature change, so Task 2 can be proven inert. No production code changes here.

**Files:**
- Create: `rust/steel-plugins/tests/golden_smhm_sfr.rs`

**Interfaces:**
- Consumes: existing `SmhmModel::stellar_mass(log_dm, z, rng)`, `SfrModel::log_sfr(log_sm, z)` (pre-widening signatures).
- Produces: `tests/golden_smhm_sfr.rs` with hardcoded expected values that Task 2 must reproduce bit-for-bit.

- [ ] **Step 1: Write the generator test that prints current values**

Create `rust/steel-plugins/tests/golden_smhm_sfr.rs`:

```rust
//! Refactor guard. These values were captured from the pre-
//! `AccretionContext` signatures and must remain bit-identical through
//! the trait widening. A change here means the widening was not inert.

use rand::rngs::StdRng;
use rand::SeedableRng;
use steel_core::{SfrModel, SmhmModel};
use steel_plugins::{
    BehrooziFormSmhm, DoublePowerLawSfr, MosterFormSmhm, RodriguezPuebla17, SchreiberFormSfr,
    TomczakFormSfr,
};

const LOG_DM: [f64; 5] = [10.0, 11.0, 12.0, 13.0, 14.0];
const Z: [f64; 4] = [0.1, 0.5, 1.0, 3.0];
const LOG_SM: [f64; 4] = [9.0, 10.0, 10.7, 11.5];

/// Prints the current values as Rust literals. Run with
/// `--ignored --nocapture` to regenerate the tables below.
#[test]
#[ignore]
fn print_golden_values() {
    let smhm: Vec<(&str, Box<dyn SmhmModel>)> = vec![
        ("g19_se", Box::new(MosterFormSmhm::g19_se(true))),
        ("moster13", Box::new(MosterFormSmhm::moster13(true))),
        ("rp17", Box::new(RodriguezPuebla17)),
    ];
    for (name, m) in &smhm {
        for &dm in &LOG_DM {
            for &z in &Z {
                println!("{name} {dm} {z} {:.17e}", m.stellar_mass(dm, z, None));
            }
        }
    }
    let sfr: Vec<(&str, Box<dyn SfrModel>)> = vec![
        ("ce", Box::new(TomczakFormSfr::ce())),
        ("t16", Box::new(TomczakFormSfr::t16())),
    ];
    for (name, s) in &sfr {
        for &sm in &LOG_SM {
            for &z in &Z {
                println!("{name} {sm} {z} {:.17e}", s.log_sfr(sm, z));
            }
        }
    }
}
```

- [ ] **Step 2: Run the generator and capture output**

Run: `cd rust && cargo test -p steel-plugins --test golden_smhm_sfr -- --ignored --nocapture`
Expected: PASS, printing one line per (model, mass, z) combination.

Save the output to the scratchpad — it becomes the expected table in Step 3.

- [ ] **Step 3: Write the assertion test using the captured values**

Append to the same file. Substitute the **actual** numbers from Step 2 into `EXPECTED_*`; the tuples below show the required shape, and the test must fail if you leave them unedited.

```rust
/// `(model, log_dm, z, expected_log_sm)` — captured from Step 2.
/// Replace every `f64::NAN` with the printed value; NAN never compares
/// equal, so an unedited table fails loudly.
const EXPECTED_SMHM: &[(&str, f64, f64, f64)] = &[
    ("g19_se", 12.0, 0.1, f64::NAN),
    // ... one row per printed line
];

const EXPECTED_SFR: &[(&str, f64, f64, f64)] = &[
    ("ce", 10.0, 0.1, f64::NAN),
    // ... one row per printed line
];

fn smhm_by_name(name: &str) -> Box<dyn SmhmModel> {
    match name {
        "g19_se" => Box::new(MosterFormSmhm::g19_se(true)),
        "moster13" => Box::new(MosterFormSmhm::moster13(true)),
        "rp17" => Box::new(RodriguezPuebla17),
        other => panic!("unknown smhm preset in golden table: {other}"),
    }
}

fn sfr_by_name(name: &str) -> Box<dyn SfrModel> {
    match name {
        "ce" => Box::new(TomczakFormSfr::ce()),
        "t16" => Box::new(TomczakFormSfr::t16()),
        other => panic!("unknown sfr preset in golden table: {other}"),
    }
}

#[test]
fn existing_smhm_plugins_are_bit_identical_to_golden() {
    for &(name, dm, z, expected) in EXPECTED_SMHM {
        let got = smhm_by_name(name).stellar_mass(dm, z, None);
        assert_eq!(got.to_bits(), expected.to_bits(), "{name} at dm={dm} z={z}: {got} != {expected}");
    }
}

#[test]
fn existing_sfr_plugins_are_bit_identical_to_golden() {
    for &(name, sm, z, expected) in EXPECTED_SFR {
        let got = sfr_by_name(name).log_sfr(sm, z);
        assert_eq!(got.to_bits(), expected.to_bits(), "{name} at sm={sm} z={z}: {got} != {expected}");
    }
}

#[test]
fn seeded_scatter_is_reproducible() {
    let m = MosterFormSmhm::g19_se(true);
    let mut a = StdRng::seed_from_u64(20260817);
    let mut b = StdRng::seed_from_u64(20260817);
    assert_eq!(
        m.stellar_mass(12.0, 0.1, Some(&mut a)).to_bits(),
        m.stellar_mass(12.0, 0.1, Some(&mut b)).to_bits()
    );
}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd rust && cargo test -p steel-plugins --test golden_smhm_sfr`
Expected: PASS (3 tests; `print_golden_values` is skipped as ignored).

If a bit-comparison fails, a value was transcribed wrong — fix the table, not the tolerance.

- [ ] **Step 5: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-plugins/tests/golden_smhm_sfr.rs
git commit -m "test: pin existing SMHM/SFR plugin values as refactor guard

Captures bit-exact output of the three SmhmModel and two SfrModel
implementations before the AccretionContext trait widening, so the
widening can be proven inert."
```

---

### Task 2: `AccretionContext` and the inert trait widening

**Files:**
- Create: `rust/steel-core/src/accretion.rs`
- Modify: `rust/steel-core/src/lib.rs`, `rust/steel-core/src/smhm.rs`, `rust/steel-core/src/sfr.rs`
- Modify: `rust/steel-plugins/src/smhm/{moster,behroozi,rodriguez_puebla}.rs`, `rust/steel-plugins/src/sfr.rs`
- Modify: `rust/steel-core/src/context.rs:642,644`, `rust/steel-core/src/baryonic.rs:197,249`, `rust/steel-postprocess/src/central_evolution.rs:84`, `rust/steel-fit/src/smf.rs:61`
- Modify: `rust/steel-plugins/tests/golden_smhm_sfr.rs` (call-site updates only)

**Interfaces:**
- Consumes: `GrowthTrack` (`steel_core::halo_growth`), `Cosmology` + `MassDefinition` (`steel_core::cosmology`).
- Produces: `steel_core::accretion::AccretionContext<'a>` with public fields `own_track: &'a GrowthTrack`, `host_track: Option<&'a GrowthTrack>`, `z_infall: Option<f64>`, `log_m_peak: Option<f64>`, `cosmology: &'a dyn Cosmology`, `mass_definition: MassDefinition`; plus `AccretionContext::central(own_track, cosmology, mass_definition)` and `::satellite(own_track, host_track, z_infall, cosmology, mass_definition)`. New signatures `SmhmModel::stellar_mass(&self, log_dm, z, ctx: &AccretionContext<'_>, rng)` and `SfrModel::log_sfr(&self, log_sm, z, ctx: &AccretionContext<'_>)`.

- [ ] **Step 1: Write the failing test for `AccretionContext` constructors**

Create `rust/steel-core/src/accretion.rs` with the test module only at first:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::halo_growth::GrowthTrack;

    struct StubCosmo;
    impl crate::cosmology::Cosmology for StubCosmo {
        fn h0(&self) -> f64 { 67.74 }
        fn omega_m0(&self) -> f64 { 0.3089 }
        fn omega_b0(&self) -> f64 { 0.0486 }
        fn omega_de0(&self) -> f64 { 0.6911 }
    }

    fn track() -> GrowthTrack {
        GrowthTrack { z: vec![0.0, 1.0, 2.0], log_mass: vec![12.0, 11.5, 11.0] }
    }

    #[test]
    fn central_context_has_no_host_or_infall() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        assert!(ctx.host_track.is_none());
        assert!(ctx.z_infall.is_none());
        assert_eq!(ctx.own_track.log_mass[0], 12.0);
    }

    #[test]
    fn satellite_context_carries_host_and_infall() {
        let own = track();
        let host = GrowthTrack { z: vec![0.0, 1.0], log_mass: vec![14.0, 13.5] };
        let c = StubCosmo;
        let ctx = AccretionContext::satellite(&own, &host, 1.5, &c, MassDefinition::Vir);
        assert_eq!(ctx.z_infall, Some(1.5));
        assert_eq!(ctx.host_track.expect("host").log_mass[0], 14.0);
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-core accretion`
Expected: FAIL — `cannot find type AccretionContext` / module not registered.

- [ ] **Step 3: Implement `AccretionContext`**

Prepend to `rust/steel-core/src/accretion.rs`:

```rust
//! Read-only accretion history and environment, passed to every
//! mass-assigning plugin.
//!
//! STEEL's `SmhmModel` is a memoryless M*(Mh,z) map, but rate-based
//! models (EMERGE, UniverseMachine) need the halo's assembly history.
//! Rather than a second parallel trait, every mass-assigning plugin
//! receives this context and ignores what it does not need — so any
//! future model has equal access to history without another interface.
//!
//! All fields are shared references or `Copy` scalars: constructing one
//! allocates nothing.

use crate::cosmology::{Cosmology, MassDefinition};
use crate::halo_growth::GrowthTrack;

pub struct AccretionContext<'a> {
    /// Main-progenitor track of *this* object treated as a central: to
    /// z0 for a central, to `z_infall` for a satellite. Always present.
    pub own_track: &'a GrowthTrack,
    /// Main-progenitor track of the host halo. `None` for centrals.
    pub host_track: Option<&'a GrowthTrack>,
    /// Infall redshift. `None` for centrals.
    pub z_infall: Option<f64>,
    /// Peak halo mass \[log10 Msun\] where it differs from the current
    /// mass. `None` when the caller cannot distinguish them.
    pub log_m_peak: Option<f64>,
    pub cosmology: &'a dyn Cosmology,
    /// Mass definition the `log_dm` / `log_mh` arguments are in.
    pub mass_definition: MassDefinition,
}

impl<'a> AccretionContext<'a> {
    pub fn central(
        own_track: &'a GrowthTrack,
        cosmology: &'a dyn Cosmology,
        mass_definition: MassDefinition,
    ) -> Self {
        Self { own_track, host_track: None, z_infall: None, log_m_peak: None, cosmology, mass_definition }
    }

    pub fn satellite(
        own_track: &'a GrowthTrack,
        host_track: &'a GrowthTrack,
        z_infall: f64,
        cosmology: &'a dyn Cosmology,
        mass_definition: MassDefinition,
    ) -> Self {
        Self {
            own_track,
            host_track: Some(host_track),
            z_infall: Some(z_infall),
            log_m_peak: None,
            cosmology,
            mass_definition,
        }
    }

    /// Redshift at which the main progenitor first exceeded `log_m`
    /// \[log10 Msun\], interpolated linearly in `log_mass` between the
    /// bracketing samples. `None` if the track never crosses it.
    ///
    /// `own_track.z` is increasing into the past and `log_mass` is
    /// decreasing, so the crossing is the first index whose mass falls
    /// below `log_m`.
    pub fn formation_redshift(&self, log_m: f64) -> Option<f64> {
        let t = self.own_track;
        let i = t.log_mass.iter().position(|&m| m < log_m)?;
        if i == 0 {
            return Some(t.z[0]);
        }
        let (m_hi, m_lo) = (t.log_mass[i - 1], t.log_mass[i]);
        let (z_lo, z_hi) = (t.z[i - 1], t.z[i]);
        let span = m_hi - m_lo;
        if span.abs() < f64::EPSILON {
            return Some(z_hi);
        }
        Some(z_lo + (m_hi - log_m) / span * (z_hi - z_lo))
    }
}
```

Register in `rust/steel-core/src/lib.rs`: add `pub mod accretion;` to the module list and `pub use accretion::AccretionContext;` to the re-exports.

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test -p steel-core accretion`
Expected: PASS (2 tests).

- [ ] **Step 5: Add a test for `formation_redshift`**

Append to the test module in `accretion.rs`:

```rust
    #[test]
    fn formation_redshift_interpolates_between_samples() {
        // log_mass 12.0 -> 11.5 -> 11.0 at z 0, 1, 2. Crossing 11.75
        // sits halfway through the first interval, i.e. z = 0.5.
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let z = ctx.formation_redshift(11.75).expect("should cross");
        assert!((z - 0.5).abs() < 1e-12, "z = {z}");
    }

    #[test]
    fn formation_redshift_is_none_when_never_crossed() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        assert!(ctx.formation_redshift(9.0).is_none());
    }
```

Run: `cd rust && cargo test -p steel-core accretion`
Expected: PASS (4 tests).

- [ ] **Step 6: Widen the two trait signatures**

`rust/steel-core/src/smhm.rs` — replace the trait body:

```rust
use rand::RngCore;

use crate::accretion::AccretionContext;

pub trait SmhmModel: Send + Sync {
    /// Stellar mass \[log10 Msun\] given halo mass `log_dm` \[log10
    /// Msun, h-free\] and redshift `z`. `ctx` supplies the object's
    /// accretion history; memoryless relations ignore it. When `rng` is
    /// `Some`, draws and adds the model's intrinsic scatter.
    fn stellar_mass(
        &self,
        log_dm: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64;
}
```

`rust/steel-core/src/sfr.rs`:

```rust
use crate::accretion::AccretionContext;

pub trait SfrModel: Send + Sync {
    /// log10 star formation rate \[Msun/yr\] on the star-forming main
    /// sequence, before quenching/scatter is applied by the caller.
    /// `ctx` supplies accretion history; M*-keyed relations ignore it.
    fn log_sfr(&self, log_sm: f64, z: f64, ctx: &AccretionContext<'_>) -> f64;
}
```

- [ ] **Step 7: Update the five existing implementations**

In each of `moster.rs`, `behroozi.rs`, `rodriguez_puebla.rs` (SMHM) and `sfr.rs` (all three SFR forms), add the parameter and ignore it. Import `steel_core::accretion::AccretionContext`, then change each signature, e.g. in `moster.rs`:

```rust
impl SmhmModel for MosterFormSmhm {
    fn stellar_mass(
        &self,
        log_dm: f64,
        z: f64,
        _ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        // body unchanged
```

Add a one-line note to each module doc: `//! Memoryless in the accretion history: `_ctx` is ignored by design, not omission.`

- [ ] **Step 8: Update all six production call sites**

`context.rs:642,644` — construct the context before the loop. The satellite own-track plumbing lands in Task 4; for now use the host track as `own_track` and mark it:

```rust
                    // TASK-4 will replace `own_track` with the
                    // satellite's own pre-infall central track. Until
                    // then this is the host track, which the three
                    // memoryless SMHM plugins ignore, so behaviour is
                    // unchanged.
                    let ctx = AccretionContext::satellite(
                        &host_track_for_bin[j],
                        &host_track_for_bin[j],
                        z[i],
                        self.cosmology.as_ref(),
                        MassDefinition::Vir,
                    );
                    for slot in sm_infall.iter_mut() {
                        let draw = if config.scatter {
                            self.smhm.stellar_mass(sm_infall_dm, z[i], &ctx, Some(&mut rng))
                        } else {
                            self.smhm.stellar_mass(sm_infall_dm, z[i], &ctx, None)
                        };
                        *slot = draw;
                    }
```

`host_track_for_bin: Vec<GrowthTrack>` is built alongside the existing precompute at `context.rs:425-433` — retain each `track` rather than only copying `track.log_mass` into `raw_host_mass`.

Apply the analogous mechanical change at `baryonic.rs:197,249`, `central_evolution.rs:84`, `smf.rs:61`, threading a context in from the caller. Where a caller has no track available (`steel-fit`'s SMF fit), build a single-point track:

```rust
// The SMF fit evaluates the mean relation only; no history is involved.
let flat = GrowthTrack { z: vec![z], log_mass: vec![dm] };
let ctx = AccretionContext::central(&flat, cosmology, MassDefinition::Vir);
```

- [ ] **Step 9: Update the golden test call sites**

In `rust/steel-plugins/tests/golden_smhm_sfr.rs`, add the same flat-track helper and pass `&ctx`. Do **not** touch `EXPECTED_SMHM` / `EXPECTED_SFR`.

```rust
use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::GrowthTrack;

/// Single-point track: these are memoryless relations, so the context
/// content is irrelevant — only that one can be built.
fn flat_ctx<'a>(t: &'a GrowthTrack, c: &'a dyn steel_core::Cosmology) -> AccretionContext<'a> {
    AccretionContext::central(t, c, MassDefinition::Vir)
}
```

- [ ] **Step 10: Verify the widening is inert**

Run: `cd rust && cargo test --workspace`
Expected: PASS, **including both bit-identity tests from Task 1**.

If either bit-identity test fails, the widening changed behaviour. Do not adjust the expected values — find the cause.

- [ ] **Step 11: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-core/src/accretion.rs rust/steel-core/src/lib.rs \
        rust/steel-core/src/smhm.rs rust/steel-core/src/sfr.rs \
        rust/steel-plugins/src/smhm/ rust/steel-plugins/src/sfr.rs \
        rust/steel-core/src/context.rs rust/steel-core/src/baryonic.rs \
        rust/steel-postprocess/src/central_evolution.rs rust/steel-fit/src/smf.rs \
        rust/steel-plugins/tests/golden_smhm_sfr.rs
git commit -m "refactor: thread read-only AccretionContext through SMHM and SFR

Widens SmhmModel::stellar_mass and SfrModel::log_sfr to carry accretion
history, so rate-based models (EMERGE, UniverseMachine) can use it while
memoryless relations ignore it. Proven inert by the Task 1 bit-identity
guard."
```

---

### Task 3: `z0 != 0` coverage for `VandenBosch14`

Spec §5.1. The satellite own-track design depends on a code path no test exercises.

**Files:**
- Modify: `rust/steel-plugins/src/halo_growth.rs` (test module only)

**Interfaces:**
- Consumes: `HaloGrowthModel::{redshift_grid, growth_history}`, `VandenBosch14::new(&cosmo)`, `N_Z`.
- Produces: no new API. Establishes that `growth_history(m, z0)` is correct for `z0 > 0`, which Task 4 relies on.

- [ ] **Step 1: Write the failing tests**

Append to `mod tests` in `rust/steel-plugins/src/halo_growth.rs`:

```rust
    #[test]
    fn growth_history_starts_at_m0_for_nonzero_z0() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        for z0 in [0.5, 1.0, 2.0, 4.0] {
            let track = model.growth_history(12.0, z0);
            assert!(
                (track.log_mass[0] - 12.0).abs() < 1e-3,
                "z0={z0}: log_mass[0] = {}",
                track.log_mass[0]
            );
        }
    }

    #[test]
    fn growth_history_grid_begins_at_z0() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        for z0 in [0.5, 1.0, 2.0, 4.0] {
            let track = model.growth_history(12.0, z0);
            assert_eq!(track.z.len(), N_Z, "z0={z0}");
            assert!((track.z[0] - z0).abs() < 1e-3, "z0={z0}: z[0] = {}", track.z[0]);
            assert!(track.z[N_Z - 1] > z0, "z0={z0}: grid must extend into the past");
        }
    }

    #[test]
    fn growth_history_is_monotonic_for_nonzero_z0() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        for z0 in [0.5, 1.0, 2.0, 4.0] {
            let track = model.growth_history(13.0, z0);
            for w in track.log_mass.windows(2) {
                assert!(w[1] <= w[0] + 1e-6, "z0={z0}: mass increased into the past: {w:?}");
            }
        }
    }

    /// A halo observed at z0 must be no more massive at a *shared* later
    /// epoch than the same-mass halo observed at z=0, because the z0 halo
    /// has had less time to assemble. Compares the two tracks at the
    /// highest redshift common to both.
    #[test]
    fn later_observed_halos_have_less_assembled_progenitors() {
        let cosmo = Planck15::new();
        let model = VandenBosch14::new(&cosmo);
        let at_z0 = model.growth_history(12.0, 0.0);
        let at_z1 = model.growth_history(12.0, 1.0);
        let z_probe = 3.0;
        let m0 = interp_at(&at_z0.z, &at_z0.log_mass, z_probe);
        let m1 = interp_at(&at_z1.z, &at_z1.log_mass, z_probe);
        assert!(m1 <= m0 + 1e-6, "m(z0=1)={m1} should not exceed m(z0=0)={m0} at z={z_probe}");
    }

    /// Linear interpolation of `y` at `x_probe`; `xs` must be increasing.
    fn interp_at(xs: &[f64], ys: &[f64], x_probe: f64) -> f64 {
        let i = xs.iter().position(|&x| x >= x_probe).unwrap_or(xs.len() - 1).max(1);
        let (x0, x1) = (xs[i - 1], xs[i]);
        let (y0, y1) = (ys[i - 1], ys[i]);
        if (x1 - x0).abs() < f64::EPSILON {
            return y1;
        }
        y0 + (x_probe - x0) / (x1 - x0) * (y1 - y0)
    }
```

- [ ] **Step 2: Run the tests**

Run: `cd rust && cargo test -p steel-plugins halo_growth`
Expected: PASS. `VandenBosch14` is believed `z0`-general, so these should pass unmodified.

**If any fails, stop and report.** A failure means the satellite design in Task 4 rests on a broken path, and the spec's §5 needs revisiting before continuing. Do not patch the test to pass.

- [ ] **Step 3: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-plugins/src/halo_growth.rs
git commit -m "test: cover z0 != 0 growth histories in VandenBosch14

The satellite own-track design calls growth_history(m, z_infall), a path
every existing test left unexercised (all passed z0=0.0)."
```

---

### Task 4: Satellite own-track plumbing

Replaces the Task 2 placeholder so satellites carry their genuine pre-infall central history.

**Files:**
- Modify: `rust/steel-core/src/context.rs:425-433` (cache), `:638-647` (context construction)

**Interfaces:**
- Consumes: `HaloGrowthModel::growth_history`, `AccretionContext::satellite`, Task 3's verified `z0 != 0` path.
- Produces: `satellite_tracks: Vec<Vec<GrowthTrack>>` indexed `[z_index][subhalo_bin]`, and correct `own_track` wiring at the SMHM call site.

- [ ] **Step 1: Write the failing integration test**

Create `rust/steel-plugins/tests/satellite_tracks.rs`:

```rust
//! A satellite's `own_track` must be its own pre-infall central
//! history, not the host's. Spec §5.

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::{Cosmology, SmhmModel};
use steel_plugins::{Planck15, VandenBosch14};

/// Records the `own_track` head mass it was called with.
struct SpySmhm {
    seen: std::sync::Mutex<Vec<f64>>,
}

impl SmhmModel for SpySmhm {
    fn stellar_mass(
        &self,
        log_dm: f64,
        _z: f64,
        ctx: &AccretionContext<'_>,
        _rng: Option<&mut dyn rand::RngCore>,
    ) -> f64 {
        self.seen.lock().unwrap().push(ctx.own_track.log_mass[0]);
        log_dm - 2.0
    }
}

#[test]
fn satellite_own_track_head_equals_its_infall_mass() {
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);
    let z_infall = 1.5;
    let log_m_sub = 11.4;

    let own = growth.growth_history(log_m_sub, z_infall);
    let host = growth.growth_history(13.8, 0.0);
    let ctx = AccretionContext::satellite(&own, &host, z_infall, &cosmo, MassDefinition::Vir);

    // own_track starts at the subhalo's own infall mass...
    assert!((ctx.own_track.log_mass[0] - log_m_sub).abs() < 1e-3);
    // ...and is distinct from the host's.
    assert!((ctx.host_track.expect("host").log_mass[0] - 13.8).abs() < 1e-3);

    let spy = SpySmhm { seen: std::sync::Mutex::new(Vec::new()) };
    let _ = spy.stellar_mass(log_m_sub, z_infall, &ctx, None);
    assert!((spy.seen.lock().unwrap()[0] - log_m_sub).abs() < 1e-3);
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-plugins --test satellite_tracks`
Expected: FAIL to compile until `steel-plugins` exposes what the test imports; if it compiles, it passes trivially and only guards `AccretionContext`. Proceed either way — Step 3 is where the production wiring changes.

- [ ] **Step 3: Cache satellite tracks in `context.rs`**

After the existing host precompute (`context.rs:425-433`), add:

```rust
        // Satellite own-tracks. At infall the object *was* a central, so
        // its pre-infall history is `growth_history(m_infall, z_infall)`
        // — the same average-MAH approximation already used for hosts,
        // applied to subhalos. Spec §5.
        //
        // One track per (redshift step, subhalo bin): ~190 x 5 root-finds
        // at startup, against 56 for hosts. Computed once, never in the
        // hot loop.
        let mut satellite_tracks: Vec<Vec<GrowthTrack>> = Vec::with_capacity(n_z);
        for &z_i in &z {
            let mut row = Vec::with_capacity(n_sub);
            for k in 0..n_sub {
                let log_m_sub = sat_mass[k] - log_h;
                row.push(self.halo_growth.growth_history(log_m_sub, z_i));
            }
            satellite_tracks.push(row);
        }
```

Place it after `z` is trimmed (`context.rs:439`) and after `sat_mass` / `n_sub` / `log_h` are in scope. If `sat_mass` is defined later, move this block to just after its definition rather than moving `sat_mass`.

- [ ] **Step 4: Use the cached track at the call site**

Replace the Task 2 placeholder at `context.rs:638-647`:

```rust
                    // ---- abundance matching at infall ----
                    let sm_infall_dm = sat_mass[k] - log_h;
                    let ctx = AccretionContext::satellite(
                        &satellite_tracks[i][k],
                        &host_track_for_bin[j],
                        z[i],
                        self.cosmology.as_ref(),
                        MassDefinition::Vir,
                    );
                    for slot in sm_infall.iter_mut() {
                        let draw = if config.scatter {
                            self.smhm.stellar_mass(sm_infall_dm, z[i], &ctx, Some(&mut rng))
                        } else {
                            self.smhm.stellar_mass(sm_infall_dm, z[i], &ctx, None)
                        };
                        *slot = draw;
                    }
```

Delete the `TASK-4` comment.

- [ ] **Step 5: Run the full suite**

Run: `cd rust && cargo test --workspace`
Expected: PASS, **including the Task 1 bit-identity tests** — the three current SMHM plugins ignore `ctx`, so results must be unchanged.

- [ ] **Step 6: Verify startup cost is acceptable**

Run: `cd rust && cargo build --release && time ./target/release/steel-cli <existing runfile>`

Substitute a runfile from `rust/runfiles/`. Compare against the same run on `HEAD~1`. Expected: startup grows by well under a second; total runtime essentially unchanged. If startup grows by more than ~5s, report rather than optimising — it means `growth_history` is costlier than the spec's estimate.

- [ ] **Step 7: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-core/src/context.rs rust/steel-plugins/tests/satellite_tracks.rs
git commit -m "feat: give satellites their own pre-infall central growth track

At infall an object was a central, so its history is
growth_history(m_infall, z_infall). Removes the need for the formation-
time proxy considered in design, making EMERGE's reionization gate exact
for satellites. Spec section 5."
```

---

### Task 5: `StellarGrowthModel` trait and the track integrator

Spec §3, §4.3. Rate-based models return dM*/dt; M* is its integral along the growth track.

**Files:**
- Create: `rust/steel-core/src/stellar_growth.rs`
- Modify: `rust/steel-core/src/lib.rs`

**Interfaces:**
- Consumes: `AccretionContext` (Task 2), `GrowthTrack`, `Cosmology::age`.
- Produces: `steel_core::stellar_growth::StellarGrowthModel` with `stellar_growth_rate(&self, log_mh, z, ctx, rng) -> f64` (log10 Msun/yr); and `integrate_stellar_mass(model: &dyn StellarGrowthModel, ctx: &AccretionContext<'_>, z_end: f64, rng: Option<&mut dyn RngCore>) -> f64` returning log10 M*/Msun. Both used by Tasks 9 and 11.

- [ ] **Step 1: Write the failing tests**

Create `rust/steel-core/src/stellar_growth.rs` with the test module only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::{Cosmology, MassDefinition};
    use crate::halo_growth::GrowthTrack;

    struct StubCosmo;
    impl Cosmology for StubCosmo {
        fn h0(&self) -> f64 { 67.74 }
        fn omega_m0(&self) -> f64 { 0.3089 }
        fn omega_b0(&self) -> f64 { 0.0486 }
        fn omega_de0(&self) -> f64 { 0.6911 }
        /// Deliberately linear in (1+z)^-1 so the exact integral of a
        /// constant rate is hand-computable.
        fn age(&self, z: f64) -> f64 { 13.8 / (1.0 + z) }
    }

    /// Constant 1 Msun/yr regardless of mass, redshift, or history.
    struct ConstantRate;
    impl StellarGrowthModel for ConstantRate {
        fn stellar_growth_rate(
            &self,
            _log_mh: f64,
            _z: f64,
            _ctx: &AccretionContext<'_>,
            _rng: Option<&mut dyn rand::RngCore>,
        ) -> f64 {
            0.0 // log10(1.0)
        }
    }

    fn track() -> GrowthTrack {
        // z decreasing into the present is NOT the convention: GrowthTrack
        // is increasing into the past, so index 0 is the observed epoch.
        GrowthTrack { z: vec![0.0, 1.0, 2.0, 3.0], log_mass: vec![12.0, 11.6, 11.2, 10.8] }
    }

    #[test]
    fn constant_rate_integrates_to_rate_times_elapsed_time() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        // Integrate from the track's earliest epoch (z=3) to z=0.
        // age(0) - age(3) = 13.8 - 3.45 = 10.35 Gyr = 1.035e10 yr.
        // At 1 Msun/yr that is 1.035e10 Msun.
        let got = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        let expected = 1.035e10f64.log10();
        assert!((got - expected).abs() < 1e-6, "got {got}, expected {expected}");
    }

    #[test]
    fn integrating_to_an_earlier_epoch_gives_less_mass() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let early = integrate_stellar_mass(&ConstantRate, &ctx, 2.0, None);
        let late = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        assert!(early < late, "early {early} should be below late {late}");
    }

    #[test]
    fn zero_elapsed_time_gives_negative_infinity_log_mass() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        // z_end at the track's oldest sample: no time has elapsed.
        let got = integrate_stellar_mass(&ConstantRate, &ctx, 3.0, None);
        assert!(got.is_infinite() && got.is_sign_negative(), "got {got}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-core stellar_growth`
Expected: FAIL — `StellarGrowthModel` and `integrate_stellar_mass` undefined.

- [ ] **Step 3: Implement the trait and integrator**

Prepend to `rust/steel-core/src/stellar_growth.rs`:

```rust
//! Rate-based stellar mass assembly.
//!
//! `SmhmModel` answers "what M* corresponds to this Mh at this z?" as a
//! memoryless map. EMERGE and UniverseMachine instead specify a *rate*:
//! EMERGE as a baryon conversion efficiency applied to the halo
//! accretion rate, UniverseMachine as an SFR drawn from halo properties.
//! For both, M* is the time integral of that rate along the object's
//! growth track.
//!
//! Keeping these separate from `SmhmModel` matters beyond tidiness:
//! EMERGE's efficiency double power law is *algebraically identical* to
//! the Moster form in `steel_plugins::smhm::moster`, but multiplies a
//! rate rather than a mass. Substituting its coefficients into
//! `MosterFormSmhm` would silently produce a wrong SMHM curve.

use rand::RngCore;

use crate::accretion::AccretionContext;

pub trait StellarGrowthModel: Send + Sync {
    /// log10 dM*/dt \[Msun/yr\] for a halo of mass `log_mh` \[log10
    /// Msun\] at redshift `z`.
    ///
    /// `rng` is present because UniverseMachine draws SFR from a bimodal
    /// PDF: the rate is intrinsically stochastic, not a mean relation
    /// with scatter added afterwards. Models with a deterministic rate
    /// ignore it.
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64;
}

/// Integrate `model`'s rate along `ctx.own_track` from the track's
/// earliest sample down to `z_end`, returning log10 M*/Msun.
///
/// Trapezoidal in cosmic time. `own_track.z` is increasing into the past
/// (index 0 is the observed epoch), so integration walks the track in
/// reverse. Samples at `z < z_end` are excluded.
///
/// Returns `f64::NEG_INFINITY` when no time elapses (zero mass, whose
/// log is negative infinity) rather than `NaN`, so callers can compare
/// and clamp without special-casing.
pub fn integrate_stellar_mass(
    model: &dyn StellarGrowthModel,
    ctx: &AccretionContext<'_>,
    z_end: f64,
    mut rng: Option<&mut dyn RngCore>,
) -> f64 {
    let t = ctx.own_track;
    debug_assert_eq!(t.z.len(), t.log_mass.len(), "GrowthTrack axes must be equal length");

    // Indices from oldest to youngest, keeping only z >= z_end.
    let idx: Vec<usize> = (0..t.z.len()).rev().filter(|&i| t.z[i] >= z_end).collect();
    if idx.len() < 2 {
        return f64::NEG_INFINITY;
    }

    let mut mass = 0.0_f64; // Msun, linear
    for w in idx.windows(2) {
        let (i0, i1) = (w[0], w[1]); // i0 older, i1 younger
        // age() is in Gyr; rates are per year.
        let dt_yr = (ctx.cosmology.age(t.z[i1]) - ctx.cosmology.age(t.z[i0])) * 1.0e9;
        if dt_yr <= 0.0 {
            continue;
        }
        let r0 = 10f64.powf(model.stellar_growth_rate(
            t.log_mass[i0],
            t.z[i0],
            ctx,
            rng.as_deref_mut(),
        ));
        let r1 = 10f64.powf(model.stellar_growth_rate(
            t.log_mass[i1],
            t.z[i1],
            ctx,
            rng.as_deref_mut(),
        ));
        mass += 0.5 * (r0 + r1) * dt_yr;
    }

    if mass <= 0.0 {
        f64::NEG_INFINITY
    } else {
        mass.log10()
    }
}
```

Register in `rust/steel-core/src/lib.rs`: `pub mod stellar_growth;` and `pub use stellar_growth::{integrate_stellar_mass, StellarGrowthModel};`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test -p steel-core stellar_growth`
Expected: PASS (3 tests).

- [ ] **Step 5: Add a mass-return caveat test**

The integral above is *formed* stellar mass, not surviving mass — mass loss from stellar evolution is applied by STEEL's existing machinery, not here. Pin that boundary so a later reader does not double-apply it. Append to the test module:

```rust
    /// `integrate_stellar_mass` returns mass *formed*, with no stellar
    /// mass-loss return fraction applied. STEEL applies mass loss in
    /// `Functions.py::StellarMassLoss` / its Rust port, so applying it
    /// here too would double-count. This test documents the boundary.
    #[test]
    fn integrator_returns_formed_mass_not_surviving_mass() {
        let t = track();
        let c = StubCosmo;
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let got = integrate_stellar_mass(&ConstantRate, &ctx, 0.0, None);
        // Exactly rate x elapsed time: no 0.6-0.8 return-fraction factor.
        assert!((got - 1.035e10f64.log10()).abs() < 1e-6);
    }
```

Run: `cd rust && cargo test -p steel-core stellar_growth`
Expected: PASS (4 tests).

- [ ] **Step 6: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-core/src/stellar_growth.rs rust/steel-core/src/lib.rs
git commit -m "feat: add StellarGrowthModel trait and growth-track integrator

Rate-based models (EMERGE, UniverseMachine) specify dM*/dt rather than
M*(Mh,z); integrate_stellar_mass integrates along the object's own track.
Documents that EMERGE's efficiency power law must not be substituted into
MosterFormSmhm despite identical algebra. Spec sections 3 and 4.3."
```

---

### Task 6: Harmonisation layer

Spec §7. Unit and definition conversions only — no new physics. These are the mismatches that make an overlay look plausible and be wrong.

**Files:**
- Create: `rust/steel-plugins/src/harmonise.rs`
- Modify: `rust/steel-plugins/src/lib.rs`

**Interfaces:**
- Consumes: `Cosmology::{m_to_r, delta_vir}`, `MassDefinition`.
- Produces: `Imf` enum (`Chabrier`, `Kroupa`, `Salpeter`, `NotApplicable`) with `Imf::log_offset_to(self, other) -> f64`; `HConvention` enum (`HFree`, `PerH`); `ConcentrationMassRelation` trait with `concentration(&self, log_mh, z) -> f64`; `DuttonMaccio14`; `mpeak_to_vmax(log_mh, z, cosmo, cm, mdef) -> f64` returning Vmax in km/s. Consumed by Tasks 7, 9, 11.

- [ ] **Step 1: Write the failing tests for IMF offsets and h-conversion**

Create `rust/steel-plugins/src/harmonise.rs` with the test module only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use steel_core::cosmology::MassDefinition;

    #[test]
    fn imf_offset_is_zero_to_itself() {
        for imf in [Imf::Chabrier, Imf::Kroupa, Imf::Salpeter] {
            assert_eq!(imf.log_offset_to(imf), 0.0);
        }
    }

    #[test]
    fn imf_offset_is_antisymmetric() {
        let a = Imf::Chabrier.log_offset_to(Imf::Salpeter);
        let b = Imf::Salpeter.log_offset_to(Imf::Chabrier);
        assert!((a + b).abs() < 1e-12, "{a} and {b} should sum to zero");
    }

    #[test]
    fn salpeter_masses_exceed_chabrier() {
        // A Salpeter IMF infers more stellar mass for the same light.
        assert!(Imf::Chabrier.log_offset_to(Imf::Salpeter) > 0.0);
    }

    #[test]
    fn not_applicable_offset_is_zero_and_does_not_panic() {
        assert_eq!(Imf::NotApplicable.log_offset_to(Imf::Chabrier), 0.0);
        assert_eq!(Imf::Chabrier.log_offset_to(Imf::NotApplicable), 0.0);
    }

    #[test]
    fn h_conversion_round_trips() {
        let h = 0.6774;
        let log_m_per_h = 12.0;
        let free = HConvention::PerH.to_h_free(log_m_per_h, h);
        let back = HConvention::PerH.from_h_free(free, h);
        assert!((back - log_m_per_h).abs() < 1e-12);
        // Msun/h -> Msun divides by h, so the h-free value is larger.
        assert!(free > log_m_per_h);
    }

    #[test]
    fn concentration_decreases_with_mass_and_redshift() {
        let cm = DuttonMaccio14;
        assert!(cm.concentration(11.0, 0.0) > cm.concentration(14.0, 0.0));
        assert!(cm.concentration(12.0, 0.0) > cm.concentration(12.0, 2.0));
    }

    #[test]
    fn vmax_increases_with_halo_mass() {
        let cosmo = Planck15::new();
        let cm = DuttonMaccio14;
        let v11 = mpeak_to_vmax(11.0, 0.0, &cosmo, &cm, MassDefinition::Vir);
        let v13 = mpeak_to_vmax(13.0, 0.0, &cosmo, &cm, MassDefinition::Vir);
        assert!(v13 > v11, "v(13)={v13} should exceed v(11)={v11}");
    }

    /// A Milky-Way-mass halo should have Vmax of order 150-250 km/s.
    /// Wide bounds: this catches unit errors, not fit quality.
    #[test]
    fn vmax_is_physically_plausible_for_a_milky_way_halo() {
        let cosmo = Planck15::new();
        let cm = DuttonMaccio14;
        let v = mpeak_to_vmax(12.1, 0.0, &cosmo, &cm, MassDefinition::Vir);
        assert!((120.0..300.0).contains(&v), "Vmax = {v} km/s");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-plugins harmonise`
Expected: FAIL — module not declared, types undefined.

- [ ] **Step 3: Implement the conversions**

Prepend to `rust/steel-plugins/src/harmonise.rs`:

```rust
//! Unit and definition conversions between STEEL and external models.
//!
//! No physics here. These are the mismatches that silently invalidate an
//! SMHM overlay: an IMF offset comparable in size to the signal being
//! compared, an `Msun/h` vs `Msun` slip, or a halo mass quoted at a
//! different overdensity. Spec section 7.

use steel_core::cosmology::{Cosmology, MassDefinition};

/// Stellar initial mass function a stellar-mass calibration assumes.
///
/// Offsets are in dex, to be *added* to log10 M* when converting. Values
/// are the conventional ones; each must be re-verified against the
/// source paper before results are published (spec section 6.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imf {
    Chabrier,
    Kroupa,
    Salpeter,
    /// The plugin's output does not carry an IMF (e.g. a halo-only model).
    /// Compatibility checks skip it.
    NotApplicable,
}

impl Imf {
    /// log10 M* offset relative to Chabrier. Chabrier is the zero point
    /// because it is STEEL's own calibration basis.
    fn dex_from_chabrier(self) -> f64 {
        match self {
            Imf::Chabrier => 0.0,
            // Kroupa masses are ~0.05 dex above Chabrier.
            Imf::Kroupa => 0.05,
            // Salpeter masses are ~0.24 dex above Chabrier.
            Imf::Salpeter => 0.24,
            Imf::NotApplicable => 0.0,
        }
    }

    /// Offset in dex to add to a log10 M* on `self` to express it on
    /// `other`. Zero if either side is `NotApplicable`.
    pub fn log_offset_to(self, other: Imf) -> f64 {
        if self == Imf::NotApplicable || other == Imf::NotApplicable {
            return 0.0;
        }
        other.dex_from_chabrier() - self.dex_from_chabrier()
    }
}

/// Whether masses carry a factor of `h`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HConvention {
    /// Masses in Msun.
    HFree,
    /// Masses in Msun/h — STEEL's internal convention.
    PerH,
}

impl HConvention {
    /// Convert a log10 mass in `self`'s convention to h-free log10 Msun.
    pub fn to_h_free(self, log_m: f64, h: f64) -> f64 {
        match self {
            HConvention::HFree => log_m,
            HConvention::PerH => log_m - h.log10(),
        }
    }

    /// Inverse of [`to_h_free`](Self::to_h_free).
    pub fn from_h_free(self, log_m: f64, h: f64) -> f64 {
        match self {
            HConvention::HFree => log_m,
            HConvention::PerH => log_m + h.log10(),
        }
    }
}

/// Halo concentration as a function of mass and redshift.
///
/// A trait rather than a constant because UniverseMachine is keyed on
/// peak circular velocity while STEEL is mass-keyed throughout, so this
/// relation sits on the conversion path and materially affects UM's
/// results. It is a selectable modelling assumption, not an
/// implementation detail (spec section 7).
pub trait ConcentrationMassRelation: Send + Sync {
    /// NFW concentration c = R_delta / r_s for `log_mh` \[log10 Msun\].
    fn concentration(&self, log_mh: f64, z: f64) -> f64;
}

/// Dutton & Maccio (2014) NFW concentration fit for a Planck cosmology,
/// virial mass definition:
///
/// ```text
/// log10 c = a + b (log10 M_vir/[1e12 h^-1 Msun])
/// a = 0.537 + (1.025 - 0.537) exp(-0.718 z^1.08)
/// b = -0.097 + 0.024 z
/// ```
///
/// Chosen as the default because it is a simple closed form calibrated on
/// Planck parameters, matching STEEL's `Planck15`. Swappable: implement
/// `ConcentrationMassRelation` and select it in the runfile.
pub struct DuttonMaccio14;

impl ConcentrationMassRelation for DuttonMaccio14 {
    fn concentration(&self, log_mh: f64, z: f64) -> f64 {
        let a = 0.537 + (1.025 - 0.537) * (-0.718 * z.powf(1.08)).exp();
        let b = -0.097 + 0.024 * z;
        // The fit is in units of 1e12 h^-1 Msun.
        10f64.powf(a + b * (log_mh - 12.0))
    }
}

/// Peak circular velocity \[km/s\] for a halo of `log_mh` \[log10
/// Msun/h\] at `z`, under an NFW profile.
///
/// For NFW, `Vmax^2 / V_delta^2 = 0.216 c / [ln(1+c) - c/(1+c)]`, with
/// `V_delta = sqrt(G M / R_delta)`. Masses in Msun/h and radii in kpc/h
/// leave the h factors cancelling in `G M / R`.
pub fn mpeak_to_vmax(
    log_mh: f64,
    z: f64,
    cosmo: &dyn Cosmology,
    cm: &dyn ConcentrationMassRelation,
    mdef: MassDefinition,
) -> f64 {
    /// kpc (km/s)^2 Msun^-1 — matches `Cosmology::rho_crit`.
    const G: f64 = 4.30091e-6;

    let m = 10f64.powf(log_mh); // Msun/h
    let r = cosmo.m_to_r(m, z, mdef); // kpc/h
    let v_delta_sq = G * m / r; // (km/s)^2

    let c = cm.concentration(log_mh, z);
    let denom = (1.0 + c).ln() - c / (1.0 + c);
    debug_assert!(denom > 0.0, "NFW mass factor must be positive, c = {c}");

    (v_delta_sq * 0.216 * c / denom).sqrt()
}
```

Declare in `rust/steel-plugins/src/lib.rs`: add `pub mod harmonise;` and `pub use harmonise::{ConcentrationMassRelation, DuttonMaccio14, HConvention, Imf};`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test -p steel-plugins harmonise`
Expected: PASS (8 tests).

If `vmax_is_physically_plausible_for_a_milky_way_halo` fails, suspect the `h` convention in `m_to_r` before adjusting bounds — that test exists to catch exactly that class of error.

- [ ] **Step 5: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-plugins/src/harmonise.rs rust/steel-plugins/src/lib.rs
git commit -m "feat: add harmonisation layer for IMF, h, and Mpeak-to-Vmax

Conversions needed for a valid cross-model SMHM overlay: IMF offsets
(comparable in size to the signal), h convention, and the NFW
Mpeak -> Vmax path UniverseMachine needs. Concentration-mass relation is
a selectable trait, defaulting to Dutton & Maccio (2014). Spec section 7."
```

---

### Task 7: Rigid composition validator

Spec §8.1. Catches silent double-counting, which produces plausible wrong science rather than an error.

**Files:**
- Create: `rust/steel-core/src/compat.rs`
- Modify: `rust/steel-core/src/lib.rs`

**Interfaces:**
- Consumes: `MassDefinition`; `Imf` and `HConvention` are re-declared here (see Step 3 note on crate direction).
- Produces: `PluginDescriptor` (fields `id: &'static str`, `imf: Imf`, `mass_definition: MassDefinition`, `h_convention: HConvention`, `calibrated_cosmology: Option<CosmologyTag>`, `provides: &'static [Capability]`); `Capability` enum (`StellarMass`, `Quenching`, `Scatter`, `StarFormationRate`); `CosmologyTag` enum (`Planck15`, `Planck18`, `Wmap7`, `Wmap9`); `Incompatibility` enum; `DescribedPlugin` trait with `descriptor(&self) -> PluginDescriptor`; `validate_composition(&[PluginDescriptor], run_cosmology: CosmologyTag) -> Result<(), Vec<Incompatibility>>`. Consumed by Task 9, 11, and the registry.

- [ ] **Step 1: Write the failing tests, one per rule**

Create `rust/steel-core/src/compat.rs` with the test module only:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn base(id: &'static str, provides: &'static [Capability]) -> PluginDescriptor {
        PluginDescriptor {
            id,
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::PerH,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            provides,
        }
    }

    #[test]
    fn a_coherent_composition_passes() {
        let d = [
            base("g19_se", &[Capability::StellarMass, Capability::Scatter]),
            base("ce_sfr", &[Capability::StarFormationRate]),
            base("wetzel13", &[Capability::Quenching]),
        ];
        assert!(validate_composition(&d, CosmologyTag::Planck15).is_ok());
    }

    #[test]
    fn duplicate_quenching_is_rejected() {
        // The live case: UniverseMachine's bimodal SFR PDF already
        // contains quenching, so pairing it with STEEL's QuenchingModel
        // double-counts.
        let d = [
            base("universe_machine", &[Capability::StellarMass, Capability::Quenching]),
            base("wetzel13", &[Capability::Quenching]),
        ];
        let err = validate_composition(&d, CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(
            e,
            Incompatibility::DuplicateCapability { capability: Capability::Quenching, .. }
        )), "{err:?}");
    }

    #[test]
    fn duplicate_stellar_mass_source_is_rejected() {
        let d = [
            base("g19_se", &[Capability::StellarMass]),
            base("emerge", &[Capability::StellarMass]),
        ];
        let err = validate_composition(&d, CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(
            e,
            Incompatibility::DuplicateCapability { capability: Capability::StellarMass, .. }
        )), "{err:?}");
    }

    #[test]
    fn duplicate_scatter_is_rejected() {
        let d = [
            base("g19_se", &[Capability::Scatter]),
            base("emerge", &[Capability::Scatter]),
        ];
        let err = validate_composition(&d, CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(
            e,
            Incompatibility::DuplicateCapability { capability: Capability::Scatter, .. }
        )), "{err:?}");
    }

    #[test]
    fn imf_mismatch_is_rejected() {
        let mut a = base("emerge", &[Capability::StellarMass]);
        a.imf = Imf::Chabrier;
        let mut b = base("steel_ssfr", &[Capability::StarFormationRate]);
        b.imf = Imf::Kroupa;
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        assert!(err.iter().any(|e| matches!(e, Incompatibility::ImfMismatch { .. })), "{err:?}");
    }

    #[test]
    fn mass_definition_mismatch_is_rejected() {
        let mut a = base("a", &[Capability::StellarMass]);
        a.mass_definition = MassDefinition::Vir;
        let mut b = base("b", &[Capability::StarFormationRate]);
        b.mass_definition = MassDefinition::Critical(200.0);
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        assert!(
            err.iter().any(|e| matches!(e, Incompatibility::MassDefinitionMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn h_convention_mismatch_is_rejected() {
        let mut a = base("a", &[Capability::StellarMass]);
        a.h_convention = HConvention::PerH;
        let mut b = base("b", &[Capability::StarFormationRate]);
        b.h_convention = HConvention::HFree;
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        assert!(
            err.iter().any(|e| matches!(e, Incompatibility::HConventionMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn cosmology_mismatch_against_the_run_is_rejected() {
        let mut a = base("fitted_on_wmap7", &[Capability::StellarMass]);
        a.calibrated_cosmology = Some(CosmologyTag::Wmap7);
        let err = validate_composition(&[a], CosmologyTag::Planck15).expect_err("must reject");
        assert!(
            err.iter().any(|e| matches!(e, Incompatibility::CosmologyMismatch { .. })),
            "{err:?}"
        );
    }

    #[test]
    fn unspecified_calibration_cosmology_is_accepted() {
        let mut a = base("cosmology_agnostic", &[Capability::StellarMass]);
        a.calibrated_cosmology = None;
        assert!(validate_composition(&[a], CosmologyTag::Planck15).is_ok());
    }

    #[test]
    fn all_violations_are_reported_not_just_the_first() {
        let mut a = base("a", &[Capability::StellarMass]);
        a.imf = Imf::Chabrier;
        let mut b = base("b", &[Capability::StellarMass]);
        b.imf = Imf::Salpeter;
        let err = validate_composition(&[a, b], CosmologyTag::Planck15).expect_err("must reject");
        // Duplicate StellarMass *and* an IMF mismatch.
        assert!(err.len() >= 2, "expected multiple violations, got {err:?}");
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-core compat`
Expected: FAIL — module not declared, types undefined.

- [ ] **Step 3: Implement the validator**

Prepend to `rust/steel-core/src/compat.rs`:

```rust
//! Composition validation for plugin sets.
//!
//! The dangerous failures when mixing models are not type errors: they
//! are *silent double-counting* of a physical effect, which yields
//! plausible-looking output. UniverseMachine's bimodal SFR PDF already
//! contains quenching, so combining it with STEEL's `QuenchingModel`
//! quenches twice and no error is raised anywhere.
//!
//! A literal N-by-N model compatibility matrix is deliberately not used:
//! it needs a new row *and* column per plugin and is no stricter than
//! this rule set over declared metadata. Spec section 8.1.
//!
//! **Limitation.** This detects conflicts only along the dimensions
//! enumerated below. It cannot detect a novel incompatibility nobody
//! declared; that is what the planned derived-contract mechanism and
//! property-based cross-validation address (spec section 8.2).

use crate::cosmology::MassDefinition;

/// Stellar IMF a calibration assumes. Mirrors
/// `steel_plugins::harmonise::Imf`; declared here because `steel-core`
/// must not depend on `steel-plugins` (the dependency runs the other
/// way). Keep the two in sync — the round-trip test in
/// `steel-plugins::harmonise` guards the numeric offsets, this copy
/// carries only the tags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Imf {
    Chabrier,
    Kroupa,
    Salpeter,
    NotApplicable,
}

/// Whether masses carry a factor of `h`. Mirrors
/// `steel_plugins::harmonise::HConvention`; see [`Imf`] on why.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HConvention {
    HFree,
    PerH,
}

/// Cosmology a plugin's parameters were fitted under.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CosmologyTag {
    Planck15,
    Planck18,
    Wmap7,
    Wmap9,
}

/// An exclusive physical effect a plugin supplies. Two plugins in one
/// run must not supply the same one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capability {
    /// Determines M*. Either an `SmhmModel` or a `StellarGrowthModel`,
    /// never both.
    StellarMass,
    /// Suppresses star formation. Held by explicit quenching models *and*
    /// by any model whose SFR prescription already encodes quenching.
    Quenching,
    /// Applies intrinsic scatter to M*.
    Scatter,
    /// Supplies the star-formation rate.
    StarFormationRate,
}

/// What a plugin declares about its own assumptions.
#[derive(Debug, Clone)]
pub struct PluginDescriptor {
    pub id: &'static str,
    pub imf: Imf,
    pub mass_definition: MassDefinition,
    pub h_convention: HConvention,
    /// `None` means the plugin is cosmology-agnostic and is not checked.
    pub calibrated_cosmology: Option<CosmologyTag>,
    pub provides: &'static [Capability],
}

pub trait DescribedPlugin {
    fn descriptor(&self) -> PluginDescriptor;
}

#[derive(Debug, Clone, PartialEq)]
pub enum Incompatibility {
    DuplicateCapability { capability: Capability, first: &'static str, second: &'static str },
    ImfMismatch { first: &'static str, first_imf: Imf, second: &'static str, second_imf: Imf },
    MassDefinitionMismatch { first: &'static str, second: &'static str },
    HConventionMismatch { first: &'static str, second: &'static str },
    CosmologyMismatch { plugin: &'static str, fitted: CosmologyTag, run: CosmologyTag },
}

impl std::fmt::Display for Incompatibility {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Incompatibility::DuplicateCapability { capability, first, second } => write!(
                f,
                "'{first}' and '{second}' both supply {capability:?}; the effect would be \
                 applied twice. Select only one, or a variant that does not supply it."
            ),
            Incompatibility::ImfMismatch { first, first_imf, second, second_imf } => write!(
                f,
                "'{first}' assumes {first_imf:?} but '{second}' assumes {second_imf:?}. The \
                 offset is comparable to the signal under comparison; convert one, or select \
                 matching calibrations."
            ),
            Incompatibility::MassDefinitionMismatch { first, second } => write!(
                f,
                "'{first}' and '{second}' use different halo mass definitions; convert via \
                 MassDefinition before combining."
            ),
            Incompatibility::HConventionMismatch { first, second } => write!(
                f,
                "'{first}' and '{second}' disagree on the h convention (Msun vs Msun/h)."
            ),
            Incompatibility::CosmologyMismatch { plugin, fitted, run } => write!(
                f,
                "'{plugin}' was fitted under {fitted:?} but the run uses {run:?}; its \
                 normalisation does not transfer."
            ),
        }
    }
}

/// Validate a plugin set. Returns **every** violation, not just the
/// first, so one run surfaces all problems.
///
/// Callers must treat an `Err` as fatal at startup. A warning in a batch
/// run is indistinguishable from silence.
pub fn validate_composition(
    descriptors: &[PluginDescriptor],
    run_cosmology: CosmologyTag,
) -> Result<(), Vec<Incompatibility>> {
    let mut errors = Vec::new();

    // Rule 1: no duplicated exclusive capability.
    for (i, a) in descriptors.iter().enumerate() {
        for b in &descriptors[i + 1..] {
            for &cap in a.provides {
                if b.provides.contains(&cap) {
                    errors.push(Incompatibility::DuplicateCapability {
                        capability: cap,
                        first: a.id,
                        second: b.id,
                    });
                }
            }
        }
    }

    // Rules 2-4: pairwise agreement on IMF, mass definition, h.
    // Compared against the first descriptor that expresses an opinion,
    // so N plugins give at most N-1 errors rather than N^2.
    if let Some(first_imf) = descriptors.iter().find(|d| d.imf != Imf::NotApplicable) {
        for d in descriptors {
            if d.imf != Imf::NotApplicable && d.imf != first_imf.imf {
                errors.push(Incompatibility::ImfMismatch {
                    first: first_imf.id,
                    first_imf: first_imf.imf,
                    second: d.id,
                    second_imf: d.imf,
                });
            }
        }
    }

    if let Some(first) = descriptors.first() {
        for d in &descriptors[1..] {
            if d.mass_definition != first.mass_definition {
                errors.push(Incompatibility::MassDefinitionMismatch {
                    first: first.id,
                    second: d.id,
                });
            }
            if d.h_convention != first.h_convention {
                errors.push(Incompatibility::HConventionMismatch { first: first.id, second: d.id });
            }
        }
    }

    // Rule 5: declared calibration cosmology must match the run.
    for d in descriptors {
        if let Some(fitted) = d.calibrated_cosmology {
            if fitted != run_cosmology {
                errors.push(Incompatibility::CosmologyMismatch {
                    plugin: d.id,
                    fitted,
                    run: run_cosmology,
                });
            }
        }
    }

    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}
```

`MassDefinition` must derive `PartialEq` for the comparisons above. Check `rust/steel-core/src/cosmology.rs:10`; if the derive list lacks it, add `PartialEq`:

```rust
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MassDefinition {
```

Register in `rust/steel-core/src/lib.rs`: `pub mod compat;` and
`pub use compat::{Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, Incompatibility, PluginDescriptor, validate_composition};`

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test -p steel-core compat`
Expected: PASS (10 tests).

- [ ] **Step 5: Add descriptors to the five existing plugins**

Implement `DescribedPlugin` for `MosterFormSmhm`, `BehrooziFormSmhm`, `RodriguezPuebla17`, `TomczakFormSfr`, `SchreiberFormSfr`. For `MosterFormSmhm` in `rust/steel-plugins/src/smhm/moster.rs`:

```rust
use steel_core::compat::{Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor};

impl DescribedPlugin for MosterFormSmhm {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "moster_form",
            // The G19 presets are PyMorph/cmodel SDSS calibrations on a
            // Chabrier IMF; verify against Grylls+2019 before publishing.
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::PerH,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            // Applies its own log-normal scatter via `self.scatter`.
            provides: &[Capability::StellarMass, Capability::Scatter],
        }
    }
}
```

For the two SFR forms, `provides: &[Capability::StarFormationRate]` and `imf: Imf::Chabrier`. For `RodriguezPuebla17`, note that it returns the mean relation with no scatter, so `provides: &[Capability::StellarMass]` only — omitting `Scatter` is meaningful, since it tells the validator another scatter source is permissible.

- [ ] **Step 6: Wire validation into the registry**

In `rust/steel-cli/src/registry.rs`, after all plugins are built, collect descriptors and validate before returning the `Simulation`:

```rust
    let descriptors = vec![smhm_descriptor, sfr_descriptor, quenching_descriptor];
    if let Err(problems) = validate_composition(&descriptors, run_cosmology_tag) {
        let detail = problems.iter().map(|p| format!("  - {p}")).collect::<Vec<_>>().join("\n");
        return Err(anyhow!(
            "incompatible plugin combination in this runfile:\n{detail}\n\n\
             See docs/model-assumptions.md for what each plugin assumes."
        ));
    }
```

Each `build_*` function returns its descriptor alongside the boxed trait object; change their return types to `Result<(Box<dyn Trait>, PluginDescriptor)>`.

- [ ] **Step 7: Add an end-to-end rejection test**

Create `rust/steel-plugins/tests/composition_rejection.rs`:

```rust
//! A runfile requesting an incompatible plugin set must fail at startup
//! with an actionable message, not run and produce wrong numbers.

use steel_core::compat::{
    validate_composition, Capability, CosmologyTag, HConvention, Imf, PluginDescriptor,
};
use steel_core::cosmology::MassDefinition;

#[test]
fn message_names_both_plugins_and_the_duplicated_effect() {
    let um = PluginDescriptor {
        id: "universe_machine",
        imf: Imf::Chabrier,
        mass_definition: MassDefinition::Vir,
        h_convention: HConvention::PerH,
        calibrated_cosmology: Some(CosmologyTag::Planck15),
        provides: &[Capability::StellarMass, Capability::Quenching],
    };
    let wetzel = PluginDescriptor {
        id: "wetzel13",
        provides: &[Capability::Quenching],
        ..um.clone()
    };
    let errs = validate_composition(&[um, wetzel], CosmologyTag::Planck15).expect_err("reject");
    let text = errs.iter().map(|e| e.to_string()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("universe_machine"), "{text}");
    assert!(text.contains("wetzel13"), "{text}");
    assert!(text.contains("twice"), "message should say the effect is applied twice: {text}");
}
```

- [ ] **Step 8: Run the full suite**

Run: `cd rust && cargo test --workspace`
Expected: PASS, including the Task 1 bit-identity guards.

- [ ] **Step 9: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-core/src/compat.rs rust/steel-core/src/lib.rs \
        rust/steel-core/src/cosmology.rs rust/steel-plugins/src/smhm/ \
        rust/steel-plugins/src/sfr.rs rust/steel-cli/src/registry.rs \
        rust/steel-plugins/tests/composition_rejection.rs
git commit -m "feat: reject incompatible plugin compositions at startup

Declared-descriptor rule set catching silent double-counting: two sources
of M*, two scatter applications, two quenching prescriptions (the live
UniverseMachine case), IMF/mass-definition/h mismatches, and plugins
fitted under a different cosmology than the run. Hard error, never a
warning. Spec section 8.1."
```

---

### Task 8: EMERGE upstream reference fixtures

Spec §6. Build and run upstream EMERGE once; commit its output as a regression target.

**Blocked on:** the `Cargo.toml` MIT vs `LICENSE` AGPL-3.0 discrepancy noted in the File Structure section. Raise with the maintainer before starting.

**Files:**
- Create: `scripts/fixtures/build_emerge_fixture.sh`
- Create: `rust/steel-plugins/tests/fixtures/emerge/provenance.toml`
- Create: `rust/steel-plugins/tests/fixtures/emerge/eps_grid.npy`, `.../smhm_grid.npy`

**Interfaces:**
- Consumes: nothing in-tree.
- Produces: committed `.npy` grids plus `provenance.toml`. Task 9 asserts against these. Grid axes are fixed here and Task 9 depends on them: `log_mh` from 10.0 to 15.0 in 0.1 dex (51 points), `z` in `[0.1, 0.5, 1.0, 2.0, 4.0, 6.0]` (6 points). `eps_grid.npy` and `smhm_grid.npy` are both `float64` with shape `(51, 6)`, C-order, indexed `[mass, redshift]`.

- [ ] **Step 1: Write the fixture build script**

Create `scripts/fixtures/build_emerge_fixture.sh`:

```bash
#!/usr/bin/env bash
# Builds upstream EMERGE and dumps reference grids for regression tests.
#
# Upstream is cloned OUT OF TREE and never committed: it is an MPI
# whole-pipeline C code, not a library, so vendoring would add a C
# toolchain for no fidelity gain over pinned outputs. Spec section 6.
#
# Usage: build_emerge_fixture.sh <scratch-dir> <output-dir>
set -euo pipefail

SCRATCH="${1:?scratch dir required}"
OUTDIR="${2:?output dir required}"

REPO="https://github.com/bmoster/emerge.git"
REF="v1.0.2"
EXPECTED_SHA="2781b54c21a80acf237daf7f2e71ff6254da8c3b"

mkdir -p "$SCRATCH" "$OUTDIR"
cd "$SCRATCH"

if [ ! -d emerge ]; then
  git clone --branch "$REF" --depth 1 "$REPO" emerge
fi
cd emerge

ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: upstream HEAD $ACTUAL_SHA != pinned $EXPECTED_SHA" >&2
  echo "Upstream moved. Do not proceed: re-pin deliberately and re-verify." >&2
  exit 1
fi

echo "== building =="
make clean || true
make

echo "== upstream built at $ACTUAL_SHA =="
echo "Next: run EMERGE with a Planck15-matched parameter file and dump"
echo "eps(M_h, z) and integrated M*(M_h, z) on the grid fixed in the plan."
```

Make it executable: `chmod +x scripts/fixtures/build_emerge_fixture.sh`

- [ ] **Step 2: Run the build and confirm the pinned SHA**

Run:
```bash
cd /Users/pgrylls/Code/STEEL
SCRATCH=/private/tmp/claude-502/-Users-pgrylls-Code-STEEL/0ef466b8-7a46-4ca2-8035-1ae84039a873/scratchpad
./scripts/fixtures/build_emerge_fixture.sh "$SCRATCH" rust/steel-plugins/tests/fixtures/emerge
```
Expected: clone succeeds, SHA matches `2781b54c...`, `make` completes.

**If `make` fails** (missing MPI, GSL, HDF5): record the exact error and stop. Per spec §12, the fallback is published tabulated data with the weakened validation stated explicitly — but that is a maintainer decision, not the executor's.

**If the SHA does not match**, stop. Upstream moved and the pin must be updated deliberately.

- [ ] **Step 3: Extract the parameter values from upstream, not from prose**

Read upstream's parameter file (typically `parameters/*.param` or the defaults in `src/`) and the O'Leary+2023 PDF tables. Record each coefficient with its source:

```
eps_N0, eps_Nz  <- <upstream file>:<line>  /  O'Leary+23 Table <n>
M_10,  M_1z     <- ...
beta_0, beta_z  <- ...
gamma_0         <- ...
tau_s, M_q, a_q, R_q  <- ...
```

Per the Global Constraints, a coefficient sourced only from an HTML summary must not be committed. The provisional values circulated during design (`beta_0 = 2.22`, `beta_z = -1.50`, `tau_s = 0.40`, `M_q = 9.33`, `a_q = 0.19`, `R_q = 2.56`) are **unverified** and exist only to be checked against the real source.

- [ ] **Step 4: Dump the reference grids**

Run upstream at a Planck15-matched cosmology and write `eps_grid.npy` and `smhm_grid.npy` on the grid fixed in the Interfaces block. If upstream emits text, convert with a short one-off python script in the scratchpad (not committed):

```python
import numpy as np
# 51 mass points x 6 redshifts, float64, C-order, [mass, redshift]
eps = np.loadtxt("eps.txt").reshape(51, 6).astype(np.float64)
np.save("rust/steel-plugins/tests/fixtures/emerge/eps_grid.npy", eps)
```

- [ ] **Step 5: Write the provenance record**

Create `rust/steel-plugins/tests/fixtures/emerge/provenance.toml`. Replace every angle-bracketed placeholder with real values — the Task 9 test parses this file and fails if any remain.

```toml
# Provenance for the committed EMERGE reference grids.
# Everything needed to regenerate them from scratch.

[upstream]
repo = "https://github.com/bmoster/emerge.git"
ref = "v1.0.2"
commit = "2781b54c21a80acf237daf7f2e71ff6254da8c3b"
build_command = "make"
run_command = "<exact command line used>"

[cosmology]
name = "Planck15"
h = 0.6774
omega_m0 = 0.3089
omega_b0 = 0.0486
omega_de0 = 0.6911

[grid]
log_mh_min = 10.0
log_mh_max = 15.0
log_mh_step = 0.1
n_mass = 51
redshifts = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
dtype = "float64"
order = "C"
axes = "[mass, redshift]"

[conventions]
# What the dumped masses actually are, so Task 9 converts rather than assumes.
halo_mass_definition = "<vir | 200c | 200m — read from upstream>"
h_convention = "<h_free | per_h>"
imf = "<chabrier | kroupa | salpeter>"

[files]
eps_grid = "eps_grid.npy"
smhm_grid = "smhm_grid.npy"

[generated]
date = "<YYYY-MM-DD>"
by = "scripts/fixtures/build_emerge_fixture.sh"
```

- [ ] **Step 6: Verify the fixtures load and are physically sane**

Create `rust/steel-plugins/tests/upstream_agreement.rs`:

```rust
//! Validates our Rust implementations against committed upstream
//! reference grids. Spec section 6.

use ndarray::Array2;
use ndarray_npy::read_npy;

const EMERGE_DIR: &str = "tests/fixtures/emerge";

fn load(name: &str) -> Array2<f64> {
    read_npy(format!("{EMERGE_DIR}/{name}")).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

#[test]
fn emerge_fixtures_have_the_documented_shape() {
    for name in ["eps_grid.npy", "smhm_grid.npy"] {
        let a = load(name);
        assert_eq!(a.shape(), &[51, 6], "{name} shape");
        assert!(a.iter().all(|v| v.is_finite()), "{name} contains non-finite values");
    }
}

#[test]
fn emerge_efficiency_is_a_physical_fraction() {
    let eps = load("eps_grid.npy");
    assert!(
        eps.iter().all(|&v| v > 0.0 && v <= 1.0),
        "conversion efficiency must lie in (0, 1]; got min {} max {}",
        eps.iter().cloned().fold(f64::INFINITY, f64::min),
        eps.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    );
}

#[test]
fn emerge_smhm_is_monotonic_in_halo_mass() {
    let smhm = load("smhm_grid.npy");
    for j in 0..smhm.ncols() {
        for i in 1..smhm.nrows() {
            assert!(
                smhm[[i, j]] >= smhm[[i - 1, j]] - 1e-9,
                "M* decreased with Mh at row {i}, col {j}"
            );
        }
    }
}

#[test]
fn emerge_provenance_has_no_unfilled_placeholders() {
    let text = std::fs::read_to_string(format!("{EMERGE_DIR}/provenance.toml"))
        .expect("provenance.toml must exist");
    assert!(!text.contains('<'), "provenance.toml still contains placeholders:\n{text}");
}
```

Add to `rust/steel-plugins/Cargo.toml` under `[dev-dependencies]`:
```toml
ndarray-npy = { workspace = true }
```

Run: `cd rust && cargo test -p steel-plugins --test upstream_agreement`
Expected: PASS (4 tests).

- [ ] **Step 7: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add scripts/fixtures/build_emerge_fixture.sh \
        rust/steel-plugins/tests/fixtures/emerge/ \
        rust/steel-plugins/tests/upstream_agreement.rs \
        rust/steel-plugins/Cargo.toml
git commit -m "test: pin upstream EMERGE reference grids as fixtures

Clones and builds upstream EMERGE v1.0.2 out of tree, dumps eps(Mh,z) and
M*(Mh,z), commits the grids plus full provenance. Keeps the build pure
cargo while pinning our implementation to upstream numerics, following the
getPWGH precedent. Spec section 6."
```

---

### Task 9: EMERGE plugin

**Files:**
- Create: `rust/steel-plugins/src/growth_models/mod.rs`, `rust/steel-plugins/src/growth_models/emerge.rs`
- Modify: `rust/steel-plugins/src/lib.rs`, `rust/steel-io/src/runfile.rs`, `rust/steel-cli/src/registry.rs`
- Modify: `rust/steel-plugins/tests/upstream_agreement.rs`

**Interfaces:**
- Consumes: `StellarGrowthModel` + `integrate_stellar_mass` (Task 5), `AccretionContext` (Task 2), `Imf`/`HConvention` (Task 6), `DescribedPlugin` (Task 7), the fixtures (Task 8).
- Note: `AccretionContext::formation_redshift` (Task 2) is **not** used here. History dependence enters through integration: the gate is evaluated pointwise at each `(log_mass, z)` along the track, so a late-forming halo is suppressed because its early-epoch mass is low. `formation_redshift` remains available for a future model that needs an explicit formation epoch.
- Produces: `steel_plugins::growth_models::EmergeGrowth` with `EmergeGrowth::o_leary23()`, implementing `StellarGrowthModel` and `DescribedPlugin`. Runfile selector `[stellar_growth] model = "emerge"`, `preset = "o_leary23"`.

- [ ] **Step 1: Write the failing unit tests**

Create `rust/steel-plugins/src/growth_models/emerge.rs` with the test module only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use steel_core::accretion::AccretionContext;
    use steel_core::cosmology::MassDefinition;
    use steel_core::halo_growth::GrowthTrack;

    fn ctx_for<'a>(t: &'a GrowthTrack, c: &'a Planck15) -> AccretionContext<'a> {
        AccretionContext::central(t, c, MassDefinition::Vir)
    }

    /// Track for a halo that assembled early: already massive at high z.
    fn early_track() -> GrowthTrack {
        GrowthTrack {
            z: vec![0.0, 1.0, 2.0, 4.0, 6.0],
            log_mass: vec![11.0, 10.95, 10.9, 10.8, 10.7],
        }
    }

    /// Same z=0 mass, assembled late: progenitors are far smaller.
    fn late_track() -> GrowthTrack {
        GrowthTrack {
            z: vec![0.0, 1.0, 2.0, 4.0, 6.0],
            log_mass: vec![11.0, 10.2, 9.4, 8.6, 8.0],
        }
    }

    #[test]
    fn efficiency_peaks_at_the_pivot_mass() {
        let m = EmergeGrowth::o_leary23();
        let at_pivot = m.efficiency(m.log_m1(0.0), 0.0);
        let below = m.efficiency(m.log_m1(0.0) - 2.0, 0.0);
        let above = m.efficiency(m.log_m1(0.0) + 2.0, 0.0);
        assert!(at_pivot > below, "{at_pivot} vs {below}");
        assert!(at_pivot > above, "{at_pivot} vs {above}");
    }

    #[test]
    fn efficiency_is_a_fraction() {
        let m = EmergeGrowth::o_leary23();
        for log_mh in [9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0] {
            for z in [0.0, 1.0, 3.0, 6.0] {
                let e = m.efficiency(log_mh, z);
                assert!(e > 0.0 && e <= 1.0, "eps({log_mh}, {z}) = {e}");
            }
        }
    }

    #[test]
    fn growth_rate_increases_with_halo_mass_below_the_pivot() {
        let m = EmergeGrowth::o_leary23();
        let c = Planck15::new();
        let t = early_track();
        let ctx = ctx_for(&t, &c);
        let r10 = m.stellar_growth_rate(10.0, 1.0, &ctx, None);
        let r11 = m.stellar_growth_rate(11.0, 1.0, &ctx, None);
        assert!(r11 > r10, "rate should rise with mass below pivot: {r10} then {r11}");
    }

    /// The reionization gate is the paper's dwarf-regime contribution: a
    /// halo that only assembled after the gate epoch is suppressed.
    #[test]
    fn gate_suppresses_late_forming_low_mass_halos() {
        let m = EmergeGrowth::o_leary23();
        let c = Planck15::new();
        let early = early_track();
        let late = late_track();
        let m_early = steel_core::integrate_stellar_mass(&m, &ctx_for(&early, &c), 0.0, None);
        let m_late = steel_core::integrate_stellar_mass(&m, &ctx_for(&late, &c), 0.0, None);
        assert!(
            m_late < m_early,
            "late-forming halo should end up less massive: {m_late} vs {m_early}"
        );
    }

    #[test]
    fn gate_does_not_suppress_massive_halos() {
        let m = EmergeGrowth::o_leary23();
        // Well above M_q, the gate must be inert (factor ~1).
        let g = m.gate_factor(13.0, 4.0);
        assert!((g - 1.0).abs() < 1e-6, "gate factor at high mass = {g}");
    }

    #[test]
    fn descriptor_declares_stellar_mass_and_not_quenching() {
        let d = EmergeGrowth::o_leary23().descriptor();
        assert!(d.provides.contains(&Capability::StellarMass));
        // EMERGE's gate suppresses early low-mass growth; it is not a
        // satellite quenching prescription, so STEEL's QuenchingModel
        // remains compatible.
        assert!(!d.provides.contains(&Capability::Quenching));
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-plugins emerge`
Expected: FAIL — `EmergeGrowth` undefined.

- [ ] **Step 3: Implement `EmergeGrowth`**

Prepend to `rust/steel-plugins/src/growth_models/emerge.rs`:

```rust
//! EMERGE: empirical galaxy formation via a baryon conversion efficiency
//! applied to the halo accretion rate.
//!
//! Moster, Naab & White (2018) for the base model; O'Leary,
//! Steinwandel, Moster, Martin & Naab (2023), arXiv:2301.07122, for the
//! dwarf-regime extension implemented here.
//!
//! ```text
//! dm*/dt   = eps(M_h, z) . f_b . dM_h/dt
//! eps(M)   = 2 eps_N / [ (M/M_1)^-beta + (M/M_1)^gamma ]
//! ```
//!
//! **Do not confuse this with `crate::smhm::moster`.** The `eps` double
//! power law is algebraically identical to `MosterFormSmhm`'s equation,
//! but `eps` multiplies an accretion *rate* whereas `MosterFormSmhm`
//! multiplies a *mass*. Substituting these coefficients there — or those
//! there here — silently yields a wrong SMHM curve with no error. The
//! stellar-to-halo mass relation in O'Leary+2023's title is an *output*
//! of integrating this rate, not an input parametrisation.
//!
//! Redshift evolution is linear in `z/(1+z)`, matching Moster's
//! convention (`ZEvo::MosterStyle` in `crate::smhm::moster`) rather than
//! STEEL's shifted `(z-0.1)/(1+z)`.
//!
//! The reionization gate penalises late-forming low-mass haloes:
//! ```text
//! M_h^min(a) = M_q / [1 + exp(-R_q (a - a_q))]
//! ```
//! Applied via `gate_factor`, using the object's own growth track, so it
//! is exact for satellites as well as centrals (spec section 5).

use rand::RngCore;

use steel_core::accretion::AccretionContext;
use steel_core::compat::{
    Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor,
};
use steel_core::cosmology::MassDefinition;
use steel_core::stellar_growth::StellarGrowthModel;

pub struct EmergeGrowth {
    /// `eps_N(z) = eps_n[0] + eps_n[1] . z/(1+z)`
    eps_n: [f64; 2],
    /// `log10 M_1(z) = log_m1[0] + log_m1[1] . z/(1+z)`
    log_m1: [f64; 2],
    /// `beta(z) = beta[0] + beta[1] . z/(1+z)`
    beta: [f64; 2],
    /// `gamma`, held fixed with redshift in O'Leary+2023.
    gamma: f64,
    /// Reionization gate: characteristic mass `M_q` \[log10 Msun\],
    /// scale factor `a_q`, and steepness `R_q`.
    log_m_q: f64,
    a_q: f64,
    r_q: f64,
    /// Baryon fraction f_b = Omega_b / Omega_m, taken from the run
    /// cosmology at construction.
    baryon_fraction: f64,
}

impl EmergeGrowth {
    /// O'Leary et al. (2023) logistic-quenching best fit.
    ///
    /// **PROVISIONAL COEFFICIENTS.** Every value below must be replaced
    /// with one read from the paper's parameter table or upstream's
    /// parameter file, and annotated with that source, before any result
    /// is published. See spec section 6.1 and Task 8 Step 3. The
    /// `baryon_fraction` default matches Planck15
    /// (0.0486/0.3089 = 0.1573).
    pub fn o_leary23() -> Self {
        Self {
            eps_n: [0.005, 0.689],   // SOURCE REQUIRED
            log_m1: [11.339, 0.692], // SOURCE REQUIRED
            beta: [2.22, -1.50],     // SOURCE REQUIRED
            gamma: 0.966,            // SOURCE REQUIRED
            log_m_q: 9.33,           // SOURCE REQUIRED
            a_q: 0.19,               // SOURCE REQUIRED
            r_q: 2.56,               // SOURCE REQUIRED
            baryon_fraction: 0.0486 / 0.3089,
        }
    }

    fn z_param(z: f64) -> f64 {
        z / (1.0 + z)
    }

    /// `log10 M_1(z)`, the pivot mass.
    pub fn log_m1(&self, z: f64) -> f64 {
        self.log_m1[0] + self.log_m1[1] * Self::z_param(z)
    }

    /// Instantaneous baryon conversion efficiency, dimensionless in (0, 1].
    pub fn efficiency(&self, log_mh: f64, z: f64) -> f64 {
        let zp = Self::z_param(z);
        let eps_n = self.eps_n[0] + self.eps_n[1] * zp;
        let beta = self.beta[0] + self.beta[1] * zp;
        let ratio = 10f64.powf(log_mh - self.log_m1(z));
        let eps = 2.0 * eps_n / (ratio.powf(-beta) + ratio.powf(self.gamma));
        eps.clamp(f64::MIN_POSITIVE, 1.0)
    }

    /// Reionization suppression in \[0, 1\]. Unity for haloes already
    /// above `M_h^min` at this epoch, falling towards zero for haloes
    /// below it — so a halo that assembled its mass late is penalised.
    pub fn gate_factor(&self, log_mh: f64, z: f64) -> f64 {
        let a = 1.0 / (1.0 + z);
        let log_m_min = self.log_m_q - (1.0 + (-self.r_q * (a - self.a_q)).exp()).log10();
        // Smooth in log mass so the rate stays differentiable; the
        // logistic width is set by R_q as in the paper.
        1.0 / (1.0 + 10f64.powf(-(log_mh - log_m_min)))
    }

    /// `dM_h/dt` \[Msun/yr\] at `z`, from the object's own track by
    /// central difference. Returns 0 where the track cannot support a
    /// finite difference.
    fn halo_accretion_rate(&self, z: f64, ctx: &AccretionContext<'_>) -> f64 {
        let t = ctx.own_track;
        if t.z.len() < 2 {
            return 0.0;
        }
        // Nearest sample to `z`; the track is increasing into the past.
        let i = t
            .z
            .iter()
            .enumerate()
            .min_by(|a, b| (a.1 - z).abs().partial_cmp(&(b.1 - z).abs()).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        let (i0, i1) = if i == 0 { (0, 1) } else { (i - 1, i) };
        // i0 is younger (lower z) than i1.
        let dt_yr = (ctx.cosmology.age(t.z[i0]) - ctx.cosmology.age(t.z[i1])) * 1.0e9;
        if dt_yr <= 0.0 {
            return 0.0;
        }
        let dm = 10f64.powf(t.log_mass[i0]) - 10f64.powf(t.log_mass[i1]);
        (dm / dt_yr).max(0.0)
    }
}

impl StellarGrowthModel for EmergeGrowth {
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        _rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        let mdot_h = self.halo_accretion_rate(z, ctx);
        if mdot_h <= 0.0 {
            return f64::NEG_INFINITY;
        }
        let rate = self.efficiency(log_mh, z)
            * self.gate_factor(log_mh, z)
            * self.baryon_fraction
            * mdot_h;
        if rate <= 0.0 {
            f64::NEG_INFINITY
        } else {
            rate.log10()
        }
    }
}

impl DescribedPlugin for EmergeGrowth {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "emerge",
            // VERIFY against O'Leary+2023 section 2 before publishing.
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::HFree,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            // Supplies M* and its own scatter. The reionization gate is
            // early-growth suppression, not satellite quenching, so
            // Quenching is deliberately absent and STEEL's
            // QuenchingModel stays compatible.
            provides: &[Capability::StellarMass, Capability::Scatter],
        }
    }
}
```

Create `rust/steel-plugins/src/growth_models/mod.rs`:

```rust
//! Rate-based stellar mass assembly models.
//!
//! These implement `steel_core::StellarGrowthModel` rather than
//! `SmhmModel`: they specify dM*/dt and M* is obtained by integration
//! along the growth track. See `steel_core::stellar_growth` for why the
//! distinction is load-bearing.

mod emerge;
mod universe_machine;

pub use emerge::EmergeGrowth;
pub use universe_machine::UniverseMachineGrowth;
```

For this task, comment out the `universe_machine` lines; Task 11 restores them.

In `rust/steel-plugins/src/lib.rs`: add `pub mod growth_models;` and `pub use growth_models::EmergeGrowth;`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test -p steel-plugins emerge`
Expected: PASS (6 tests).

- [ ] **Step 5: Add the fixture agreement test**

Append to `rust/steel-plugins/tests/upstream_agreement.rs`:

```rust
use steel_core::accretion::AccretionContext;
use steel_core::cosmology::MassDefinition;
use steel_core::halo_growth::{GrowthTrack, HaloGrowthModel};
use steel_plugins::{EmergeGrowth, Planck15, VandenBosch14};

const REDSHIFTS: [f64; 6] = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0];

/// Largest absolute deviation in dex between our efficiency and
/// upstream's, printed so the achieved figure can be recorded as this
/// plugin's reference tolerance (spec section 6, step 5).
#[test]
fn emerge_efficiency_agrees_with_upstream() {
    let eps_ref = load("eps_grid.npy");
    let m = EmergeGrowth::o_leary23();
    let mut worst = 0.0_f64;
    let mut worst_at = (0.0, 0.0);

    for (i, log_mh) in (0..51).map(|i| (i, 10.0 + i as f64 * 0.1)) {
        for (j, &z) in REDSHIFTS.iter().enumerate() {
            let ours = m.efficiency(log_mh, z).log10();
            let theirs = eps_ref[[i, j]].log10();
            let d = (ours - theirs).abs();
            if d > worst {
                worst = d;
                worst_at = (log_mh, z);
            }
        }
    }

    println!("worst eps deviation {worst:.6} dex at log_mh={} z={}", worst_at.0, worst_at.1);
    assert!(
        worst < 0.01,
        "worst deviation {worst:.6} dex at log_mh={} z={} exceeds 0.01; identify the cause \
         rather than widening the bound (spec section 6)",
        worst_at.0,
        worst_at.1
    );
}

#[test]
fn emerge_integrated_smhm_agrees_with_upstream() {
    let smhm_ref = load("smhm_grid.npy");
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);
    let m = EmergeGrowth::o_leary23();
    let mut worst = 0.0_f64;

    // Compare at z=0.1 (column 0) across the mass axis.
    for (i, log_mh) in (0..51).map(|i| (i, 10.0 + i as f64 * 0.1)) {
        let track = growth.growth_history(log_mh, 0.1);
        let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
        let ours = steel_core::integrate_stellar_mass(&m, &ctx, 0.1, None);
        if !ours.is_finite() {
            continue;
        }
        worst = worst.max((ours - smhm_ref[[i, 0]]).abs());
    }

    println!("worst integrated M* deviation {worst:.6} dex");
    assert!(worst < 0.05, "worst integrated M* deviation {worst:.6} dex exceeds 0.05");
}
```

The `0.01` and `0.05` bounds are the spec's investigate-above thresholds, not targets. Record the *achieved* figures in `docs/VALIDATION.md` and tighten the assertions to just above them.

- [ ] **Step 6: Run and record achieved agreement**

Run: `cd rust && cargo test -p steel-plugins --test upstream_agreement -- --nocapture`
Expected: PASS, printing the worst deviations.

**If either fails**, the likely causes in order: (a) unverified coefficients from Step 3 of Task 8; (b) an `h` or mass-definition mismatch against `provenance.toml` `[conventions]`; (c) the redshift-evolution convention (`z/(1+z)` vs `(z-0.1)/(1+z)`). Diagnose before touching the bound.

Add the achieved numbers to `docs/VALIDATION.md` under a new "External model agreement" section.

- [ ] **Step 7: Wire into runfile and registry**

In `rust/steel-io/src/runfile.rs`, add an optional section:

```rust
/// Rate-based stellar growth model, an alternative to `[smhm]`. Exactly
/// one of `[smhm]` and `[stellar_growth]` may be present; the composition
/// validator rejects both (duplicate `Capability::StellarMass`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct StellarGrowthConfig {
    pub model: String,
    pub preset: String,
}
```

Add `pub stellar_growth: Option<StellarGrowthConfig>` to `RunFile`, and a parse test:

```rust
    #[test]
    fn parses_stellar_growth_section() {
        let run: RunFile = toml::from_str(
            r#"
            [stellar_growth]
            model = "emerge"
            preset = "o_leary23"
            "#,
        )
        .expect("should parse");
        let sg = run.stellar_growth.expect("section present");
        assert_eq!(sg.model, "emerge");
        assert_eq!(sg.preset, "o_leary23");
    }
```

In `registry.rs`, add `build_stellar_growth` returning `Result<(Box<dyn StellarGrowthModel>, PluginDescriptor)>`, matching `("emerge", "o_leary23") => EmergeGrowth::o_leary23()`, and include its descriptor in the `validate_composition` call from Task 7 Step 6.

- [ ] **Step 8: Run the full suite**

Run: `cd rust && cargo test --workspace`
Expected: PASS, including the Task 1 bit-identity guards.

- [ ] **Step 9: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-plugins/src/growth_models/ rust/steel-plugins/src/lib.rs \
        rust/steel-plugins/tests/upstream_agreement.rs \
        rust/steel-io/src/runfile.rs rust/steel-cli/src/registry.rs \
        docs/VALIDATION.md
git commit -m "feat: add EMERGE as a rate-based stellar growth plugin

Baryon conversion efficiency applied to the halo accretion rate, with
O'Leary+2023's logistic reionization gate evaluated on the object's own
growth track (exact for satellites, not proxied). Validated against
committed upstream fixtures. Module docs warn against substituting these
coefficients into MosterFormSmhm despite identical algebra.

Coefficients remain PROVISIONAL pending verification against the paper
tables; see spec section 6.1."
```

---

### Task 10: UM-SAGA upstream reference fixtures

**Files:**
- Create: `scripts/fixtures/build_um_fixture.sh`
- Create: `rust/steel-plugins/tests/fixtures/um_saga/provenance.toml`
- Create: `.../sfr_sf_grid.npy`, `.../quenched_fraction_grid.npy`

**Interfaces:**
- Consumes: nothing in-tree.
- Produces: committed grids on a **velocity** axis (UM is `vMpeak`-keyed): `log_vmpeak` from 1.4 to 3.0 in 0.04 dex (41 points), `z` in `[0.1, 0.5, 1.0, 2.0, 4.0, 6.0]` (6 points). Both files `float64`, shape `(41, 6)`, C-order, `[velocity, redshift]`. `sfr_sf_grid.npy` is log10 SFR \[Msun/yr\] for the star-forming mode; `quenched_fraction_grid.npy` is `f_Q` in \[0,1\]. Task 11 asserts against these.

- [ ] **Step 1: Write the fixture build script**

Create `scripts/fixtures/build_um_fixture.sh`:

```bash
#!/usr/bin/env bash
# Builds upstream UniverseMachine (UM-SAGA branch) and dumps reference
# grids. Cloned out of tree; never committed. Spec section 6.
#
# Usage: build_um_fixture.sh <scratch-dir> <output-dir>
set -euo pipefail

SCRATCH="${1:?scratch dir required}"
OUTDIR="${2:?output dir required}"

REPO="https://bitbucket.org/RW-Stanford/universemachine-saga.git"
REF="saga"
EXPECTED_SHA="6aff8d792e81bf6049058e3e1bc6f2cfa616b525"

mkdir -p "$SCRATCH" "$OUTDIR"
cd "$SCRATCH"

if [ ! -d universemachine-saga ]; then
  git clone --branch "$REF" "$REPO" universemachine-saga
fi
cd universemachine-saga

ACTUAL_SHA="$(git rev-parse HEAD)"
if [ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]; then
  echo "FATAL: upstream $REF HEAD $ACTUAL_SHA != pinned $EXPECTED_SHA" >&2
  echo "Upstream moved. Re-pin deliberately and re-verify." >&2
  exit 1
fi

echo "== locating the UM-SAGA best-fit parameter file =="
find . -name '*.param' -o -name '*fit*' -o -name '*param*' | head -40

echo "== building =="
make clean || true
make

echo "== upstream built at $ACTUAL_SHA =="
echo "Next: evaluate SFR(vMpeak, z) and f_Q(vMpeak, z) on the grid fixed"
echo "in the plan, using the UM-SAGA best-fit parameters."
```

`chmod +x scripts/fixtures/build_um_fixture.sh`

- [ ] **Step 2: Run the build and confirm the pinned SHA**

Run:
```bash
cd /Users/pgrylls/Code/STEEL
SCRATCH=/private/tmp/claude-502/-Users-pgrylls-Code-STEEL/0ef466b8-7a46-4ca2-8035-1ae84039a873/scratchpad
./scripts/fixtures/build_um_fixture.sh "$SCRATCH" rust/steel-plugins/tests/fixtures/um_saga
```
Expected: clone succeeds, SHA matches `6aff8d79...`, build completes, parameter file located.

Note: cloning without `--depth 1` here because a specific non-default branch is needed and the pinned SHA must be verifiable.

**If the build fails or the parameter file is absent**, record the exact error and stop, as in Task 8 Step 2.

- [ ] **Step 3: Extract the 15 fitted parameters from the upstream parameter file**

UM-SAGA has 15 explored parameters: 9 inherited from UM DR1 plus 6 new low-mass quenching terms. Record each with its source, in this order (Task 11 consumes exactly these names):

```
alpha_0, alpha_a, alpha_la, alpha_z   <- <file>:<line>
r_min, r_width                        <- ...
v_r_0, v_r_a                          <- ...
t_merge_300                           <- ...
v_q2_0, v_q2_a, v_q2_z                <- ...   (new low-mass quenching)
sigma_vq2_0, sigma_vq2_a, sigma_vq2_z <- ...   (new low-mass quenching)
```

The values circulated during design (`alpha_0 = -6.14`, `alpha_a = -3.93`, `alpha_z = -0.54`, `alpha_la = 6.37`, `r_min = 0.48`, `r_width = 0.19`, `v_r_0 = 2.24`, `v_r_a = -5.72`, `t_merge_300 = 0.71`, `v_q2_0 = 1.66`, `v_q2_a = -0.23`, `v_q2_z = -0.63`, `sigma_vq2_0 = 0.14`, `sigma_vq2_a = 0.38`, `sigma_vq2_z = 0.22`) are **unverified** and exist only to be checked. Also record the **beta** power-law parameters, which the design summary did not capture at all — the SFR double power law needs both `alpha` and `beta` branches.

- [ ] **Step 4: Dump the reference grids**

Evaluate upstream's SFR and quenched-fraction functions on the grid from the Interfaces block and save as `.npy`, as in Task 8 Step 4.

- [ ] **Step 5: Write the provenance record**

Create `rust/steel-plugins/tests/fixtures/um_saga/provenance.toml`, same shape as EMERGE's but with the velocity axis and one extra field:

```toml
[upstream]
repo = "https://bitbucket.org/RW-Stanford/universemachine-saga.git"
ref = "saga"
commit = "6aff8d792e81bf6049058e3e1bc6f2cfa616b525"
build_command = "make"
run_command = "<exact command line used>"
parameter_file = "<path within the repo>"

[cosmology]
name = "Planck15"
h = 0.6774
omega_m0 = 0.3089
omega_b0 = 0.0486
omega_de0 = 0.6911

[grid]
log_vmpeak_min = 1.4
log_vmpeak_max = 3.0
log_vmpeak_step = 0.04
n_velocity = 41
redshifts = [0.1, 0.5, 1.0, 2.0, 4.0, 6.0]
dtype = "float64"
order = "C"
axes = "[velocity, redshift]"

[conventions]
velocity_units = "km/s"
# Which velocity definition vMpeak means upstream: peak over history of
# Vmax, or Vmax at the epoch of peak mass. These differ and Task 11's
# Mpeak -> Vmax conversion must match whichever this is.
vmpeak_definition = "<read from upstream>"
halo_mass_definition = "<vir | 200c | 200m>"
h_convention = "<h_free | per_h>"
imf = "<chabrier | kroupa | salpeter>"

[files]
sfr_sf_grid = "sfr_sf_grid.npy"
quenched_fraction_grid = "quenched_fraction_grid.npy"

[generated]
date = "<YYYY-MM-DD>"
by = "scripts/fixtures/build_um_fixture.sh"
```

- [ ] **Step 6: Add fixture sanity tests**

Append to `rust/steel-plugins/tests/upstream_agreement.rs`:

```rust
const UM_DIR: &str = "tests/fixtures/um_saga";

fn load_um(name: &str) -> Array2<f64> {
    read_npy(format!("{UM_DIR}/{name}")).unwrap_or_else(|e| panic!("load {name}: {e}"))
}

#[test]
fn um_fixtures_have_the_documented_shape() {
    for name in ["sfr_sf_grid.npy", "quenched_fraction_grid.npy"] {
        let a = load_um(name);
        assert_eq!(a.shape(), &[41, 6], "{name} shape");
        assert!(a.iter().all(|v| v.is_finite()), "{name} has non-finite values");
    }
}

#[test]
fn um_quenched_fraction_is_a_fraction() {
    let f = load_um("quenched_fraction_grid.npy");
    assert!(f.iter().all(|&v| (0.0..=1.0).contains(&v)), "f_Q outside [0,1]");
}

#[test]
fn um_sfr_rises_with_velocity_at_low_mass() {
    // Below the pivot, more massive (faster) haloes form more stars.
    let s = load_um("sfr_sf_grid.npy");
    assert!(s[[20, 0]] > s[[0, 0]], "SFR should rise with vMpeak at the low end");
}

#[test]
fn um_provenance_has_no_unfilled_placeholders() {
    let text = std::fs::read_to_string(format!("{UM_DIR}/provenance.toml"))
        .expect("provenance.toml must exist");
    assert!(!text.contains('<'), "provenance.toml still contains placeholders:\n{text}");
}
```

Run: `cd rust && cargo test -p steel-plugins --test upstream_agreement`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add scripts/fixtures/build_um_fixture.sh \
        rust/steel-plugins/tests/fixtures/um_saga/ \
        rust/steel-plugins/tests/upstream_agreement.rs
git commit -m "test: pin upstream UM-SAGA reference grids as fixtures

SFR(vMpeak,z) and quenched fraction from upstream UniverseMachine's saga
branch, on a velocity axis since UM is vMpeak-keyed. Records the vMpeak
definition explicitly: the Mpeak->Vmax conversion in the Rust plugin must
match it. Spec section 6."
```

---

### Task 11: UniverseMachine plugin

**Files:**
- Create: `rust/steel-plugins/src/growth_models/universe_machine.rs`
- Modify: `rust/steel-plugins/src/growth_models/mod.rs`, `rust/steel-plugins/src/lib.rs`, `rust/steel-cli/src/registry.rs`, `rust/steel-plugins/tests/upstream_agreement.rs`

**Interfaces:**
- Consumes: `StellarGrowthModel` (Task 5), `mpeak_to_vmax` + `DuttonMaccio14` (Task 6), `DescribedPlugin` (Task 7), fixtures (Task 10).
- Produces: `steel_plugins::growth_models::UniverseMachineGrowth` with `UniverseMachineGrowth::um_saga(cm: Arc<dyn ConcentrationMassRelation>)`, implementing `StellarGrowthModel` and `DescribedPlugin`. Runfile selector `[stellar_growth] model = "universe_machine"`, `preset = "um_saga"`.

- [ ] **Step 1: Write the failing unit tests**

Create `rust/steel-plugins/src/growth_models/universe_machine.rs` with the test module only:

```rust
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cosmology::Planck15;
    use crate::harmonise::DuttonMaccio14;
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::sync::Arc;
    use steel_core::accretion::AccretionContext;
    use steel_core::cosmology::MassDefinition;
    use steel_core::halo_growth::GrowthTrack;

    fn model() -> UniverseMachineGrowth {
        UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14))
    }

    fn track() -> GrowthTrack {
        GrowthTrack {
            z: vec![0.0, 1.0, 2.0, 4.0],
            log_mass: vec![12.0, 11.6, 11.2, 10.6],
        }
    }

    #[test]
    fn quenched_fraction_is_a_fraction() {
        let m = model();
        for log_v in [1.4, 1.8, 2.2, 2.6, 3.0] {
            for z in [0.0, 1.0, 3.0] {
                let f = m.quenched_fraction(log_v, z);
                assert!((0.0..=1.0).contains(&f), "f_Q({log_v}, {z}) = {f}");
            }
        }
    }

    /// UM-SAGA's contribution is enhanced quenching at low mass, so f_Q
    /// must be non-monotonic: high at low velocity (the new term), low in
    /// the middle, high again at cluster scales (the DR1 term).
    #[test]
    fn quenched_fraction_is_elevated_at_both_extremes() {
        let m = model();
        let low = m.quenched_fraction(1.5, 0.0);
        let mid = m.quenched_fraction(2.1, 0.0);
        let high = m.quenched_fraction(2.9, 0.0);
        assert!(low > mid, "low-mass quenching: f_Q({low}) should exceed mid ({mid})");
        assert!(high > mid, "high-mass quenching: f_Q({high}) should exceed mid ({mid})");
    }

    #[test]
    fn star_forming_sfr_rises_then_falls_with_velocity() {
        let m = model();
        let a = m.log_sfr_star_forming(1.6, 0.0);
        let b = m.log_sfr_star_forming(2.2, 0.0);
        let c = m.log_sfr_star_forming(2.9, 0.0);
        assert!(b > a, "SFR should rise below the pivot: {a} then {b}");
        assert!(b > c, "SFR should fall above the pivot: {b} then {c}");
    }

    /// The rate is drawn from a bimodal PDF, so it is stochastic — but
    /// reproducible for a fixed seed.
    #[test]
    fn rate_is_stochastic_but_seed_reproducible() {
        let m = model();
        let c = Planck15::new();
        let t = track();
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let mut r1 = StdRng::seed_from_u64(7);
        let mut r2 = StdRng::seed_from_u64(7);
        let a = m.stellar_growth_rate(12.0, 0.5, &ctx, Some(&mut r1));
        let b = m.stellar_growth_rate(12.0, 0.5, &ctx, Some(&mut r2));
        assert_eq!(a.to_bits(), b.to_bits(), "same seed must give the same draw");
    }

    /// Without an RNG the model must still return something usable: the
    /// quenched-fraction-weighted mean rather than panicking or picking a
    /// mode arbitrarily.
    #[test]
    fn no_rng_gives_the_population_mean_rate() {
        let m = model();
        let c = Planck15::new();
        let t = track();
        let ctx = AccretionContext::central(&t, &c, MassDefinition::Vir);
        let mean = m.stellar_growth_rate(12.0, 0.5, &ctx, None);
        assert!(mean.is_finite(), "mean rate should be finite, got {mean}");
        // Must sit at or below the pure star-forming rate, since some of
        // the population is quenched.
        let log_v = m.log_vmpeak(12.0, 0.5, &ctx);
        assert!(mean <= m.log_sfr_star_forming(log_v, 0.5) + 1e-9);
    }

    /// UM's SFR encodes quenching, so its descriptor must claim the
    /// Quenching capability — that is what makes the validator reject
    /// pairing it with STEEL's QuenchingModel.
    #[test]
    fn descriptor_claims_quenching_to_prevent_double_counting() {
        let d = model().descriptor();
        assert!(d.provides.contains(&Capability::StellarMass));
        assert!(
            d.provides.contains(&Capability::Quenching),
            "UM's bimodal SFR PDF contains quenching; not declaring it would let a run \
             silently quench twice"
        );
    }
}
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd rust && cargo test -p steel-plugins universe_machine`
Expected: FAIL — `UniverseMachineGrowth` undefined.

- [ ] **Step 3: Implement `UniverseMachineGrowth`**

Prepend to `rust/steel-plugins/src/growth_models/universe_machine.rs`:

```rust
//! UniverseMachine: SFR assigned from halo properties and assembly
//! history, drawn from a bimodal star-forming / quenched distribution.
//!
//! Behroozi, Wechsler, Hearin & Conroy (2019) for the base model; Wang,
//! Nadler, Mao, Wechsler, Behroozi et al. (2024), arXiv:2404.14500, for
//! the UM-SAGA low-mass quenching extension implemented here.
//!
//! ```text
//! SFR_sf(v, z) proportional to 1 / [ (v/v_pivot)^-alpha + (v/v_pivot)^-beta ]
//! alpha(z) = alpha_0 + alpha_a (1-a) + alpha_la ln(1+z) + alpha_z z
//! f_Q(v, z) = DR1 high-mass term + UM-SAGA low-mass term
//! ```
//!
//! **Velocity-keyed, not mass-keyed.** UM is parametrised in peak
//! circular velocity while STEEL works in mass throughout, so every call
//! converts `log_mh -> vMpeak` through
//! `crate::harmonise::mpeak_to_vmax`, which needs a concentration-mass
//! relation. That relation is an injected assumption that materially
//! affects results, not an implementation detail: it is constructor-
//! supplied and recorded in the runfile (spec section 7). It must match
//! the `vmpeak_definition` recorded in the fixture provenance.
//!
//! **Quenching is internal.** The bimodal PDF already contains
//! quenching, so this plugin declares `Capability::Quenching` and the
//! composition validator will reject pairing it with STEEL's separate
//! `QuenchingModel`. Removing that declaration would let a run quench
//! twice with no error raised.
//!
//! Assembly-history dependence enters through the rank correlation with
//! `Delta vmax`, derived from the object's own growth track — exact for
//! satellites as well as centrals (spec section 5).

use std::sync::Arc;

use rand::RngCore;
use rand_distr::{Distribution, Normal};

use steel_core::accretion::AccretionContext;
use steel_core::compat::{
    Capability, CosmologyTag, DescribedPlugin, HConvention, Imf, PluginDescriptor,
};
use steel_core::cosmology::MassDefinition;
use steel_core::stellar_growth::StellarGrowthModel;

use crate::harmonise::{mpeak_to_vmax, ConcentrationMassRelation};

pub struct UniverseMachineGrowth {
    /// `alpha(z) = alpha[0] + alpha[1](1-a) + alpha[2] ln(1+z) + alpha[3] z`
    alpha: [f64; 4],
    /// `beta(z)`, same expansion.
    beta: [f64; 4],
    /// log10 pivot velocity `v_pivot(z) = v_pivot[0] + v_pivot[1](1-a)`.
    v_pivot: [f64; 2],
    /// Normalisation, log10 SFR at the pivot.
    log_sfr_norm: [f64; 2],
    /// DR1 high-mass quenching: `log10 v_Q(z) = v_q[0] + v_q[1](1-a)`,
    /// width `sigma_vq`.
    v_q: [f64; 2],
    sigma_vq: f64,
    /// UM-SAGA low-mass quenching:
    /// `log10 v_Q2(z) = v_q2[0] + v_q2[1](1-a) + v_q2[2] z`.
    v_q2: [f64; 3],
    /// `sigma_vQ2(z) = sigma_vq2[0] + sigma_vq2[1](1-a) + sigma_vq2[2] z`.
    sigma_vq2: [f64; 3],
    /// Floor on the quenched fraction.
    q_min: f64,
    /// Lognormal scatter in SFR within the star-forming mode \[dex\].
    sfr_scatter: f64,
    cm: Arc<dyn ConcentrationMassRelation>,
}

impl UniverseMachineGrowth {
    /// UM-SAGA best fit (Wang et al. 2024).
    ///
    /// **PROVISIONAL COEFFICIENTS.** Replace every value with one read
    /// from the upstream parameter file or the paper's posterior table,
    /// annotated with its source, before publishing any result. See spec
    /// section 6.1 and Task 10 Step 3. In particular the `beta`,
    /// `v_pivot`, `log_sfr_norm`, `v_q`, `sigma_vq`, and `q_min` values
    /// were **not** captured during design and must be sourced.
    pub fn um_saga(cm: Arc<dyn ConcentrationMassRelation>) -> Self {
        Self {
            alpha: [-6.14, -3.93, 6.37, -0.54], // SOURCE REQUIRED
            beta: [0.0, 0.0, 0.0, 0.0],         // SOURCE REQUIRED
            v_pivot: [2.2, 0.0],                // SOURCE REQUIRED
            log_sfr_norm: [0.5, 0.0],           // SOURCE REQUIRED
            v_q: [2.4, 0.0],                    // SOURCE REQUIRED
            sigma_vq: 0.2,                      // SOURCE REQUIRED
            v_q2: [1.66, -0.23, -0.63],         // SOURCE REQUIRED
            sigma_vq2: [0.14, 0.38, 0.22],      // SOURCE REQUIRED
            q_min: 0.0,                         // SOURCE REQUIRED
            sfr_scatter: 0.3,                   // SOURCE REQUIRED
            cm,
        }
    }

    fn scale_factor(z: f64) -> f64 {
        1.0 / (1.0 + z)
    }

    /// `c[0] + c[1](1-a) + c[2] ln(1+z) + c[3] z`
    fn expand4(c: [f64; 4], z: f64) -> f64 {
        let a = Self::scale_factor(z);
        c[0] + c[1] * (1.0 - a) + c[2] * (1.0 + z).ln() + c[3] * z
    }

    /// `c[0] + c[1](1-a)`
    fn expand2(c: [f64; 2], z: f64) -> f64 {
        c[0] + c[1] * (1.0 - Self::scale_factor(z))
    }

    /// log10 vMpeak \[km/s\] for a halo of `log_mh` \[log10 Msun/h\].
    ///
    /// Uses `ctx.log_m_peak` when the caller distinguishes peak from
    /// current mass, else `log_mh`.
    pub fn log_vmpeak(&self, log_mh: f64, z: f64, ctx: &AccretionContext<'_>) -> f64 {
        let m = ctx.log_m_peak.unwrap_or(log_mh);
        mpeak_to_vmax(m, z, ctx.cosmology, self.cm.as_ref(), ctx.mass_definition).log10()
    }

    /// log10 SFR \[Msun/yr\] for the star-forming mode.
    pub fn log_sfr_star_forming(&self, log_v: f64, z: f64) -> f64 {
        let alpha = Self::expand4(self.alpha, z);
        let beta = Self::expand4(self.beta, z);
        let x = log_v - Self::expand2(self.v_pivot, z);
        // Double power law in v, written in logs: the denominator is
        // 10^(-alpha x) + 10^(-beta x).
        let denom = 10f64.powf(-alpha * x) + 10f64.powf(-beta * x);
        Self::expand2(self.log_sfr_norm, z) - denom.log10()
    }

    /// Quenched fraction: the DR1 high-mass error function plus the
    /// UM-SAGA low-mass one. Clamped to \[0,1\] because the sum of two
    /// independent terms can exceed unity where both are active.
    pub fn quenched_fraction(&self, log_v: f64, z: f64) -> f64 {
        let hi = {
            let arg = (log_v - Self::expand2(self.v_q, z)) / (self.sigma_vq * 2f64.sqrt());
            0.5 + 0.5 * erf(arg)
        };
        let lo = {
            let a = Self::scale_factor(z);
            let v_q2 = self.v_q2[0] + self.v_q2[1] * (1.0 - a) + self.v_q2[2] * z;
            let s = (self.sigma_vq2[0] + self.sigma_vq2[1] * (1.0 - a) + self.sigma_vq2[2] * z)
                .max(1e-3);
            let arg = (log_v - v_q2) / (s * 2f64.sqrt());
            0.5 - 0.5 * erf(arg)
        };
        (self.q_min + (1.0 - self.q_min) * (hi + lo)).clamp(0.0, 1.0)
    }

    /// Rank correlation between SFR and the assembly-history proxy
    /// `Delta vmax`, from the object's own track: the fractional change
    /// in halo mass over the most recent track interval, as a stand-in
    /// for recent `vmax` growth.
    fn delta_vmax_proxy(ctx: &AccretionContext<'_>) -> f64 {
        let t = ctx.own_track;
        if t.log_mass.len() < 2 {
            return 0.0;
        }
        t.log_mass[0] - t.log_mass[1]
    }
}

/// Abramowitz & Stegun 7.1.26 error-function approximation, max absolute
/// error 1.5e-7 — well below the scatter this feeds into.
fn erf(x: f64) -> f64 {
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let y = 1.0
        - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t
            + 0.254829592)
            * t
            * (-x * x).exp();
    sign * y
}

impl StellarGrowthModel for UniverseMachineGrowth {
    fn stellar_growth_rate(
        &self,
        log_mh: f64,
        z: f64,
        ctx: &AccretionContext<'_>,
        rng: Option<&mut dyn RngCore>,
    ) -> f64 {
        let log_v = self.log_vmpeak(log_mh, z, ctx);
        let f_q = self.quenched_fraction(log_v, z);
        let log_sfr_sf = self.log_sfr_star_forming(log_v, z);

        match rng {
            // Stochastic: draw a mode, then scatter within it. Haloes
            // with faster recent growth are pushed towards star-forming,
            // which is UM's assembly-history correlation.
            Some(r) => {
                let boost = Self::delta_vmax_proxy(ctx).clamp(0.0, 0.5);
                let f_q_eff = (f_q - boost).clamp(0.0, 1.0);
                let u = Normal::new(0.0, 1.0)
                    .expect("unit normal is valid")
                    .sample(r)
                    .abs()
                    .min(1.0);
                if u < f_q_eff {
                    // Quenched mode: SFR suppressed far below the main
                    // sequence. UM keeps it finite rather than zero.
                    log_sfr_sf - 2.0
                } else if self.sfr_scatter > 0.0 && self.sfr_scatter.is_finite() {
                    let n = Normal::new(0.0, self.sfr_scatter).expect("checked finite positive");
                    log_sfr_sf + n.sample(r)
                } else {
                    log_sfr_sf
                }
            }
            // Deterministic: the population mean over both modes, so
            // callers without an RNG get a usable value rather than an
            // arbitrary mode. Averaged in linear SFR, then re-logged.
            None => {
                let sf = 10f64.powf(log_sfr_sf);
                let q = 10f64.powf(log_sfr_sf - 2.0);
                let mean = (1.0 - f_q) * sf + f_q * q;
                if mean <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    mean.log10()
                }
            }
        }
    }
}

impl DescribedPlugin for UniverseMachineGrowth {
    fn descriptor(&self) -> PluginDescriptor {
        PluginDescriptor {
            id: "universe_machine",
            // VERIFY against Behroozi+2019 section 3 before publishing.
            imf: Imf::Chabrier,
            mass_definition: MassDefinition::Vir,
            h_convention: HConvention::HFree,
            calibrated_cosmology: Some(CosmologyTag::Planck15),
            // Quenching is declared because the bimodal SFR PDF already
            // contains it. Dropping it here would let a run pair UM with
            // STEEL's QuenchingModel and quench twice, silently.
            provides: &[
                Capability::StellarMass,
                Capability::Quenching,
                Capability::Scatter,
                Capability::StarFormationRate,
            ],
        }
    }
}
```

Uncomment the `universe_machine` lines in `growth_models/mod.rs`, and add `pub use growth_models::UniverseMachineGrowth;` to `rust/steel-plugins/src/lib.rs`.

- [ ] **Step 4: Run to verify it passes**

Run: `cd rust && cargo test -p steel-plugins universe_machine`
Expected: PASS (7 tests).

Note the placeholder `beta = [0,0,0,0]` makes the double power law degenerate. If `star_forming_sfr_rises_then_falls_with_velocity` fails, that is the cause — it is the expected failure mode until Step 3 of Task 10 supplies real coefficients. Record it and proceed; the fixture test in Step 5 is the real gate.

- [ ] **Step 5: Add the fixture agreement test**

Append to `rust/steel-plugins/tests/upstream_agreement.rs`:

```rust
use std::sync::Arc;
use steel_plugins::harmonise::DuttonMaccio14;
use steel_plugins::UniverseMachineGrowth;

#[test]
fn um_star_forming_sfr_agrees_with_upstream() {
    let sfr_ref = load_um("sfr_sf_grid.npy");
    let m = UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14));
    let mut worst = 0.0_f64;
    let mut worst_at = (0.0, 0.0);

    for (i, log_v) in (0..41).map(|i| (i, 1.4 + i as f64 * 0.04)) {
        for (j, &z) in REDSHIFTS.iter().enumerate() {
            let d = (m.log_sfr_star_forming(log_v, z) - sfr_ref[[i, j]]).abs();
            if d > worst {
                worst = d;
                worst_at = (log_v, z);
            }
        }
    }

    println!("worst UM SFR deviation {worst:.6} dex at log_v={} z={}", worst_at.0, worst_at.1);
    assert!(
        worst < 0.01,
        "worst UM SFR deviation {worst:.6} dex at log_v={} z={} exceeds 0.01",
        worst_at.0,
        worst_at.1
    );
}

#[test]
fn um_quenched_fraction_agrees_with_upstream() {
    let f_ref = load_um("quenched_fraction_grid.npy");
    let m = UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14));
    let mut worst = 0.0_f64;

    for (i, log_v) in (0..41).map(|i| (i, 1.4 + i as f64 * 0.04)) {
        for (j, &z) in REDSHIFTS.iter().enumerate() {
            // Absolute, not dex: f_Q is a fraction and can be zero.
            worst = worst.max((m.quenched_fraction(log_v, z) - f_ref[[i, j]]).abs());
        }
    }

    println!("worst UM f_Q deviation {worst:.6} (absolute)");
    assert!(worst < 0.02, "worst f_Q deviation {worst:.6} exceeds 0.02");
}
```

- [ ] **Step 6: Run and record achieved agreement**

Run: `cd rust && cargo test -p steel-plugins --test upstream_agreement -- --nocapture`
Expected: PASS, printing worst deviations.

**If either fails**, check in order: (a) unverified coefficients; (b) `vmpeak_definition` in `provenance.toml` versus what `mpeak_to_vmax` computes — a peak-over-history `vMpeak` is *not* `Vmax` at peak mass, and conflating them is the most likely failure; (c) `h` convention; (d) the concentration-mass relation.

Add the achieved figures to `docs/VALIDATION.md`.

- [ ] **Step 7: Wire into the registry and prove the double-quenching rejection**

In `registry.rs`, extend `build_stellar_growth`:

```rust
        ("universe_machine", "um_saga") => {
            let cm: Arc<dyn ConcentrationMassRelation> = match cfg.concentration.as_deref() {
                None | Some("dutton_maccio14") => Arc::new(DuttonMaccio14),
                Some(other) => return Err(anyhow!("unknown concentration relation: {other}")),
            };
            let m = UniverseMachineGrowth::um_saga(cm);
            let d = m.descriptor();
            (Box::new(m) as Box<dyn StellarGrowthModel>, d)
        }
```

Add `pub concentration: Option<String>` to `StellarGrowthConfig` in `runfile.rs`.

Add to `rust/steel-plugins/tests/composition_rejection.rs`:

```rust
#[test]
fn real_um_descriptor_conflicts_with_steel_quenching() {
    use std::sync::Arc;
    use steel_core::compat::DescribedPlugin;
    use steel_plugins::harmonise::DuttonMaccio14;
    use steel_plugins::UniverseMachineGrowth;

    let um = UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14)).descriptor();
    let wetzel = PluginDescriptor {
        id: "wetzel13",
        imf: Imf::Chabrier,
        mass_definition: MassDefinition::Vir,
        h_convention: HConvention::HFree,
        calibrated_cosmology: Some(CosmologyTag::Planck15),
        provides: &[Capability::Quenching],
    };
    validate_composition(&[um, wetzel], CosmologyTag::Planck15)
        .expect_err("UM plus a separate quenching model must be rejected");
}
```

- [ ] **Step 8: Run the full suite**

Run: `cd rust && cargo test --workspace`
Expected: PASS, including the Task 1 bit-identity guards.

- [ ] **Step 9: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/steel-plugins/src/growth_models/ rust/steel-plugins/src/lib.rs \
        rust/steel-plugins/tests/upstream_agreement.rs \
        rust/steel-plugins/tests/composition_rejection.rs \
        rust/steel-io/src/runfile.rs rust/steel-cli/src/registry.rs \
        docs/VALIDATION.md
git commit -m "feat: add UniverseMachine (UM-SAGA) as a stellar growth plugin

Bimodal SFR PDF keyed on vMpeak with UM-SAGA's low-mass quenching term,
converted from STEEL's mass axis via a selectable concentration-mass
relation. Declares Capability::Quenching so the composition validator
rejects pairing it with STEEL's QuenchingModel, which would otherwise
quench twice silently.

Coefficients remain PROVISIONAL pending verification; several were not
captured during design at all. See spec section 6.1."
```

---

### Task 12: Assumptions catalogue

Spec §9.

**Files:**
- Create: `docs/model-assumptions.md`

**Interfaces:**
- Consumes: the descriptors from Tasks 7, 9, 11; the `[conventions]` blocks of both `provenance.toml` files.
- Produces: `docs/model-assumptions.md`, referenced by the registry's error message (Task 7 Step 6).

- [ ] **Step 1: Write the catalogue**

Create `docs/model-assumptions.md`:

```markdown
# Model assumptions: STEEL, EMERGE, UniverseMachine

What each model assumes, per `rs-steel` trait, so cross-model comparisons
are like-for-like. Referenced by the composition validator's error output.

Classification: **identical** (no action) / **convertible** (handled by
`steel_plugins::harmonise`) / **needs plugin** (candidate follow-up spec)
/ **structurally absent** (no analogue; a documented asymmetry).

| Trait | STEEL | EMERGE | UniverseMachine | Status |
|---|---|---|---|---|
| `Cosmology` | Planck15 | <verify> | <verify> | convertible |
| `HaloGrowthModel` | van den Bosch 2014 average MAH | N-body trees | N-body trees | needs plugin |
| `HaloMassFunctionModel` | Despali 2016 | <verify> | implicit in the simulation | needs plugin |
| `SubhaloMassFunctionModel` | Jiang & van den Bosch 2016 | resolved subhaloes | resolved subhaloes | structurally absent upstream |
| `MergerTimescaleModel` | Boylan-Kolchin 2008 / McCavana 2012 | tracked directly | tracked directly | structurally absent upstream |
| `HaloStrippingModel` | van den Bosch 2005 / Cattaneo 2011 | from the tree | from the tree | structurally absent upstream |
| `SmhmModel` | Moster form, G19_SE default | not used (rate-based) | not used (rate-based) | n/a |
| `StellarGrowthModel` | not used | eps x baryonic accretion | SFR from halo properties | identical interface |
| `SfrModel` | Tomczak/Schreiber forms, M*-keyed | implied by the rate | bimodal PDF, v-keyed | needs plugin |
| `QuenchingModel` | Wetzel 2013 timescales | reionization gate (early growth, not satellite) | internal to the SFR PDF | conflict — see below |
| `GasMassModel` | Stewart scaling | <verify> | <verify> | needs plugin |
| `StellarStrippingModel` | STEEL prescription | <verify> | <verify> | needs plugin |

## Unit and definition conventions

| | STEEL | EMERGE | UniverseMachine |
|---|---|---|---|
| Halo mass definition | `Vir` (Bryan & Norman 98) | <from provenance.toml> | <from provenance.toml> |
| `h` convention | `Msun/h` internally | <from provenance.toml> | <from provenance.toml> |
| IMF | Chabrier | <from provenance.toml> | <from provenance.toml> |
| Primary halo key | mass | mass | **peak velocity** |

The velocity key is the one genuinely unavoidable conversion: see
`steel_plugins::harmonise::mpeak_to_vmax`. The concentration-mass
relation it needs (default Dutton & Maccio 2014) is a modelling choice
that affects UM's results and is recorded per run.

## Known limitations

1. **Subhalo histories are average MAHs, not individual trees.** STEEL is
   statistical by design: a satellite's pre-infall history comes from
   `growth_history(m_infall, z_infall)`, the same average-MAH
   approximation already used for hosts. Both upstreams resolve
   individual subhaloes. This is the principal asymmetry in any
   comparison, and it is STEEL's pre-existing limitation rather than one
   introduced here.
2. **Concentration-mass relation on the UM path.** Any UM result inherits
   a dependence on the chosen c(M,z).
3. **Quenching is not separable in UM.** UM's bimodal SFR PDF contains
   quenching, so UM cannot be combined with STEEL's `QuenchingModel`; the
   validator rejects it. A comparison of STEEL-plus-Wetzel13 against UM is
   therefore comparing two quenching treatments as well as two SFR
   treatments. EMERGE differs: its reionization gate suppresses early
   low-mass growth and is *not* a satellite quenching prescription, so
   EMERGE and Wetzel13 remain compatible.
4. **Parameter provenance.** Every coefficient must be traceable to a
   paper table or upstream parameter file. Values sourced only from
   automated summaries are not acceptable.

## Recommended follow-up plugins, by comparison impact

1. **`HaloMassFunctionModel` parity** — sets the abundance normalisation;
   the largest single lever on an SMHM overlay.
2. **`SfrModel` parity (v-keyed)** — lets STEEL's own SFR be evaluated on
   UM's velocity key, separating the key from the prescription.
3. **`HaloGrowthModel` parity** — an average-MAH versus tree-resolved
   comparison bounds limitation 1 quantitatively.
4. **`GasMassModel` / `StellarStrippingModel` parity** — smaller effects;
   worth doing only after the above.
```

- [ ] **Step 2: Fill in every `<verify>` and `<from provenance.toml>`**

Read the values from the two committed `provenance.toml` files and the papers. The doc must contain no angle brackets when done.

- [ ] **Step 3: Verify no placeholders remain**

Run: `grep -n '<' docs/model-assumptions.md`
Expected: no output.

- [ ] **Step 4: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add docs/model-assumptions.md
git commit -m "docs: catalogue STEEL/EMERGE/UM assumptions per trait

Per-trait gap table, unit conventions, and four known limitations,
including that UM's quenching is inseparable from its SFR while EMERGE's
reionization gate is not a quenching prescription. Ranks follow-up parity
plugins by comparison impact. Spec section 9."
```

---

### Task 13: End-to-end self-consistency validation

Spec §10, closing against `paper/main.tex:177-183`'s three criteria.

**Files:**
- Create: `rust/runfiles/emerge_o_leary23.toml`, `rust/runfiles/um_saga.toml`
- Create: `rust/steel-plugins/tests/baryon_budget.rs`
- Modify: `docs/VALIDATION.md`

**Interfaces:**
- Consumes: everything above.
- Produces: two committed runfiles and a validation section reporting the three self-consistency checks for each model.

- [ ] **Step 1: Write the two runfiles**

Copy an existing runfile from `rust/runfiles/` as the base so every unrelated key matches the established configuration, then change only the model selection. `rust/runfiles/emerge_o_leary23.toml`:

```toml
# EMERGE (O'Leary et al. 2023) through STEEL's accretion machinery.
# Base configuration copied from the G19_SE reference runfile; only the
# stellar-mass prescription differs, so any difference in output is
# attributable to the model rather than the setup.
#
# No [smhm] section: EMERGE supplies Capability::StellarMass, and having
# both would be rejected by the composition validator.

[stellar_growth]
model = "emerge"
preset = "o_leary23"

# [quenching] is retained: EMERGE's reionization gate suppresses early
# low-mass growth and is not a satellite quenching prescription, so this
# is not double-counting. See docs/model-assumptions.md.
```

`rust/runfiles/um_saga.toml`:

```toml
# UniverseMachine UM-SAGA (Wang et al. 2024) through STEEL's accretion
# machinery. Base configuration copied from the G19_SE reference runfile.
#
# No [smhm] and NO [quenching]: UM's bimodal SFR PDF already contains
# quenching, so adding STEEL's QuenchingModel would quench twice. The
# composition validator rejects it; this comment records why the section
# is absent rather than overlooked.

[stellar_growth]
model = "universe_machine"
preset = "um_saga"
concentration = "dutton_maccio14"
```

- [ ] **Step 2: Confirm the validator rejects a deliberately broken runfile**

Create a scratch copy of `um_saga.toml` with a `[quenching]` section added, and run it.

Run: `cd rust && cargo run --release -p steel-cli -- <scratch runfile>`
Expected: **exits non-zero before any computation**, printing a message naming `universe_machine`, `wetzel13`, and the duplicated effect, plus the pointer to `docs/model-assumptions.md`.

If it runs to completion, the validator is not wired into the startup path — fix that before proceeding. This is the single most important behaviour in the plan.

- [ ] **Step 3: Run both models end to end**

Run:
```bash
cd rust
cargo run --release -p steel-cli -- runfiles/emerge_o_leary23.toml
cargo run --release -p steel-cli -- runfiles/um_saga.toml
```
Expected: both complete and write output directories.

- [ ] **Step 4: Check the three self-consistency criteria**

For each model, using the existing postprocessing scripts in `Scripts/Validation/`:

1. **Satellite counts and pair fractions** — compare against the STEEL G19_SE baseline. These need not *agree*; they must be finite, smooth in mass and redshift, and free of discontinuities that would indicate a plumbing error rather than a physics difference.
2. **Central mass accretion plausibility** — integrated M* must never exceed `f_b . M_h`. Add `rust/steel-plugins/tests/baryon_budget.rs`:

```rust
//! No model may convert more than the available baryons into stars.
//! Applies to both rate-based models over the full mass and redshift
//! range STEEL runs on. Spec section 10.

use std::sync::Arc;

use steel_core::accretion::AccretionContext;
use steel_core::cosmology::{Cosmology, MassDefinition};
use steel_core::halo_growth::HaloGrowthModel;
use steel_core::{integrate_stellar_mass, StellarGrowthModel};
use steel_plugins::harmonise::DuttonMaccio14;
use steel_plugins::{EmergeGrowth, Planck15, UniverseMachineGrowth, VandenBosch14};

fn assert_within_baryon_budget(model: &dyn StellarGrowthModel, label: &str) {
    let cosmo = Planck15::new();
    let growth = VandenBosch14::new(&cosmo);
    let f_b = cosmo.omega_b0() / cosmo.omega_m0();

    for i in 0..=50 {
        let log_mh = 10.0 + i as f64 * 0.1;
        for &z_end in &[0.1, 0.5, 1.0, 2.0, 4.0] {
            let track = growth.growth_history(log_mh, z_end);
            let ctx = AccretionContext::central(&track, &cosmo, MassDefinition::Vir);
            let log_sm = integrate_stellar_mass(model, &ctx, z_end, None);
            if !log_sm.is_finite() {
                continue; // no elapsed time on this track segment
            }
            let ceiling = log_mh + f_b.log10();
            assert!(
                log_sm <= ceiling,
                "{label}: M*={log_sm:.3} exceeds baryon budget {ceiling:.3} \
                 at log_mh={log_mh} z={z_end}"
            );
        }
    }
}

#[test]
fn emerge_respects_the_baryon_budget() {
    assert_within_baryon_budget(&EmergeGrowth::o_leary23(), "emerge");
}

#[test]
fn universe_machine_respects_the_baryon_budget() {
    assert_within_baryon_budget(
        &UniverseMachineGrowth::um_saga(Arc::new(DuttonMaccio14)),
        "universe_machine",
    );
}
```

Run: `cd rust && cargo test -p steel-plugins --test baryon_budget`
Expected: PASS. A failure means either the coefficients are wrong or the integrator is double-counting a time interval — diagnose, do not relax the bound.
3. **SFR consistent with the driving accretion history** — the sSFR check of `Grylls2020`. Regenerate the sSFR self-consistency figure for each model.

- [ ] **Step 5: Record results in `docs/VALIDATION.md`**

Add an "External model agreement" section containing, for each model: the upstream fixture agreement in dex from Tasks 9 and 11; the three self-consistency outcomes; and any criterion not met, stated plainly rather than omitted.

Per `paper/main.tex:186-189`, agreement with observational data is explicitly **not** the bar. Do not present observational comparisons as validation.

- [ ] **Step 6: Commit**

```bash
cd /Users/pgrylls/Code/STEEL
git add rust/runfiles/emerge_o_leary23.toml rust/runfiles/um_saga.toml \
        rust/steel-plugins/tests/baryon_budget.rs docs/VALIDATION.md
git commit -m "test: run EMERGE and UM end to end with self-consistency checks

Two runfiles differing from the G19_SE reference only in the stellar-mass
prescription, so output differences are attributable to the model. Records
upstream fixture agreement and the three self-consistency criteria from
the paper's motivation section for each."
```

---

## Notes for the executor

**Order matters.** Tasks 1-2 establish that the trait widening is inert; do not begin Task 5 or later before the Task 1 bit-identity tests pass post-widening. Task 3 gates Task 4. Tasks 8 and 10 gate 9 and 11 respectively.

**Parallelism.** After Task 7, the EMERGE chain (8, 9) and the UM chain (10, 11) are independent and can run concurrently. Task 12 needs both.

**Stop and report rather than working around**, in these cases:
- Task 3's `z0 != 0` tests fail — the satellite design rests on that path.
- An upstream build fails or a pinned SHA does not match.
- A fixture agreement test cannot be met without widening its bound.
- The Task 1 bit-identity guards break at any point.
- Task 13 Step 2 does not reject the broken runfile.

**The provisional-coefficient issue is the largest correctness risk.** Both `o_leary23()` and `um_saga()` ship with values marked `SOURCE REQUIRED`, and several UM parameters were never captured. Fixture agreement tests will fail until real values are supplied — that is the intended mechanism, not an obstacle to route around.
