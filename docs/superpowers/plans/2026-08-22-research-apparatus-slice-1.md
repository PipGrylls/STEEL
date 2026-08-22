# Research Apparatus Slice 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce the ICL-ceiling → maximum-stripping-strength result through the research harness, with the M500↔Mvir conversion actually performed instead of flagged as a caveat.

**Architecture:** A Rust `steel-harmonise-cli` exposes the existing `harmonise.rs` conversions (plus a new NFW overdensity mass converter) as JSON-in/JSON-out. A thin Python kernel shells out to it for every numeric conversion, so no physics is duplicated. Findings live in MongoDB behind a `research-store` MCP server that is the sole write path and enforces the spec's gates server-side. Two agents (`data-curator`, `referee`) and one derivation close the loop.

**Tech Stack:** Rust (existing workspace), Python 3.11+ via a `research/.venv`, MongoDB 7 via docker-compose, `pymongo`, `mcp` (Python SDK), `pytest`, `colossus` (test-only, for independent validation).

**Spec:** `docs/superpowers/specs/2026-08-22-research-apparatus-design.md`

## Global Constraints

- STEEL's internal halo-mass convention is **log10 Msun/h**; `SmhmModel::stellar_mass` and `harmonise` conversions taking "h-free" want **log10 Msun**. Never pass one where the other is expected.
- `Cosmology::m_to_r(m, z, mdef)` takes **m in Msun/h** and returns **kpc/h**. `Cosmology::rho_crit(z)` is **Msun h²/kpc³**.
- Every numeric unit/definition conversion goes through `steel-harmonise-cli`. Python must not reimplement a conversion formula.
- The MCP server is the **only** write path to Mongo. No Python module outside `research/mcp/server.py` may open a `pymongo` connection for writes.
- `definition` fields are mandatory; unknown values must be the literal string `"unknown"`, which blocks comparison.
- `extraction` is one of `table | figure | text | abstract`.
- Existing Rust tests must stay green: `cargo test --workspace` (183 passing as of `e7ae3df`).
- Do not modify existing physics behaviour. `harmonise.rs` gains new functions; existing ones keep their current numerics.

---

## File Structure

| File | Responsibility |
|---|---|
| `rust/steel-plugins/src/harmonise.rs` (modify) | Pin the concentration h-convention; add `convert_mass_definition` |
| `rust/steel-harmonise-cli/Cargo.toml` (create) | New binary crate manifest |
| `rust/steel-harmonise-cli/src/main.rs` (create) | JSON stdin → conversion → JSON stdout |
| `rust/Cargo.toml` (modify) | Add workspace member and `serde_json` |
| `docker-compose.yml` (create) | MongoDB 7 service |
| `research/pyproject.toml` (create) | Python package + deps |
| `research/kernel/definitions.py` (create) | `Definition`, compatibility rules |
| `research/kernel/convert.py` (create) | Subprocess bridge to `steel-harmonise-cli` |
| `research/kernel/store.py` (create) | Mongo schema, gates, query/put — imported only by the MCP server |
| `research/mcp/server.py` (create) | MCP server exposing `store.query`/`store.put`/`store.verify` |
| `research/derivations/icl_stripping_bound.py` (create) | The Slice 1 derivation |
| `research/tests/` (create) | pytest suite |
| `.claude/agents/data-curator.md`, `.claude/agents/referee.md` (create) | Agent roles |
| `.mcp.json` (modify) | Register `research-store` |

---

### Task 1: Pin the concentration mass convention

`ConcentrationMassRelation`'s trait doc says `log_mh [log10 Msun]` but `DuttonMaccio14`'s body comment says the fit is in `1e12 h^-1 Msun`, and sibling `mpeak_to_vmax` documents `Msun/h`. The convention is ambiguous inside the module that exists to prevent exactly this bug. Resolve it before anything builds on it.

**Files:**
- Modify: `rust/steel-plugins/src/harmonise.rs:91-94` (trait doc), `:117-118` (impl comment)
- Test: `rust/steel-plugins/src/harmonise.rs` (in-file `#[cfg(test)] mod tests`)

**Interfaces:**
- Consumes: nothing
- Produces: `ConcentrationMassRelation::concentration(&self, log_mh: f64, z: f64) -> f64` with `log_mh` documented and tested as **log10 Msun/h**

- [ ] **Step 1: Write the failing test**

Dutton & Maccio (2014) eq. 7 gives `log10 c_vir = a + b (log10 M_vir / [10^12 h^-1 Msun])` with `a = 0.537 + 0.488 exp(-0.718 z^1.08)`, `b = -0.097 + 0.024 z`. At `z = 0` and `M_vir = 10^12 h^-1 Msun` the bracket is zero, so `log10 c = a = 1.025`, i.e. `c = 10^1.025`.

Add to `rust/steel-plugins/src/harmonise.rs` tests module:

```rust
#[test]
fn dutton_maccio_pivot_is_defined_at_1e12_msun_per_h() {
    // D&M14 eq. 7 pivot: at z=0 and M_vir = 1e12 h^-1 Msun the mass term
    // vanishes, leaving log10 c = a(0) = 0.537 + 0.488 = 1.025. This test
    // pins the argument convention as log10 Msun/h -- passing an h-free
    // mass here would silently shift c.
    let c = DuttonMaccio14.concentration(12.0, 0.0);
    assert!((c - 10f64.powf(1.025)).abs() < 1e-9, "c = {c}");
}

#[test]
fn concentration_falls_with_mass_and_redshift() {
    let c_low = DuttonMaccio14.concentration(11.0, 0.0);
    let c_high = DuttonMaccio14.concentration(14.0, 0.0);
    let c_z1 = DuttonMaccio14.concentration(12.0, 1.0);
    assert!(c_low > c_high, "concentration must fall with mass");
    assert!(c_z1 < DuttonMaccio14.concentration(12.0, 0.0));
}
```

- [ ] **Step 2: Run the tests**

Run: `cd rust && cargo test -p steel-plugins harmonise`
Expected: both PASS (they document existing behaviour). If `dutton_maccio_pivot_is_defined_at_1e12_msun_per_h` fails, the implementation disagrees with D&M14 — stop and report rather than editing the formula.

- [ ] **Step 3: Correct the trait documentation**

In `rust/steel-plugins/src/harmonise.rs`, change the trait method doc from `log10 Msun` to:

```rust
    /// NFW concentration c = R_delta / r_s for `log_mh`
    /// \[log10 **Msun/h**\], virial mass definition.
    ///
    /// The h-convention is load-bearing: `DuttonMaccio14`'s fit is
    /// pivoted at 1e12 h^-1 Msun, so passing an h-free mass shifts the
    /// concentration by `b * log10(h)`. Pinned by
    /// `dutton_maccio_pivot_is_defined_at_1e12_msun_per_h`.
    fn concentration(&self, log_mh: f64, z: f64) -> f64;
```

- [ ] **Step 4: Verify the workspace is still green**

Run: `cd rust && cargo test --workspace 2>&1 | grep -E "test result|FAILED"`
Expected: no `FAILED`; totals unchanged except +2 in `steel-plugins`.

- [ ] **Step 5: Commit**

```bash
git add rust/steel-plugins/src/harmonise.rs
git commit -m "Pin ConcentrationMassRelation's mass argument to Msun/h

The trait doc said log10 Msun while DuttonMaccio14's fit is pivoted at
1e12 h^-1 Msun. Two tests now pin the convention at the D&M14 pivot."
```

---

### Task 2: NFW overdensity mass conversion

The capability gap. `MassDefinition`, `m_to_r`, `rho_crit` and `DuttonMaccio14` all exist; nothing composes them. Because the concentration relation is **virial-calibrated**, conversions anchor on virial: `from → Mvir → to`.

**Files:**
- Modify: `rust/steel-plugins/src/harmonise.rs` (append)
- Test: same file, tests module

**Interfaces:**
- Consumes: `ConcentrationMassRelation::concentration(log_mh: f64, z: f64) -> f64` (Task 1)
- Produces: `pub fn convert_mass_definition(log_m_from: f64, from: MassDefinition, to: MassDefinition, z: f64, cosmology: &dyn Cosmology, concentration: &dyn ConcentrationMassRelation) -> f64` — takes and returns **log10 Msun/h**

- [ ] **Step 1: Write the failing tests**

```rust
#[test]
fn converting_to_the_same_definition_is_the_identity() {
    let c = crate::cosmology::Planck15::new();
    let got = convert_mass_definition(
        14.0, MassDefinition::Vir, MassDefinition::Vir, 0.1, &c, &DuttonMaccio14);
    assert!((got - 14.0).abs() < 1e-6, "got {got}");
}

#[test]
fn virial_mass_exceeds_m500c() {
    // Delta_vir(z~0) ~ 100x critical, well below 500x, so the virial
    // radius encloses more mass than r500c.
    let c = crate::cosmology::Planck15::new();
    let log_mvir = convert_mass_definition(
        14.0, MassDefinition::Critical(500.0), MassDefinition::Vir, 0.1, &c, &DuttonMaccio14);
    assert!(log_mvir > 14.0, "Mvir {log_mvir} should exceed M500c 14.0");
    assert!(log_mvir < 14.5, "conversion should be a modest shift, got {log_mvir}");
}

#[test]
fn mass_definition_conversion_round_trips() {
    let c = crate::cosmology::Planck15::new();
    let fwd = convert_mass_definition(
        14.0, MassDefinition::Critical(500.0), MassDefinition::Vir, 0.3, &c, &DuttonMaccio14);
    let back = convert_mass_definition(
        fwd, MassDefinition::Vir, MassDefinition::Critical(500.0), 0.3, &c, &DuttonMaccio14);
    assert!((back - 14.0).abs() < 1e-4, "round trip gave {back}");
}

#[test]
fn m200m_exceeds_m200c() {
    // Mean-density overdensities enclose more mass than critical ones at
    // the same Delta, since rho_mean < rho_crit.
    let c = crate::cosmology::Planck15::new();
    let m200m = convert_mass_definition(
        14.0, MassDefinition::Vir, MassDefinition::Mean(200.0), 0.0, &c, &DuttonMaccio14);
    let m200c = convert_mass_definition(
        14.0, MassDefinition::Vir, MassDefinition::Critical(200.0), 0.0, &c, &DuttonMaccio14);
    assert!(m200m > m200c, "M200m {m200m} should exceed M200c {m200c}");
}
```

- [ ] **Step 2: Run to verify they fail**

Run: `cd rust && cargo test -p steel-plugins convert_mass`
Expected: FAIL — `cannot find function convert_mass_definition`

- [ ] **Step 3: Implement**

Append to `rust/steel-plugins/src/harmonise.rs`:

```rust
use std::f64::consts::PI;

/// NFW characteristic mass function, `mu(x) = ln(1+x) - x/(1+x)`.
///
/// The enclosed mass of an NFW halo is `M(<r) = 4 pi rho_s r_s^3 mu(r/r_s)`,
/// so mass ratios between radii reduce to ratios of `mu`.
fn nfw_mu(x: f64) -> f64 {
    (1.0 + x).ln() - x / (1.0 + x)
}

/// Overdensity threshold for `mdef`, relative to the critical density —
/// the same convention `Cosmology::m_to_r` uses.
fn delta_wrt_critical(mdef: MassDefinition, z: f64, cosmology: &dyn Cosmology) -> f64 {
    match mdef {
        MassDefinition::Vir => cosmology.delta_vir(z),
        MassDefinition::Critical(d) => d,
        MassDefinition::Mean(d) => d * cosmology.omega_m(z),
    }
}

/// Mass \[Msun/h\] enclosed by radius `r` \[kpc/h\] under definition
/// `mdef` — the exact inverse of [`Cosmology::m_to_r`].
fn mass_from_radius(r: f64, z: f64, mdef: MassDefinition, cosmology: &dyn Cosmology) -> f64 {
    let delta = delta_wrt_critical(mdef, z, cosmology);
    (4.0 / 3.0) * PI * r.powi(3) * cosmology.rho_crit(z) * delta
}

/// Mass at definition `mdef` implied by the NFW halo whose virial mass is
/// `m_vir` \[Msun/h\].
///
/// Solves for the radius where the profile's enclosed mass equals the
/// mass that `mdef` itself assigns to that radius. Both sides are
/// continuous and cross exactly once for physical concentrations, so
/// bisection is safe.
fn implied_mass_at(
    m_vir: f64,
    mdef: MassDefinition,
    z: f64,
    cosmology: &dyn Cosmology,
    concentration: &dyn ConcentrationMassRelation,
) -> f64 {
    let c_vir = concentration.concentration(m_vir.log10(), z);
    let r_vir = cosmology.m_to_r(m_vir, z, MassDefinition::Vir);
    let r_s = r_vir / c_vir;
    let mu_c = nfw_mu(c_vir);

    // f(r) > 0 while the profile encloses more than the definition
    // demands. At tiny r the profile term dominates; at large r the r^3
    // term does. Bracket generously around r_vir.
    let f = |r: f64| m_vir * nfw_mu(r / r_s) / mu_c - mass_from_radius(r, z, mdef, cosmology);

    let (mut lo, mut hi) = (1.0e-4 * r_vir, 1.0e2 * r_vir);
    debug_assert!(f(lo) > 0.0 && f(hi) < 0.0, "root not bracketed for {mdef:?}");
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let r = 0.5 * (lo + hi);
    mass_from_radius(r, z, mdef, cosmology)
}

/// Convert a spherical-overdensity halo mass between definitions under an
/// NFW profile, in log10 Msun/h.
///
/// Conversions anchor on the virial definition because
/// [`ConcentrationMassRelation`] is virial-calibrated: `from` is first
/// inverted to a virial mass, then the profile is re-evaluated at `to`.
/// The inversion is a bisection on virial mass, since `implied_mass_at`
/// increases monotonically with it.
pub fn convert_mass_definition(
    log_m_from: f64,
    from: MassDefinition,
    to: MassDefinition,
    z: f64,
    cosmology: &dyn Cosmology,
    concentration: &dyn ConcentrationMassRelation,
) -> f64 {
    if from == to {
        return log_m_from;
    }
    let m_from = 10f64.powf(log_m_from);

    // Invert `from` to a virial mass, unless it already is one.
    let m_vir = if from == MassDefinition::Vir {
        m_from
    } else {
        let (mut lo, mut hi) = (log_m_from - 3.0, log_m_from + 3.0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            let implied =
                implied_mass_at(10f64.powf(mid), from, z, cosmology, concentration);
            if implied < m_from {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        10f64.powf(0.5 * (lo + hi))
    };

    if to == MassDefinition::Vir {
        return m_vir.log10();
    }
    implied_mass_at(m_vir, to, z, cosmology, concentration).log10()
}
```

- [ ] **Step 4: Run the tests**

Run: `cd rust && cargo test -p steel-plugins convert_mass`
Expected: all four PASS

- [ ] **Step 5: Verify the workspace is green**

Run: `cd rust && cargo test --workspace 2>&1 | grep -E "test result|FAILED"`
Expected: no `FAILED`

- [ ] **Step 6: Commit**

```bash
git add rust/steel-plugins/src/harmonise.rs
git commit -m "Add NFW spherical-overdensity mass conversion

Composes m_to_r, rho_crit and DuttonMaccio14 into convert_mass_definition,
anchoring on the virial definition because the concentration relation is
virial-calibrated. Retires the M500-vs-Mvir caveat."
```

---

### Task 3: `steel-harmonise-cli`

**Files:**
- Create: `rust/steel-harmonise-cli/Cargo.toml`, `rust/steel-harmonise-cli/src/main.rs`
- Modify: `rust/Cargo.toml`
- Test: `rust/steel-harmonise-cli/tests/cli.rs`

**Interfaces:**
- Consumes: `convert_mass_definition` (Task 2), `HConvention::{to_h_free, from_h_free}`, `Imf::log_offset_to`
- Produces: a binary reading one JSON object on stdin and writing one on stdout. Ops: `convert_mass`, `convert_stellar`.

- [ ] **Step 1: Add the workspace member and `serde_json`**

In `rust/Cargo.toml`, add `"steel-harmonise-cli",` to `members`, and under `[workspace.dependencies]` add:

```toml
serde_json = "1"
```

- [ ] **Step 2: Create the crate manifest**

`rust/steel-harmonise-cli/Cargo.toml`:

```toml
[package]
name = "steel-harmonise-cli"
version.workspace = true
edition.workspace = true
authors.workspace = true
license.workspace = true

[dependencies]
steel-core = { path = "../steel-core" }
steel-plugins = { path = "../steel-plugins" }
serde = { workspace = true }
serde_json = { workspace = true }
anyhow = { workspace = true }
```

- [ ] **Step 3: Write the failing integration test**

`rust/steel-harmonise-cli/tests/cli.rs`:

```rust
use std::io::Write;
use std::process::{Command, Stdio};

fn run(input: &str) -> serde_json::Value {
    let mut child = Command::new(env!("CARGO_BIN_EXE_steel-harmonise-cli"))
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .expect("spawn cli");
    child.stdin.as_mut().unwrap().write_all(input.as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(out.status.success(), "cli failed: {}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).expect("cli emitted valid json")
}

#[test]
fn converts_m500c_to_virial_and_reports_the_path() {
    let v = run(r#"{"op":"convert_mass","log_m":14.0,
        "from":{"mass_def":"M500c","h_convention":"h_free"},
        "to":{"mass_def":"Mvir","h_convention":"per_h"},"z":0.1}"#);
    let log_m = v["log_m"].as_f64().unwrap();
    assert!(log_m > 14.0 && log_m < 15.0, "got {log_m}");
    assert!(v["path"].as_array().unwrap().len() >= 2, "path must record each step");
}

#[test]
fn rejects_an_unknown_mass_definition() {
    let mut child = Command::new(env!("CARGO_BIN_EXE_steel-harmonise-cli"))
        .stdin(Stdio::piped()).stdout(Stdio::piped()).stderr(Stdio::piped())
        .spawn().unwrap();
    child.stdin.as_mut().unwrap()
        .write_all(br#"{"op":"convert_mass","log_m":14.0,
            "from":{"mass_def":"unknown","h_convention":"h_free"},
            "to":{"mass_def":"Mvir","h_convention":"per_h"},"z":0.1}"#).unwrap();
    let out = child.wait_with_output().unwrap();
    assert!(!out.status.success(), "unknown definition must be an error, not a guess");
}

#[test]
fn converts_stellar_mass_between_imfs() {
    let v = run(r#"{"op":"convert_stellar","log_m":10.0,
        "from":{"imf":"salpeter","h_convention":"h_free"},
        "to":{"imf":"chabrier","h_convention":"h_free"}}"#);
    // Salpeter masses sit ~0.24 dex above Chabrier, so the offset is negative.
    assert!((v["log_m"].as_f64().unwrap() - 9.76).abs() < 1e-6);
}
```

- [ ] **Step 4: Run to verify it fails**

Run: `cd rust && cargo test -p steel-harmonise-cli`
Expected: FAIL — crate has no `main.rs` yet

- [ ] **Step 5: Implement the binary**

`rust/steel-harmonise-cli/src/main.rs`:

```rust
//! JSON-in/JSON-out wrapper over `steel_plugins::harmonise`.
//!
//! The research apparatus performs no unit or definition conversion of its
//! own; it shells out here, so the conversion logic stays in one tested
//! place rather than being reimplemented in Python.

use anyhow::{anyhow, bail, Result};
use serde::Deserialize;
use serde_json::json;
use steel_core::cosmology::MassDefinition;
use steel_plugins::harmonise::{convert_mass_definition, DuttonMaccio14, HConvention, Imf};
use steel_plugins::Planck15;

#[derive(Deserialize)]
struct Endpoint {
    mass_def: Option<String>,
    imf: Option<String>,
    h_convention: String,
}

#[derive(Deserialize)]
struct Request {
    op: String,
    log_m: f64,
    from: Endpoint,
    to: Endpoint,
    #[serde(default)]
    z: f64,
}

fn parse_mass_def(s: &str) -> Result<MassDefinition> {
    match s {
        "Mvir" => Ok(MassDefinition::Vir),
        _ if s.ends_with('c') => s[1..s.len() - 1]
            .parse()
            .map(MassDefinition::Critical)
            .map_err(|_| anyhow!("bad critical overdensity: {s}")),
        _ if s.ends_with('m') => s[1..s.len() - 1]
            .parse()
            .map(MassDefinition::Mean)
            .map_err(|_| anyhow!("bad mean overdensity: {s}")),
        // "unknown" lands here too: refuse rather than guess.
        _ => bail!("unrecognised mass definition: {s}"),
    }
}

fn parse_h(s: &str) -> Result<HConvention> {
    match s {
        "h_free" => Ok(HConvention::HFree),
        "per_h" => Ok(HConvention::PerH),
        _ => bail!("unrecognised h convention: {s}"),
    }
}

fn parse_imf(s: &str) -> Result<Imf> {
    match s {
        "chabrier" => Ok(Imf::Chabrier),
        "kroupa" => Ok(Imf::Kroupa),
        "salpeter" => Ok(Imf::Salpeter),
        _ => bail!("unrecognised IMF: {s}"),
    }
}

fn main() -> Result<()> {
    let req: Request = serde_json::from_reader(std::io::stdin())?;
    let cosmo = Planck15::new();
    let h = cosmo.h();
    let mut path: Vec<String> = Vec::new();

    let from_h = parse_h(&req.from.h_convention)?;
    let to_h = parse_h(&req.to.h_convention)?;

    let log_m = match req.op.as_str() {
        "convert_mass" => {
            // The conversion works in Msun/h, matching m_to_r.
            let per_h = from_h.from_h_free(from_h.to_h_free(req.log_m, h), h);
            let as_per_h = match from_h {
                HConvention::HFree => {
                    path.push("h_free->per_h".into());
                    HConvention::PerH.from_h_free(req.log_m, h)
                }
                _ => per_h,
            };
            let from_def = parse_mass_def(
                req.from.mass_def.as_deref().ok_or_else(|| anyhow!("from.mass_def required"))?,
            )?;
            let to_def = parse_mass_def(
                req.to.mass_def.as_deref().ok_or_else(|| anyhow!("to.mass_def required"))?,
            )?;
            let converted =
                convert_mass_definition(as_per_h, from_def, to_def, req.z, &cosmo, &DuttonMaccio14);
            path.push(format!(
                "{}->{} (DuttonMaccio14, NFW)",
                req.from.mass_def.as_deref().unwrap(),
                req.to.mass_def.as_deref().unwrap()
            ));
            match to_h {
                HConvention::HFree => {
                    path.push("per_h->h_free".into());
                    HConvention::PerH.to_h_free(converted, h)
                }
                _ => converted,
            }
        }
        "convert_stellar" => {
            let from_imf = parse_imf(
                req.from.imf.as_deref().ok_or_else(|| anyhow!("from.imf required"))?,
            )?;
            let to_imf =
                parse_imf(req.to.imf.as_deref().ok_or_else(|| anyhow!("to.imf required"))?)?;
            let offset = from_imf.log_offset_to(to_imf);
            path.push(format!("imf {from_imf:?}->{to_imf:?} ({offset:+.3} dex)"));
            let h_free = from_h.to_h_free(req.log_m, h);
            if from_h != to_h {
                path.push(format!("{}->{}", req.from.h_convention, req.to.h_convention));
            }
            to_h.from_h_free(h_free, h) + offset
        }
        other => bail!("unrecognised op: {other}"),
    };

    println!("{}", json!({"log_m": log_m, "path": path}));
    Ok(())
}
```

- [ ] **Step 6: Run the tests**

Run: `cd rust && cargo test -p steel-harmonise-cli`
Expected: all four PASS

- [ ] **Step 7: Commit**

```bash
git add rust/Cargo.toml rust/steel-harmonise-cli
git commit -m "Add steel-harmonise-cli: JSON wrapper over the harmonise layer

Single conversion path for the research apparatus, so no unit or
definition logic is reimplemented outside tested Rust."
```

---

### Task 4: Python package and the `Definition` compatibility rules

**Files:**
- Create: `research/pyproject.toml`, `research/kernel/__init__.py`, `research/kernel/definitions.py`, `research/tests/test_definitions.py`

**Interfaces:**
- Consumes: nothing
- Produces: `Definition` (frozen dataclass, fields `quantity, component, mass_def, aperture, h_convention, imf, cosmology, z_range`), `Definition.from_dict(d) -> Definition`, `Definition.is_comparable_to(other) -> bool`, `IncompatibleDefinitions(Exception)`, `require_comparable(a, b) -> None`

- [ ] **Step 1: Create the package**

`research/pyproject.toml`:

```toml
[project]
name = "steel-research"
version = "0.1.0"
requires-python = ">=3.11"
dependencies = ["pymongo>=4.6", "mcp>=1.2"]

[project.optional-dependencies]
dev = ["pytest>=8", "colossus>=1.3"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

`research/kernel/__init__.py`: empty file.

- [ ] **Step 2: Write the failing tests**

`research/tests/test_definitions.py`:

```python
import pytest
from kernel.definitions import Definition, IncompatibleDefinitions, require_comparable

GZZ07 = dict(quantity="f_bcg_icl", component="bcg+icl", mass_def="M500c",
             aperture="r500", h_convention="h_free", imf="chabrier",
             cosmology="wmap7", z_range=[0.0, 0.13])


def test_identical_definitions_are_comparable():
    a, b = Definition.from_dict(GZZ07), Definition.from_dict(GZZ07)
    assert a.is_comparable_to(b)
    require_comparable(a, b)


def test_differing_mass_definition_blocks_comparison():
    other = Definition.from_dict({**GZZ07, "mass_def": "Mvir"})
    assert not Definition.from_dict(GZZ07).is_comparable_to(other)
    with pytest.raises(IncompatibleDefinitions, match="mass_def"):
        require_comparable(Definition.from_dict(GZZ07), other)


def test_unknown_never_compares_even_with_itself():
    """`unknown` is missing information, not a value that happens to match."""
    d = Definition.from_dict({**GZZ07, "imf": "unknown"})
    assert not d.is_comparable_to(d)
    with pytest.raises(IncompatibleDefinitions, match="imf"):
        require_comparable(d, d)


def test_component_mismatch_blocks_comparison():
    """BCG+ICL is not ICL-only -- the caveat that went unenforced all session."""
    icl_only = Definition.from_dict({**GZZ07, "component": "icl"})
    with pytest.raises(IncompatibleDefinitions, match="component"):
        require_comparable(Definition.from_dict(GZZ07), icl_only)


def test_missing_field_is_rejected_at_construction():
    incomplete = {k: v for k, v in GZZ07.items() if k != "imf"}
    with pytest.raises(ValueError, match="imf"):
        Definition.from_dict(incomplete)
```

- [ ] **Step 3: Run to verify failure**

Run: `cd research && python -m venv .venv && .venv/bin/pip install -e ".[dev]" -q && .venv/bin/pytest tests/test_definitions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'kernel.definitions'`

- [ ] **Step 4: Implement**

`research/kernel/definitions.py`:

```python
"""The comparability fingerprint.

Two quantities may only be combined when every field of their `Definition`
agrees. `"unknown"` is missing information rather than a matching value, so
it never compares -- including against itself. That asymmetry is
deliberate: it turns an unrecorded assumption into a hard stop instead of a
silent pass.
"""
from dataclasses import dataclass, fields
from typing import Any

FIELDS = ("quantity", "component", "mass_def", "aperture",
          "h_convention", "imf", "cosmology", "z_range")

UNKNOWN = "unknown"


class IncompatibleDefinitions(Exception):
    """Raised when two definitions cannot be compared without conversion."""


@dataclass(frozen=True)
class Definition:
    quantity: str
    component: str
    mass_def: str
    aperture: str
    h_convention: str
    imf: str
    cosmology: str
    z_range: tuple

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Definition":
        missing = [f for f in FIELDS if f not in d]
        if missing:
            raise ValueError(f"definition missing required field(s): {', '.join(missing)}")
        z = d["z_range"]
        return cls(**{**{f: d[f] for f in FIELDS if f != "z_range"},
                      "z_range": tuple(z)})

    def differences(self, other: "Definition") -> list[str]:
        """Field names that block comparison, including any `unknown`."""
        out = []
        for f in fields(self):
            mine, theirs = getattr(self, f.name), getattr(other, f.name)
            if mine == UNKNOWN or theirs == UNKNOWN or mine != theirs:
                out.append(f.name)
        return out

    def is_comparable_to(self, other: "Definition") -> bool:
        return not self.differences(other)


def require_comparable(a: Definition, b: Definition) -> None:
    diff = a.differences(b)
    if diff:
        raise IncompatibleDefinitions(
            "cannot compare without explicit conversion; differing or unknown: "
            + ", ".join(diff))
```

- [ ] **Step 5: Run the tests**

Run: `cd research && .venv/bin/pytest tests/test_definitions.py -q`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add research/pyproject.toml research/kernel research/tests
git commit -m "Add Definition compatibility rules

Every field must agree; 'unknown' never compares, including with itself,
so an unrecorded assumption becomes a hard stop rather than a silent pass."
```

---

### Task 5: Conversion bridge, validated against COLOSSUS

**Files:**
- Create: `research/kernel/convert.py`, `research/tests/test_convert.py`

**Interfaces:**
- Consumes: `steel-harmonise-cli` (Task 3), `Definition` (Task 4)
- Produces: `convert(log_m: float, frm: Definition, to: Definition, z: float) -> tuple[float, list[str]]` returning `(log_m, path)`; `ConversionError(Exception)`

- [ ] **Step 1: Write the failing tests**

`research/tests/test_convert.py`:

```python
import pytest
from kernel.convert import convert, ConversionError
from kernel.definitions import Definition

BASE = dict(quantity="m_halo", component="halo", aperture="r500",
            imf="chabrier", cosmology="planck15", z_range=[0.1, 0.1])


def d(**over):
    return Definition.from_dict({**BASE, "mass_def": "M500c",
                                 "h_convention": "h_free", **over})


def test_mass_definition_conversion_increases_virial_mass():
    log_m, path = convert(14.0, d(), d(mass_def="Mvir"), z=0.1)
    assert 14.0 < log_m < 15.0
    assert any("Mvir" in step for step in path)


def test_unknown_mass_definition_raises():
    with pytest.raises(ConversionError):
        convert(14.0, d(mass_def="unknown"), d(mass_def="Mvir"), z=0.1)


def test_agrees_with_colossus():
    """Independent validation -- the conversion is new physics, so it is
    checked against an established implementation, not just itself."""
    colossus = pytest.importorskip("colossus.halo.mass_defs")
    from colossus.cosmology import cosmology as ccosmo
    ccosmo.setCosmology("planck15")
    from colossus.halo.concentration import concentration as c_of_m

    m500c_per_h = 1e14  # Msun/h
    c500 = c_of_m(m500c_per_h, "500c", 0.1, model="duffy08")
    m_vir_ref, _, _ = colossus.changeMassDefinition(
        m500c_per_h, c500, 0.1, "500c", "vir")

    got, _ = convert(14.0, d(h_convention="per_h"),
                     d(mass_def="Mvir", h_convention="per_h"), z=0.1)
    # Different concentration relations (DuttonMaccio14 vs Duffy08) shift
    # the answer, so agree to 0.05 dex rather than exactly.
    assert abs(got - (m_vir_ref ** 0 * 0 + __import__("math").log10(m_vir_ref))) < 0.05
```

- [ ] **Step 2: Run to verify failure**

Run: `cd research && .venv/bin/pytest tests/test_convert.py -q`
Expected: FAIL — no module `kernel.convert`

- [ ] **Step 3: Implement**

`research/kernel/convert.py`:

```python
"""Bridge to `steel-harmonise-cli`.

No conversion arithmetic lives here. Python owns definitions and
provenance; every number is converted by the Rust layer, so the formulas
have exactly one tested implementation.
"""
import json
import subprocess
from pathlib import Path

from .definitions import Definition

CLI = Path(__file__).resolve().parents[2] / "rust" / "target" / "release" / "steel-harmonise-cli"


class ConversionError(Exception):
    """The conversion was refused or the CLI failed."""


def _endpoint(defn: Definition) -> dict:
    return {"mass_def": defn.mass_def, "imf": defn.imf,
            "h_convention": defn.h_convention}


def convert(log_m: float, frm: Definition, to: Definition, z: float) -> tuple[float, list[str]]:
    """Convert `log_m` from one definition to another.

    Returns the converted value and the ordered list of steps taken, which
    the caller records as provenance.
    """
    if not CLI.exists():
        raise ConversionError(
            f"{CLI} not built; run: cargo build --release -p steel-harmonise-cli")
    op = "convert_mass" if frm.mass_def != to.mass_def or to.quantity.startswith("m_") \
        else "convert_stellar"
    req = {"op": op, "log_m": log_m, "z": z,
           "from": _endpoint(frm), "to": _endpoint(to)}
    proc = subprocess.run([str(CLI)], input=json.dumps(req),
                          capture_output=True, text=True)
    if proc.returncode != 0:
        raise ConversionError(proc.stderr.strip() or "steel-harmonise-cli failed")
    out = json.loads(proc.stdout)
    return out["log_m"], out["path"]
```

- [ ] **Step 4: Build the CLI and run the tests**

Run:
```bash
cd rust && cargo build --release -p steel-harmonise-cli
cd ../research && .venv/bin/pytest tests/test_convert.py -q
```
Expected: 3 passed (the COLOSSUS test skips if `colossus` is unavailable)

- [ ] **Step 5: Commit**

```bash
git add research/kernel/convert.py research/tests/test_convert.py
git commit -m "Add conversion bridge to steel-harmonise-cli

Python owns definitions and provenance; Rust owns every formula. Includes
an independent COLOSSUS cross-check of the new mass conversion."
```

---

### Task 6: Mongo store with the gates

**Files:**
- Create: `docker-compose.yml`, `research/kernel/store.py`, `research/tests/test_store.py`

**Interfaces:**
- Consumes: `Definition` (Task 4)
- Produces: `Store(uri: str, db: str)` with `put(doc: dict) -> str`, `query(spec: dict) -> list[dict]`, `verify_source(source_id: str, method: str) -> dict`; `GateViolation(Exception)`

- [ ] **Step 1: Create the Mongo service**

`docker-compose.yml`:

```yaml
services:
  research-store:
    image: mongo:7
    container_name: steel-research-store
    ports:
      - "27017:27017"
    volumes:
      - research-store-data:/data/db

volumes:
  research-store-data:
```

- [ ] **Step 2: Write the failing tests**

`research/tests/test_store.py`:

```python
import pytest
from kernel.store import Store, GateViolation

pytestmark = pytest.mark.integration

GZZ07_DEF = dict(quantity="f_bcg_icl", component="bcg+icl", mass_def="M500c",
                 aperture="r500", h_convention="h_free", imf="chabrier",
                 cosmology="wmap7", z_range=[0.0, 0.13])


@pytest.fixture
def store():
    s = Store("mongodb://localhost:27017", db="steel_research_test")
    s.drop()
    s.ensure_schema()
    return s


def measurement(**over):
    return {"_id": "gzz07-f-bcg-icl-r500", "kind": "measurement",
            "definition": GZZ07_DEF,
            "payload": {"value": 0.40},
            "source_id": "arxiv:0705.1726",
            "source_snapshot": {"arxiv": "0705.1726",
                                "verified_at": "2026-08-21T00:00:00Z",
                                "verification_method": "arxiv-api-resolved",
                                "extraction": "abstract"},
            "caveats": ["not-icl-only"], **over}


def test_measurement_requires_a_verified_source(store):
    """Gate 1 -- the fabricated-citation failure."""
    with pytest.raises(GateViolation, match="verified source"):
        store.put(measurement())  # source not registered yet


def test_measurement_accepted_once_source_verified(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    assert store.put(measurement()) == "gzz07-f-bcg-icl-r500"


def test_blank_definition_field_is_rejected(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement()
    del bad["definition"]["imf"]
    with pytest.raises(GateViolation, match="imf"):
        store.put(bad)


def test_extraction_method_is_mandatory_and_enumerated(store):
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    bad = measurement()
    bad["source_snapshot"]["extraction"] = "remembered"
    with pytest.raises(GateViolation, match="extraction"):
        store.put(bad)


def test_query_returns_definition_and_verification_state(store):
    """A cache hit must carry enough context to judge reuse."""
    store.verify_source("arxiv:0705.1726", method="arxiv-api-resolved")
    store.put(measurement())
    hits = store.query({"definition.quantity": "f_bcg_icl"})
    assert len(hits) == 1
    assert hits[0]["definition"]["mass_def"] == "M500c"
    assert hits[0]["source_snapshot"]["extraction"] == "abstract"
```

- [ ] **Step 3: Start Mongo and verify the tests fail**

Run:
```bash
docker compose up -d research-store
cd research && .venv/bin/pytest tests/test_store.py -q
```
Expected: FAIL — no module `kernel.store`

- [ ] **Step 4: Implement**

`research/kernel/store.py`:

```python
"""MongoDB result store and its gates.

Imported only by `research/mcp/server.py`. Agents reach the store through
MCP tools, never through this module -- a library can be bypassed by an
agent writing its own script, which is the behaviour the apparatus exists
to prevent.
"""
from typing import Any

from pymongo import MongoClient

from .definitions import FIELDS as DEFINITION_FIELDS

EXTRACTION_METHODS = {"table", "figure", "text", "abstract"}
VERIFICATION_METHODS = {"arxiv-api-resolved", "doi-resolved", "manual-pdf"}


class GateViolation(Exception):
    """A write was refused because it would break a spec gate."""


class Store:
    def __init__(self, uri: str, db: str = "steel_research"):
        self._client = MongoClient(uri, serverSelectionTimeoutMS=3000)
        self._db = self._client[db]

    def drop(self) -> None:
        self._client.drop_database(self._db.name)

    def ensure_schema(self) -> None:
        """Indexes for the pre-check query path."""
        self._db.artifacts.create_index("kind")
        self._db.artifacts.create_index("definition.quantity")
        self._db.artifacts.create_index("definition.mass_def")
        self._db.sources.create_index("source_id", unique=True)

    def verify_source(self, source_id: str, method: str) -> dict:
        """Register a source as verified. Recollection is not verification."""
        if method not in VERIFICATION_METHODS:
            raise GateViolation(
                f"verification_method must be one of {sorted(VERIFICATION_METHODS)}")
        doc = {"source_id": source_id, "verification_method": method}
        self._db.sources.update_one({"source_id": source_id},
                                    {"$set": doc}, upsert=True)
        return doc

    def _check(self, doc: dict[str, Any]) -> None:
        if doc.get("kind") != "measurement":
            return
        source_id = doc.get("source_id")
        if not source_id or not self._db.sources.find_one({"source_id": source_id}):
            raise GateViolation(
                f"measurement requires a verified source; {source_id!r} is not registered")
        missing = [f for f in DEFINITION_FIELDS if f not in doc.get("definition", {})]
        if missing:
            raise GateViolation(
                f"definition missing required field(s): {', '.join(missing)}")
        extraction = doc.get("source_snapshot", {}).get("extraction")
        if extraction not in EXTRACTION_METHODS:
            raise GateViolation(
                f"extraction must be one of {sorted(EXTRACTION_METHODS)}, got {extraction!r}")

    def put(self, doc: dict[str, Any]) -> str:
        self._check(doc)
        self._db.artifacts.replace_one({"_id": doc["_id"]}, doc, upsert=True)
        return doc["_id"]

    def query(self, spec: dict[str, Any]) -> list[dict]:
        return list(self._db.artifacts.find(spec))
```

- [ ] **Step 5: Run the tests**

Run: `cd research && .venv/bin/pytest tests/test_store.py -q`
Expected: 5 passed

- [ ] **Step 6: Commit**

```bash
git add docker-compose.yml research/kernel/store.py research/tests/test_store.py
git commit -m "Add Mongo result store with the spec's write gates

Measurements require a verified source, a complete definition, and an
enumerated extraction method. Each gate maps to a failure observed in the
ICL/SMHM work."
```

---

### Task 7: `research-store` MCP server

**Files:**
- Create: `research/mcp/__init__.py`, `research/mcp/server.py`, `research/tests/test_mcp_server.py`
- Modify: `.mcp.json`

**Interfaces:**
- Consumes: `Store`, `GateViolation` (Task 6)
- Produces: MCP tools `store_query(spec: dict)`, `store_put(doc: dict)`, `store_verify_source(source_id: str, method: str)`

- [ ] **Step 1: Write the failing test**

`research/tests/test_mcp_server.py`:

```python
import pytest
from mcp_server_under_test import build_server  # re-exported below

pytestmark = pytest.mark.integration


@pytest.fixture
def tools():
    return build_server("mongodb://localhost:27017", db="steel_research_mcp_test")


def test_put_is_refused_without_a_verified_source(tools):
    """The gate must live server-side, not in a bypassable library."""
    result = tools["store_put"]({"_id": "x", "kind": "measurement",
                                 "definition": {}, "source_id": "arxiv:9999.99999"})
    assert result["ok"] is False
    assert "verified source" in result["error"]


def test_verify_then_put_succeeds(tools):
    tools["store_verify_source"]("arxiv:0705.1726", "arxiv-api-resolved")
    doc = {"_id": "m1", "kind": "measurement",
           "definition": {"quantity": "f_bcg_icl", "component": "bcg+icl",
                          "mass_def": "M500c", "aperture": "r500",
                          "h_convention": "h_free", "imf": "chabrier",
                          "cosmology": "wmap7", "z_range": [0.0, 0.13]},
           "payload": {"value": 0.40}, "source_id": "arxiv:0705.1726",
           "source_snapshot": {"extraction": "abstract"}}
    assert tools["store_put"](doc)["ok"] is True
    assert len(tools["store_query"]({"kind": "measurement"})["results"]) == 1


def test_bad_verification_method_is_refused(tools):
    result = tools["store_verify_source"]("arxiv:1.1", "i-remember-it")
    assert result["ok"] is False
```

- [ ] **Step 2: Run to verify failure**

Run: `cd research && .venv/bin/pytest tests/test_mcp_server.py -q`
Expected: FAIL — no module `mcp_server_under_test`

- [ ] **Step 3: Implement**

`research/mcp/__init__.py`: empty file.

`research/mcp/server.py`:

```python
"""The research-store MCP server: the only write path to the store.

Gates are enforced here rather than in a Python library because an agent
holding a library can write its own script around it; an agent holding an
MCP tool cannot.
"""
import sys

from mcp.server.fastmcp import FastMCP

from kernel.store import GateViolation, Store


def build_server(uri: str, db: str = "steel_research") -> dict:
    """Return the tool callables, so tests can exercise them directly."""
    store = Store(uri, db=db)
    store.ensure_schema()

    def store_verify_source(source_id: str, method: str) -> dict:
        try:
            return {"ok": True, "source": store.verify_source(source_id, method)}
        except GateViolation as e:
            return {"ok": False, "error": str(e)}

    def store_put(doc: dict) -> dict:
        try:
            return {"ok": True, "id": store.put(doc)}
        except GateViolation as e:
            return {"ok": False, "error": str(e)}

    def store_query(spec: dict) -> dict:
        results = store.query(spec)
        for r in results:
            r.pop("_id", None) if not isinstance(r.get("_id"), str) else None
        return {"ok": True, "results": results}

    return {"store_verify_source": store_verify_source,
            "store_put": store_put, "store_query": store_query}


def main() -> None:
    uri = "mongodb://localhost:27017"
    tools = build_server(uri)
    server = FastMCP("research-store")
    for name, fn in tools.items():
        server.add_tool(fn, name=name)
    server.run()


if __name__ == "__main__":
    main()
```

Add `research/tests/mcp_server_under_test.py` so the test import resolves without a running MCP transport:

```python
"""Test shim: exposes the server's tool callables without a transport."""
from mcp.server import build_server  # noqa: F401
```

- [ ] **Step 4: Run the tests**

Run: `cd research && .venv/bin/pytest tests/test_mcp_server.py -q`
Expected: 3 passed

- [ ] **Step 5: Register the server**

Modify `.mcp.json` to:

```json
{
  "mcpServers": {
    "arxiv": { "type": "sse", "url": "http://localhost:8050/sse" },
    "research-store": {
      "command": "research/.venv/bin/python",
      "args": ["-m", "mcp.server"],
      "cwd": "research"
    }
  }
}
```

- [ ] **Step 6: Commit**

```bash
git add research/mcp research/tests/test_mcp_server.py research/tests/mcp_server_under_test.py .mcp.json
git commit -m "Add research-store MCP server as the sole write path

Gates are enforced at the tool boundary, which an agent cannot bypass."
```

---

### Task 8: `data-curator` and `referee` agents

**Files:**
- Create: `.claude/agents/data-curator.md`, `.claude/agents/referee.md`

**Interfaces:**
- Consumes: MCP tools from Task 7
- Produces: two agent definitions

- [ ] **Step 1: Write the data-curator**

`.claude/agents/data-curator.md`:

```markdown
---
name: data-curator
description: Extracts measurements with complete definitions from an already-verified source and writes them to the research store. Use when a verified source needs numbers pulled out of it.
tools: mcp__research-store__store_put, mcp__research-store__store_query, WebFetch, Read
model: sonnet
---

You extract measurements from **already-verified** sources and record them.

## You must
- Read the source's full text. Prefer tables, then figures, then body text.
- Set `extraction` to exactly what you used: `table`, `figure`, `text`, or
  `abstract`. `abstract` is a permanent flag on the value, not a
  placeholder — never label an abstract-derived number as `table`.
- Record `locator` precisely: "Table 3, row 2, column f_ICL".
- Fill every `definition` field. Where the paper does not state one, write
  the literal string `"unknown"`. This blocks comparison, which is the
  correct outcome — a guessed aperture or IMF is worse than a blocked one.
- Record the paper's own cosmology and IMF, not STEEL's.
- Put every stated systematic into `caveats`.

## You must never
- Invent, recall, or infer a number that is not in the source text.
- Convert units or mass definitions. Record what the paper states; the
  conversion layer handles the rest, and it needs the original.
- Write a measurement for an unverified source. `store_put` will refuse it;
  do not work around the refusal.

Report the artifact IDs written and every field you set to `"unknown"`.
```

- [ ] **Step 2: Write the referee**

`.claude/agents/referee.md`:

```markdown
---
name: referee
description: Adversarially audits a draft claim before it may leave draft status. Use after an analyst produces a claim and before it is relied upon.
tools: mcp__research-store__store_query, Read, Grep
model: opus
---

You are an adversarial referee. Your job is to find the reason this claim
is wrong. You do not fix anything — fixing is someone else's job, and an
auditor who edits loses independence.

## Checklist, in order

1. **Provenance.** Is every input a registered artifact with a verified
   source? Any value with `extraction: abstract` is a weakness — say so
   explicitly, with the artifact ID.
2. **Definitions.** Were any two quantities compared across differing
   definitions? If a conversion was applied, is it recorded in the
   derivation's `path`? An unrecorded conversion is a defect.
3. **Circularity.** Does any input appear on *both* sides of the
   comparison? A model calibrated on relation X cannot be used to test
   relation X. State the shared input by ID.
4. **Caveat completeness.** Does the claim carry the union of its inputs'
   caveats? List any dropped.
5. **Overreach.** Does the wording claim more than the data supports?
   A bound derived under one stripping model is not "the" bound.

## Verdict

Emit `PASS` or `REVISE`, then a numbered list of findings, each naming the
artifact ID it concerns. `REVISE` on any unresolved item in 1-4.
Overreach alone may be `PASS` with a required wording change.
```

- [ ] **Step 3: Verify the agents are discoverable**

Run: `ls .claude/agents/`
Expected: `data-curator.md`, `referee.md`

- [ ] **Step 4: Commit**

```bash
git add .claude/agents
git commit -m "Add data-curator and referee agent definitions

The curator may not convert or invent; the referee may not fix. Both
restrictions are the point."
```

---

### Task 9: The Slice 1 derivation and refereed claim

Reproduce `icl_stripping_bound` through the harness, with the M500→Mvir conversion actually applied.

**Files:**
- Create: `research/derivations/__init__.py`, `research/derivations/icl_stripping_bound.py`, `research/tests/test_icl_derivation.py`

**Interfaces:**
- Consumes: `convert` (Task 5), `Definition` (Task 4), the sweep CSV `falsification_lowmass.csv`
- Produces: `run(sweep_csv: str, gzz07_value: float, out_png: str) -> dict` returning `{"figure": path, "max_strength_at": {...}, "path": [...], "caveats": [...]}`

- [ ] **Step 1: Write the failing test**

`research/tests/test_icl_derivation.py`:

```python
import csv

from derivations.icl_stripping_bound import max_allowed_strength, run


def test_max_allowed_strength_interpolates_the_crossing():
    """The helper previously copy-pasted between two plotting scripts."""
    strengths = [0.0, 1.0, 2.0]
    f_icl = [0.0, 0.30, 0.50]
    # bound 0.40 sits halfway between strengths 1 and 2
    assert abs(max_allowed_strength(strengths, f_icl, 0.40) - 1.5) < 1e-9


def test_ceiling_below_every_sample_gives_zero():
    assert max_allowed_strength([0.0, 1.0], [0.5, 0.9], 0.1) == 0.0


def test_ceiling_above_every_sample_returns_the_tested_maximum():
    assert max_allowed_strength([0.0, 4.0], [0.0, 0.1], 0.9) == 4.0


def test_run_applies_the_mass_conversion_and_records_it(tmp_path):
    csv_path = tmp_path / "sweep.csv"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["satellite_sf", "strength", "log_mh_perh", "log_sm_smhm",
                    "log_accreted", "ratio", "log_icl", "f_icl"])
        for s, f in [(0.0, 0.0), (1.0, 0.30), (2.0, 0.50)]:
            w.writerow(["False", s, 14.0, 11.5, 11.9, 1.0, 11.2, f])
    out = run(str(csv_path), gzz07_value=0.40, out_png=str(tmp_path / "f.png"))
    assert any("Mvir" in step for step in out["path"]), \
        "the M500->Mvir conversion must be recorded, not skipped"
    assert "not-icl-only" in out["caveats"]
```

- [ ] **Step 2: Run to verify failure**

Run: `cd research && .venv/bin/pytest tests/test_icl_derivation.py -q`
Expected: FAIL — no module `derivations.icl_stripping_bound`

- [ ] **Step 3: Implement**

`research/derivations/__init__.py`: empty file.

`research/derivations/icl_stripping_bound.py`:

```python
"""Slice 1: the empirical ICL ceiling as a bound on stripping strength.

Reproduces the 2026-08-21 result, with one substantive change: the
Gonzalez+07 ceiling is quoted at M500c and STEEL's grid is Mvir, so the
halo-mass axis is now genuinely converted rather than annotated as a
caveat. The conversion is performed by the Rust layer and its steps are
returned for provenance.
"""
import csv

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from kernel.convert import convert
from kernel.definitions import Definition

GZZ07_CAVEATS = ["not-icl-only", "extraction-abstract"]

_HALO = dict(quantity="m_halo", component="halo", aperture="r500",
             imf="chabrier", cosmology="planck15", z_range=[0.1, 0.1])


def max_allowed_strength(strengths, f_icl, bound):
    """Largest stripping strength keeping `f_icl` at or below `bound`.

    `f_icl` increases monotonically with strength, so linear interpolation
    against the bound is safe. Returns 0.0 when even zero stripping
    exceeds the bound, and the largest tested strength when the bound is
    never reached (i.e. unconstrained by this data).
    """
    if f_icl[0] > bound:
        return 0.0
    if f_icl[-1] <= bound:
        return strengths[-1]
    for i in range(1, len(f_icl)):
        if f_icl[i] > bound:
            s0, s1 = strengths[i - 1], strengths[i]
            f0, f1 = f_icl[i - 1], f_icl[i]
            return s0 + (bound - f0) * (s1 - s0) / (f1 - f0)
    return strengths[-1]


def run(sweep_csv: str, gzz07_value: float, out_png: str) -> dict:
    rows = [r for r in csv.DictReader(open(sweep_csv))
            if r["satellite_sf"].lower() == "false"]

    by_mass: dict[float, list[tuple[float, float]]] = {}
    for r in rows:
        by_mass.setdefault(float(r["log_mh_perh"]), []).append(
            (float(r["strength"]), float(r["f_icl"])))

    m500c = Definition.from_dict({**_HALO, "mass_def": "M500c",
                                  "h_convention": "per_h"})
    mvir = Definition.from_dict({**_HALO, "mass_def": "Mvir",
                                 "h_convention": "h_free"})

    xs, ys, path = [], [], []
    for log_mh_perh in sorted(by_mass):
        pairs = sorted(by_mass[log_mh_perh])
        strengths = [p[0] for p in pairs]
        f_icl = [p[1] for p in pairs]
        # The ceiling is an M500c measurement; STEEL's axis is Mvir. Convert
        # the axis onto the measurement's definition before comparing.
        log_mh_converted, steps = convert(log_mh_perh, mvir, m500c, z=0.1)
        path = steps
        xs.append(log_mh_converted)
        ys.append(max_allowed_strength(strengths, f_icl, gzz07_value))

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.plot(xs, ys, "-", color="crimson", lw=2.0)
    ax.axhline(1.0, color="0.5", ls=":", lw=1.2)
    ax.annotate("published (Cattaneo+11) baseline", xy=(xs[0], 1.03),
                fontsize=8, color="0.4")
    ax.set_xlabel(r"host halo mass  $\log_{10} M_{500c}$  [$\mathrm{M}_\odot$]")
    ax.set_ylabel("max. stripping strength\nallowed by the ICL ceiling")
    ax.set_title("ICL ceiling as a bound on stripping strength\n"
                 "(halo mass converted to the measurement's own definition)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    return {"figure": out_png,
            "max_strength_at": dict(zip(xs, ys)),
            "path": path,
            "caveats": GZZ07_CAVEATS}
```

- [ ] **Step 4: Run the tests**

Run: `cd research && .venv/bin/pytest tests/test_icl_derivation.py -q`
Expected: 4 passed

- [ ] **Step 5: Regenerate against the real sweep and compare**

Run:
```bash
cd research && .venv/bin/python -c "
from derivations.icl_stripping_bound import run
out = run('$SWEEP/falsification_lowmass.csv', 0.40, 'icl_bound_harness.png')
print('conversion path:', out['path'])
print('caveats:', out['caveats'])
"
```
where `$SWEEP` is the directory holding `falsification_lowmass.csv`.

Expected: the conversion path names `M500c`/`Mvir`, and the halo-mass axis is **shifted relative to the 2026-08-21 figure** — that shift is the previously-skipped mass-definition conversion and is the slice's success criterion. Record the shift in dex.

- [ ] **Step 6: Referee the result**

Dispatch the `referee` agent (Task 8) against the derivation output and the artifacts it consumed. Expected findings, which must all appear: the GZZ07 value is `extraction: abstract`; the ceiling is BCG+ICL while the model quantity is satellite-stripped-only (`component` mismatch); the bound holds only for the `Cattaneo11` stripping family.

- [ ] **Step 7: Commit**

```bash
git add research/derivations research/tests/test_icl_derivation.py
git commit -m "Add the Slice 1 ICL-ceiling derivation

Reproduces the 2026-08-21 stripping bound with the M500c/Mvir conversion
actually applied and its steps recorded, rather than annotated as an
unresolved caveat."
```

---

## Self-Review

**Spec coverage.** Gate 1 (verified source) → Task 6. Gate 2 (definition compatibility) → Tasks 4, 5. Gate 3 (figures only via derivations) → Task 9. Gate 4 (caveat inheritance) → Task 9 returns `caveats`; full automatic union is Slice 4. Gate 5 (dirty-tree claims) → deferred to Slice 3 with `model-runner`. Gate 6 (referee mandatory) → Task 8 defines the agent; Task 9 Step 6 exercises it. `lit-scout`, the question register, the ledger and `research export` are explicitly Slices 2-4 and correctly absent here.

**Known gaps, deliberate.** Gates 4 and 5 are partially deferred; this is consistent with the spec's slice ordering and noted so a reviewer does not read it as an omission.

**Type consistency.** `Definition.from_dict` / `is_comparable_to` / `require_comparable` (Task 4) are used with those names in Tasks 5 and 9. `convert(log_m, frm, to, z) -> (float, list[str])` (Task 5) matches its call in Task 9. `Store.put/query/verify_source` (Task 6) match the MCP tool wrappers (Task 7). `max_allowed_strength(strengths, f_icl, bound)` (Task 9) matches its tests.
