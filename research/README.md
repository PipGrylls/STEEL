# STEEL research apparatus

## What this is for

This directory is a research harness whose purpose is to make unsourced,
mis-defined or unreproducible claims *structurally impossible* rather than
merely discouraged. Every research artifact — a source, a measurement, a
model run, a derivation run, a claim — is a typed document in MongoDB with
a mandatory `definition` (the eight-field comparability fingerprint:
quantity, component, mass_def, aperture, h_convention, imf, cosmology,
z_range) and, for measurements, a *verified* source. Six gates enforce
this: (1) a measurement cannot be written without a verified source;
(2) combining incompatible definitions raises, so conversion must be
explicit and is itself recorded; (3) a figure may only be produced as a
`derivation_run` output; (4) a claim inherits the union of its inputs'
caveats automatically; (5) a `model_run` from a dirty git tree may back a
draft claim but never a published one; (6) a claim cannot leave `draft`
without a referee verdict. The gates live in the MCP server, not in a
Python library, because a library can be bypassed by an agent writing its
own script — which is the exact behaviour the apparatus exists to prevent.
The design document is
`docs/superpowers/specs/2026-08-22-research-apparatus-design.md`.

**Current state of slice 1:** the flagship derivation's headline
comparison is *refused*. `derivations/icl_stripping_bound.py` compares
STEEL's `f_icl` with Gonzalez+07's `f_BCG+ICL`, and those two definitions
differ on seven fields at once, so the claim is recorded as `draft`
carrying the referee's `REVISE` verdict. That refusal is the apparatus
working, not failing. **The claim must not be published or cited.**

## Setup

Four things are required. Skipping any of them makes the suite fail
loudly rather than silently degrade — that is deliberate.

### 1. MongoDB

```sh
docker compose up -d research-store      # from the repo root
```

Brings up `mongo:7` as `steel-research-store` on `localhost:27017`
(`docker-compose.yml` at the repo root). The store is the source of truth;
there is no file-based fallback, because a silent fallback to unvalidated
writes is worse than an outage.

### 2. Python environment

Python 3.13 (`requires-python = ">=3.11"`; 3.13 is what this was built
and tested against).

```sh
cd research
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install -e '.[dev]'
```

The editable install matters: `.mcp.json` launches the server as
`research/.venv/bin/python -m mcp_server.server` from the repo root, which
only resolves because the package is installed editable.

`[dev]` pulls in `pytest` and `colossus`. Colossus is not optional
decoration — it is the *independent* reference the M500c↔Mvir conversion
is validated against, and that conversion is the one piece of genuinely
new physics in the design.

### 3. The Rust conversion CLI

```sh
cd rust
cargo build --release -p steel-harmonise-cli
```

**Without this every conversion raises.** `kernel/convert.py` performs no
arithmetic of its own by design: Python owns definitions and provenance,
and every number is converted by `rust/steel-plugins/src/harmonise.rs`
through the CLI, so the formulas have exactly one tested implementation.
`kernel.convert.CLI` points at
`rust/target/release/steel-harmonise-cli` and refuses with the build
command in the message if it is missing.

### 4. The MCP server (optional for tests, required for agents)

`.mcp.json` at the repo root registers `research-store`. To check it by
hand:

```sh
research/.venv/bin/python -m mcp_server.server   # speaks MCP over stdio
```

It should complete `initialize` as `research-store` and answer
`tools/list` with `store_verify_source`, `store_put`, `store_query`.
Those three tools are the only write path to the store.

## Running the tests

```sh
cd research
.venv/bin/pytest -q                      # everything
.venv/bin/pytest -q -m "not integration" # no MongoDB needed
```

Tests marked `integration` need a running MongoDB (step 1); everything
else — the conversion kernel, the definition algebra, the derivation
arithmetic — is pure and runs without it. The colossus cross-check is
skipped if colossus is not installed.

## Running the derivation

```sh
cd research
.venv/bin/python -m derivations.icl_stripping_bound
```

Needs MongoDB *and* the Rust CLI. It reads the committed sweep
`data/falsification_lowmass.csv`, writes the figure to
`figures/icl_stripping_bound.png` (git-ignored — the figure is a
`derivation_run` output, and the run record in Mongo carries the code
hash and input ids that make it reproducible), and records a `model_run`,
a `derivation_run` and a `draft` claim. It prints the artifact ids and the
inherited caveats.

## Layout

| Path | What it is |
|---|---|
| `kernel/definitions.py` | the eight-field `Definition` and the comparability gate. `"unknown"` never compares, including against itself |
| `kernel/convert.py` | bridge to `steel-harmonise-cli`. Refuses any difference in a field the CLI cannot convert |
| `kernel/store.py` | MongoDB store and its gates. Imported only by the MCP server |
| `mcp_server/server.py` | the `research-store` MCP server — the only write path |
| `derivations/` | versioned analysis functions; a `derivation_run` records the code hash of the one that ran |
| `data/` | committed inputs, so results reproduce from a clean clone |
| `figures/` | generated, git-ignored |
| `tests/` | `-m "not integration"` for the Mongo-free subset |
