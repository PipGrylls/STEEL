# STEEL Research Apparatus — Design

**Date:** 2026-08-22
**Status:** draft, awaiting review

## Goal

A multi-agent research harness that makes unsourced, mis-defined, or
unreproducible research claims *structurally impossible* rather than
merely discouraged, while letting independent lines of enquiry run in
parallel.

## Motivation

This design is not speculative. Every requirement below traces to a
failure observed during the ICL/SMHM work of 2026-08-17..21.

| Observed failure | Consequence |
|---|---|
| Two of seven references ("Zhang et al. 2019", "Sampaio-Santos et al. 2021") could not be shown to exist; a third had the wrong year | Fabricated citations nearly entered a paper |
| The f_ICL = 40%/33% numbers carrying three figures were read from an **abstract**, never a data table | Load-bearing values with no real provenance |
| M500 (literature) vs Mvir (STEEL) never reconciled; flagged as a caveat three times, never fixed | Known-invalid comparison, repeatedly shipped |
| The `Msun/h` → h-free bug appeared independently in `dump_central_smhm_grid.rs` and `dump_merger_time.rs` | Same defect class twice, nothing checking for it |
| `max_allowed_strength()` copy-pasted across two plotting scripts | Divergence waiting to happen |
| All figures written to `/private/tmp/...` with no link to model commit or config | Results not reproducible |
| Caveats ("BCG+ICL ≠ ICL-only") restated by hand in every note | Silent drift; a dropped caveat is invisible |
| Satellite masses and the tested central curve both came from Moster+13 | Circularity found by luck, not by process |

A decisive constraint already exists and went unused:
`rust/steel-plugins/src/harmonise.rs` implements `Imf` offsets,
`HConvention` (`to_h_free`/`from_h_free`), and `DuttonMaccio14`
concentration. Its own header names exactly the failures above. It was
never invoked by any analysis this session. **The apparatus does not
reimplement this layer; it makes bypassing it impossible.**

## Architecture

Four subsystems, one spine.

1. **Result store** — MongoDB. Heterogeneous findings as JSON documents.
2. **Enforcement boundary** — a `research-store` MCP server. The *only*
   write path. Gates live server-side, so an agent cannot bypass them by
   writing its own script.
3. **Conversion kernel** — thin Python, delegating every numeric
   conversion to a Rust `steel-harmonise-cli`. No duplicated physics.
4. **Agent roster** — five roles, each with one output type and an
   explicit forbidden list.

Spine: *every research claim is a typed artifact with a verified source
and a complete definition.*

### Why Mongo, not files or SQL

Findings vary too much for a fixed relational schema: an SMHM analytical
form, an ICL fraction, a quenched-fraction table and a merger timescale
share almost no columns. A rigid schema would force lossy flattening.

Mongo is the **source of truth**; `research export` writes canonical JSON
into `research/export/` (git-tracked) for review, diffing and disaster
recovery. This preserves the review gate without fighting the database.

### Why the MCP server, not a Python library

A library can be bypassed by an agent writing a one-off script — the
exact behaviour this apparatus exists to prevent. Tool-level gates
cannot be bypassed: an agent holds `store.put`, not a database
connection.

## Data model

Envelope validated by Mongo `$jsonSchema`; payload unconstrained.

```json
{
  "_id": "gzz07-f-bcg-icl-r500",
  "kind": "measurement",
  "definition": {
    "quantity": "f_bcg_icl", "component": "bcg+icl",
    "mass_def": "M500c", "aperture": "r500",
    "h_convention": "h_free", "imf": "chabrier",
    "cosmology": "wmap7", "z_range": [0.0, 0.13]
  },
  "payload": {
    "rel_type": "smhm",
    "analytical_form": "log M* = log(eps) + ...",
    "parameters": {"M1": 11.59, "alpha": -1.99}
  },
  "source_id": "arxiv:0705.1726",
  "source_snapshot": {"arxiv": "0705.1726", "verified_at": "2026-08-21T00:00:00Z",
                      "verification_method": "arxiv-api-resolved",
                      "extraction": "abstract", "locator": "abstract, sentence 6"},
  "caveats": ["not-icl-only"],
  "created_by": "data-curator", "created_at": "..."
}
```

**Artifact kinds:** `source`, `measurement`, `model_run`, `derivation_run`,
`claim`, `question`.

**Sources are their own documents.** A `measurement` references one by
`source_id`; `source_snapshot` is a denormalised copy carrying the
verification state *at extraction time*, so a later re-verification
cannot retroactively rewrite what a published claim rested on.

**What "verified" means.** A `source` is verified when its identifier
resolves against an authority — `arxiv-api-resolved`, `doi-resolved`, or
`manual-pdf` (a human confirmed a local copy). Recollection is never
verification. `verification_method` is mandatory and recorded.

**`definition` is the comparability fingerprint.** Every field is
mandatory; unknown values must be the explicit string `"unknown"`, which
*blocks* comparison rather than silently permitting it.

**`extraction`** is one of `table | figure | text | abstract`.
`abstract` is a permanent flag, never a temporary state — the GZZ07
values carry it today and would continue to.

**Identity/dedup:** unique index on
`(source.arxiv|doi, definition.quantity, hash(definition))`. Re-finding
a measurement updates rather than duplicates.

**Indexes:** `definition.quantity`, `definition.mass_def`,
`definition.z_range`, `kind`, `caveats` — so the pre-check is precise.

**Derivations are code, not documents.** A derivation lives in
`research/derivations/*.py` under git. Mongo stores a `derivation_run`:
code hash, input artifact IDs, output figure path, timestamp.

## Enforcement gates

Each gate maps to an observed failure.

1. A `measurement` cannot be written without a **verified** `source`
   → fabricated citations.
2. Combining incompatible `definition`s **raises**. Conversion must be
   explicit and is itself recorded → M500/Mvir, h-convention.
3. A figure may only be produced as a `derivation_run` output
   → `/tmp` orphans, copy-pasted analysis.
4. A `claim` **inherits the union of its inputs' caveats** automatically
   → caveat drift.
5. A `model_run` from a dirty git tree may back a `draft` claim, never a
   `published` one → reproducibility.
6. A `claim` cannot leave `draft` without a `referee` verdict.

## Agents

| Agent | Produces | Forbidden | Model tier |
|---|---|---|---|
| `lit-scout` | `source` only | **Emitting any numeric value.** Finds and verifies papers; never reads results out of them | mid |
| `data-curator` | `measurement` | Extracting without setting `extraction`; leaving `definition` fields blank | mid |
| `model-runner` | `model_run` | Interpreting results; editing model source (must stop and escalate) | cheap |
| `analyst` | `derivation_run`, figures | Numeric literals for physical quantities; bypassing the conversion API | high |
| `referee` | verdict only | **Fixing anything itself** — audit stays independent | high |

Splitting `lit-scout` from `data-curator` costs an extra hop and is the
point: a remembered number cannot launder itself into a citation, because
the agent that finds papers cannot emit values.

**Referee checklist:** all inputs verified; no definitional mismatch
silently crossed; **circularity** — does an input appear on both sides of
the comparison?; stated caveats complete against inherited set; claim
strength within what the data supports.

## Research loop

```
Question ──> store.query (pre-check)
               ├── exact definition hit ───────────> reuse, no search
               ├── near hit, other definition ─────> registered conversion (logged)
               └── miss ───────────────────────────> lit-scout ──> data-curator ──> store.put
                                                                          │
   model-runner ──> model_run ────────────────────────────────────────────┤
                                                                          ↓
                                                                       analyst
                                                                          ↓
                                     claim{refereed} <──── referee <───────┘
```

`lit-scout` and `model-runner` run **in parallel** — independent inputs;
this is the throughput gain. `data-curator` is gated on verified sources,
`analyst` on both, `referee` is mandatory.

**Cache hits are not free reuse.** `store.query` returns candidates with
definition and verification status attached; a definition mismatch is a
conversion decision. Otherwise the store becomes a fast way to launder
mismatched numbers.

**Question register.** Questions are artifacts with status
`open | answered | abandoned`, linked to the claim answering them. This
is the direct remedy for scattergunning: open questions become a visible
queue. (Live example: "does our rising f_ICL(Mh) contradict GZZ07's
falling group-to-cluster trend?" — raised, unresolved, currently tracked
only in prose.)

**Ledger.** Every transition appends to `research/ledger.jsonl`, so the
loop is recoverable from disk after context compaction.

## Layout

```
research/
  kernel/        definitions.py artifacts.py convert.py figures.py derive.py
  mcp/server.py  research-store MCP server (the write gate)
  derivations/   versioned analysis functions
  export/        canonical JSON, git-tracked
  ledger.jsonl
rust/steel-harmonise-cli/   JSON-in/JSON-out over harmonise.rs
.claude/agents/  lit-scout.md data-curator.md model-runner.md analyst.md referee.md
.mcp.json        + research-store entry
```

### `steel-harmonise-cli`

```
$ echo '{"op":"convert_mass","from":{"mass_def":"M500c","h_convention":"h_free"},
         "to":{"mass_def":"Mvir","h_convention":"per_h"},"log_m":14.0,"z":0.1}' \
  | steel-harmonise-cli
{"log_m": 14.117, "path":["M500c->Mvir (DuttonMaccio14, NFW)","h_free->per_h"]}
```

**Capability gap:** `MassDefinition::{Vir,Critical,Mean}`, `m_to_r` and
`DuttonMaccio14` all exist, but nothing composes them into an overdensity
mass conversion. This must be built (NFW enclosed-mass inversion) and
tested against an independent reference (COLOSSUS). It is the one piece
of genuinely new physics in this design.

## Vertical slices

**Slice 1 — redo the ICL ceiling result through the harness.**
Exercises every component; the current answer is known, so any
discrepancy is diagnostic; and it retires the M500/Mvir defect.
Contents: harmonise CLI with M500↔Mvir; Mongo + MCP server with envelope
validation; `data-curator` and `referee` agents; one derivation
reproducing `icl_stripping_bound`; a refereed claim.
*Success:* the regenerated figure differs from the current one **by the
mass-definition conversion**, and the claim carries its caveats
automatically.

**Slice 2 —** `lit-scout` + store pre-check.
**Slice 3 —** `model-runner` + run registry; migrate existing sweeps.
**Slice 4 —** `analyst` hardening (handle-only figure API), question register.

## Testing

- Kernel: incompatible definitions must raise; conversions round-trip.
- MCP server: invalid documents rejected (missing source, blank
  definition field, unverified source).
- Harmonise CLI: numerical agreement with COLOSSUS within tolerance.
- Derivations: deterministic — identical inputs give an identical figure
  hash.
- Referee: fixture claims with known defects (circular input, dropped
  caveat) must be caught.

## Risks

- **Mongo daemon availability.** Document a `docker-compose.yml`; the
  apparatus must fail loudly, never silently fall back to unvalidated
  writes.
- **MCP server is a new failure point.** Keep it thin; no analysis logic.
- **M500↔Mvir is real physics.** Must be independently validated before
  any claim depends on it.
- **Agents attempting bypass.** No filesystem write access to the store;
  the referee checks provenance completeness.
- **Over-formalisation.** If the harness makes exploratory work slower
  than it makes it safer, it has failed. Slice 1 is the test of that.

## Out of scope

Web UI; multi-user access control; distributed execution; a generic
plugin system; migrating historical results (only what Slice 1 needs).
